"""python -m pytest tests/test_queue.py: the experiment queue (`experiments/queue.py`) and the plan files it runs.

Three parts:
  * the plan files of `experiments/plans` against `fixtures/plans_dry/<plan>.txt`, the `run` and `cpu baselines` lines that the
    shell queues of the 3-block-level-counting branch printed with `--dry` (e48efb7: queue_gnn.sh --stage <stage> --dry,
    scripts/noapi_runs.sh, model1_runs.sh, team_gpu_runs.sh; l2_nocall_seeds and r2_noapi have no --dry and were taken from
    the two scripts of the run host with `run` replaced by `echo`; model1_variants has no source script: its recording is the `--dry`
    output when the plan was written). The queue's `--dry` has to print the same run names with the same arguments, in the same order;
  * the plan grammar;
  * the queue loop, through the real subprocess launch with `queue_stub.py` in the place of the interpreter: skip, STOP, budget,
    retry, logs, exit codes.
"""
import io
import json
import re
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

from common.config import RunConfig
from experiments import queue as q

HERE = Path(__file__).resolve().parent
PLAN_FILES = sorted(q.PLANS.glob('*.txt'))
FIXTURES = HERE / 'fixtures' / 'plans_dry'
STUB = HERE / 'queue_stub.py'
PLANS = ['day1', 'day1b', 'v3', 'subgraph', 'paths', 'paths_codebert', 'vocab', 'imports', 'imports_header', 'noapi', 'team',
         'model1', 'model1_variants', 'l2_nocall_seeds', 'r2_noapi']


def dry_lines(plan, **kw):
    out = io.StringIO()
    queue = q.Queue(q.load_plan(plan), Path('unused'), dry=True, out=out, **kw)
    assert queue.run() == 0
    return out.getvalue().splitlines()


def recorded(plan):
    return (FIXTURES / f'{plan}.txt').read_text(encoding='utf-8').splitlines()


def test_every_plan_has_a_recording_and_the_other_way_round():
    assert [p.stem for p in PLAN_FILES] == sorted(PLANS)
    assert sorted(p.stem for p in FIXTURES.glob('*.txt')) == sorted(PLANS)


@pytest.mark.parametrize('plan', PLANS)
def test_dry_run_prints_the_runs_of_the_source_queue(plan):
    got = [l for l in dry_lines(plan) if l.startswith(('run ', 'cpu baselines '))]
    want = recorded(plan)
    if plan == 'model1':        # the shell script called token_sequence.train; the one CLI takes the model by name
        got = [l.replace(' --model tokenseq', '', 1) for l in got]
    assert got == want


def test_the_recordings_have_the_run_counts_of_the_experiments():
    counts = {p: sum(l.startswith('run ') for l in recorded(p)) for p in PLANS}
    assert counts == {'day1': 25, 'day1b': 65, 'v3': 45, 'subgraph': 25, 'paths': 7, 'paths_codebert': 30, 'vocab': 10,
                      'imports': 0, 'imports_header': 5, 'noapi': 15, 'team': 30, 'model1': 15, 'model1_variants': 45, 'l2_nocall_seeds': 20,
                      'r2_noapi': 15}


@pytest.mark.parametrize('plan', PLANS)
def test_run_names_are_the_run_config_names(plan):
    from experiments.train import parse_args
    for step in q.load_plan(plan):
        if step.verb == 'train':
            cfg, out, _ = parse_args(list(step.args))
            assert step.label == cfg.name and out is None and step.done_files == (f'{cfg.name}/metrics.json',)


def test_a_plan_ends_in_the_summaries_of_its_source_queue():
    verbs = {p: [s.verb for s in q.load_plan(p)] for p in PLANS}
    for plan in ('day1', 'day1b', 'v3', 'subgraph', 'paths', 'paths_codebert', 'vocab', 'imports', 'imports_header'):
        assert verbs[plan][-3:] == ['report', 'gates', 'val_threshold']
    assert verbs['noapi'][-1] == 'gates' and verbs['model1'][-2:] == ['report', 'gates']
    assert verbs['model1_variants'][-2:] == ['report', 'gates']
    gates = {p: [s.stdout for s in q.load_plan(p) if s.verb == 'gates'] for p in PLANS}
    assert gates['imports_header'] == ['gates_imports_header.md'] and gates['imports'] == ['gates_imports.md']
    assert gates['team'] == ['gates_team.md'] and gates['noapi'] == ['gates_noapi.md']
    assert gates['model1_variants'] == ['gates_model1_variants.md']


def test_plans_name_no_machine_path():
    for path in PLAN_FILES:
        text = path.read_text(encoding='utf-8')
        assert not re.search(r'[A-Za-z]:[/\\]|/Users/|/home/', text), path.name


# the dry run, the budget guard ------------------------------------------------------------------------------------------

def test_dry_run_charges_minutes_per_train_line_and_keeps_the_cpu_lines():
    lines = dry_lines('day1', budget_min=290, dry_min=100)
    runs = [l for l in lines if l.startswith('run ')]
    skipped = [l for l in lines if l.startswith('skipped (budget) ')]
    assert len(runs) == 3 and len(skipped) == 22                          # 100 + 100 + 100 minutes pass 290, then nothing runs
    assert lines[0].startswith('cpu baselines') and lines[-3:] == ['cpu report', 'cpu gates day1 > <runs>/gates_day1.md',
                                                                  'cpu val_threshold > <runs>/val_threshold.md']
    assert not any(l.startswith('skipped') for l in dry_lines('day1', budget_min=290))
    assert dry_lines('day1', budget_min=0, dry_min=1)[1].startswith('skipped (budget) kfold0_x86_gnn-noedges_s0')


def test_dry_run_writes_nothing_and_skips_nothing(tmp_path):
    (tmp_path / 'kfold0_x86_gnn_s0').mkdir()
    (tmp_path / 'kfold0_x86_gnn_s0' / 'metrics.json').write_text('{}')
    steps = q.parse_plan('train --protocol kfold --fold 0 --arch x86 --seed 0 --model gnn\nreport\n')
    out = io.StringIO()
    assert q.Queue(steps, tmp_path, dry=True, out=out).run() == 0
    assert out.getvalue().splitlines()[0].startswith('run kfold0_x86_gnn_s0 ')
    assert sorted(p.name for p in tmp_path.iterdir()) == ['kfold0_x86_gnn_s0']


# the grammar --------------------------------------------------------------------------------------------------------------

def test_plan_grammar():
    steps = q.parse_plan('# a comment\n\ntrain --protocol kfold --fold 1 --model gnn --out smoke_a\nreport\n'
                         'gates noapi\ngates imports --out imports_header\nval_threshold\n')
    assert [s.verb for s in steps] == ['train', 'report', 'gates', 'gates', 'val_threshold']
    assert steps[0].label == 'smoke_a' and steps[0].done_files == ('smoke_a/metrics.json',)
    assert steps[2].stdout == 'gates_noapi.md' and steps[3].stdout == 'gates_imports_header.md'
    assert steps[3].module_args == ('--stage', 'imports') and steps[4].stdout == 'val_threshold.md'


@pytest.mark.parametrize('text, message', [
    ('', 'no commands'), ('# only a comment\n', 'no commands'),
    ('python -m experiments.train\n', "unknown command 'python'"),
    ('train --no-such-flag 1\n', 'not accepted by experiments.train'),
    ('train --protocol kfold --fold 0 --control bogus\n', 'not accepted by experiments.train'),
    ('baselines --arch nowhere\n', 'not accepted by s04_evaluation.baselines'),
    ('report now\n', 'takes no arguments'), ('val_threshold x\n', 'takes no arguments'),
    ('gates\n', 'usage'), ('gates noapi extra\n', 'usage'), ('gates v4\n', "no gates stage 'v4'"),
])
def test_a_bad_plan_is_refused_with_its_line_number(text, message):
    with pytest.raises(q.PlanError, match=message) as e:
        q.parse_plan('# first\n' + text if text else text, 'p.txt')
    if text.strip() and not text.startswith('#'):
        assert 'p.txt:2' in str(e.value)


def test_baselines_lines_name_the_directories_the_baselines_write():
    from s04_evaluation import baselines
    cases = {'--protocol kfold --fold 0 1 --arch x86 --seed 0': ['kfold0_x86_s0', 'kfold1_x86_s0'],
             '--protocol kfold --fold 0 --arch x86 --seed 0 --train-fraction 0.5': ['kfold0_x86_s0_tf50'],
             '--protocol kfold_v3 --fold 2 --arch all --seed 0 --arch-balance': ['v3kfold2_all_s0_ab'],
             '--protocol kfold --fold 3 --arch x86 --seed 1 --vocab noapi': ['kfold3_x86_s1_voc-noapi'],
             '--protocol kfold --fold 4 --arch x86 --seed 0 --imports': ['kfold4_x86_s0_imp'],
             '--protocol kfold --fold 0 --arch x86 --seed 0 --stack-logits': ['kfold0_x86_s0']}
    for args, dirs in cases.items():
        step = q.parse_line(f'baselines {args}')
        marker = 'stack_logits.csv' if '--stack-logits' in args else 'metrics.json'
        assert step.done_files == tuple(f'baselines/{d}/{marker}' for d in dirs), args
        assert step.label.startswith('baselines_') and 'kfold' in step.label
    cfg = RunConfig(protocol='kfold', fold=1, arch='x86', seed=0, train_fraction=0.25)
    assert baselines.output_name(cfg) == 'kfold1_x86_s0_tf25'


# the loop, through real subprocesses ------------------------------------------------------------------------------------

LINES = ['train --out stub_a --protocol kfold --fold 0 --arch x86 --seed 0 --model gnn',
         'train --out stub_b --protocol kfold --fold 1 --arch x86 --seed 0 --model gnn',
         'train --out stub_c --protocol kfold --fold 2 --arch x86 --seed 0 --model gnn',
         'baselines --protocol kfold --fold 0 --arch x86 --seed 0',
         'report', 'gates noapi', 'val_threshold']


def run_queue(tmp_path, lines=LINES, env=None, monkeypatch=None, **kw):
    """(exit code, stdout lines, runs directory) of a queue over `lines` with the stub for interpreter."""
    runs = tmp_path / 'runs'
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    out = io.StringIO()
    queue = q.Queue(q.parse_plan('\n'.join(lines) + '\n'), runs, python=[sys.executable, str(STUB)], out=out, **kw)
    rc = queue.run()
    return rc, out.getvalue().splitlines(), runs


def calls(runs):
    path = runs / 'calls.log'
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def test_a_run_writes_its_log_and_the_summaries_go_where_the_source_queue_put_them(tmp_path):
    rc, out, runs = run_queue(tmp_path)
    assert rc == 0
    assert [l.split()[0] for l in out].count('start') == 7 and out[-1].startswith('queue done')
    assert [c['name'] for c in calls(runs)] == ['stub_a', 'stub_b', 'stub_c', 's04_evaluation.baselines', 's04_evaluation.report',
                                                  's04_evaluation.gates', 's04_evaluation.val_threshold']
    assert 'stub experiments.train stub_a' in (runs / 'logs' / 'stub_a.log').read_text()
    assert 'stub stderr' in (runs / 'logs' / 'stub_a.log').read_text()                  # stderr goes to the same log
    assert '# gates noapi' in (runs / 'gates_noapi.md').read_text().splitlines()      # gates: stdout is the table
    assert 'stub stderr' in (runs / 'logs' / 'gates_noapi.log').read_text()             # ... stderr the log
    assert 'stub' in (runs / 'val_threshold.md').read_text() and (runs / 'logs' / 'report.log').is_file()
    assert re.fullmatch(r'start stub_a \d\d:\d\d:\d\d', out[0]) and re.match(r'finished stub_a rc=0 \d\d:\d\d:\d\d gpu_min=0', out[1])


def test_every_child_runs_with_expandable_segments_and_unbuffered_output(tmp_path, monkeypatch):
    rc, _, runs = run_queue(tmp_path, env={'PYTORCH_CUDA_ALLOC_CONF': 'something_else', 'GT_RUNS': 'ignored'}, monkeypatch=monkeypatch)
    assert rc == 0 and len(calls(runs)) == 7
    assert {c['alloc'] for c in calls(runs)} == {'expandable_segments:True'} and {c['unbuffered'] for c in calls(runs)} == {'1'}


def test_a_finished_run_is_skipped_on_rerun_and_its_log_is_kept(tmp_path):
    run_queue(tmp_path)
    log = (tmp_path / 'runs' / 'logs' / 'stub_a.log').read_text()
    before = len(calls(tmp_path / 'runs'))
    rc, out, runs = run_queue(tmp_path)
    assert rc == 0 and out[:3] == ['skip stub_a', 'skip stub_b', 'skip stub_c']
    assert (runs / 'logs' / 'stub_a.log').read_text() == log
    assert len(calls(runs)) - before == 4                                   # the baselines line and the three summaries run again


def test_a_baselines_line_is_skipped_when_every_directory_holds_a_validation_threshold(tmp_path):
    d = tmp_path / 'runs' / 'baselines' / 'kfold0_x86_s0'
    d.mkdir(parents=True)
    (d / 'metrics.json').write_text('{"methods": {}}')
    _, out, _ = run_queue(tmp_path)
    assert 'start baselines_kfold_x86_s0' in ' '.join(out)                  # no threshold_val: it runs
    (d / 'metrics.json').write_text('{"methods": {"m": {"threshold_val": 0.5}}}')
    _, out, runs = run_queue(tmp_path)
    assert 'skip baselines_kfold_x86_s0' in out
    stack = q.parse_line('baselines --protocol kfold --fold 0 --arch x86 --seed 0 --stack-logits')
    queue = q.Queue([stack], runs)
    assert not queue.finished(stack)
    (d / 'stack_logits.csv').write_text('x')
    assert queue.finished(stack)


def test_a_stale_stop_file_is_cleared_and_logged_at_the_start(tmp_path):
    runs = tmp_path / 'runs'
    runs.mkdir()
    (runs / 'STOP').touch()
    rc, out, _ = run_queue(tmp_path)
    assert rc == 0 and out[0] == f'cleared stale {runs / "STOP"}' and not any(l.startswith('stopped') for l in out)
    assert (runs / 'stub_c' / 'metrics.json').is_file()


def test_the_stop_file_during_a_run_stops_before_the_next_work_line_but_not_the_summaries(tmp_path, monkeypatch):
    rc, out, runs = run_queue(tmp_path, env={'STUB_STOP_AT': 'stub_b'}, monkeypatch=monkeypatch)
    assert rc == 0
    assert 'stopped before stub_c' in out and 'stopped before baselines_kfold_x86_s0' in out
    assert not (runs / 'stub_c').exists()
    assert [c['name'] for c in calls(runs)][-3:] == ['s04_evaluation.report', 's04_evaluation.gates', 's04_evaluation.val_threshold']
    assert (runs / 'gates_noapi.md').is_file() and out[-1].startswith('queue done')


def test_a_custom_stop_file_is_honoured(tmp_path):
    stop = tmp_path / 'elsewhere' / 'STOP_ME'
    stop.parent.mkdir()
    stop.touch()
    rc, out, runs = run_queue(tmp_path, stop_file=stop)
    assert rc == 0 and out[0] == f'cleared stale {stop}' and (runs / 'stub_c' / 'metrics.json').is_file()


def test_no_retry_after_stop_and_a_kill_leaves_no_metrics(tmp_path, monkeypatch):
    rc, out, runs = run_queue(tmp_path, env={'STUB_KILL': 'stub_b'}, monkeypatch=monkeypatch)
    assert rc == 1 and 'retry stub_b' not in out
    assert not (runs / 'stub_b' / 'metrics.json').exists() and 'stopped before stub_c' in out
    assert [c['name'] for c in calls(runs)].count('stub_b') == 1
    assert 'finished stub_b rc=1' in ' '.join(out) and 'failed: stub_b' in out[-1]


def test_a_failed_run_is_retried_once_and_the_first_log_is_kept(tmp_path, monkeypatch):
    rc, out, runs = run_queue(tmp_path, env={'STUB_FAIL_ONCE': 'stub_b'}, monkeypatch=monkeypatch)
    assert rc == 0 and 'retry stub_b' in out and out.index('retry stub_b') < out.index(next(l for l in out if l.startswith('finished stub_b')))
    assert (runs / 'logs' / 'stub_b.log.try1').is_file() and 'stub experiments.train stub_b' in (runs / 'logs' / 'stub_b.log').read_text()
    assert [c['name'] for c in calls(runs)].count('stub_b') == 2 and (runs / 'stub_b' / 'metrics.json').is_file()


def test_a_run_that_fails_after_its_retry_fails_the_queue_but_the_queue_goes_on(tmp_path, monkeypatch):
    rc, out, runs = run_queue(tmp_path, env={'STUB_FAIL_ALWAYS': 'stub_b'}, monkeypatch=monkeypatch)
    assert rc == 1
    assert [c['name'] for c in calls(runs)].count('stub_b') == 2             # one attempt and one retry
    assert (runs / 'stub_c' / 'metrics.json').is_file() and (runs / 'gates_noapi.md').is_file()
    assert out[-1].endswith('failed: stub_b') and 'finished stub_b rc=1' in ' '.join(out)


def test_the_budget_counts_train_lines_and_the_summaries_still_run(tmp_path):
    rc, out, runs = run_queue(tmp_path, budget_min=0)
    assert rc == 0 and out[:3] == ['skipped (budget) stub_a', 'skipped (budget) stub_b', 'skipped (budget) stub_c']
    assert [c['name'] for c in calls(runs)] == ['s04_evaluation.baselines', 's04_evaluation.report', 's04_evaluation.gates',
                                                  's04_evaluation.val_threshold']
    rc, out, runs = run_queue(tmp_path / 'b', budget_min=60)
    assert rc == 0 and not any('budget' in l for l in out) and out[1].endswith('gpu_min=0')


def test_the_budget_stops_the_train_lines_after_the_minutes_are_spent(tmp_path, monkeypatch):
    clock = iter(range(0, 10_000, 40))                                      # each call of monotonic is 40 s later
    monkeypatch.setattr(q, 'time', types.SimpleNamespace(monotonic=lambda: float(next(clock)), strftime=time.strftime))
    rc, out, runs = run_queue(tmp_path, budget_min=1)                      # one run takes 40 s: stub_c starts at 80 s > 60 s
    assert rc == 0 and out[1].endswith('gpu_min=0') and out[3].endswith('gpu_min=1') and 'skipped (budget) stub_c' in out
    assert not (runs / 'stub_c').exists()


# the command line ---------------------------------------------------------------------------------------------------------

def cli(*args, **kw):
    return subprocess.run([sys.executable, '-m', 'experiments.queue', *args], cwd=q.PIPELINE_ROOT, capture_output=True,
                          text=True, **kw)


def test_cli_dry_run_and_plan_by_name():
    r = cli('--plan', 'noapi', '--dry')
    assert r.returncode == 0 and r.stdout.splitlines()[0].startswith('cpu baselines --protocol kfold --fold 0 1 2 3 4 --arch x86 --vocab noapi')
    assert r.stdout.count('\nrun ') == 15 and r.stdout.splitlines()[-1] == 'cpu gates noapi > <runs>/gates_noapi.md'
    assert cli('--plan', str(q.PLANS / 'noapi.txt'), '--dry').stdout == r.stdout


def test_cli_refuses_a_bad_plan_with_exit_code_2(tmp_path):
    plan = tmp_path / 'bad.txt'
    plan.write_text('report\ngates nonesuch\n')
    r = cli('--plan', str(plan), '--dry')
    assert r.returncode == 2 and 'bad.txt:2' in r.stderr and "no gates stage 'nonesuch'" in r.stderr
    assert cli('--plan', str(tmp_path / 'missing.txt')).returncode == 2
    assert cli().returncode == 2                                           # argparse: --plan is required


def test_cli_stop_creates_the_stop_file(tmp_path):
    r = cli('--stop', '--runs', str(tmp_path / 'r'))
    assert r.returncode == 0 and (tmp_path / 'r' / 'STOP').is_file()
    assert cli('--stop', '--stop-file', str(tmp_path / 'x' / 'S')).returncode == 0 and (tmp_path / 'x' / 'S').is_file()


def test_cli_runs_a_plan_end_to_end_with_the_python_option(tmp_path):
    plan = tmp_path / 'p.txt'
    plan.write_text('# two lines\ntrain --out stub_a --protocol kfold --fold 0 --model gnn\nreport\n')
    runs = tmp_path / 'runs'
    r = cli('--plan', str(plan), '--runs', str(runs), '--python', f'{Path(sys.executable).as_posix()} {STUB.as_posix()}')
    assert r.returncode == 0, r.stderr
    assert (runs / 'stub_a' / 'metrics.json').is_file() and 'queue done' in r.stdout
