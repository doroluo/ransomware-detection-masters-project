"""python -m pytest tests/test_queue.py: the experiment queue (`experiments/queue.py`) and the plan files it runs.

Three parts:
  * the plan files of `experiments/plans` against `fixtures/plans_dry/<plan>.txt`, the `run` lines that `--dry` printed when the
    plan was written (model1_variants: the fivechannel lines of the plan of that name; team_b: the 15 sig500v3-e2e runs). The
    queue's `--dry` has to print the same run names with the same arguments, in the same order;
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
PLANS = ['model1_variants', 'team_b']
FIVECHANNEL = 'train --protocol kfold --fold {} --arch x86 --seed 0 --model fivechannel'


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
def test_dry_run_prints_the_runs_of_the_recording(plan):
    assert [l for l in dry_lines(plan) if l.startswith('run ')] == recorded(plan)


def test_the_recordings_have_the_run_counts_of_the_experiments():
    counts = {p: sum(l.startswith('run ') for l in recorded(p)) for p in PLANS}
    assert counts == {'model1_variants': 30, 'team_b': 15}


@pytest.mark.parametrize('plan', PLANS)
def test_run_names_are_the_run_config_names(plan):
    from experiments.train import parse_args
    for step in q.load_plan(plan):
        if step.verb == 'train':
            cfg, out, _ = parse_args(list(step.args))
            assert step.label == cfg.name and out is None and step.done_files == (f'{cfg.name}/metrics.json',)


def test_a_plan_ends_in_the_summaries_of_its_stage():
    assert [s.verb for s in q.load_plan('model1_variants')][-2:] == ['report', 'gates']
    assert [s.stdout for s in q.load_plan('model1_variants') if s.verb == 'gates'] == ['gates_model1_variants.md']


def test_plans_name_no_machine_path():
    for path in PLAN_FILES:
        text = path.read_text(encoding='utf-8')
        assert not re.search(r'[A-Za-z]:[/\\]|/Users/|/home/', text), path.name


# the dry run, the budget guard ------------------------------------------------------------------------------------------

def test_dry_run_charges_minutes_per_train_line_and_keeps_the_cpu_lines():
    lines = dry_lines('model1_variants', budget_min=290, dry_min=100)
    runs = [l for l in lines if l.startswith('run ')]
    skipped = [l for l in lines if l.startswith('skipped (budget) ')]
    assert len(runs) == 3 and len(skipped) == 27                          # 100 + 100 + 100 minutes pass 290, then nothing runs
    assert lines[-2:] == ['cpu report', 'cpu gates model1_variants > <runs>/gates_model1_variants.md']
    assert not any(l.startswith('skipped') for l in dry_lines('model1_variants', budget_min=290))
    assert dry_lines('model1_variants', budget_min=0, dry_min=1)[0].startswith('skipped (budget) kfold0_x86_fivechannel_s0')


def test_dry_run_writes_nothing_and_skips_nothing(tmp_path):
    (tmp_path / 'kfold0_x86_fivechannel_s0').mkdir()
    (tmp_path / 'kfold0_x86_fivechannel_s0' / 'metrics.json').write_text('{}')
    steps = q.parse_plan(FIVECHANNEL.format(0) + '\nreport\n')
    out = io.StringIO()
    assert q.Queue(steps, tmp_path, dry=True, out=out).run() == 0
    assert out.getvalue().splitlines()[0].startswith('run kfold0_x86_fivechannel_s0 ')
    assert sorted(p.name for p in tmp_path.iterdir()) == ['kfold0_x86_fivechannel_s0']


# the grammar --------------------------------------------------------------------------------------------------------------

def test_plan_grammar():
    steps = q.parse_plan('# a comment\n\ntrain --protocol kfold --fold 1 --model fivechannel --out smoke_a\nreport\n'
                         'gates team\ngates model1_variants --out variants\nval_threshold\n')
    assert [s.verb for s in steps] == ['train', 'report', 'gates', 'gates', 'val_threshold']
    assert steps[0].label == 'smoke_a' and steps[0].done_files == ('smoke_a/metrics.json',)
    assert steps[2].stdout == 'gates_team.md' and steps[3].stdout == 'gates_variants.md'
    assert steps[3].module_args == ('--stage', 'model1_variants') and steps[4].stdout == 'val_threshold.md'


@pytest.mark.parametrize('text, message', [
    ('', 'no commands'), ('# only a comment\n', 'no commands'),
    ('python -m experiments.train\n', "unknown command 'python'"),
    ('train --no-such-flag 1\n', 'not accepted by experiments.train'),
    ('train --protocol kfold --fold 0 --model fivechannel --control bogus\n', 'not accepted by experiments.train'),
    ('train --protocol kfold --fold 0\n', 'not accepted by experiments.train'),                    # --model is required
    ('train --protocol kfold --fold 0 --model gnn\n', 'not accepted by experiments.train'),         # on a branch, not here
    ('baselines --arch nowhere\n', "unknown command 'baselines'"),
    ('report now\n', 'takes no arguments'), ('val_threshold x\n', 'takes no arguments'),
    ('gates\n', 'usage'), ('gates team extra\n', 'usage'), ('gates v4\n', "no gates stage 'v4'"),
])
def test_a_bad_plan_is_refused_with_its_line_number(text, message):
    with pytest.raises(q.PlanError, match=message) as e:
        q.parse_plan('# first\n' + text if text else text, 'p.txt')
    if text.strip() and not text.startswith('#'):
        assert 'p.txt:2' in str(e.value)


# the loop, through real subprocesses ------------------------------------------------------------------------------------

LINES = ['train --out stub_a --protocol kfold --fold 0 --arch x86 --seed 0 --model fivechannel',
         'train --out stub_b --protocol kfold --fold 1 --arch x86 --seed 0 --model fivechannel',
         'train --out stub_c --protocol kfold --fold 2 --arch x86 --seed 0 --model fivechannel',
         'report', 'gates team', 'val_threshold']


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
    assert [l.split()[0] for l in out].count('start') == 6 and out[-1].startswith('queue done')
    assert [c['name'] for c in calls(runs)] == ['stub_a', 'stub_b', 'stub_c', 's04_evaluation.report', 's04_evaluation.gates',
                                                  's04_evaluation.val_threshold']
    assert 'stub experiments.train stub_a' in (runs / 'logs' / 'stub_a.log').read_text()
    assert 'stub stderr' in (runs / 'logs' / 'stub_a.log').read_text()                  # stderr goes to the same log
    assert '# gates team' in (runs / 'gates_team.md').read_text().splitlines()      # gates: stdout is the table
    assert 'stub stderr' in (runs / 'logs' / 'gates_team.log').read_text()             # ... stderr the log
    assert 'stub' in (runs / 'val_threshold.md').read_text() and (runs / 'logs' / 'report.log').is_file()
    assert re.fullmatch(r'start stub_a \d\d:\d\d:\d\d', out[0]) and re.match(r'finished stub_a rc=0 \d\d:\d\d:\d\d gpu_min=0', out[1])


def test_every_child_runs_with_expandable_segments_and_unbuffered_output(tmp_path, monkeypatch):
    rc, _, runs = run_queue(tmp_path, env={'PYTORCH_CUDA_ALLOC_CONF': 'something_else', 'GT_RUNS': 'ignored'}, monkeypatch=monkeypatch)
    assert rc == 0 and len(calls(runs)) == 6
    assert {c['alloc'] for c in calls(runs)} == {'expandable_segments:True'} and {c['unbuffered'] for c in calls(runs)} == {'1'}


def test_a_finished_run_is_skipped_on_rerun_and_its_log_is_kept(tmp_path):
    run_queue(tmp_path)
    log = (tmp_path / 'runs' / 'logs' / 'stub_a.log').read_text()
    before = len(calls(tmp_path / 'runs'))
    rc, out, runs = run_queue(tmp_path)
    assert rc == 0 and out[:3] == ['skip stub_a', 'skip stub_b', 'skip stub_c']
    assert (runs / 'logs' / 'stub_a.log').read_text() == log
    assert len(calls(runs)) - before == 3                                   # the three summaries run again


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
    assert 'stopped before stub_c' in out
    assert not (runs / 'stub_c').exists()
    assert [c['name'] for c in calls(runs)][-3:] == ['s04_evaluation.report', 's04_evaluation.gates', 's04_evaluation.val_threshold']
    assert (runs / 'gates_team.md').is_file() and out[-1].startswith('queue done')


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
    assert (runs / 'stub_c' / 'metrics.json').is_file() and (runs / 'gates_team.md').is_file()
    assert out[-1].endswith('failed: stub_b') and 'finished stub_b rc=1' in ' '.join(out)


def test_the_budget_counts_train_lines_and_the_summaries_still_run(tmp_path):
    rc, out, runs = run_queue(tmp_path, budget_min=0)
    assert rc == 0 and out[:3] == ['skipped (budget) stub_a', 'skipped (budget) stub_b', 'skipped (budget) stub_c']
    assert [c['name'] for c in calls(runs)] == ['s04_evaluation.report', 's04_evaluation.gates', 's04_evaluation.val_threshold']
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
    r = cli('--plan', 'model1_variants', '--dry')
    assert r.returncode == 0 and r.stdout.splitlines()[0].startswith('run kfold0_x86_fivechannel_s0 --model fivechannel')
    assert r.stdout.count('\nrun ') == 29 and r.stdout.splitlines()[-1] == 'cpu gates model1_variants > <runs>/gates_model1_variants.md'
    assert cli('--plan', str(q.PLANS / 'model1_variants.txt'), '--dry').stdout == r.stdout


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
    plan.write_text('# two lines\ntrain --out stub_a --protocol kfold --fold 0 --model fivechannel\nreport\n')
    runs = tmp_path / 'runs'
    r = cli('--plan', str(plan), '--runs', str(runs), '--python', f'{Path(sys.executable).as_posix()} {STUB.as_posix()}')
    assert r.returncode == 0, r.stderr
    assert (runs / 'stub_a' / 'metrics.json').is_file() and 'queue done' in r.stdout


# the train_b verb -------------------------------------------------------------------------------------------------------------

TRAIN_B = ['train_b --fold 0 --seed 0 --arm sig500v3-emb', 'train_b --fold 1 --seed 2 --arm sigwhole-feat --build RUNS/team/b_sigwhole_v3']


def test_train_b_lines_are_named_like_the_runs_they_write():
    from experiments.team_pipelines import common as team
    steps = q.parse_plan('\n'.join(TRAIN_B) + '\n')
    assert [s.label for s in steps] == ['kfold0_x86_gnn_s0_team-sig500v3-emb', 'kfold1_x86_gnn_s2_team-sigwhole-feat']
    assert [s.label for s in steps] == [team.team_config(0, 0, 'sig500v3-emb').name, team.team_config(1, 2, 'sigwhole-feat').name]
    assert [s.label for s in steps] == [RunConfig(protocol='kfold', fold=f, arch='x86', seed=s, model='gnn', tag=f'team-{a}').name
                                        for f, s, a in ((0, 0, 'sig500v3-emb'), (1, 2, 'sigwhole-feat'))]
    assert steps[0].module == 'experiments.team_pipelines.train_b' and steps[1].module_args[-2:] == ('--build', 'RUNS/team/b_sigwhole_v3')
    assert steps[0].done_files == ('kfold0_x86_gnn_s0_team-sig500v3-emb/metrics.json',)
    assert all(s.verb in q.TRAIN for s in steps)


@pytest.mark.parametrize('text, message', [
    ('train_b --seed 0 --arm sig500v3\n', 'required'), ('train_b --fold 0 --seed 0\n', 'required'),
    ('train_b --fold 0 --arm nonesuch\n', 'invalid choice'), ('train_b --fold x --arm sig500v3\n', 'invalid int'),
    ('train_b --fold 0 --arm sig500v3 --smoke 4\n', 'unrecognized'),
])
def test_a_bad_train_b_line_is_refused_with_its_line_number(text, message):
    with pytest.raises(q.PlanError, match=message) as e:
        q.parse_plan('# first\n' + text, 'p.txt')
    assert 'p.txt:2' in str(e.value)


def test_dry_run_prints_the_two_run_names_of_a_train_b_plan(tmp_path):
    plan = tmp_path / 'two.txt'
    plan.write_text('\n'.join(TRAIN_B) + '\n', encoding='utf-8')
    r = cli('--plan', str(plan), '--dry')
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [
        'run kfold0_x86_gnn_s0_team-sig500v3-emb train_b --fold 0 --seed 0 --arm sig500v3-emb',
        'run kfold1_x86_gnn_s2_team-sigwhole-feat train_b --fold 1 --seed 2 --arm sigwhole-feat --build RUNS/team/b_sigwhole_v3']


def test_train_b_lines_are_charged_to_the_budget_and_skipped_or_stopped_like_train_lines(tmp_path):
    lines = [*TRAIN_B, 'train --out stub_a --protocol kfold --fold 0 --model fivechannel']
    out = io.StringIO()
    assert q.Queue(q.parse_plan('\n'.join(lines) + '\n'), tmp_path, dry=True, dry_min=100, budget_min=150, out=out).run() == 0
    said = out.getvalue().splitlines()
    assert said[0].startswith('run kfold0_x86_gnn_s0_team-sig500v3-emb train_b ') and said[1].startswith('run kfold1_x86_gnn_s2_')
    assert said[2] == 'skipped (budget) stub_a'                                    # the two train_b lines were charged 200 minutes
    done = tmp_path / 'kfold0_x86_gnn_s0_team-sig500v3-emb'
    done.mkdir()
    (done / 'metrics.json').write_text('{}')
    (tmp_path / 'logs').mkdir()
    ran = []

    class StopAfterOne(q.Queue):
        def launch(self, command, stdout_path, stderr_path):
            ran.append(command[-1])
            (self.runs / 'STOP').touch()                                                 # as `queue --stop` does while a run is going
            return 0
    out = io.StringIO()
    assert StopAfterOne(q.parse_plan('\n'.join(lines) + '\n'), tmp_path, out=out).run() == 0
    said = out.getvalue().splitlines()
    assert said[0] == 'skip kfold0_x86_gnn_s0_team-sig500v3-emb' and ran == ['RUNS/team/b_sigwhole_v3']     # the finished line is skipped
    assert said[1].startswith('start kfold1_x86_gnn_s2_team-sigwhole-feat') and 'stopped before stub_a' in said


def test_a_train_b_line_runs_its_module_with_its_arguments_and_counts_toward_the_budget(tmp_path):
    started = []

    class Fake(q.Queue):
        def launch(self, command, stdout_path, stderr_path):
            started.append(command)
            name = (f"kfold{command[command.index('--fold') + 1]}_x86_gnn_s{command[command.index('--seed') + 1]}_team-"
                    f"{command[command.index('--arm') + 1]}")
            (self.runs / name).mkdir(parents=True, exist_ok=True)
            (self.runs / name / 'metrics.json').write_text('{}')
            return 0
    out = io.StringIO()
    runs = tmp_path / 'runs'
    queue = Fake(q.parse_plan(TRAIN_B[0] + '\n' + TRAIN_B[0].replace('--fold 0', '--fold 3') + '\n'), runs, python=['py'], out=out)
    assert queue.run() == 0
    assert [c[:3] for c in started] == [['py', '-m', 'experiments.team_pipelines.train_b']] * 2
    assert started[0][3:] == ['--fold', '0', '--seed', '0', '--arm', 'sig500v3-emb']
    said = out.getvalue().splitlines()
    assert any(l.startswith('finished kfold0_x86_gnn_s0_team-sig500v3-emb rc=0') and 'gpu_min=' in l for l in said)
    assert (runs / 'kfold3_x86_gnn_s0_team-sig500v3-emb' / 'metrics.json').is_file() 
    out = io.StringIO()
    again = Fake(q.parse_plan(TRAIN_B[0] + '\n'), runs, python=['py'], out=out)
    assert again.run() == 0 and out.getvalue().splitlines()[0] == 'skip kfold0_x86_gnn_s0_team-sig500v3-emb' and len(started) == 2
