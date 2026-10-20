"""python -m pytest tests/test_gates_real.py -m real_data: the gates engine, `report.py` and `stats.py` on the real run directory.

Equivalence with the 3-block-level-counting branch (3764cc7): `tests/fixtures/gates/` holds what that branch's `gates.py --stage
<name>`, `report.py` and `stats.py` printed on the author's run directory (default 2000 resamples, UTF-8, line ends normalised);
the engine here must print the same text. `gates/recorded.json` holds the fingerprint of the run directory they were recorded on
(file names and sizes of the runs and baselines); on any other run directory the comparison is skipped, not failed.

The second half runs the gate stages on synthetic runs that sit beside the real TF-IDF baselines and the real fold manifests.
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from common.config import RUNS, RunConfig
from common.manifests import load_index, make_manifest
from common.run_io import write_run
from eval_synth import val_threshold_of
from s04_evaluation import gate_layouts, gate_tables, gates, report, stats

pytestmark = pytest.mark.real_data
if not RUNS.is_dir():
    pytest.skip('needs the run directory (GT_RUNS or datasets.json)', allow_module_level=True)
FIXTURES = Path(__file__).resolve().parent / 'fixtures' / 'gates'
RECORDED = json.loads((FIXTURES / 'recorded.json').read_text(encoding='utf-8'))
FILES = ('config.json', 'metrics.json', 'predictions.csv', 'predictions_auc.csv', 'subgraph_chain.json')
FOLDS = range(5)
needs_baselines = pytest.mark.skipif(not (RUNS / 'baselines' / 'kfold0_x86_s0' / 'predictions.csv').exists(),
                                     reason='needs the tfidf kfold baselines')
PAIRS = {'stats_r7_r2': ('kfold_x86_gnn-noedges', 'kfold_x86_gnn'),
         'stats_tfidf_r7': ('tfidf', 'kfold_x86_gnn-mblk15'),
         'stats_r7_noapi': ('kfold_x86_gnn-mblk15', 'kfold_x86_gnn-mblk15-voc-noapi')}


def runs_fingerprint(runs=RUNS):
    """sha256 of 'relative path size' for the files the gates read: the run directories and the baselines."""
    if not runs.is_dir():
        return None
    entries = []
    for top in sorted(p for p in runs.iterdir() if p.is_dir()):
        dirs = [top] + (sorted(p for p in top.iterdir() if p.is_dir()) if top.name.startswith('baselines') else [])
        for d in dirs:
            entries += [f'{f.relative_to(runs).as_posix()} {f.stat().st_size}' for f in sorted(d.iterdir()) if f.name in FILES]
    chain = runs / 'subgraph_chain.json'
    if chain.exists():
        entries.append(f'subgraph_chain.json {chain.stat().st_size}')
    return hashlib.sha256('\n'.join(entries).encode('utf-8')).hexdigest()


def same_runs_as_recorded():
    return runs_fingerprint() == RECORDED['runs_fingerprint']


needs_recorded_runs = pytest.mark.skipif(not same_runs_as_recorded(),
                                         reason='the run directory is not the one the expected outputs were recorded on')


def recorded(name):
    return (FIXTURES / f'{name}.md').read_bytes().decode('utf-8').replace('\r\n', '\n')


@pytest.fixture(autouse=True)
def fresh_caches():
    for cache in (gate_tables.pooled, gate_layouts.v3_cell, report.v3_floors):
        cache.cache_clear()
    yield
    for cache in (gate_tables.pooled, gate_layouts.v3_cell, report.v3_floors):
        cache.cache_clear()


# ---- equivalence with the source branch ---------------------------------------------------------------------------------

def test_the_recording_covers_every_stage_the_report_and_three_pairs():
    assert set(RECORDED['outputs']) == set(gates.stage_names()) | {'report'} | set(PAIRS)
    for name, digest in RECORDED['outputs'].items():
        assert hashlib.sha256(recorded(name).encode('utf-8')).hexdigest() == digest, name        # the files are the ones recorded


@needs_recorded_runs
@pytest.mark.parametrize('name', gates.stage_names())
def test_gates_stage_prints_what_the_source_branch_printed(name):
    got = '\n'.join(gates.run_stage(name)) + '\n'
    assert got == recorded(name)


@needs_recorded_runs
def test_report_prints_and_writes_what_the_source_branch_printed(tmp_path, capsys):
    report.main(['--out', str(tmp_path / 'RESULTS.md')])
    out = capsys.readouterr().out
    assert out == recorded('report')
    assert (tmp_path / 'RESULTS.md').read_text(encoding='utf-8') == out[:-1]                 # the file has no final newline


@needs_recorded_runs
@pytest.mark.parametrize('name', sorted(PAIRS))
def test_stats_prints_what_the_source_branch_printed(name):
    a, b = PAIRS[name]
    result, info = stats.compare(a, b, 'auc', 2000)
    assert stats.fmt_block(a, b, 'auc', 2000, result, info) + '\n' == recorded(name)


# ---- synthetic runs next to the real baselines and fold manifests --------------------------------------------------------

def fake_rows(manifest, rng, signal):
    out = {}
    for split in ('val', 'test'):
        rows = getattr(manifest, split)
        probs = np.clip(0.5 + signal * (np.array([r['label'] for r in rows]) - 0.5) + rng.normal(0, 0.2, len(rows)), 0.01, 0.99)
        out[split] = ([r['sample_id'] for r in rows], probs, [(10, 500, 700)] * len(rows))
    return out


def fake_group(runs, cfg_of, signal, seeds=(0, 1, 2)):
    """Synthetic five-fold runs for every seed through write_run: the probabilities follow the label by `signal`."""
    index = load_index()
    for seed in seeds:
        rng = np.random.default_rng(100 * seed + int(10 * signal))
        for fold in FOLDS:
            cfg = cfg_of(fold, seed)
            manifest = make_manifest(RunConfig(protocol='kfold', fold=fold, arch='x86', seed=seed), index)
            f1, auc = fake_rows(manifest, rng, signal), fake_rows(manifest, rng, signal)
            write_run(runs / cfg.name, cfg, f1, auc, val_threshold_of(manifest.val, *f1['val'][:2]),
                      val_threshold_of(manifest.val, *auc['val'][:2]),
                      {'best_epoch': 2, 'epochs_run': 7, 'auc_best_epoch': 3, 'val_auc': 0.9, 'val_macro_f1': 0.8})


def copy_baselines(runs):
    (runs / 'baselines').mkdir(parents=True)
    for f in FOLDS:
        shutil.copytree(RUNS / 'baselines' / f'kfold{f}_x86_s0', runs / 'baselines' / f'kfold{f}_x86_s0')


def team_config(fold, seed, arm):
    return RunConfig(protocol='kfold', fold=fold, arch='x86', seed=seed, model='gnn', tag=f'team-{arm}')


def table_cells(text, columns, labels):
    return {l.split('|')[1].strip(): [c.strip() for c in l.split('|')[2:-1]]
            for l in text.splitlines() if l.startswith('| ') and l.count('|') == columns + 1 and l.split('|')[1].strip() in labels}


@needs_baselines
def test_team_gates_from_synthetic_runs(tmp_path, monkeypatch):
    runs = tmp_path / 'runs'
    copy_baselines(runs)
    base = dict(protocol='kfold', arch='x86', model='gnn')
    fake_group(runs, lambda f, s: RunConfig(**base, fold=f, seed=s, mask_blocks=0.15), 0.5)                  # R7, 3 seeds
    fake_group(runs, lambda f, s: RunConfig(**base, fold=f, seed=s, mask_blocks=0.15, subgraph='cap', budget=500,
                                            tag='b500'), 0.5)                                                  # ours b500 typed
    fake_group(runs, lambda f, s: team_config(f, s, 'gnn500'), 0.4)                                            # weak
    fake_group(runs, lambda f, s: team_config(f, s, 'cb500'), 1.5)                                             # strong
    fake_group(runs, lambda f, s: team_config(f, s, 'sig500'), 0.7, seeds=(0, 1))                              # seed 2 missing
    fake_group(runs, lambda f, s: team_config(f, s, 'sig500v2'), 0.9)                                          # the v2 arm, 3 seeds
    monkeypatch.setattr(stats, 'RUNS', runs)
    text = '\n'.join(gates.run_stage('team', 60))
    team = gates.load_stage('team')
    arms = dict(team.arms)
    ops = table_cells(text, 11, arms)
    assert len(ops) == len(arms)
    for arm in ('TF-IDF', 'R7 typed, whole graph', 'ours b500 typed', 'team gnn500', 'team cb500', 'team sig500', 'team sig500v2'):
        auc, pr, f1, acc, prec, rec, tp_, fn, fp, tn = ops[arm]
        tp_, fn, fp, tn = int(tp_), int(fn), int(fp), int(tn)
        assert tp_ + fn == 1134 and fp + tn == 676                          # ransomware / goodware in the five test folds
        assert float(acc) == pytest.approx((tp_ + tn) / 1810, abs=6e-4)
        assert float(prec) == pytest.approx(tp_ / max(tp_ + fp, 1), abs=6e-4)
        assert float(rec) == pytest.approx(tp_ / 1134, abs=6e-4)
    assert ops['TF-IDF'][0] == f"{gate_tables.pooled('tfidf', 'auc')[0]:.3f}"
    assert ops['ours b500 no-edges'][0].startswith('missing') and ops['team gnn500'][0] != ops['team cb500'][0]
    pair = {(first, second): cells[-1] for first, second, cells in
            ((l.split('|')[1].strip(), l.split('|')[2].strip(), [c.strip() for c in l.split('|')[3:-1]])
             for l in text.splitlines() if l.startswith('| ') and l.count('|') == 6) if (first, second) in team.pairs}
    assert len(pair) == len(team.pairs)
    assert pair[('R7 typed, whole graph', 'team cb500')] == 'beats'                                       # 3 seeds in both
    assert pair[('R7 typed, whole graph', 'team gnn500')] == 'does not beat'
    tf_cb = pair[('TF-IDF', 'team cb500')]
    assert tf_cb.startswith(('beats', 'does not beat')) and 'seeds [0] only: a direction, not the test' in tf_cb
    assert pair[('R7 typed, whole graph', 'ours b500 typed')] in ('non-inferior', 'not shown non-inferior')
    assert 'seeds [0, 1] only' in pair[('R7 typed, whole graph', 'team sig500')]                          # the seeds complete in both
    assert pair[('path 2: typed, CodeBERT vectors', 'team cb500')] == 'missing'
    assert 'seeds [0, 1] only' in pair[('team sig500', 'team sig500v2')]                                  # sig500 has no seed 2
    assert pair[('R7 typed, whole graph', 'team sig500v2')] in ('beats', 'does not beat')                 # 3 seeds in both
    assert 'seeds [0] only' in pair[('TF-IDF', 'team sig500v2')]


@needs_baselines
def test_noapi_gates_from_synthetic_runs(tmp_path, monkeypatch):
    runs = tmp_path / 'runs'
    copy_baselines(runs)
    for f in FOLDS:
        noapi_dir = runs / 'baselines' / f'kfold{f}_x86_s0_voc-noapi'                    # a TF-IDF noapi that is the plain one renamed
        shutil.copytree(runs / 'baselines' / f'kfold{f}_x86_s0', noapi_dir)
        m = json.loads((noapi_dir / 'metrics.json').read_text(encoding='utf-8'))
        m['methods'] = {'tfidf_noapi_lr': m['methods']['tfidf_sig_api_lr']}
        (noapi_dir / 'metrics.json').write_text(json.dumps(m), encoding='utf-8')
        lines = (noapi_dir / 'predictions.csv').read_text(encoding='utf-8').splitlines()
        (noapi_dir / 'predictions.csv').write_text('\n'.join([lines[0]] + [l.replace('tfidf_sig_api_lr', 'tfidf_noapi_lr')
                                                                         for l in lines[1:] if 'tfidf_sig_api_lr' in l]),
                                                   encoding='utf-8')
    base = dict(protocol='kfold', arch='x86', model='gnn', mask_blocks=0.15)
    fake_group(runs, lambda f, s: RunConfig(**base, fold=f, seed=s), 1.2)                              # R7, 3 seeds
    fake_group(runs, lambda f, s: RunConfig(**base, fold=f, seed=s, vocab='noapi'), 0.2)               # R7 noapi, much weaker
    monkeypatch.setattr(stats, 'RUNS', runs)
    text = '\n'.join(gates.run_stage('noapi', 60))
    noapi = gates.load_stage('noapi')
    pair = {(l.split('|')[1].strip(), l.split('|')[2].strip()): l.split('|')[-2].strip()
            for l in text.splitlines() if l.startswith('| ') and l.count('|') == 6
            and (l.split('|')[1].strip(), l.split('|')[2].strip()) in noapi.pairs}
    assert len(pair) == 4
    assert pair[('R7 typed', 'R7 noapi')] == 'api loss shown'                                          # 3 seeds in both arms
    assert pair[('TF-IDF', 'TF-IDF noapi')].startswith('api loss not shown (seed 0 only: a direction, not the test)')
    assert pair[('TF-IDF noapi', 'R7 noapi')].endswith('(seed 0 only: a direction, not the test)')
    assert pair[('TF-IDF noapi', 'R7 noapi')].startswith('does not beat')
    assert pair[('header-imports LR', 'R7 noapi')] == 'missing'                                         # header-imports LR was not copied
    arms = dict(noapi.arms)
    pooled_rows = {l.split('|')[1].strip(): l.split('|')[3].strip() for l in text.splitlines()
                   if l.startswith('| ') and l.count('|') == 6 and arms.get(l.split('|')[1].strip()) == l.split('|')[2].strip()}
    want = f"{gate_tables.pooled('tfidf', 'auc')[0]:.3f}"
    assert pooled_rows['TF-IDF'] == pooled_rows['TF-IDF noapi'] == want
    assert float(pooled_rows['R7 typed']) > float(pooled_rows['R7 noapi'])
    assert pooled_rows['header-imports LR'].startswith('missing')
    assert 'seed 0 only: a direction' in text and 'seeds [0, 1, 2]' in text


def test_the_stage_commands_run_on_the_real_run_directory_from_the_pipeline_root():
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, '-m', 's04_evaluation.gates', '--stage', 'paths', '--resamples', '20'], cwd=root,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert '## Pooled, AUC checkpoint' in r.stdout and '## Paired differences, second minus first' in r.stdout
