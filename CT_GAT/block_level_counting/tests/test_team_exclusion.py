"""Explicit exclusions of pipeline B: `build_b --exclude`, `write_run(excluded=)`, the run_arm / train_b handling, the
"excluded from both arms" line of stats / gates, and the sig500 lane's build check (processed + excluded == 1810)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from common.config import RunConfig
from evaluation import gates, stats
from experiments.team_pipelines import common as tp, fit, train_a, train_b
from test_team_pipelines import (BASH, FOLDS, ROOT, fake_group, fake_rows, needs_bash, needs_real_data, pinned_team_modules,  # noqa: F401
                                 synthetic_graphs)
from training.train import val_threshold_of

WHY = 'process_sample raises MemoryError reading the listing'


def fake_rows_without(manifest, rng, signal, skip):
    rows = fake_rows(manifest, rng, signal)
    out = {}
    for split, (ids, probs, sizes) in rows.items():
        keep = [k for k, i in enumerate(ids) if i not in skip]
        out[split] = ([ids[k] for k in keep], probs[keep], [sizes[k] for k in keep])
    return out


def fake_group_without(runs, cfg_of, signal, skip, seeds=(0,)):
    for seed in seeds:
        rng = np.random.default_rng(100 * seed + int(10 * signal))
        for fold in FOLDS:
            cfg, manifest = cfg_of(fold, seed), tp.fold_manifest(fold, seed)
            f1, auc = (fake_rows_without(manifest, rng, signal, skip) for _ in range(2))
            tp.write_run(runs / cfg.name, cfg, f1, auc, val_threshold_of(manifest.val, *f1['val'][:2]),
                         val_threshold_of(manifest.val, *auc['val'][:2]),
                         {'best_epoch': 2, 'epochs_run': 7, 'auc_best_epoch': 3, 'val_auc': 0.9, 'val_macro_f1': 0.8},
                         excluded=sorted(skip), excluded_reason=WHY)


@needs_real_data
def test_write_run_takes_an_explicit_excluded_list(tmp_path):
    cfg = tp.team_config(0, 0, 'fake')
    manifest = tp.fold_manifest(0, 0)
    victim = manifest.test[0]['sample_id']
    other = tp.fold_manifest(1, 0).test[0]['sample_id']                     # in the train or val set of fold 0
    assert other in {r['sample_id'] for r in manifest.train + manifest.val}
    extra = {'best_epoch': 1, 'epochs_run': 1}
    rows = fake_rows_without(manifest, np.random.default_rng(0), 0.8, {victim})
    assert victim not in rows['test'][0]
    with pytest.raises(AssertionError, match='missing'):                      # not declared: the coverage check still fails
        tp.write_run(tmp_path / 'a' / cfg.name, cfg, rows, rows, 0.5, 0.5, extra)
    with pytest.raises(ValueError, match='excluded_reason'):
        tp.write_run(tmp_path / 'b' / cfg.name, cfg, rows, rows, 0.5, 0.5, extra, excluded=[victim])
    with pytest.raises(ValueError, match='no split'):
        tp.write_run(tmp_path / 'c' / cfg.name, cfg, rows, rows, 0.5, 0.5, extra, excluded=['0' * 64], excluded_reason='x')
    full = fake_rows(manifest, np.random.default_rng(0), 0.8)
    with pytest.raises(AssertionError, match='extra'):                         # declared but its rows are there
        tp.write_run(tmp_path / 'd' / cfg.name, cfg, full, full, 0.5, 0.5, extra, excluded=[victim], excluded_reason='x')
    metrics = tp.write_run(tmp_path / 'e' / cfg.name, cfg, rows, rows, 0.5, 0.5, extra, excluded=[victim, other],
                           excluded_reason=WHY)
    assert metrics['excluded'] == sorted([victim, other]) and metrics['excluded_reason'] == WHY
    saved = json.loads((tmp_path / 'e' / cfg.name / 'metrics.json').read_text(encoding='utf-8'))
    assert saved['excluded'] == metrics['excluded'] and saved['excluded_reason'] == WHY
    predictions = (tmp_path / 'e' / cfg.name / 'predictions.csv').read_text(encoding='utf-8')
    assert victim not in predictions and len(predictions.splitlines()) - 1 == len(manifest.val) + len(manifest.test) - 1
    plain = tp.write_run(tmp_path / 'f' / cfg.name, cfg, full, full, 0.5, 0.5, extra)
    assert 'excluded' not in plain and 'excluded_reason' not in plain         # an ordinary run records nothing extra


@needs_real_data
def test_run_arm_leaves_the_excluded_ids_out_of_every_split(tmp_path, monkeypatch):
    from torch import nn
    from torch_geometric.nn import global_mean_pool

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 2)

        def forward(self, data):
            return self.lin(global_mean_pool(data.x, data.batch))

    monkeypatch.setattr(fit, 'RUNS', tmp_path)
    manifest = tp.fold_manifest(2, 1)
    in_test, in_train = manifest.test[0]['sample_id'], manifest.train[0]['sample_id']
    graphs, sizes = synthetic_graphs(tp.all_ids(), np.random.default_rng(0))
    for i in (in_test, in_train):
        del graphs[i]
    kw = dict(arm='fake18', fold=2, seed=1, graphs=graphs, sizes=sizes, build_model=Tiny,
              hp={**train_a.HP, 'epochs': 2}, device=torch.device('cpu'), log=lambda *_: None)
    with pytest.raises(SystemExit, match='no graph'):                           # not declared: still an error
        fit.run_arm(**kw)
    out = fit.run_arm(**kw, excluded=[in_test, in_train], excluded_reason=WHY)
    d = tmp_path / 'kfold2_x86_gnn_s1_team-fake18'
    assert out['metrics']['excluded'] == sorted([in_test, in_train]) and out['metrics']['excluded_reason'] == WHY
    for name in ('predictions.csv', 'predictions_auc.csv'):
        text = (d / name).read_text(encoding='utf-8')
        assert in_test not in text and in_train not in text
        assert len(text.splitlines()) - 1 == len(manifest.val) + len(manifest.test) - 1
    assert json.loads((d / 'manifest.json').read_text(encoding='utf-8'))      # the manifest file is the fold's own, unchanged


def test_train_b_reads_the_exclusions_of_the_build(tmp_path):
    assert train_b.read_excluded(tmp_path) == {}                                # no settings.json: nothing excluded
    (tmp_path / 'settings.json').write_text(json.dumps({'sample': 'signal', 'excluded': {'a' * 64: 'why'}}), encoding='utf-8')
    assert train_b.read_excluded(tmp_path) == {'a' * 64: 'why'}
    (tmp_path / 'settings.json').write_text(json.dumps({'sample': 'signal', 'excluded': {}}), encoding='utf-8')
    assert train_b.read_excluded(tmp_path) == {}


@needs_real_data
def test_stats_and_gates_say_what_was_excluded_from_both_arms(tmp_path, monkeypatch):
    runs = tmp_path / 'runs'
    runs.mkdir()
    victim = tp.fold_manifest(3, 0).test[0]['sample_id']
    base = dict(protocol='kfold', arch='x86', model='gnn')
    fake_group(runs, lambda f, s: RunConfig(**base, fold=f, seed=s, mask_blocks=0.15), 0.5, seeds=(0,))        # R7: all 1810
    fake_group(runs, lambda f, s: RunConfig(**base, fold=f, seed=s, mask_blocks=0.15, subgraph='cap', budget=500, tag='b500'),
               0.5, seeds=(0,))                                                                                 # b500: all 1810
    fake_group_without(runs, lambda f, s: tp.team_config(f, s, 'sig500'), 0.7, {victim})                          # sig500: 1809
    monkeypatch.setattr(stats, 'RUNS', runs)
    gates.pooled.cache_clear()
    arms = dict(gates.TEAM_ARMS)
    sig, r7, b500 = arms['team sig500'], arms['R7 typed, whole graph'], arms['ours b500 typed']
    assert stats.group_exclusions(sig) == {victim: WHY} and stats.group_exclusions(r7) == {}
    result, info = stats.compare(r7, sig, 'auc', 30, seeds=[0])
    assert info['n'] == 1809 and info['excluded'] == [victim] and info['excluded_reason'] == WHY
    block = stats.fmt_block(r7, sig, 'auc', 30, result, info)
    assert f'1 excluded from both arms: {victim} ({WHY})' in block and '1809 test binaries' in block
    result, plain = stats.compare(r7, b500, 'auc', 30, seeds=[0])             # two arms without exclusions: nothing printed
    assert plain['n'] == 1810 and plain['excluded'] == []
    assert 'excluded from both arms' not in stats.fmt_block(r7, b500, 'auc', 30, result, plain)
    text = '\n'.join(gates.team(30))
    gates.pooled.cache_clear()
    assert text.count('excluded from both arms') == 2                          # R7 and ours b500 against sig500
    assert f'{victim} ({WHY})' in text


def test_sig500_lane_accepts_processed_plus_excluded(tmp_path):
    if BASH is None:
        pytest.skip('needs bash')
    runs = tmp_path / 'runs'
    (runs / 'team' / 'b_sig500').mkdir(parents=True)
    env = {**os.environ, 'GT_RUNS': runs.as_posix(), 'GT_PY': Path(sys.executable).as_posix()}

    def check():
        return subprocess.run([BASH, 'scripts/team_runs.sh', '--check-build'], cwd=ROOT, capture_output=True, text=True,
                              env=env).stdout.strip()

    summary = runs / 'team' / 'b_sig500' / 'build_summary.json'
    assert check() == 'build summary absent'
    for body, want in (({'processed': 1810, 'total': 1810}, 'build complete'),                      # the old accounting holds
                       ({'processed': 1809, 'total': 1810, 'excluded': {'a' * 64: 'why'}}, 'build complete'),
                       ({'processed': 1809, 'total': 1810}, 'build incomplete'),
                       ({'processed': 1808, 'total': 1810, 'excluded': {'a' * 64: 'why'}}, 'build incomplete'),
                       ({'processed': 5, 'total': 5, 'excluded': {}}, 'build incomplete')):             # not the 1810 build
        summary.write_text(json.dumps(body, indent=2), encoding='utf-8')
        assert check() == want, body


@needs_bash
@needs_real_data
def test_build_b_excludes_by_id_and_records_why(tmp_path):
    from experiments.team_pipelines import build_b
    rows = sorted(tp.sample_rows().values(), key=lambda r: int(r['num_blocks']))
    small, other = rows[0]['sample_id'], rows[1]['sample_id']
    out = tmp_path / 'b'
    with pytest.raises(SystemExit):                                              # --exclude and --exclude-reason go together
        build_b.main(['--out', str(out), '--ids', small, other, '--exclude', other])
    with pytest.raises(SystemExit):                                              # an id outside this build
        build_b.main(['--out', str(out), '--ids', small, '--exclude', other, '--exclude-reason', 'x'])
    args = ['--out', str(out), '--ids', small, other, '--exclude', other, '--exclude-reason', WHY]
    summary = build_b.main(args)
    assert (summary['processed'], summary['total'], summary['excluded']) == (1, 2, {other: WHY})
    assert summary['built_this_run'] == 1                                        # the excluded id is neither built nor failed
    assert (out / f'{small}.pt').is_file() and not (out / f'{other}.pt').exists()
    saved = json.loads((out / 'build_summary.json').read_text(encoding='utf-8'))
    settings = json.loads((out / 'settings.json').read_text(encoding='utf-8'))
    assert saved['excluded'] == settings['excluded'] == {other: WHY}
    assert settings['sample'] == 'signal' and settings['num_blocks'] == 500
    ckpt = json.loads((out / 'checkpoint.json').read_text(encoding='utf-8'))
    assert ckpt['settings'] == {'sample': 'signal', 'num_blocks': 500, 'seed': 0} and ckpt['done'] == [small]   # format unchanged
    assert train_b.read_excluded(out) == {other: WHY}
    again = build_b.main(args)
    assert again['built_this_run'] == 0 and again['processed'] == 1
    with pytest.raises(SystemExit, match='cannot be excluded'):                  # a built id cannot be excluded afterwards
        build_b.main(['--out', str(out), '--ids', small, other, '--exclude', small, '--exclude-reason', 'x'])
    plain = build_b.main(['--out', str(tmp_path / 'p'), '--ids', small])
    assert plain['excluded'] == {} and plain['processed'] == plain['total'] == 1
    assert train_b.read_excluded(tmp_path / 'p') == {}
