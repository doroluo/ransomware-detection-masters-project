"""python -m pytest tests/test_matched_excluded.py: a `--arch matched` run on the `all` build records only the build's excluded ids
that the matched subset holds.

A kfold_v3 build is made for `all`, so its settings exclude ids of the whole dataset; the matched manifest holds a part of it, and
`write_run` refuses an excluded id that is in no split of the fold (production, step 62: 34 ids excluded by the build, 11 of them
matched, every matched run failed after training). `tiny_v3`'s own tests exclude ids that all lie in the matched subset, so here the
excluded set is two ids inside it and two outside it. The runs: the train_b arms (plain and `-m1frozen`) and Model 1 (plain, per
block, `-ab`) on matched; the same on `all`, which keeps all four."""
import json

import pytest
import torch

from experiments.team_pipelines import common as tp, fit, train_b
from s03a_transformer.five_channel import train as fc
from test_datasets_train import WHY, five_run, quiet
from test_team_m1frozen import model1_run
from test_team_pipelines import fresh_team_caches, needs_pyg  # noqa: F401
from test_team_v3 import synthetic_build
from tiny_cache import tiny_v3  # noqa: F401  (a fixture)

FOLD, SEED = 2, 1


def victims_of(rows):
    """(inside, outside): two excluded ids that the matched manifest of fold 2 holds, and two that only the `all` manifest holds."""
    matched, everything = tp.fold_manifest(FOLD, SEED, 'kfold_v3', 'matched'), tp.fold_manifest(FOLD, SEED, 'kfold_v3', 'all')
    in_matched = {r['sample_id'] for r in matched.train + matched.val + matched.test}
    inside = {matched.train[0]['sample_id'], matched.test[0]['sample_id']}
    outside = {next(r['sample_id'] for r in split if r['sample_id'] not in in_matched) for split in (everything.train, everything.test)}
    assert outside and not outside & in_matched and all(str(rows[i]['arch_matched']) != '1' for i in outside)
    return inside, outside


def test_dataset_excluded_keeps_the_ids_of_the_manifest_with_their_reasons(tiny_v3):   # noqa: F811
    rows = tp.sample_rows('kfold_v3', 'all')
    inside, outside = victims_of(rows)
    excluded = {i: f'why {n}' for n, i in enumerate(sorted(inside | outside | {'0' * 64}))}
    matched, everything = (tp.fold_manifest(FOLD, SEED, 'kfold_v3', arch) for arch in ('matched', 'all'))
    assert tp.dataset_excluded(excluded, matched) == {i: excluded[i] for i in inside}
    assert tp.dataset_excluded(excluded, everything) == {i: excluded[i] for i in inside | outside}      # not the id of no dataset
    assert tp.dataset_excluded({}, matched) == {}


@needs_pyg
@pytest.mark.parametrize('arch', ['matched', 'all'])
@pytest.mark.parametrize('arm', ['sig500v3-emb', 'sig500v3-m1frozen'])
def test_a_train_b_arm_records_the_excluded_ids_of_its_dataset(tiny_v3, tmp_path, monkeypatch, arm, arch):   # noqa: F811
    rows = tp.sample_rows('kfold_v3', 'all')
    inside, outside = victims_of(rows)
    folder = synthetic_build(tmp_path / 'b', sorted(set(rows) - inside - outside), rows, dataset=('kfold_v3', 'all'),
                             excluded={v: WHY for v in sorted(inside | outside)})
    monkeypatch.setattr(fit, 'RUNS', tmp_path / 'runs')
    monkeypatch.setitem(train_b.ARM_HP, arm, {**train_b.ARM_HP[arm], 'epochs': 2})
    kw = {}
    if arm == 'sig500v3-m1frozen':
        names = json.loads((folder / 'api_vocab.json').read_text())
        run = model1_run(tmp_path / 'm1', folder, names=names)
        tp.fold_manifest(FOLD, SEED, 'kfold_v3', arch).save(run / 'manifest.json')
        kw['model1_run'] = run
    out = train_b.run(arm, FOLD, SEED, device='cpu', b_dir=folder, log=quiet, protocol='kfold_v3', arch=arch, **kw)
    expect = sorted(inside if arch == 'matched' else inside | outside)
    metrics = json.loads((out['dir'] / 'metrics.json').read_text(encoding='utf-8'))
    assert metrics['excluded'] == expect and metrics['excluded_reason'] == WHY == out['metrics']['excluded_reason']
    for file in ('predictions.csv', 'predictions_auc.csv'):
        text = (out['dir'] / file).read_text(encoding='utf-8')
        assert not any(v in text for v in inside | outside)
    if arch == 'matched':
        assert not set(metrics['excluded']) & outside


@pytest.mark.parametrize('flags', [{}, {'per_block': True}, {'arch_balance': True}], ids=['plain', 'per-block', 'ab'])
@pytest.mark.parametrize('arch', ['matched', 'all'])
def test_model1_records_the_excluded_ids_of_its_dataset(tiny_v3, tmp_path, arch, flags):   # noqa: F811
    rows = tp.sample_rows('kfold_v3', 'all')
    inside, outside = victims_of(rows)
    cfg = five_run(FOLD, 'kfold_v3', arch, seed=SEED, max_instructions=64 if flags.get('per_block') else 16, epochs=2, **flags)
    build = synthetic_build(tmp_path / 'b', sorted(set(rows) - inside - outside), rows, dataset=('kfold_v3', 'all'),
                            excluded={v: WHY for v in sorted(inside | outside)})
    metrics = fc.run(cfg, build, 'cpu', None, tmp_path / 'runs', log=quiet)
    assert metrics['excluded'] == sorted(inside if arch == 'matched' else inside | outside) and metrics['excluded_reason'] == WHY
    saved = json.loads((tmp_path / 'runs' / cfg.name / 'metrics.json').read_text(encoding='utf-8'))
    assert saved['excluded'] == metrics['excluded']
    for file in ('predictions.csv', 'predictions_auc.csv'):
        text = (tmp_path / 'runs' / cfg.name / file).read_text(encoding='utf-8')
        assert not any(v in text for v in inside | outside)
