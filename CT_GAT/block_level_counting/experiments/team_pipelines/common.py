"""Shared adapter for running the teammates' GAT pipelines under our 5-fold x86 family holdout.

Four pieces: a sha256-pinned import of the teammates' code (`import_team`), the fold manifests of the
`kfold_x86_gnn-mblk15_s{seed}` runs (`fold_manifest`, `all_ids`), the sample_id -> cfg / asm file lookup
(`paths_for`) and `write_run`, which writes a run directory that `evaluation/stats.py`, `report.py` and `gates.py`
read like one of ours.

Run scripts as modules from the block_level_counting root (`python -m experiments.team_pipelines.build_b`): run as a
file, this directory would sit first on sys.path and shadow our `common` package.
"""
import csv
import hashlib
import importlib
import json
import sys
from functools import lru_cache
from pathlib import Path

if __name__ == 'common':
    raise ImportError('run experiments/team_pipelines scripts with python -m experiments.team_pipelines.<script> '
                      'from the block_level_counting root, not as files')

import numpy as np

from common.config import RUNS, RunConfig
from common.manifests import load_index, make_manifest
from evaluation.metrics import family_recall, floors
from training.train import AUC_RULES, evaluate, truncated_count, val_threshold_of, write_predictions  # noqa: F401

DOWNLOADS = Path('C:/Users/chaoa/Downloads')
TEAM_ROOT = DOWNLOADS / 'rdmp-team-gnn'
TEAM_COMMIT = '331ce0b'
MASTERS = DOWNLOADS / 'ransomware-detection-masters-project/reports'
BUNDLE = MASTERS / 'google_drive_upload/2026-10-07/Mendeley_GNN_v1'
ASM_DIR = MASTERS / 'ida_asm_mendeley_20261006/asm'
TEAM_RUNS = RUNS / 'team'

PREP = 'CT_GAT/preprocessing/'
PINNED_SHA256 = {
    'construct_blocks.py': 'd922bee97d82f3b258313299939cd726b8372823199dae82c0b80a2bcdc58a12',
    'train_cfg.py': '8ee82aa89711a5c213d95a8632b40a7b964dfee1e6d062062ffb234583a7fdb6',
    PREP + 'preprocess_full.py': '27e841c9b8bdf94f2b1604e51c16f105b08aef9a49687c56a2536e1cead1499a',
    PREP + 'tokenize_asm_gz.py': '50044ef6b1a7d48be5193fa5211dc246401ed0a4067e2d59b1308045d60db3ab',
    PREP + 'graph_full_loader.py': '53d6f30a8e120e15801069cab716d71bb11e2a580135b9e6e17d3775c863bd45',
    PREP + 'gat_full_model.py': '222f7946835bf7206aa6987c49220a77cb632e7832e9b49176272bd697fceb1b',
    PREP + 'train_full_gat.py': '90e03abff1dd58ec62c6eec50a7867a6fb65f904eec78af710d473f34e254127',
    # run by importing preprocess_full (it imports transformer_model, which imports token_mapping)
    PREP + 'transformer_model.py': '599a6fc97b7115fcc49985121d6dc11a83a0c607de5f516d5c470cae9ed03fc0',
    PREP + 'token_mapping.py': 'b931df0eb3052dda610111bfd0914ffe9da16c20eed925268741c9214bb79a56',
}

# module -> the teammate files that importing it runs (itself first)
CLOSURE = {
    'construct_blocks': ('construct_blocks.py',),
    'train_cfg': ('train_cfg.py',),
    'tokenize_asm_gz': (PREP + 'tokenize_asm_gz.py',),
    'graph_full_loader': (PREP + 'graph_full_loader.py',),
    'gat_full_model': (PREP + 'gat_full_model.py',),
    'preprocess_full': (PREP + 'preprocess_full.py', PREP + 'tokenize_asm_gz.py', PREP + 'transformer_model.py',
                        PREP + 'token_mapping.py'),
    'train_full_gat': (PREP + 'train_full_gat.py', PREP + 'gat_full_model.py', PREP + 'graph_full_loader.py'),
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_pins(module, root=TEAM_ROOT):
    """Raise unless every teammate file that importing `module` runs has its pinned sha256 under `root`."""
    if module not in CLOSURE:
        raise KeyError(f'{module} is not a pinned teammate module; known: {sorted(CLOSURE)}')
    for rel in CLOSURE[module]:
        path = Path(root) / rel
        if not path.is_file():
            raise FileNotFoundError(f'pinned teammate file missing: {path}')
        got = _sha256(path)
        if got != PINNED_SHA256[rel]:
            raise RuntimeError(f'{rel}: sha256 {got} does not match the pin {PINNED_SHA256[rel]} '
                               f'(teammate commit {TEAM_COMMIT}) under {root}')


def import_team(module, root=TEAM_ROOT):
    """Import teammate `module` (a key of CLOSURE) from the review worktree `root` after checking the sha256 of every
    file the import runs. The repository root and CT_GAT/preprocessing go first on sys.path, so the teammates' own
    `tokenize_asm_gz`, `transformer_model` and `token_mapping` win over same-named modules of ours. Nothing is written
    under `root`: bytecode caching is switched off for the import."""
    check_pins(module, root)
    for p in (Path(root) / 'CT_GAT/preprocessing', Path(root)):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    keep, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        mod = importlib.import_module(module)
    finally:
        sys.dont_write_bytecode = keep
    for dep in CLOSURE[module]:
        loaded = sys.modules.get(Path(dep).stem)
        if loaded is not None and Path(loaded.__file__).resolve() != (Path(root) / dep).resolve():
            raise ImportError(f'{Path(dep).stem} was imported from {loaded.__file__}, not the pinned '
                              f'{Path(root) / dep}')
    return mod


def fold_config(fold, seed):
    """The RunConfig whose manifest the `kfold_x86_gnn-mblk15_s{seed}` runs use (val_seed None: the validation carve
    follows the seed)."""
    return RunConfig(protocol='kfold', fold=fold, arch='x86', seed=seed, model='gnn', mask_blocks=0.15)


@lru_cache(maxsize=1)
def _index():
    return load_index()


@lru_cache(maxsize=1)
def sample_rows():
    """{sample_id: index row} (label, arch, family, fold, ...) of the x86 binaries that have a fold."""
    return {r['sample_id']: r for r in _index() if r['arch'] == 'x86' and r['fold'] is not None}


@lru_cache(maxsize=None)
def fold_manifest(fold, seed):
    return make_manifest(fold_config(fold, seed), _index())


@lru_cache(maxsize=1)
def all_ids():
    """Sorted sample ids of the five folds' test sets; the test sets are disjoint (asserted)."""
    ids = [r['sample_id'] for f in range(5) for r in fold_manifest(f, 0).test]
    if len(ids) != len(set(ids)):
        raise AssertionError('a sample is in the test set of two folds')
    return sorted(ids)


@lru_cache(maxsize=1)
def _bundle_rows():
    with (BUNDLE / 'samples.csv').open(encoding='utf-8', newline='') as f:
        return {r['sample_id']: r for r in csv.DictReader(f)}


def paths_for(sample_id):
    """(cfg_path, asm_path) of a binary: the CFG in the Mendeley bundle (samples.csv cfg_path), the IDA listing in
    ida_asm_mendeley_20261006/asm. A missing row or file is an error."""
    row = _bundle_rows().get(sample_id)
    if row is None:
        raise KeyError(f'{sample_id}: no row in {BUNDLE / "samples.csv"}')
    cfg_path, asm_path = BUNDLE / row['cfg_path'], ASM_DIR / f'{sample_id}.asm.gz'
    for p in (cfg_path, asm_path):
        if not p.is_file():
            raise FileNotFoundError(f'{sample_id}: missing {p}')
    return cfg_path, asm_path


def team_row(sample_id):
    """The row `construct_blocks.gnn_graph_from_export` / `graph_from_export` expect, from the bundle's samples.csv (the
    fields `load_split_rows` fills: the csv columns, an int `label`, `cfg_file`). A missing row or file is an error."""
    cfg_path, _ = paths_for(sample_id)
    row = dict(_bundle_rows()[sample_id])
    row['label'] = int(row['label'])
    row['cfg_file'] = cfg_path
    return row


def team_config(fold, seed, arm):
    """The RunConfig of a team arm; its `.name` is the run directory name (`kfold{f}_x86_gnn_s{seed}_team-<arm>`)."""
    return RunConfig(protocol='kfold', fold=fold, arch='x86', seed=seed, model='gnn', tag=f'team-{arm}')


def _check_cover(name, by_split, manifest, excluded=()):
    """The predictions hold exactly the manifest's validation and test ids minus the explicitly `excluded` ones."""
    skip = set(excluded)
    for split, rows in (('val', manifest.val), ('test', manifest.test)):
        ids = list(by_split[split][0])
        want = sorted(r['sample_id'] for r in rows if r['sample_id'] not in skip)
        if sorted(ids) != want:
            raise AssertionError(f'{name}: {split} predictions hold {len(set(ids))} distinct of {len(ids)} ids, the '
                                 f'manifest has {len(want)}; missing {sorted(set(want) - set(ids))[:5]}, '
                                 f'extra {sorted(set(ids) - set(want))[:5]}')


def write_run(run_dir, cfg, rows_f1, rows_auc, thr_f1, thr_auc, extra, excluded=(), excluded_reason=''):
    """Write one team run directory that the harness reads like its own.

    run_dir     the directory; its name must equal cfg.name (team_config(...).name)
    rows_f1     {'val': (ids, probs, sizes), 'test': (ids, probs, sizes)} of the best-validation-macro-F1 state, with
                sizes = (n_segments_total, n_blocks_kept, n_blocks_total) per sample (as training/train.py's predict)
    rows_auc    the same for the best-validation-AUC state
    thr_f1      validation threshold (GRID, val_threshold_of) of the F1 state: metrics only. predictions.csv is
                thresholded at cfg.threshold (0.5), as training/train.py writes it
    thr_auc     validation threshold of the AUC state: predictions_auc.csv and the AUC block of metrics.json
    extra       must hold best_epoch and epochs_run; auc_best_epoch, val_auc, val_macro_f1 and other keys are kept
    excluded    sample ids deliberately left out of this run (a binary the arm's build could not process): the row sets
                must cover the manifest's validation and test ids minus these, no more and no fewer; metrics.json records
                them as `excluded` with `excluded_reason`, and evaluation/stats.py prints them where it pairs the run
                with an arm that has those binaries. An empty list keeps the exact-coverage check.
    Both row sets must cover exactly the manifest's validation and test ids (less `excluded`). The metrics come from
    evaluation/metrics.py through training.train's helpers."""
    run_dir = Path(run_dir)
    if run_dir.name != cfg.name:
        raise ValueError(f'run directory {run_dir.name} is not the RunConfig name {cfg.name}')
    for key in ('best_epoch', 'epochs_run'):
        if key not in extra:
            raise KeyError(f'extra needs {key}')
    manifest = fold_manifest(cfg.fold, cfg.seed)
    excluded = sorted(set(excluded))
    if excluded and not excluded_reason:
        raise ValueError('excluded ids need an excluded_reason')
    unknown = sorted(set(excluded) - {r['sample_id'] for r in manifest.train + manifest.val + manifest.test})
    if unknown:
        raise ValueError(f'excluded ids that are in no split of the fold: {unknown}')
    for name, rows in (('rows_f1', rows_f1), ('rows_auc', rows_auc)):
        _check_cover(name, rows, manifest, excluded)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / 'config.json')
    manifest.save(run_dir / 'manifest.json')
    index_rows = manifest.val + manifest.test
    write_predictions(run_dir / 'predictions.csv', index_rows, rows_f1, cfg.threshold)
    write_predictions(run_dir / 'predictions_auc.csv', index_rows, rows_auc, thr_auc)

    by_id = {r['sample_id']: r for r in manifest.test}
    labels = np.array([r['label'] for r in manifest.train])

    def test_scores(rows, threshold):
        ids, probs, _ = rows['test']
        return evaluate(manifest.test, ids, probs, threshold)

    def test_family_recall(rows, threshold):
        ids, probs, _ = rows['test']
        return family_recall([by_id[i]['label'] for i in ids], (np.asarray(probs) >= threshold).astype(int),
                             [by_id[i]['family'] for i in ids])

    ids = rows_f1['test'][0]
    metrics = {
        'run': cfg.name, 'select_on': cfg.select_on, 'best_epoch': extra['best_epoch'],
        'epochs_run': extra['epochs_run'], 'val_macro_f1': extra.get('val_macro_f1'),
        'truncated_test': truncated_count(rows_f1['test'][2]),
        'test': test_scores(rows_f1, cfg.threshold),
        'floors': floors(labels, [by_id[i]['label'] for i in ids], [by_id[i]['arch'] for i in ids]),
        'family_recall': test_family_recall(rows_f1, cfg.threshold),
        'selection': {
            'f1': {'best_epoch': extra['best_epoch'], 'threshold_val': thr_f1,
                   'test_at_val_threshold': test_scores(rows_f1, thr_f1)},
            'auc': {'rule': AUC_RULES[cfg.select_on], 'best_epoch': extra.get('auc_best_epoch'),
                    'val_auc': extra.get('val_auc'), 'threshold_val': thr_auc,
                    'test': test_scores(rows_auc, thr_auc), 'test_at_half': test_scores(rows_auc, 0.5),
                    'family_recall': test_family_recall(rows_auc, thr_auc)},
        },
    }
    metrics.update({k: v for k, v in extra.items() if k not in metrics})
    if excluded:
        metrics.update(excluded=excluded, excluded_reason=excluded_reason)
    (run_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    return metrics
