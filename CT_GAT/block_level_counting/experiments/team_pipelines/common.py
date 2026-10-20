"""Shared adapter for running the teammates' GAT pipelines under our 5-fold x86 family holdout.

Four pieces: a sha256-pinned import of the teammates' code (`import_team`, three pinned sources), the fold manifests of the
`kfold_x86_gnn-mblk15_s{seed}` runs (`fold_manifest`, `all_ids`), the sample_id -> cfg / asm file lookup
(`paths_for`) and `write_run` (now `evaluation/run_io.py`, re-exported here), which writes a run directory that
`evaluation/stats.py`, `report.py` and `gates.py` read like one of ours.

Run scripts as modules from the block_level_counting root (`python -m experiments.team_pipelines.build_b`): run as a
file, this directory would sit first on sys.path and shadow our `common` package.
"""
import csv
import hashlib
import importlib
import os
import sys
from functools import lru_cache
from pathlib import Path

if __name__ == 'common':
    raise ImportError('run experiments/team_pipelines scripts with python -m experiments.team_pipelines.<script> '
                      'from the block_level_counting root, not as files')

from common.config import RUNS, RunConfig
from common.manifests import load_index, make_manifest
from evaluation.run_io import _check_cover, write_run  # noqa: F401  (re-exported: the team trainers and tests import them from here)
from training.train import AUC_RULES, evaluate, truncated_count, val_threshold_of, write_predictions  # noqa: F401

DOWNLOADS = Path('C:/Users/chaoa/Downloads')
TEAM_ROOT = DOWNLOADS / 'rdmp-team-gnn'
TEAM_COMMIT = '331ce0b'
MAIN_ROOT = DOWNLOADS / 'rdmp-team-main'
MAIN_COMMIT = 'c7899f0'
FIX_ROOT = DOWNLOADS / 'rdmp-team-yfix'       # detached review worktree of origin s02-endp-library-fix (the PR into main)
FIX_COMMIT = '145ed35'
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


# The second source: Yanping's streaming preprocess on main. The two files sit under CT_GAT/pipeline/s02_preprocess and
# preprocess_full imports only tokenize_asm_gz (the OPCODE / OPERAND vocabulary sizes are literals there). The trainer's
# graph loader and model, construct_blocks and train_cfg are not on main: they always come from the gnn source.
STAGE = 'CT_GAT/pipeline/s02_preprocess/'
MAIN_PINNED_SHA256 = {
    STAGE + 'preprocess_full.py': 'dbddaae1e2bf14a386cdd90be1bfb4f1b5f3f0c013798f09ab8b36977ed2eb75',
    STAGE + 'tokenize_asm_gz.py': 'c8e39522839688d15563d2da61634e51d1e086e18950fce1f07f808b4c87a05f',
}
MAIN_CLOSURE = {
    'tokenize_asm_gz': (STAGE + 'tokenize_asm_gz.py',),
    'preprocess_full': (STAGE + 'preprocess_full.py', STAGE + 'tokenize_asm_gz.py'),
}
# The third source: the same two files with the `endp` and library-function fixes (branch s02-endp-library-fix). Hashes of the
# files as checked out in FIX_ROOT (CRLF), like the pins above.
FIX_PINNED_SHA256 = {
    STAGE + 'preprocess_full.py': '2ad59852376d7849d417ccd90a2c16d496b375cbf851ffae4902f6215d8bc23a',
    STAGE + 'tokenize_asm_gz.py': 'e7b13fccfa512d11fba1105b4e2125fe720604175fcf9cb166e04ea1eb50acda',
}

DEFAULT_SOURCE = 'gnn-331ce0b'
SOURCES = {
    'gnn-331ce0b': {'root': TEAM_ROOT, 'commit': TEAM_COMMIT, 'pins': PINNED_SHA256, 'closure': CLOSURE,
                    'dirs': ('CT_GAT/preprocessing', '.')},
    'main-c7899f0': {'root': MAIN_ROOT, 'commit': MAIN_COMMIT, 'pins': MAIN_PINNED_SHA256, 'closure': MAIN_CLOSURE,
                     'dirs': ('CT_GAT/pipeline/s02_preprocess', '.')},
    'main-fix': {'root': FIX_ROOT, 'commit': FIX_COMMIT, 'pins': FIX_PINNED_SHA256, 'closure': MAIN_CLOSURE,
                 'dirs': ('CT_GAT/pipeline/s02_preprocess', '.')},
}


def _source(source):
    if source not in SOURCES:
        raise KeyError(f'unknown teammate source {source}; known: {sorted(SOURCES)}')
    return SOURCES[source]


def source_pins(source=DEFAULT_SOURCE, module='preprocess_full'):
    """{relative path: sha256} of the files that importing `module` from `source` runs (what a build records)."""
    src = _source(source)
    return {rel: src['pins'][rel] for rel in src['closure'][module]}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_pins(module, root=None, source=DEFAULT_SOURCE):
    """Raise unless every teammate file that importing `module` runs has its pinned sha256 under `root` (default: the
    review worktree of `source`)."""
    src = _source(source)
    root = src['root'] if root is None else root
    if module not in src['closure']:
        raise KeyError(f'{module} is not a pinned teammate module of {source}; known: {sorted(src["closure"])}')
    for rel in src['closure'][module]:
        path = Path(root) / rel
        if not path.is_file():
            raise FileNotFoundError(f'pinned teammate file missing: {path}')
        got = _sha256(path)
        if got != src['pins'][rel]:
            raise RuntimeError(f'{rel}: sha256 {got} does not match the pin {src["pins"][rel]} '
                               f'(teammate commit {src["commit"]}) under {root}')


def _all_stems(module):
    """Module names that importing `module` runs in any source (the stems to evict before a source switch)."""
    return {Path(rel).stem for src in SOURCES.values() for rel in src['closure'].get(module, ())}


def import_team(module, root=None, source=DEFAULT_SOURCE):
    """Import teammate `module` from the review worktree `root` (default: that of `source`, 'gnn-331ce0b',
    'main-c7899f0' or 'main-fix') after checking the sha256 of every file the import runs. The source's directories go first on sys.path,
    so its own `tokenize_asm_gz` (and, for the gnn source, `transformer_model` and `token_mapping`) win over same-named
    modules of ours or of the other source. Before the import, the modules that `module` runs in any source are evicted
    from sys.modules unless they are already the pinned file of this source, so importing one source never leaves the
    other's modules loaded. Nothing is written under `root`: bytecode caching is switched off for the import."""
    src = _source(source)
    root = Path(src['root'] if root is None else root)
    check_pins(module, root, source)
    wanted = {Path(rel).stem: (root / rel).resolve() for rel in src['closure'][module]}
    roots = [Path(o['root']).resolve() for o in SOURCES.values()] + [root.resolve()]
    for stem in _all_stems(module):
        loaded = sys.modules.get(stem)
        if loaded is None:
            continue
        where = Path(getattr(loaded, '__file__', None) or '.').resolve()
        if where != wanted.get(stem) and any(r in where.parents for r in roots):
            del sys.modules[stem]           # another source's (or another root's) module: it must not stay loaded
    own = [os.path.normcase(os.path.abspath(root / d)) for d in src['dirs']]
    foreign = {os.path.normcase(os.path.abspath(Path(o['root']) / d)) for o in SOURCES.values() for d in o['dirs']} | set(own)
    sys.path[:] = [p for p in sys.path if os.path.normcase(os.path.abspath(p)) not in foreign]
    sys.path[0:0] = [str(Path(root / d)) for d in src['dirs']]
    keep, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        mod = importlib.import_module(module)
    finally:
        sys.dont_write_bytecode = keep
    for stem, path in wanted.items():
        loaded = sys.modules.get(stem)
        if loaded is not None and Path(loaded.__file__).resolve() != path:
            raise ImportError(f'{stem} was imported from {loaded.__file__}, not the pinned {path}')
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
