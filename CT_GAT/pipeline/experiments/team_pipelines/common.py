"""Shared adapter for running the teammates' GAT pipelines under our 5-fold x86 family holdout.

Four pieces: a sha256-pinned import of the teammates' code (`import_team`, two pinned sources and this repository's own `s02_preprocess` files, `repo`), the fold manifests of the
`kfold_x86_gnn-mblk15_s{seed}` runs (`fold_manifest`, `all_ids`), the sample_id -> cfg / asm file lookup
(`paths_for`) and `write_run` (`common/run_io.py`, re-exported here), which writes a run directory that
`s04_evaluation/stats.py`, `report.py` and `gates.py` read like one of ours.

The dataset is a flag, not an arm: `protocol` ('kfold' or 'kfold_v3') and `arch` ('x86', 'all' or 'matched') pick the rows
(`sample_rows`), the folds (`fold_manifest`, `all_ids`) and the run names (`team_config`); `DATASET_OF`, `DEFAULT_ARCH` and
`BUILD_ARCHES` name the dataset of a protocol, its default architecture and the architectures a pipeline-B build is made for.
Every function keeps its zero-argument meaning, the Mendeley kfold x86 binaries, and `paths_for` takes the corpus of the row
(`mendeley` unless told otherwise). Vocabulary clash: `sig500v3` is the s02-fix *build* version, `kfold_v3` the fold *protocol*
and `_v3.zip` the preprocessing version of the zips.

Run scripts as modules from CT_GAT/pipeline (`python -m experiments.team_pipelines.build_b`): run as a file, this
directory would sit first on sys.path and shadow our `common` package.

Nothing here knows a machine path, and importing this module needs neither the teammates' worktrees nor torch_geometric.
Four locations are resolved when first used, never at import:

  GT_TEAM_GNN_ROOT   the worktree of the teammates' gnn code at commit 331ce0b          (`root_of('gnn-331ce0b')`)
  GT_TEAM_MAIN_ROOT  the worktree of the teammates' main at commit c7899f0              (`root_of('main-c7899f0')`)
  GT_BUNDLE          the Mendeley_GNN_v1 bundle (`common.paths.source_for('bundle')`)    (`bundle_dir()`)
  GT_TEAM_ASM_DIR    the directory of the IDA listings `<sample_id>.asm.gz`              (`asm_dir()`)

When one is unset the function that needs it raises `TeamSourceNotConfigured`, a FileNotFoundError whose message names the
environment variable (for the bundle also the datasets.json route of common/paths.py). The exception is the pair the bundle and
GT_TEAM_ASM_DIR make: with neither of them configured, `paths_for` takes the CFG and the listing of each binary from the
Mendeley collections (`common.paths.source_for('collections')`, `CT_GAT/data` by default); with one of them it asks for the other.
"""
import csv
import hashlib
import importlib
import os
import sys
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

if __name__ == 'common':
    raise ImportError('run experiments/team_pipelines scripts with python -m experiments.team_pipelines.<script> '
                      'from CT_GAT/pipeline, not as files')

from common.catalog import CT_GAT
from common.config import RUNS, RunConfig
from common import manifests
from common.manifests import arch_rows, load_index, make_manifest
from common.paths import MissingSource, source_for
from common.run_io import AUC_RULES, _check_cover, evaluate, truncated_count, write_predictions, write_run  # noqa: F401  (re-exported: the team trainers and tests import them from here)
from experiments.train import val_threshold_of  # noqa: F401

TEAM_COMMIT = '331ce0b'
MAIN_COMMIT = 'c7899f0'
TEAM_RUNS = RUNS / 'team'
ASM_DIR_ENV = 'GT_TEAM_ASM_DIR'


class TeamSourceNotConfigured(FileNotFoundError):
    """A teammate worktree, the Mendeley bundle or the IDA listing directory is needed and its environment variable is unset
    (or names no directory). The message names the variable."""


def _env_dir(var, what):
    value = os.environ.get(var)
    if not value:
        raise TeamSourceNotConfigured(f'{what} is not configured: set {var} to its directory')
    path = Path(value)
    if not path.is_dir():
        raise TeamSourceNotConfigured(f'{var}={value} is not a directory ({what})')
    return path


def bundle_dir():
    """The Mendeley_GNN_v1 bundle: GT_BUNDLE, else the `bundle` source of datasets.json (common.paths.source_for)."""
    try:
        path = source_for('bundle')
    except MissingSource as e:
        raise TeamSourceNotConfigured(str(e)) from e
    if not path.is_dir():
        raise TeamSourceNotConfigured(f'the Mendeley bundle {path} (GT_BUNDLE) is not a directory')
    return path


def asm_dir():
    """The directory of the IDA listings `<sample_id>.asm.gz` (GT_TEAM_ASM_DIR)."""
    return _env_dir(ASM_DIR_ENV, 'the directory of the IDA listings (<sample_id>.asm.gz)')


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

# The third source: the same two files in this repository (the checkout, `REPO_ROOT`). There is no fixed pin: the files are the
# reference, `FilePins` hashes them when a pin is looked up, and a build records the hashes it ran with (a resume refuses a
# changed file, build_b.load_state). It needs no environment variable.
REPO_ROOT = CT_GAT.parent


class FilePins(Mapping):
    """{relative path: sha256 of that file under `root`}, read on each lookup; the pin table of the `repo` source."""

    def __init__(self, root, rels):
        self.root, self.rels = Path(root), tuple(rels)

    def __getitem__(self, rel):
        if rel not in self.rels:
            raise KeyError(rel)
        return hashlib.sha256((self.root / rel).read_bytes()).hexdigest()

    def __iter__(self):
        return iter(self.rels)

    def __len__(self):
        return len(self.rels)


DEFAULT_SOURCE = 'gnn-331ce0b'
SOURCES = {
    'gnn-331ce0b': {'env': 'GT_TEAM_GNN_ROOT', 'commit': TEAM_COMMIT, 'pins': PINNED_SHA256, 'closure': CLOSURE,
                    'dirs': ('CT_GAT/preprocessing', '.')},
    'main-c7899f0': {'env': 'GT_TEAM_MAIN_ROOT', 'commit': MAIN_COMMIT, 'pins': MAIN_PINNED_SHA256, 'closure': MAIN_CLOSURE,
                     'dirs': ('CT_GAT/pipeline/s02_preprocess', '.')},
    'repo': {'env': None, 'root': REPO_ROOT, 'commit': None, 'pins': FilePins(REPO_ROOT, MAIN_PINNED_SHA256), 'closure': MAIN_CLOSURE,
             'dirs': ('CT_GAT/pipeline/s02_preprocess',)},
}


# The labels a build's settings.json may name (`source`) and the source whose code a run imports for each. `repo` is the files of
# this repository; `main-fix` is the label of the builds made from the s02-endp-library-fix branch (145ed35: `b_sig500_v3`,
# `b_sigwhole_v3`; pins 2ad59852... / e7b13fcc...), which merged into main: a run of such a build imports the embedder and the
# pooling from this repository's files, and `train_b.check_repool` (one sample re-pooled with the stored ids must reproduce the
# stored block embeddings to 1e-5 and the api-kind columns exactly) is what guards the alias. A build records its own pins in
# its settings.json; a run records them as `build_pins` and the sha256 of the files it imported as `code_pins`.
SOURCE_CODE = {'gnn-331ce0b': 'gnn-331ce0b', 'main-c7899f0': 'main-c7899f0', 'main-fix': 'repo', 'repo': 'repo'}


def source_label_ok(label, allowed=None):
    """True when `label` is a source label a build may record (SOURCE_CODE), and, if `allowed` is given, one of those."""
    return label in SOURCE_CODE and (allowed is None or label in allowed)


def code_source(label):
    """The key of SOURCES whose files a run imports for a build that records `label`."""
    if label not in SOURCE_CODE:
        raise KeyError(f'unknown build source {label}; known: {sorted(SOURCE_CODE)}')
    return SOURCE_CODE[label]


def _origin(src):
    return f"teammate commit {src['commit']}" if src['commit'] else 'the files of this repository'


def _source(source):
    if source not in SOURCES:
        raise KeyError(f'unknown teammate source {source}; known: {sorted(SOURCES)}')
    return SOURCES[source]


def root_of(source=DEFAULT_SOURCE):
    """The worktree of a pinned teammate source, from its environment variable (SOURCES[source]['env']), resolved at call
    time; the root of the `repo` source is this repository. Raises TeamSourceNotConfigured naming the variable when it is
    unset or not a directory."""
    src = _source(source)
    if src['env'] is None:
        return Path(src['root'])
    return _env_dir(src['env'], f"the teammates' {source} worktree (commit {src['commit']})")


def _configured_roots():
    """Roots of the sources whose variable is set (the others cannot have modules loaded from them)."""
    return [Path(r) for r in (os.environ.get(src['env']) for src in SOURCES.values() if src['env']) if r]


def source_pins(source=DEFAULT_SOURCE, module='preprocess_full'):
    """{relative path: sha256} of the files that importing `module` from `source` runs (what a build records)."""
    src = _source(source)
    return {rel: src['pins'][rel] for rel in src['closure'][module]}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_pins(module, root=None, source=DEFAULT_SOURCE):
    """Raise unless every teammate file that importing `module` runs has its pinned sha256 under `root` (default: the
    review worktree of `source`, `root_of(source)`)."""
    src = _source(source)
    root = root_of(source) if root is None else root
    if module not in src['closure']:
        raise KeyError(f'{module} is not a pinned teammate module of {source}; known: {sorted(src["closure"])}')
    for rel in src['closure'][module]:
        path = Path(root) / rel
        if not path.is_file():
            raise FileNotFoundError(f'pinned teammate file missing: {path}')
        got = _sha256(path)
        if got != src['pins'][rel]:
            raise RuntimeError(f'{rel}: sha256 {got} does not match the pin {src["pins"][rel]} '
                               f'({_origin(src)}) under {root}')


def _all_stems(module):
    """Module names that importing `module` runs in any source (the stems to evict before a source switch)."""
    return {Path(rel).stem for src in SOURCES.values() for rel in src['closure'].get(module, ())}


def import_team(module, root=None, source=DEFAULT_SOURCE):
    """Import teammate `module` from the review worktree `root` (default: `root_of(source)`, 'gnn-331ce0b' or
    'main-c7899f0') after checking the sha256 of every file the import runs. The source's directories go first on sys.path,
    so its own `tokenize_asm_gz` (and, for the gnn source, `transformer_model` and `token_mapping`) win over same-named
    modules of ours or of the other source. Before the import, the modules that `module` runs in any source are evicted
    from sys.modules unless they are already the pinned file of this source, so importing one source never leaves the
    other's modules loaded (the files of the `repo` source, which sits inside our own tree, are told apart by path, not by
    root). Nothing is written under `root`: bytecode caching is switched off for the import."""
    src = _source(source)
    root = Path(root_of(source) if root is None else root)
    check_pins(module, root, source)
    wanted = {Path(rel).stem: (root / rel).resolve() for rel in src['closure'][module]}
    roots = [r.resolve() for r in _configured_roots()] + ([root.resolve()] if src['env'] else [])
    repo_files = {(Path(o['root']) / rel).resolve() for o in SOURCES.values() if o['env'] is None
                  for files in o['closure'].values() for rel in files}
    for stem in _all_stems(module):
        loaded = sys.modules.get(stem)
        if loaded is None:
            continue
        where = Path(getattr(loaded, '__file__', None) or '.').resolve()
        if where != wanted.get(stem) and (where in repo_files or any(r in where.parents for r in roots)):
            del sys.modules[stem]           # another source's (or another root's) module: it must not stay loaded
    own = [os.path.normcase(os.path.abspath(root / d)) for d in src['dirs']]
    foreign = {os.path.normcase(os.path.abspath(r / d)) for r in _configured_roots() for o in SOURCES.values()
               for d in o['dirs']} | set(own)
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


DATASET_OF = {'kfold': 'mendeley_x86', 'kfold_v3': 'combined_v3'}          # catalog.DATASETS keys, for messages and settings
DEFAULT_ARCH = {'kfold': 'x86', 'kfold_v3': 'all'}
BUILD_ARCHES = {'kfold': ('x86', 'all'), 'kfold_v3': ('all',)}             # a build is made for one of these; `matched` is never built


def _check_dataset(protocol, arch):
    if protocol not in DATASET_OF:
        raise ValueError(f'protocol {protocol!r}: the team pipelines run on {" or ".join(DATASET_OF)}')
    if arch not in ('x86', 'all', 'matched'):
        raise ValueError(f'arch {arch!r}')


def fold_config(fold, seed, protocol='kfold', arch='x86'):
    """The RunConfig whose manifest the `kfold_x86_gnn-mblk15_s{seed}` runs use (val_seed None: the validation carve
    follows the seed); `protocol` and `arch` name another dataset."""
    _check_dataset(protocol, arch)
    return RunConfig(protocol=protocol, fold=fold, arch=arch, seed=seed, model='gnn', mask_blocks=0.15)


@lru_cache(maxsize=2)
def _index(protocol='kfold'):
    return load_index(protocol=protocol)


@lru_cache(maxsize=None)
def sample_rows(protocol='kfold', arch='x86'):
    """{sample_id: index row} (label, arch, family, corpus, fold, ...) of the binaries of a dataset that have a fold under its
    protocol: the rows of `arch` (`common.manifests.arch_rows`), then those with a fold. Under kfold_v3 `fold` is the v3 fold
    (rows are copies, as in `make_manifest`), so a build records the fold the models use. The default is the Mendeley x86 binaries."""
    _check_dataset(protocol, arch)
    index = _index(protocol)
    if protocol == 'kfold_v3':
        if not any(r['fold_v3'] is not None for r in index):
            raise ValueError(f'{manifests.CACHE} has no fold_v3 column; set GT_CACHE to the combined cache')
        index = [{**r, 'fold': r['fold_v3']} for r in index]
    return {r['sample_id']: r for r in arch_rows(arch, index) if r['fold'] is not None}


@lru_cache(maxsize=None)
def fold_manifest(fold, seed, protocol='kfold', arch='x86'):
    return make_manifest(fold_config(fold, seed, protocol, arch), _index(protocol))


@lru_cache(maxsize=None)
def all_ids(protocol='kfold', arch='x86'):
    """Sorted sample ids of the five folds' test sets; the test sets are disjoint (asserted)."""
    ids = [r['sample_id'] for f in range(5) for r in fold_manifest(f, 0, protocol, arch).test]
    if len(ids) != len(set(ids)):
        raise AssertionError('a sample is in the test set of two folds')
    return sorted(ids)


@lru_cache(maxsize=2)
def _rows_of(bundle):
    with (Path(bundle) / 'samples.csv').open(encoding='utf-8', newline='') as f:
        return {r['sample_id']: r for r in csv.DictReader(f)}


def _zero_config():
    """True when neither the bundle nor GT_TEAM_ASM_DIR is configured: the CFGs and listings are then the ones of the
    Mendeley collections (`build_cache.corpus_samples('mendeley')`)."""
    try:
        source_for('bundle')
    except MissingSource:
        return not os.environ.get(ASM_DIR_ENV)
    return False


@lru_cache(maxsize=4)
def _collection_rows(collections, corpus='mendeley'):
    """{sample_id: samples.csv row, with its `root`} of a corpus (default mendeley) under the `collections` folder (the
    argument only keys the cache; `corpus_samples` resolves the folder itself)."""
    from s02_preprocess.cache import build_cache
    try:
        if corpus != 'mendeley':
            build_cache.require_collections([corpus])             # names the missing folder and the zip that holds it
        return {r['sample_id']: r for r in build_cache.corpus_samples(corpus)}
    except MissingSource as e:
        raise TeamSourceNotConfigured(f'{e}; or set GT_BUNDLE and GT_TEAM_ASM_DIR' if corpus == 'mendeley' else str(e)) from e


def _sample_row(sample_id, corpus='mendeley'):
    """(samples.csv row, cfg_path, asm_path) of a binary, before any existence check. A mendeley binary: the bundle's row with
    the listing in `asm_dir()`, or without a bundle and GT_TEAM_ASM_DIR the collection row with its CFG and listing, each
    relative to the folder of the row (`root`). A binary of another corpus (the `corpus` of its index row): always the row of
    its collection, whatever the bundle and GT_TEAM_ASM_DIR say."""
    if corpus != 'mendeley' or _zero_config():
        collections = source_for('collections')
        row = _collection_rows(str(collections), corpus).get(sample_id)
        if row is None:
            raise KeyError(f'{sample_id}: no row of the {corpus} corpus under {collections}')
        if not row['asm_path']:
            raise FileNotFoundError(f'{sample_id}: the samples.csv of {row["root"]} has no listing for it (asm_path is empty)')
        root = Path(row['root'])
        return row, root / row['cfg_path'], root / row['asm_path']
    bundle = bundle_dir()
    row = _rows_of(bundle).get(sample_id)
    if row is None:
        raise KeyError(f'{sample_id}: no row in {bundle / "samples.csv"}')
    return row, bundle / row['cfg_path'], asm_dir() / f'{sample_id}.asm.gz'


def paths_for(sample_id, corpus='mendeley'):
    """(cfg_path, asm_path) of a binary: the CFG in the Mendeley bundle (samples.csv cfg_path, `bundle_dir()`), the IDA
    listing in `asm_dir()`. Without a bundle and without GT_TEAM_ASM_DIR both come from the Mendeley collections
    (`common.paths.source_for('collections')`, `CT_GAT/data` by default). A missing row or file is an error; a half-configured
    machine (the bundle without GT_TEAM_ASM_DIR or the reverse) is a TeamSourceNotConfigured naming the missing one.
    `corpus` is the corpus of the binary's index row: a binary of `ransomware_balanced` or `balanced_goodware` has both files in
    its collection (`build_cache.corpus_samples`), and the bundle and GT_TEAM_ASM_DIR do not apply to it."""
    _, cfg_path, asm_path = _sample_row(sample_id, corpus)
    for p in (cfg_path, asm_path):
        if not p.is_file():
            raise FileNotFoundError(f'{sample_id}: missing {p}')
    return cfg_path, asm_path


def team_row(sample_id, corpus='mendeley'):
    """The row `construct_blocks.gnn_graph_from_export` / `graph_from_export` expect, from the bundle's samples.csv or the
    collections' (the fields `load_split_rows` fills: the csv columns, an int `label`, `cfg_file`). A missing row or file is
    an error. `corpus` as in `paths_for`."""
    cfg_path, _ = paths_for(sample_id, corpus)
    row = {k: v for k, v in _sample_row(sample_id, corpus)[0].items() if k != 'root'}
    row['label'] = int(row['label'])
    row['cfg_file'] = cfg_path
    return row


def team_config(fold, seed, arm, protocol='kfold', arch='x86', arch_balance=False):
    """The RunConfig of a team arm; its `.name` is the run directory name (`kfold{f}_x86_gnn_s{seed}_team-<arm>`, and
    `v3kfold{f}_all_gnn-ab_s{seed}_team-<arm>` for kfold_v3, all and `arch_balance`)."""
    _check_dataset(protocol, arch)
    return RunConfig(protocol=protocol, fold=fold, arch=arch, seed=seed, model='gnn', tag=f'team-{arm}', arch_balance=arch_balance)
