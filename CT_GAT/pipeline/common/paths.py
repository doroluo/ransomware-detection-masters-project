"""Where the CFG cache, the run directories and the raw exports live on this machine.

Nothing here knows a machine-specific path. The effective registry (`registry()`) is the per-machine `datasets.json` followed
by the datasets of `common/catalog.py` that the file does not name; a name in the file replaces the catalog's. `datasets.json`
(next to the pipeline root, gitignored, optional, created per machine from `datasets.json.example`; `GT_DATASETS` names
another file) maps a dataset name to its cache and to the protocols that read it:

    {"v2": {"cache": "<dir>", "protocols": ["bundle", "kfold"], "sources": {"bundle": "<dir>", ...}},
     "combined": {"cache": "<dir>", "protocols": ["kfold_v3"]}}

A relative path in the file is relative to the file. A catalog dataset has the cache `CT_GAT/cache/<name>` and the source
defaults of the catalog (the committed folds, `CT_GAT/data` for the collections, `CT_GAT/processed/cfg_subgraphs` for the
pipeline-B build). Everything is resolved in the order environment variable > datasets.json > catalog default.
Resolution of the cache of a run, first match wins:
  1. GT_CACHE (a directory; empty counts as unset);
  2. `--cache <name>` on the command line, a dataset name of the registry;
  3. the first dataset of the registry (file order, then catalog order) that lists the run's protocol;
  4. `UNSET`, a relative path that does not exist, so that a command for a protocol nobody lists fails with a missing-file
     error that names GT_CACHE.

`GT_RUNS` overrides the run directories. The raw inputs a cache is built from (`sources`) are looked up by `source_for`: the
environment variable of that source, then the first dataset of the registry that defines it (the file's, then the catalog's).
"""
import json
import os
import sys
from pathlib import Path

from common.catalog import CACHE_ROOT, DATA_ROOT, DATASETS, FOLDS_ROOT, PIPELINE_ROOT, PROCESSED

UNSET = Path('GT_CACHE-not-set')
RESERVED = {'cache', 'protocols', 'sources'}

# source name -> (environment variable, what it is)
SOURCES = {
    'bundle': ('GT_BUNDLE', 'the Mendeley_GNN_v1 bundle (samples.csv, cfg/, provenance/)'),
    'collections': ('GT_COLLECTIONS', 'the folder holding the unzipped collections (common/catalog.py COLLECTIONS)'),
    'folds': ('GT_FOLDS', 'the Mendeley family-holdout fold CSV (folds_mendeley.csv)'),
    'folds_v3': ('GT_FOLDS_V3', 'the folder with folds_all.csv and arch_matched.csv'),
    'reader_root': ('GT_READER_ROOT', "the team's operand-role reader output (nothing reads it now; kept so that a datasets.json that names it loads)"),
    'runs': ('GT_RUNS', 'the run directories'),
    'team_b_build': ('GT_TEAM_B_BUILD', 'a pipeline-B build folder of <sample_id>.pt files (experiments/team_pipelines/build_b.py), the input of --model fivechannel'),
}


class MissingSource(RuntimeError):
    pass


def datasets_file(env=None):
    env = os.environ if env is None else env
    return Path(env['GT_DATASETS']) if env.get('GT_DATASETS') else PIPELINE_ROOT / 'datasets.json'


def load_registry(env=None):
    """{name: {'cache': Path, 'protocols': tuple, 'sources': {name: Path}}} from datasets.json; {} when the file is absent."""
    path = datasets_file(env)
    if not path.is_file():
        return {}
    doc = json.loads(path.read_text(encoding='utf-8'))
    registry = {}
    for name, entry in doc.items():
        if name.startswith('_'):
            continue                                        # "_comment" and the like
        extra = set(entry) - RESERVED
        if extra or 'cache' not in entry:
            raise ValueError(f'{path}: dataset {name} needs "cache" and may only have {sorted(RESERVED)}, got {sorted(entry)}')
        unknown = set(entry.get('sources', {})) - set(SOURCES)
        if unknown:
            raise ValueError(f'{path}: dataset {name} has unknown sources {sorted(unknown)}; known: {sorted(SOURCES)}')
        registry[name] = {'cache': path.parent / entry['cache'], 'protocols': tuple(entry.get('protocols', ())),
                          'sources': {k: path.parent / v for k, v in entry.get('sources', {}).items()}}
    return registry


def registry(env=None):
    """The effective registry: datasets.json in file order, then every catalog dataset the file does not name, with the cache
    `CT_GAT/cache/<name>` and the catalog's source defaults."""
    merged = load_registry(env)
    for name, ds in DATASETS.items():
        if name not in merged:
            sources = {'collections': DATA_ROOT, 'team_b_build': PROCESSED / 'cfg_subgraphs',
                       **{key: FOLDS_ROOT / rel for key, rel in ds.folds.items()}}
            merged[name] = {'cache': CACHE_ROOT / name, 'protocols': ds.protocols, 'sources': sources}
    return merged


_effective = registry            # `registry` is also a parameter name below


def cache_for(protocol, env=None, registry=None, name=None):
    """The cache directory of a run (see the module docstring for the order). `registry` and `name` default to the
    effective registry of this machine and to no name."""
    env = os.environ if env is None else env
    if env.get('GT_CACHE'):
        return Path(env['GT_CACHE'])
    registry = _effective(env) if registry is None else registry
    if name:
        if name not in registry:
            raise ValueError(f'unknown dataset {name!r}; the registry has {sorted(registry) or "none (see datasets.json.example)"}')
        return Path(registry[name]['cache'])
    for entry in registry.values():
        if protocol is None or protocol in entry['protocols']:
            return Path(entry['cache'])
    return UNSET


def source_for(key, env=None, registry=None, dataset=None):
    """A raw input or output location by source name (`SOURCES`): its environment variable, else the first dataset of the
    registry that defines it (datasets.json, then the catalog). With `dataset`, that dataset's entry comes first among the
    datasets.json ones when the file names it, and first among the catalog's otherwise: the order is the environment
    variable, the file's entry of `dataset`, the file's other entries, the catalog row of `dataset`, the other catalog rows.
    `runs` falls back to `runs/` inside the pipeline root. Raises MissingSource naming both ways to set it."""
    env = os.environ if env is None else env
    var, what = SOURCES[key]
    if env.get(var):
        return Path(env[var])
    registry = _effective(env) if registry is None else registry
    names = list(registry)
    if dataset in registry:
        in_file = set(load_registry(env))
        names = sorted(names, key=lambda n: (n not in in_file, n != dataset))         # stable: file order, catalog order kept
    for name in names:
        if key in registry[name]['sources']:
            return Path(registry[name]['sources'][key])
    if key == 'runs':
        return PIPELINE_ROOT / 'runs'
    raise MissingSource(f'{what} is not configured: set {var} or add "{key}" under "sources" of a dataset in '
                        f'{datasets_file(env)} (see datasets.json.example)')


def runs_for(env=None, registry=None):
    return source_for('runs', env, registry)


def _value_in(argv, flag):
    for i, a in enumerate(argv):
        if a.startswith(flag + '='):
            return a.partition('=')[2]
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]


def protocol_in(argv):
    """The value of --protocol in a command line, or None."""
    return _value_in(argv, '--protocol')


def cache_in(argv, registry):
    """The value of --cache in a command line when it names a dataset of `registry`, else None (a command whose own --cache
    is a directory is not a dataset selection)."""
    value = _value_in(argv, '--cache')
    return value if value in registry else None


_REGISTRY = registry()
DEFAULT_CACHE = cache_for('bundle', env={}, registry=_REGISTRY)           # the registry's caches, whatever GT_CACHE says
COMBINED_CACHE = cache_for('kfold_v3', env={}, registry=_REGISTRY)
CACHE = cache_for(protocol_in(sys.argv[1:]), registry=_REGISTRY, name=cache_in(sys.argv[1:], _REGISTRY))
RUNS = runs_for(registry=_REGISTRY)
