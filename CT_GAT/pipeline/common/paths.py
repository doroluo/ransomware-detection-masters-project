"""Where the CFG cache, the run directories, the block exports and the raw exports live on this machine.

Nothing here knows a machine-specific path. `datasets.json` (next to the pipeline root, gitignored, created per machine from
`datasets.json.example`; `GT_DATASETS` names another file) maps a dataset name to its cache and to the protocols that read it:

    {"v2": {"cache": "<dir>", "protocols": ["bundle", "kfold"], "sources": {"bundle": "<dir>", ...}},
     "combined": {"cache": "<dir>", "protocols": ["kfold_v3"]}}

A relative path in the file is relative to the file. Resolution of the cache of a run, first match wins:
  1. GT_CACHE (a directory; empty counts as unset);
  2. `--cache <name>` on the command line, a dataset name of the registry;
  3. the first dataset (in file order) that lists the run's protocol;
  4. `UNSET`, a relative path that does not exist, so that an unconfigured machine imports cleanly and a command that needs
     data fails with a missing-file error that names GT_CACHE.

`GT_RUNS` overrides the run directories and `GT_EXPORT` the block exports (default: `export/` inside the run directories,
so GT_RUNS alone moves both). The raw inputs a cache is built from (`sources`) are looked up by `source_for`: the
environment variable of that source, then the first dataset of the registry that defines it.
"""
import json
import os
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
UNSET = Path('GT_CACHE-not-set')
RESERVED = {'cache', 'protocols', 'sources'}

# source name -> (environment variable, what it is)
SOURCES = {
    'bundle': ('GT_BUNDLE', 'the Mendeley_GNN_v1 bundle (samples.csv, cfg/, provenance/)'),
    'collections': ('GT_COLLECTIONS', 'the folder holding Ransomware_Balanced/ and Balanced_Goodware/'),
    'mendeley_collections': ('GT_MENDELEY_COLLECTIONS', 'the folder holding Mendeley_Ransomware/ and Mendeley_Goodware/ (used when no bundle is configured)'),
    'folds': ('GT_FOLDS', 'the Mendeley family-holdout fold CSV (folds_mendeley.csv)'),
    'folds_v3': ('GT_FOLDS_V3', 'the folder with folds_all.csv and arch_matched.csv'),
    'reader_root': ('GT_READER_ROOT', "the team's operand-role reader output"),
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


def cache_for(protocol, env=None, registry=None, name=None):
    """The cache directory of a run (see the module docstring for the order). `registry` and `name` default to the
    datasets.json of this machine and to no name."""
    env = os.environ if env is None else env
    if env.get('GT_CACHE'):
        return Path(env['GT_CACHE'])
    registry = load_registry(env) if registry is None else registry
    if name:
        if name not in registry:
            raise ValueError(f'unknown dataset {name!r}; datasets.json has {sorted(registry) or "none (create it from datasets.json.example)"}')
        return Path(registry[name]['cache'])
    for entry in registry.values():
        if protocol is None or protocol in entry['protocols']:
            return Path(entry['cache'])
    return UNSET


def source_for(key, env=None, registry=None):
    """A raw input or output location by source name (`SOURCES`): its environment variable, else the first dataset that
    defines it. `runs` falls back to `runs/` inside the pipeline root. Raises MissingSource naming both ways to set it."""
    env = os.environ if env is None else env
    var, what = SOURCES[key]
    if env.get(var):
        return Path(env[var])
    registry = load_registry(env) if registry is None else registry
    for entry in registry.values():
        if key in entry['sources']:
            return Path(entry['sources'][key])
    if key == 'runs':
        return PIPELINE_ROOT / 'runs'
    raise MissingSource(f'{what} is not configured: set {var} or add "{key}" under "sources" of a dataset in '
                        f'{datasets_file(env)} (see datasets.json.example)')


def runs_for(env=None, registry=None):
    return source_for('runs', env, registry)


def export_for(env=None, registry=None):
    """GT_EXPORT when set, else `export/` inside the run directory (`runs_for`)."""
    env = os.environ if env is None else env
    return Path(env['GT_EXPORT']) if env.get('GT_EXPORT') else runs_for(env, registry) / 'export'


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
    is a directory, as side_io's, is not a dataset selection)."""
    value = _value_in(argv, '--cache')
    return value if value in registry else None


_REGISTRY = load_registry()
DEFAULT_CACHE = cache_for('bundle', env={}, registry=_REGISTRY)           # the registry's caches, whatever GT_CACHE says
COMBINED_CACHE = cache_for('kfold_v3', env={}, registry=_REGISTRY)
CACHE = cache_for(protocol_in(sys.argv[1:]), registry=_REGISTRY, name=cache_in(sys.argv[1:], _REGISTRY))
RUNS = runs_for(registry=_REGISTRY)
EXPORT = export_for(registry=_REGISTRY)
