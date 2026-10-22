"""What the repository knows about its datasets: the collection folders, the corpora built from them, the datasets (an
evaluation unit: protocols, corpora, fold files, base cache) and where each one lives by default.

Nothing here is a machine path: every root is computed from this file. `common/paths.py` imports this module and merges it
with the per-machine datasets.json (`paths.registry`), so this one does not import `common.paths`. A new dataset is a row of
`DATASETS` plus its fold files; a new corpus is a row of `CORPORA`.

    python -m common.catalog        # which collections, bundle, cache and fold files each dataset has on this machine
"""
import json
from dataclasses import dataclass
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]           # CT_GAT/pipeline
CT_GAT = PIPELINE_ROOT.parent                                  # the layout tokenize_asm_gz.PROJECT_ROOT already uses
FOLDS_ROOT = PIPELINE_ROOT / 's02_preprocess' / 'folds'        # the committed fold CSVs
DATA_ROOT = CT_GAT / 'data'                                    # the unzipped collections
CACHE_ROOT = CT_GAT / 'cache'                                  # CT_GAT/cache/<dataset>
PROCESSED = CT_GAT / 'processed'                               # Yanping's scripts, the preprocessed zips

COLLECTIONS = ('Mendeley_Ransomware', 'Mendeley_Goodware', 'Ransomware_Balanced', 'Balanced_Goodware')   # folder names, as the zips unpack
BUNDLE_SCHEMA = 'mendeley-gnn-bundle/1'                        # summary.json of Mendeley_GNN_v1
V3_FILES = ('folds_all.csv', 'arch_matched.csv')               # what the folds_v3 source holds


@dataclass(frozen=True)
class Corpus:
    collections: tuple          # folders under the collections source, each with samples.csv
    rule: str                   # 'split' or 'eligible': the branch of build_cache.exclusion
    bundle: str = ''            # a folder that replaces `collections` when it is configured or found (summary.json schema BUNDLE_SCHEMA)


@dataclass(frozen=True)
class Dataset:
    protocols: tuple            # --protocol values that read the cache by default
    corpora: tuple
    folds: dict                 # source key -> path under FOLDS_ROOT
    base: str = None            # the dataset whose cache seeds this one


CORPORA = {
    'mendeley': Corpus(('Mendeley_Ransomware', 'Mendeley_Goodware'), 'split', bundle='Mendeley_GNN_v1'),
    'ransomware_balanced': Corpus(('Ransomware_Balanced',), 'eligible'),
    'balanced_goodware': Corpus(('Balanced_Goodware',), 'eligible'),
}
DATASETS = {                    # order = default protocol resolution order
    'mendeley_x86': Dataset(('bundle', 'kfold'), ('mendeley',), {'folds': 'folds_mendeley.csv', 'folds_v3': 'v3'}),
    'combined_v3': Dataset(('kfold_v3',), ('mendeley', 'ransomware_balanced', 'balanced_goodware'),
                           {'folds': 'folds_mendeley.csv', 'folds_v3': 'v3'}, base='mendeley_x86'),
}


def read_summary(folder):
    """The parsed summary.json of a folder, or {} when it has none or it is unreadable."""
    try:
        return json.loads((Path(folder) / 'summary.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def find_bundle(folder):
    """`folder` when its summary.json carries the bundle's schema, else None."""
    return Path(folder) if read_summary(folder).get('schema') == BUNDLE_SCHEMA else None


def collection_state(data_root, name):
    """Where the collection `name` stands under `data_root`: 'present' (its samples.csv is there), 'nested' (only
    `<name>/<name>/samples.csv` is: the unzip went one folder too deep, as Windows Explorer's "Extract All" does with its
    default destination), 'zip' (`<name>.zip` is there and no folder: not unzipped yet) or 'missing'."""
    data_root = Path(data_root)
    if (data_root / name / 'samples.csv').is_file():
        return 'present'
    if (data_root / name / name / 'samples.csv').is_file():
        return 'nested'
    if (data_root / f'{name}.zip').is_file() and not (data_root / name).exists():
        return 'zip'
    return 'missing'


def collection_hint(data_root, name, state=None):
    """What to do about a collection that is nested or still zipped; '' when it is present or plainly missing."""
    data_root = Path(data_root)
    state = state or collection_state(data_root, name)
    if state == 'nested':
        return (f'{(data_root / name / name).as_posix()}/samples.csv is one folder too deep (the unzip went into a folder named '
                f'after the zip): move the inner {name} folder up so that {(data_root / name).as_posix()}/samples.csv exists')
    if state == 'zip':
        return f'{(data_root / (name + ".zip")).as_posix()} is not unzipped: unzip it into {data_root.as_posix()}'
    return ''


def status(data_root=DATA_ROOT, cache_root=CACHE_ROOT, folds_root=FOLDS_ROOT, bundle=None, caches=None, folds=None):
    """Per dataset, what is on this machine: {name: {'collections': {folder: {'present', 'state', 'snapshot'}}, 'bundle': (kind,
    path), 'cache': path, 'built': bool, 'folds': {key: (path, found)}}}. A collection is present when its samples.csv exists
    under `data_root`; its `state` is also 'nested', 'zip' or 'missing' (`collection_state`); `bundle` is the configured bundle folder or None (kind 'configured', else 'discovered' under `data_root`,
    else 'absent'); `caches` maps a dataset name to a path that replaces its default cache and `folds` maps a dataset name to
    {source key: path} for the fold files that replace its own."""
    caches, folds = caches or {}, folds or {}
    found = find_bundle(data_root / CORPORA['mendeley'].bundle)
    kind, path = ('configured', Path(bundle)) if bundle else ('discovered', found) if found else ('absent', None)
    report = {}
    for name, ds in DATASETS.items():
        folders = dict.fromkeys(f for c in ds.corpora for f in CORPORA[c].collections)
        cache = Path(caches.get(name, cache_root / name))
        fold_files = {}
        for key, rel in ds.folds.items():
            path_ = Path(folds.get(name, {}).get(key, folds_root / rel))
            fold_files[key] = (path_, all((path_ / f).is_file() for f in V3_FILES) if key == 'folds_v3' else path_.is_file())
        states = {f: collection_state(data_root, f) for f in folders}
        report[name] = {'collections': {f: {'present': states[f] == 'present', 'state': states[f],
                                           'snapshot': read_summary(data_root / f).get('snapshot_utc')} for f in folders},
                        'bundle': (kind, path), 'cache': cache, 'built': (cache / 'index.csv').is_file(), 'folds': fold_files}
    return report


def show(path):
    path = Path(path)
    try:
        return path.relative_to(CT_GAT.parent).as_posix()
    except ValueError:
        return path.as_posix()


def render(report, data_root=DATA_ROOT):
    lines = []
    for name, info in report.items():
        ds = DATASETS[name]
        lines.append(f"{name}: protocols {', '.join(ds.protocols)}; corpora {', '.join(ds.corpora)}"
                     + (f'; base {ds.base}' if ds.base else ''))
        lines.append(f'  collections under {show(data_root)}:')
        for folder, c in info['collections'].items():
            when = f" (snapshot {c['snapshot']})" if c['present'] and c['snapshot'] else ''
            hint = collection_hint(data_root, folder, c['state'])
            lines.append(f"    {folder}: {'present' if c['present'] else c['state']}{when}" + (f'; {hint}' if hint else ''))
        kind, path = info['bundle']
        lines.append(f"  bundle: {kind}{f' {show(path)}' if path else ''}")
        lines.append(f"  cache {show(info['cache'])}: {'built' if info['built'] else 'not built'}")
        for key, (path, ok) in info['folds'].items():
            lines.append(f"  {key} {show(path)}: {'found' if ok else 'missing'}")
    return '\n'.join(lines)


def main():
    from common import paths        # here and not at the top: paths imports this module
    data = paths.source_for('collections')
    try:
        bundle = paths.source_for('bundle')
    except paths.MissingSource:
        bundle = None
    registry = paths.registry()
    folds = {name: {key: paths.source_for(key, dataset=name) for key in ds.folds} for name, ds in DATASETS.items()}
    caches = {name: paths.cache_for(None, registry=registry, name=name) for name in DATASETS}   # GT_CACHE first, as the trainers do
    print(render(status(data, bundle=bundle, caches=caches, folds=folds), data))


if __name__ == '__main__':
    main()
