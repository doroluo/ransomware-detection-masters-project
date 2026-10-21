"""Manifests: built rows of the cache, the validation carve, the training subsample and the per-binary arrays."""
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common import catalog
from common.config import CACHE, RunConfig
from common.paths import MissingSource, cache_in, protocol_in, registry


class MissingCache(FileNotFoundError):
    """The index of a cache is not there: the cache is not built. `str(e)` is the whole message, with the build command."""


def missing_cache_message(cache, protocol=None):
    """One paragraph: the cache that is not built and the command that builds it. The dataset is the registry entry with this
    cache, else the one `--cache` names on the command line, else the first one that lists `protocol` (GT_CACHE can point
    anywhere)."""
    reg = registry()
    cache = Path(cache)
    name = next((n for n, e in reg.items() if Path(e['cache']).resolve() == cache.resolve()), None)
    name = name or cache_in(sys.argv[1:], reg)
    name = name or next((n for n, e in reg.items() if (protocol or protocol_in(sys.argv[1:])) in e['protocols']), None)
    shown = catalog.show(cache)
    text = f'no cache index at {shown}/index.csv: the cache is not built.'
    if name not in catalog.DATASETS:
        return (f"{text} Build a dataset's cache with python -m s02_preprocess.cache.build_cache --cache <name> from "
                f'CT_GAT/pipeline (GT_CACHE or datasets.json says where the cache goes).')
    ds = catalog.DATASETS[name]
    own = [c for c in ds.corpora if not ds.base or c not in catalog.DATASETS[ds.base].corpora]
    folders = [f for c in own for f in catalog.CORPORA[c].collections]
    needs = f"needs the {' and '.join(folders)} collections under CT_GAT/data"
    if ds.base:
        needs += f', and the {ds.base} cache built first'
    text += f' Build it with python -m s02_preprocess.cache.build_cache --cache {name} from CT_GAT/pipeline ({needs}).'
    if os.environ.get('GT_CACHE'):
        text += f' GT_CACHE is set, so the build writes there.'
    return text


def index_file(cache=None, protocol=None):
    """The index.csv of `cache` (default: the run's cache); MissingCache when the cache is not built."""
    path = Path(cache or CACHE) / 'index.csv'
    if not path.is_file():
        raise MissingCache(missing_cache_message(path.parent, protocol))
    return path


def run_command(main, *args):
    """`main(*args)` as a command: a missing cache or a missing source folder ends it with its message on stderr and exit
    code 1, no traceback."""
    try:
        return main(*args)
    except (MissingCache, MissingSource) as e:
        raise SystemExit(str(e)) from None


def load_index(cache=None, protocol=None):
    """Built rows of `cache` (default: the run's cache). `fold` (Mendeley family holdout) and `fold_v3`
    (family_holdout_v3 folds_all, every corpus) are ints, or None where the row has none; a cache without the corpus
    columns reads as all-Mendeley. Raises MissingCache (a FileNotFoundError) when the cache is not built; `protocol` only
    helps name the dataset in its message."""
    with index_file(cache, protocol).open(encoding='utf-8', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['status'] == 'ok']
    for r in rows:
        r['label'] = int(r['label'])
        r.setdefault('corpus', 'mendeley')
        for k in ('fold', 'fold_v3'):
            r[k] = int(r[k]) if r.get(k) else None
    return rows


@dataclass
class Manifest:
    train: list
    val: list
    test: list
    val_families: list

    def save(self, path):
        doc = {name: [r['sample_id'] for r in getattr(self, name)] for name in ('train', 'val', 'test')}
        doc['val_families'] = self.val_families
        path.write_text(json.dumps(doc, indent=1), encoding='utf-8')


def carve_validation(rows, fraction, seed):
    """Hold out whole ransomware families and whole goodware identity groups, about `fraction` of each class."""
    rng = np.random.default_rng(seed)
    held = set()
    val_families = []
    for label, key in ((1, 'family'), (0, 'identity_group')):
        members = {}
        for r in rows:
            if r['label'] == label:
                members.setdefault(r[key], []).append(r['sample_id'])
        groups = sorted(members)
        target, taken = fraction * sum(len(v) for v in members.values()), 0
        for i in rng.permutation(len(groups)):
            if taken >= target:
                break
            held.update(members[groups[i]])
            taken += len(members[groups[i]])
            if label == 1:
                val_families.append(groups[i])
    return [r for r in rows if r['sample_id'] not in held], [r for r in rows if r['sample_id'] in held], sorted(val_families)


def grouped_folds(rows, k, seed):
    """`rows` split into `k` held-out sets of whole groups (ransomware family, goodware identity group: the keys of
    `carve_validation`) that partition them. Per class the groups go largest first into the fold with the fewest binaries of
    that class, groups of equal size in a permutation from default_rng([seed, 4093]); the rows of a fold keep their order in
    `rows`. A class with fewer than `k` groups cannot be split in `k`."""
    rng = np.random.default_rng([seed, 4093])
    fold_of = {}
    for label, key in ((1, 'family'), (0, 'identity_group')):
        members = {}
        for r in rows:
            if r['label'] == label:
                members.setdefault(r[key], []).append(r['sample_id'])
        groups = sorted(members)
        if len(groups) < k:
            raise ValueError(f"{len(groups)} {key} group(s) among the {'ransomware' if label else 'goodware'} binaries: "
                             f'too few for {k} folds that keep a group whole')
        load = [0] * k
        for i in sorted(rng.permutation(len(groups)), key=lambda i: -len(members[groups[i]])):
            f = int(np.argmin(load))
            load[f] += len(members[groups[i]])
            fold_of.update(dict.fromkeys(members[groups[i]], f))
    return [[r for r in rows if fold_of[r['sample_id']] == f] for f in range(k)]


def subsample_train(train, fraction, seed):
    """Whole groups (ransomware family, goodware identity group) per class: in a permutation from
    default_rng([seed, 1729]), the shortest prefix whose binaries reach `fraction` of the class. Nested in `fraction`."""
    rng = np.random.default_rng([seed, 1729])
    kept = set()
    for label, key in ((1, 'family'), (0, 'identity_group')):
        members = {}
        for r in train:
            if r['label'] == label:
                members.setdefault(r[key], []).append(r['sample_id'])
        groups = sorted(members)
        target, taken = fraction * sum(len(v) for v in members.values()), 0
        for i in rng.permutation(len(groups)):
            if taken >= target:
                break
            kept.update(members[groups[i]])
            taken += len(members[groups[i]])
    return [r for r in train if r['sample_id'] in kept]


def arch_rows(arch, index):
    """'all': every row. 'matched': rows with arch_matched == 1, any architecture; the cache must have the column.
    Otherwise the rows of that architecture."""
    if arch == 'matched':
        if not any('arch_matched' in r for r in index):
            raise ValueError(f'{CACHE} has no arch_matched column; set GT_CACHE to the combined cache')
        return [r for r in index if r.get('arch_matched') in (1, '1', True)]
    return [r for r in index if arch == 'all' or r['arch'] == arch]


def make_manifest(cfg: RunConfig, index):
    """Rows without a fold under the run's protocol are left out of train, val and test alike. kfold_v3 rows are
    copies whose `fold` is their v3 fold, so everything downstream reports the fold the run used. The validation carve
    and the train-fraction subsample use `cfg.val_seed` (default: `cfg.seed`), so with a fixed `val_seed` train, val
    and test do not change with the seed."""
    rows = arch_rows(cfg.arch, index)
    if cfg.protocol == 'bundle':
        pool, test = [r for r in rows if r['split'] == 'train'], [r for r in rows if r['split'] == 'test']
    else:
        if cfg.protocol == 'kfold_v3':
            if not any(r['fold_v3'] is not None for r in index):
                raise ValueError(f'{CACHE} has no fold_v3 column; set GT_CACHE to the combined cache')
            rows = [{**r, 'fold': r['fold_v3']} for r in rows]
        rows = [r for r in rows if r['fold'] is not None]
        pool, test = [r for r in rows if r['fold'] != cfg.fold], [r for r in rows if r['fold'] == cfg.fold]
    split_seed = cfg.seed if cfg.val_seed is None else cfg.val_seed
    train, val, val_families = carve_validation(pool, cfg.val_fraction, split_seed)
    if cfg.train_fraction < 1:
        train = subsample_train(train, cfg.train_fraction, split_seed)
    return Manifest(train, val, test, val_families)


def load_sample(sample_id, ytok=False):
    """The cache arrays of one binary; with `ytok` also the reader's per-instruction triplet (side/yanping), aligned with
    `mn`. A binary without it raises (FileNotFoundError naming the sample)."""
    with np.load(CACHE / 'samples' / f'{sample_id}.npz') as z:
        out = {k: z[k] for k in ('mn', 'sig', 'beh', 'api', 'blk_insn_ptr', 'blk_api_ptr', 'blk_feat',
                                 'fn_blk_ptr', 'fn_flags', 'cfg_edges', 'call_edges', 'edge_kind', 'edge_back')
               if k in z.files}
    if ytok:
        from experiments.side_arrays.import_reader_side import load_ytok
        out['ytok'] = load_ytok(CACHE, sample_id)
        if len(out['ytok']) != len(out['mn']):
            raise ValueError(f'{sample_id}: {len(out["ytok"])} reader instructions, {len(out["mn"])} in the cache')
    return out
