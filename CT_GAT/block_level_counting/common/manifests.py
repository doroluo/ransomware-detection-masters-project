"""Manifests: built rows of the cache, the validation carve, the training subsample and the per-binary arrays."""
import csv
import json
from dataclasses import dataclass

import numpy as np

from common.config import CACHE, RunConfig


def load_index(cache=None):
    """Built rows of `cache` (default: the run's cache). `fold` (Mendeley family holdout) and `fold_v3`
    (family_holdout_v3 folds_all, every corpus) are ints, or None where the row has none; a cache without the corpus
    columns reads as all-Mendeley."""
    with ((cache or CACHE) / 'index.csv').open(encoding='utf-8', newline='') as f:
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
    copies whose `fold` is their v3 fold, so everything downstream reports the fold the run used."""
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
    train, val, val_families = carve_validation(pool, cfg.val_fraction, cfg.seed)
    if cfg.train_fraction < 1:
        train = subsample_train(train, cfg.train_fraction, cfg.seed)
    return Manifest(train, val, test, val_families)


def load_sample(sample_id):
    with np.load(CACHE / 'samples' / f'{sample_id}.npz') as z:
        return {k: z[k] for k in ('mn', 'sig', 'beh', 'api', 'blk_insn_ptr', 'blk_api_ptr', 'blk_feat',
                                  'fn_blk_ptr', 'fn_flags', 'cfg_edges', 'call_edges', 'edge_kind', 'edge_back')
                if k in z.files}
