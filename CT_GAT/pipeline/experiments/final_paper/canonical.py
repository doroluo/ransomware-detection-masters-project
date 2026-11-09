"""Fit a shared semantic API namespace using each outer fold's training samples only."""
import hashlib
import json
from pathlib import Path
import time
import argparse

import torch

from experiments.model_grid import runner
from experiments.model_grid.data import assert_disjoint, digest, manifest_for, subset, tok


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def vocabulary_lookup(source_names, fitted_names):
    """Keep reserved tokens; unknown semantic names map to UNK, regardless of source ID order."""
    canonical = {name: i + tok.FIRST_API_ID for i, name in enumerate(fitted_names)}
    return torch.tensor(list(range(tok.FIRST_API_ID)) + [canonical.get(name, tok.UNK_API) for name in source_names])


def fitting_rows(index, dataset, fold, val_seed, exclusions, smoke=0):
    splits = manifest_for(index, dataset, fold, val_seed)
    splits = {s: [r for r in rows if r['sample_id'] not in exclusions] for s, rows in splits.items()}
    assert_disjoint(splits)
    return subset(splits['train'], smoke) if smoke else splits['train']


def fit_common_vocabulary(builds, index, dataset, fold, val_seed, target, smoke=0):
    """Fit over the union of training representations; never inspect validation or test payloads."""
    builds = {k: Path(v) for k, v in builds.items()}
    settings = {k: json.loads((p / 'settings.json').read_text()) for k, p in builds.items()}
    excluded = next(iter(settings.values())).get('excluded', {})
    if any(s.get('excluded', {}) != excluded for s in settings.values()):
        raise ValueError('Vocabulary-fitting builds have different exclusions')
    rows = fitting_rows(index, dataset, fold, val_seed, excluded, smoke)
    train_ids = [r['sample_id'] for r in rows]
    contract = dict(dataset=dataset, fold=fold, val_seed=val_seed, smoke=smoke,
        train_ids=train_ids, index_sha256=file_sha256(index), fitting_scope='training representations only',
        api_base=tok.FIRST_API_ID,
        builds={k: dict(path=str(p.resolve()), settings_sha256=file_sha256(p / 'settings.json'),
                        vocabulary_sha256=file_sha256(p / 'api_vocab.json')) for k, p in builds.items()})
    target = Path(target)
    if target.exists():
        saved = json.loads(target.read_text())
        if saved['contract'] != contract:
            raise ValueError('Training vocabulary contract changed')
        return saved
    names, hashes = set(), {}
    for name, build in builds.items():
        source_names = json.loads((build / 'api_vocab.json').read_text())
        if len(source_names) != len(set(source_names)):
            raise ValueError('Source vocabulary contains duplicate names')
        used = set()
        hashes[name] = {}
        last = time.monotonic()
        for i, row in enumerate(rows, 1):
            sid = row['sample_id']
            path = build / f'{sid}.pt'
            hashes[name][sid] = file_sha256(path)
            payload = torch.load(path, mmap=True, map_location='cpu', weights_only=False)
            if int(payload['label']) != row['label']:
                raise ValueError(f'Label mismatch: {sid}')
            ids = torch.unique(payload['instruction_ids'][:, 3]).tolist()
            for api in ids:
                if api >= tok.FIRST_API_ID:
                    offset = api - tok.FIRST_API_ID
                    if offset >= len(source_names):
                        raise ValueError(f'API token outside source vocabulary: {sid}')
                    used.add(offset)
            del payload
            if time.monotonic() - last > 30:
                print(f'Vocabulary fold {fold}, {name}: {i}/{len(rows)} training samples', flush=True)
                last = time.monotonic()
        names.update(source_names[i] for i in used)
    result = dict(contract=contract, names=sorted(names), training_payload_sha256=hashes,
                  validation_or_test_used_for_fitting=False)
    result['fingerprint'] = digest(result)
    runner.atomic_json(target, result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', nargs='+', required=True, help='name=path for each fitting representation')
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--dataset', default='combined')
    p.add_argument('--fold', type=int, choices=range(5), required=True)
    p.add_argument('--val-seed', type=int, default=0)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--smoke', type=int, default=0)
    a = p.parse_args(argv)
    pairs = [s.split('=', 1) for s in a.build]
    if any(len(pair) != 2 for pair in pairs) or len({pair[0] for pair in pairs}) != len(pairs):
        p.error('Each build must have a unique name=path')
    torch.set_num_threads(1)
    result = fit_common_vocabulary(dict(pairs), a.index, a.dataset, a.fold, a.val_seed, a.out, a.smoke)
    print(json.dumps(dict(fold=a.fold, fitted_api_names=len(result['names']),
        training_samples=len(result['contract']['train_ids']), fingerprint=result['fingerprint'])), flush=True)


if __name__ == '__main__':
    main()
