"""A small synthetic CFG cache for the tests of the reader wiring: 40 binaries (20 ransomware in 10 families, 20 goodware, five
folds), the cache arrays the pipeline reads, an index.csv, the reader side arrays (`side/yanping`, `side/imports`) with a
label-correlated import signal, and a patch that points every module-level CACHE at it."""
import csv
import hashlib
import json

import numpy as np
import pytest

from common.side_io import atomic_npz
from experiments.side_arrays import import_reader_side as yan
from pipeline.s10_unused_import_features import import_side as imp

N_MN, N_SIG, N_API, N_BEH = 30, 50, 24, 9
IMPORT_POOL = [f'lib{i % 4}.dll!fn{i}' for i in range(40)]
RANSOM_ONLY = IMPORT_POOL[:6]                        # unreferenced in most ransomware, in no goodware


def sample_id(i):
    return hashlib.sha256(f'tiny{i}'.encode()).hexdigest()


def make_binary(rng):
    n_fn = int(rng.integers(5, 10))
    sizes = rng.integers(0, 4, n_fn)
    sizes[0] = 3
    ptr = np.concatenate([[0], np.cumsum(sizes)])
    n_blk = int(ptr[-1])
    n_ins, n_api = rng.integers(0, 6, n_blk), rng.integers(0, 3, n_blk)
    edges = [ptr[f] + rng.integers(0, sizes[f], (int(rng.integers(sizes[f], 2 * sizes[f] + 1)), 2))
             for f in range(n_fn) if sizes[f] >= 1]
    cfg_edges = np.concatenate(edges).astype(np.int32)
    return {
        'mn': rng.integers(0, N_MN, int(n_ins.sum())).astype(np.int32),
        'sig': rng.integers(0, N_SIG, int(n_ins.sum())).astype(np.int32),
        'beh': rng.integers(0, N_BEH, int(n_ins.sum())).astype(np.int8),
        'api': rng.integers(0, N_API, int(n_api.sum())).astype(np.int32),
        'blk_insn_ptr': np.concatenate([[0], np.cumsum(n_ins)]).astype(np.int32),
        'blk_api_ptr': np.concatenate([[0], np.cumsum(n_api)]).astype(np.int32),
        'blk_feat': (rng.random((n_blk, 6)) + 0.1).astype(np.float16),
        'fn_blk_ptr': ptr.astype(np.int32), 'fn_flags': rng.integers(0, 4, n_fn).astype(np.int8),
        'cfg_edges': cfg_edges, 'call_edges': rng.integers(0, n_fn, (7, 2)).astype(np.int32),
        'edge_kind': rng.integers(0, 6, len(cfg_edges)).astype(np.int8),
        'edge_back': rng.integers(0, 2, len(cfg_edges)).astype(np.int8),
    }


def build(root, seed=3):
    """Write the cache under `root`; returns {'ids', 'labels', 'rows'}."""
    rng = np.random.default_rng(seed)
    (root / 'samples').mkdir(parents=True)
    vocab = {'mn': [f'mn{i}' for i in range(N_MN)], 'sig': [f'sig{i}' for i in range(N_SIG)],
             'api': [f'lib{i % 4}.dll!Fn{i}' for i in range(N_API)], 'beh': [f'beh{i}' for i in range(N_BEH)]}
    (root / 'vocab.json').write_text(json.dumps(vocab), encoding='utf-8')
    rows = []
    for i in range(40):
        sid, label = sample_id(i), int(i < 20)
        family = f'fam{i // 2}' if label else ''
        z = make_binary(rng)
        n_ins = len(z['mn'])
        z['mn'][3:] %= N_MN - 1                      # the last mnemonic and signature occur in the first binary only: df 1, cut
        z['sig'][3:] %= N_SIG - 1
        z['mn'][:3], z['sig'][:3] = np.where(i == 0, [N_MN - 1] * 3, z['mn'][:3] % (N_MN - 1)), np.where(
            i == 0, [N_SIG - 1] * 3, z['sig'][:3] % (N_SIG - 1))
        arrays = {**z, 'mn_u': np.unique(z['mn']), 'sig_u': np.unique(z['sig']), 'api_u': np.unique(z['api']),
                  'num_functions': np.array(len(z['fn_flags'])), 'num_blocks': np.array(len(z['blk_feat'])),
                  'num_instructions': np.array(n_ins), 'num_edges': np.array(len(z['cfg_edges'])),
                  'num_api': np.array(len(z['api']))}
        np.savez_compressed(root / 'samples' / f'{sid}.npz', **arrays)
        ytok = np.stack([rng.integers(0, 256, n_ins), rng.integers(0, 76, n_ins), rng.integers(0, 76, n_ins)], 1)
        ytok[:, 1] = ytok[:, 1] % 70
        ytok[:3, 1] = 75 if i == 0 else ytok[:3, 1]                                     # a reader id seen in one binary only
        ytok[:, 0] = np.where(label == 1, ytok[:, 0] % 40, 100 + ytok[:, 0] % 100)      # opcode ids separate the classes
        atomic_npz(yan.yanping_path(root, sid), dict(schema=np.array(yan.SCHEMA), sample_id=np.array(sid),
                                                     ytok=ytok.astype(np.int16)))
        header = IMPORT_POOL[6:6 + int(rng.integers(8, 20))]
        unref = list(rng.choice(IMPORT_POOL[6:], int(rng.integers(0, 6)), replace=False))
        if label and rng.random() < 0.85:
            unref += list(rng.choice(RANSOM_ONLY, 3, replace=False))
        header = sorted(set(header) | set(unref))
        unref = sorted(set(unref))
        atomic_npz(imp.imports_path(root, sid), dict(
            schema=np.array(imp.SCHEMA), sample_id=np.array(sid), n_entries=np.array(len(header)),
            header_imports=np.array(header), unreferenced_imports=np.array(unref) if unref else np.zeros(0, dtype='<U1'),
            referenced_imports=np.array(sorted(set(header) - set(unref)))))
        group = family if label else sid
        fold = (i // 2) % 5 if label else (i - 20) % 5
        rows.append({'sample_id': sid, 'label': label, 'split': 'train', 'arch': 'x86', 'family': family,
                     'identity_group': group, 'fold': fold, 'fold_sha256': sid, 'num_functions': len(z['fn_flags']),
                     'num_blocks': len(z['blk_feat']), 'num_edges': len(z['cfg_edges']), 'num_instructions': n_ins,
                     'num_api': len(z['api']), 'num_call_edges': 7, 'status': 'ok', 'error': ''})
    with (root / 'index.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return {'ids': [r['sample_id'] for r in rows], 'rows': rows}


@pytest.fixture(scope='module')
def tiny(tmp_path_factory):
    """The tiny cache, with every module-level CACHE (and the runs directory of stats / gates) pointed at it."""
    from common import manifests
    from evaluation import baselines, stats
    from pipeline.s03_tokenizer import vocab as vocab_module
    from pipeline.s10_unused_import_features import imports_feat
    root = tmp_path_factory.mktemp('tiny') / 'cache'
    info = build(root)
    runs = root.parent / 'runs'
    (runs / 'baselines').mkdir(parents=True)
    with pytest.MonkeyPatch.context() as mp:
        for module in (manifests, vocab_module, baselines, imports_feat):
            mp.setattr(module, 'CACHE', root)
        mp.setattr(stats, 'RUNS', runs)
        yield type('Tiny', (), {'root': root, 'runs': runs, **info})
