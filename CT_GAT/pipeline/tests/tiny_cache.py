"""A small synthetic CFG cache: 40 binaries (20 ransomware in 10 families, 20 goodware, five folds), the cache arrays the
pipeline reads, an index.csv and a vocab.json, and a patch that points every module-level CACHE at it.

`build(root, v3=True)` (the `tiny_v3` fixture) grows it into a second dataset. The 40 rows and their arrays are unchanged and
32 rows follow them: 8 Mendeley x64 rows with a Mendeley fold (so `kfold` + `all` has 48 rows and `kfold` + `x86` still the
40), 12 `ransomware_balanced` and 12 `balanced_goodware` rows (x64 and x86 mixed, no Mendeley fold), and the columns `corpus`,
`fold_v3`, `fold_v3_sha256` and `arch_matched`. Two balanced rows have no `fold_v3`, so `kfold_v3` + `all` has 70 rows.
Families and identity groups stay whole in both fold columns; a balanced ransomware family may also be a Mendeley one."""
import csv
import hashlib
import json

import numpy as np
import pytest

N_MN, N_SIG, N_API, N_BEH = 30, 50, 24, 9


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


def write_sample(root, rng, i):
    """Binary `i`: its arrays are written under root/samples; returns (the arrays, their instruction count)."""
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
    np.savez_compressed(root / 'samples' / f'{sample_id(i)}.npz', **arrays)
    return z, n_ins


def index_row(i, z, n_ins, label, arch, family, group, fold):
    sid = sample_id(i)
    return {'sample_id': sid, 'label': label, 'split': 'train', 'arch': arch, 'family': family,
            'identity_group': group, 'fold': fold, 'fold_sha256': sid if fold != '' else '', 'num_functions': len(z['fn_flags']),
            'num_blocks': len(z['blk_feat']), 'num_edges': len(z['cfg_edges']), 'num_instructions': n_ins,
            'num_api': len(z['api']), 'num_call_edges': 7, 'status': 'ok', 'error': ''}


# The rows `build(v3=True)` adds after the 40: (corpus, label, arch, family, identity group (None: the sample id), Mendeley
# fold or '', fold_v3 or ''). A Mendeley x64 family is one of the x86 ones (fam1, fam4) or new, with the fold the family has.
V3_ROWS = (
    [('mendeley', 1, 'x64', 'fam1', 'fam1', 1, 2), ('mendeley', 1, 'x64', 'fam1', 'fam1', 1, 2),
     ('mendeley', 1, 'x64', 'fam4', 'fam4', 4, 0), ('mendeley', 1, 'x64', 'x64fam', 'x64fam', 0, 3)]
    + [('mendeley', 0, 'x64', '', None, fold, fold_v3) for fold, fold_v3 in ((0, 4), (1, 0), (2, 1), (3, 2))]
    + [('ransomware_balanced', 1, 'x64', 'fam2', 'fam2', '', 3), ('ransomware_balanced', 1, 'x64', 'fam2', 'fam2', '', 3),
       ('ransomware_balanced', 1, 'x64', 'fam7', 'fam7', '', 3), ('ransomware_balanced', 1, 'x64', 'balfam0', 'balfam0', '', 0),
       ('ransomware_balanced', 1, 'x64', 'balfam0', 'balfam0', '', 0), ('ransomware_balanced', 1, 'x64', 'balfam9', 'balfam9', '', ''),
       ('ransomware_balanced', 1, 'x86', 'balfam1', 'balfam1', '', 1), ('ransomware_balanced', 1, 'x86', 'balfam2', 'balfam2', '', 2),
       ('ransomware_balanced', 1, 'x86', 'balfam2', 'balfam2', '', 2), ('ransomware_balanced', 1, 'x86', 'balfam3', 'balfam3', '', 4),
       ('ransomware_balanced', 1, 'x86', 'balfam4', 'balfam4', '', 1), ('ransomware_balanced', 1, 'x86', 'balfam4', 'balfam4', '', 1)]
    + [('balanced_goodware', 0, 'x64', '', 'balgood0', '', 2), ('balanced_goodware', 0, 'x64', '', 'balgood0', '', 2),
       ('balanced_goodware', 0, 'x64', '', 'balgood1', '', 3), ('balanced_goodware', 0, 'x64', '', 'balgood1', '', 3)]
    + [('balanced_goodware', 0, 'x64', '', None, '', fold_v3) for fold_v3 in (4, 0, 1, 2)]
    + [('balanced_goodware', 0, 'x86', '', None, '', fold_v3) for fold_v3 in (3, 4, 0, '')])
FOLD_V3_OF_THE_FIRST_40 = [(i // 2 + 1) % 5 if i < 20 else (i - 20 + 2) % 5 for i in range(40)]


def build(root, seed=3, v3=False):
    """Write the cache under `root`; returns {'ids', 'rows'}. With `v3` it is the second dataset of the module docstring."""
    rng = np.random.default_rng(seed)
    (root / 'samples').mkdir(parents=True)
    vocab = {'mn': [f'mn{i}' for i in range(N_MN)], 'sig': [f'sig{i}' for i in range(N_SIG)],
             'api': [f'lib{i % 4}.dll!Fn{i}' for i in range(N_API)], 'beh': [f'beh{i}' for i in range(N_BEH)]}
    (root / 'vocab.json').write_text(json.dumps(vocab), encoding='utf-8')
    rows = []
    for i in range(40):
        label = int(i < 20)
        family = f'fam{i // 2}' if label else ''
        z, n_ins = write_sample(root, rng, i)
        group = family if label else sample_id(i)
        fold = (i // 2) % 5 if label else (i - 20) % 5
        rows.append(index_row(i, z, n_ins, label, 'x86', family, group, fold))
    fields = list(rows[0])
    if v3:
        for i, (corpus, label, arch, family, group, fold, _) in enumerate(V3_ROWS, 40):
            z, n_ins = write_sample(root, rng, i)
            rows.append({**index_row(i, z, n_ins, label, arch, family, group or sample_id(i), fold), 'corpus': corpus})
        fields += ['corpus', 'fold_v3', 'fold_v3_sha256', 'arch_matched']
        folds_v3 = FOLD_V3_OF_THE_FIRST_40 + [v3_row[-1] for v3_row in V3_ROWS]
        for i, (row, fold_v3) in enumerate(zip(rows, folds_v3)):
            row.setdefault('corpus', 'mendeley')
            row.update(fold_v3=fold_v3, fold_v3_sha256=row['sample_id'] if fold_v3 != '' else '',
                       arch_matched='' if fold_v3 == '' else int(i % 4 != 3))
    with (root / 'index.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return {'ids': [r['sample_id'] for r in rows], 'rows': rows}


@pytest.fixture(scope='module')
def tiny(tmp_path_factory):
    """The tiny cache, with every module-level CACHE pointed at it."""
    from common import manifests
    root = tmp_path_factory.mktemp('tiny') / 'cache'
    info = build(root)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(manifests, 'CACHE', root)
        yield type('Tiny', (), {'root': root, **info})


@pytest.fixture(scope='module')
def tiny_v3(tmp_path_factory):
    """The second dataset (`build(v3=True)`), with every module-level CACHE pointed at it."""
    from common import manifests
    root = tmp_path_factory.mktemp('tiny_v3') / 'cache'
    info = build(root, v3=True)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(manifests, 'CACHE', root)
        yield type('TinyV3', (), {'root': root, **info})
