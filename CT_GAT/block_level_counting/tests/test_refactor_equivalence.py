"""python -m pytest tests/test_refactor_equivalence.py: the restructured code against the code at 118237a.

config.py, data.py and model_gnn.py of BASE are read with `git show` into temp modules and compared on real v2 binaries
(manifests, vocabulary, per-binary arrays, batches, loader), on HierGNN state_dict keys, shapes, initial values and
logits, and on three real checkpoints. Checks that need the cache or the runs directory are skipped without them.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from common.config import COMBINED_CACHE, RUNS, RunConfig
from common.manifests import load_index, load_sample, make_manifest
from pipeline.batching import collate, encode_binary, loader
from pipeline.model import HierGNN
from pipeline.s03_tokenizer.vocab import TokenVocab
from pipeline.s04_select_subgraph import subgraph

ROOT = Path(__file__).resolve().parents[1]
BASE = '118237a'
OLD_DIR = 'CT_GAT/graph_transformer'
CHECKPOINTS = ('kfold0_x86_gnn_s0', 'kfold0_x86_gnn-mblk15_s1', 'kfold0_x86_gnn-noedges_s0')
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
needs_runs = pytest.mark.skipif(not all((RUNS / c / 'best.pt').exists() for c in CHECKPOINTS), reason='needs the runs')


@pytest.fixture(autouse=True)
def deterministic():
    """CUDA index_add_ sums in a run-dependent order, so two runs of one model differ in the last bits."""
    before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(before)


@pytest.fixture(scope='module')
def old(tmp_path_factory):
    """The BASE config, data and model_gnn modules; the old data and model import `config` by its flat name."""
    tmp = tmp_path_factory.mktemp('old')
    mods = {}
    with pytest.MonkeyPatch.context() as mp:
        for name in ('config', 'data', 'model_gnn'):
            src = subprocess.run(['git', 'show', f'{BASE}:{OLD_DIR}/{name}.py'], cwd=ROOT, check=True,
                                 capture_output=True).stdout
            path = tmp / f'old_{name}.py'
            path.write_bytes(src)
            spec = importlib.util.spec_from_file_location(f'old_{name}', path)
            mod = importlib.util.module_from_spec(spec)
            mp.setitem(sys.modules, f'old_{name}', mod)
            if name == 'config':
                mp.setitem(sys.modules, 'config', mod)
            spec.loader.exec_module(mod)
            mods[name] = mod
    return SimpleNamespace(**mods)


def gnn(**kw):
    return RunConfig(model='gnn', num_workers=0, **kw)


def same_array(a, b, what):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), what


def same_dict(a, b):
    assert a.keys() == b.keys()
    for k in a:
        same_array(a[k], b[k], k)


def same_batch(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if torch.is_tensor(a[k]):
            assert a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and torch.equal(a[k], b[k]), k
        else:
            assert a[k] == b[k], k


def same_vocab(a, b):
    assert (a.size, a.offset, a.total, a.mask_id) == (b.size, b.offset, b.total, b.mask_id)
    assert a.lut.keys() == b.lut.keys()
    for k in a.lut:
        same_array(a.lut[k], b.lut[k], f'lut {k}')
    assert (a.idf is None) == (b.idf is None)
    if a.idf is not None:
        same_array(a.idf, b.idf, 'idf')


@pytest.fixture(scope='module')
def index():
    return load_index()


@pytest.fixture(scope='module')
def train_rows(index):
    return make_manifest(RunConfig(), index).train


@pytest.fixture(scope='module')
def real_ids(index):
    rows = sorted((r for r in index if r['arch'] == 'x86' and r['split'] == 'train'
                   and 300 <= int(r['num_blocks']) <= 8000 and int(r['num_edges']) > 50),
                  key=lambda r: r['sample_id'])
    return [r['sample_id'] for r in rows[:4]]


@pytest.fixture(scope='module')
def tiny_ids(index):
    return [r['sample_id'] for r in sorted(index, key=lambda r: (int(r['num_blocks']), r['sample_id']))[:3]]


# manifests and vocabulary

@pytest.mark.parametrize('kw', [dict(), dict(train_fraction=0.5), dict(arch='all'), dict(arch='all', seed=2),
                                dict(protocol='kfold', fold=0), dict(protocol='kfold', fold=3, arch='all'),
                                dict(protocol='kfold', fold=1, train_fraction=0.25, arch_balance=True)])
def test_manifests_match(old, index, kw):
    cfg = RunConfig(**kw)
    new_m, old_m = make_manifest(cfg, index), old.data.make_manifest(cfg, index)
    for name in ('train', 'val', 'test'):
        assert [r['sample_id'] for r in getattr(new_m, name)] == [r['sample_id'] for r in getattr(old_m, name)]
    assert new_m.val_families == old_m.val_families


@pytest.mark.skipif(not (COMBINED_CACHE / 'index.csv').exists(), reason='needs the combined cache')
@pytest.mark.parametrize('arch', ['all', 'matched'])
def test_v3_manifests_match(old, arch):
    index = load_index(COMBINED_CACHE)
    assert index == old.data.load_index(COMBINED_CACHE)
    for fold in range(5):
        cfg = RunConfig(protocol='kfold_v3', fold=fold, arch=arch)
        new_m, old_m = make_manifest(cfg, index), old.data.make_manifest(cfg, index)
        for name in ('train', 'val', 'test'):
            assert [r['sample_id'] for r in getattr(new_m, name)] == [r['sample_id'] for r in getattr(old_m, name)]


@pytest.mark.parametrize('kw', [dict(), dict(encoder='mean'), dict(min_df=5)])
def test_vocab_matches(old, train_rows, kw):
    cfg = gnn(**kw)
    same_vocab(TokenVocab(train_rows[:100], cfg), old.data.TokenVocab(train_rows[:100], cfg))


# per-binary arrays and batches

ENCODE_CASES = {
    'default': dict(),
    'mask_blocks': dict(mask_blocks=0.15),
    'mask_tokens': dict(mask_tokens=0.15),
    'control_shuffle': dict(control='shuffle'),
    'control_chain': dict(control='chain'),
    'control_permute_types': dict(control='permute_types'),
    'control_no_edges': dict(control='no_edges'),
    'drop_flip_edges': dict(drop_edges=0.2, flip_edges=0.2),
    'drop_functions': dict(drop_functions=0.15),
    'drop_library': dict(drop_functions=0.05, drop_library=0.4),
    'encoder_mean': dict(encoder='mean'),
    'train_fraction': dict(train_fraction=0.5),
    'all_augmentations': dict(mask_blocks=0.1, mask_tokens=0.1, drop_functions=0.1, drop_edges=0.1, flip_edges=0.1,
                              control='shuffle'),
}


@pytest.mark.parametrize('case', ENCODE_CASES)
@pytest.mark.parametrize('which', ['real', 'tiny'])
def test_encode_and_collate_match(old, train_rows, real_ids, tiny_ids, case, which):
    cfg = gnn(**ENCODE_CASES[case])
    vocab, old_vocab = TokenVocab(train_rows[:100], cfg), old.data.TokenVocab(train_rows[:100], cfg)
    same_vocab(vocab, old_vocab)
    ids = real_ids if which == 'real' else tiny_ids
    for epoch in (None, 1):
        new_items, old_items = [], []
        for i, sid in enumerate(ids):
            z = load_sample(sid)
            rng = lambda: None if epoch is None else np.random.default_rng([cfg.seed, epoch, i])
            new, ref = encode_binary(z, vocab, cfg, rng(), sid), old.data.encode_binary(z, old_vocab, cfg, rng(), sid)
            same_dict(new, ref)
            new['label'] = ref['label'] = i % 2
            new['sample_id'] = ref['sample_id'] = sid
            new_items.append(new)
            old_items.append(ref)
        same_batch(collate(new_items), old.data.collate(old_items))


def test_block_cap_matches(old, train_rows, real_ids, monkeypatch):
    monkeypatch.setattr(subgraph, 'MAX_BINARY_BLOCKS', 150)
    monkeypatch.setattr(old.data, 'MAX_BINARY_BLOCKS', 150)
    cfg = gnn(drop_functions=0.1)
    vocab = TokenVocab(train_rows[:100], cfg)
    for epoch in (None, 1):
        for i, sid in enumerate(real_ids):
            z = load_sample(sid)
            rng = lambda: None if epoch is None else np.random.default_rng([cfg.seed, epoch, i])
            same_dict(encode_binary(z, vocab, cfg, rng(), sid), old.data.encode_binary(z, vocab, cfg, rng(), sid))


@pytest.mark.parametrize('train', [True, False])
def test_loader_batches_match(old, train_rows, train, monkeypatch):
    monkeypatch.setattr(subgraph, 'MAX_BINARY_BLOCKS', 20_000)
    monkeypatch.setattr(old.data, 'MAX_BINARY_BLOCKS', 20_000)
    cfg = gnn(batch_blocks=30_000, mask_blocks=0.15, drop_edges=0.1)
    rows = train_rows[:12]
    vocab = TokenVocab(train_rows[:100], cfg)
    new_batches, old_batches = list(loader(rows, vocab, cfg, train)), list(old.data.loader(rows, vocab, cfg, train))
    assert len(new_batches) == len(old_batches) > 1
    for a, b in zip(new_batches, old_batches):
        same_batch(a, b)


# HierGNN

MODEL_CASES = {
    'default': dict(), 'gated': dict(readout='gated'), 'stack_tfidf': dict(stack_tfidf=True),
    'forward': dict(direction='forward'), 'backward': dict(direction='backward'),
    'undirected': dict(direction='undirected'), 'both': dict(direction='both'), 'L8': dict(gnn_fn_layers=8),
    'B3_L1': dict(gnn_bin_layers=3, gnn_fn_layers=1), 'd64': dict(d_model=64),
}


@pytest.mark.parametrize('case', MODEL_CASES)
def test_state_dict_matches(old, case):
    cfg = gnn(**MODEL_CASES[case])
    torch.manual_seed(0)
    ref = old.model_gnn.HierGNN(cfg, 1000)
    torch.manual_seed(0)
    new = HierGNN(cfg, 1000)
    a, b = new.state_dict(), ref.state_dict()
    assert list(a) == list(b)
    for k in a:
        assert a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and torch.equal(a[k], b[k]), k
    assert [n for n, _ in new.named_parameters()] == [n for n, _ in ref.named_parameters()]
    new.load_state_dict(b, strict=True)
    ref.load_state_dict(a, strict=True)


def contract_batch(cfg, train_rows, ids, device):
    vocab = TokenVocab(train_rows[:100], cfg)
    items = []
    for i, sid in enumerate(ids):
        item = encode_binary(load_sample(sid), vocab, cfg, None, sid)
        item['label'], item['sample_id'] = i % 2, sid
        items.append(item)
    batch = collate(items)
    batch['stack_logit'] = torch.linspace(-1, 1, len(items))
    return vocab, {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def logits(model, batch, device, train):
    model.train(train)
    torch.manual_seed(1)
    with torch.no_grad():
        if device == 'cuda':
            with torch.autocast('cuda', dtype=torch.bfloat16):
                return model(batch)
        return model(batch)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=needs_cuda)])
@pytest.mark.parametrize('case', ['default', 'gated', 'stack_tfidf', 'forward', 'undirected', 'L8'])
def test_logits_equal(old, train_rows, real_ids, case, device):
    cfg = gnn(**MODEL_CASES[case])
    vocab, batch = contract_batch(cfg, train_rows, real_ids, device)
    torch.manual_seed(0)
    ref = old.model_gnn.HierGNN(cfg, vocab.total).to(device)
    torch.manual_seed(0)
    new = HierGNN(cfg, vocab.total).to(device)
    for train in (False, True):
        assert torch.equal(logits(new, batch, device, train), logits(ref, batch, device, train)), train
    for control, untyped, no_call in (('no_edges', False, False), ('', True, False), ('', False, True)):
        cfg = gnn(control=control, untyped=untyped, no_call_mp=no_call)
        torch.manual_seed(0)
        ref = old.model_gnn.HierGNN(cfg, vocab.total).to(device)
        torch.manual_seed(0)
        new = HierGNN(cfg, vocab.total).to(device)
        assert torch.equal(logits(new, batch, device, False), logits(ref, batch, device, False)), control


@needs_runs
@pytest.mark.parametrize('run', CHECKPOINTS)
def test_real_checkpoints_load_strict_and_agree(old, index, run):
    cfg = RunConfig.load(RUNS / run / 'config.json')
    n_tokens = json.loads((RUNS / run / 'vocab.json').read_text(encoding='utf-8'))['total']
    state = torch.load(RUNS / run / 'best.pt', map_location='cpu')
    new, ref = HierGNN(cfg, n_tokens), old.model_gnn.HierGNN(cfg, n_tokens)
    new.load_state_dict(state, strict=True)
    ref.load_state_dict(state, strict=True)
    assert list(new.state_dict()) == list(state)
    manifest = make_manifest(cfg, index)
    vocab = TokenVocab(manifest.train, cfg)
    assert vocab.total == n_tokens
    rows = manifest.test[:6]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    new, ref = new.to(device), ref.to(device)
    for batch in loader(rows, vocab, cfg, train=False):
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        assert torch.equal(logits(new, batch, device, False), logits(ref, batch, device, False))

