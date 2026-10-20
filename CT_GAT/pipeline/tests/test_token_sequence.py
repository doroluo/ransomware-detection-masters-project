"""python -m pytest tests/test_token_sequence.py: Model 1 (s03a_transformer/token_sequence/): the regrouping of kind-major blocks
into instruction triplets and API positions, windows, collation, the model and the split isolation of FoldSequences.

CPU only. The export is built from the small synthetic cache of tests/tiny_cache.py (40 binaries, fold 0); no test needs
the real CFG cache or a GPU.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tiny_cache import tiny  # noqa: F401  (a fixture)

from common.config import RunConfig
from common.manifests import load_index, make_manifest
from common.reader import BlockStore, FoldView
from s03a_transformer.token_sequence import (BinaryTokenSequence, FoldSequences, InstructionEmbedder, Model1TokenSequence, Stream,
                                             collate, instruction_stream, regroup, windows)
from s03c_cfg_transformer import block_export as be

torch.set_num_threads(2)


@pytest.fixture(scope='module')
def export(tiny, tmp_path_factory):  # noqa: F811
    """A block export of the 40 binaries of the tiny cache, fold 0."""
    for path in (tiny.root / 'samples').glob('*.npz'):                    # the tiny samples carry no vocabulary hash
        with np.load(path) as z:
            arrays = dict(z)
        np.savez_compressed(path, **arrays, vocab_sha=np.array('tiny'))
    cfg = RunConfig(model='gnn', protocol='kfold', fold=0, arch='x86', seed=0, val_seed=0)
    index = load_index()
    manifest = make_manifest(cfg, index)
    rows = sorted((r for r in index if r['arch'] == 'x86' and r['fold'] is not None), key=lambda r: r['sample_id'])
    out = tmp_path_factory.mktemp('export') / 'tiny'
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(be, 'CACHE', tiny.root)
        be.build_export(cfg, out, rows, {0: manifest}, log=lambda *_: None)
    return SimpleNamespace(root=out, store=BlockStore(out), manifest=manifest)


def decode_block(tokens, kinds):
    """The positions of one block by direct decoding: [(mn, sig, beh) ...] then [api ...]."""
    tokens, kinds = list(map(int, tokens)), list(map(int, kinds))
    n = kinds.count(0)
    assert kinds == [0] * n + [1] * n + [2] * n + [3] * (len(kinds) - 3 * n)
    triplets = [(tokens[i], tokens[n + i], tokens[2 * n + i]) for i in range(n)]
    return triplets, tokens[3 * n:]


def as_rows(stream):
    """Stream -> list of ('i', (mn, sig, beh)) / ('a', api) for comparison."""
    return [('a', int(i[0])) if k == 1 else ('i', tuple(int(x) for x in i)) for i, k in zip(stream.ids, stream.kind)]


def test_regroup_equals_a_direct_decode_of_each_block(export):
    store = export.store
    view = FoldView(store, 0, 'all')
    seen_instructions = seen_api = 0
    for binary in range(3):
        want = []
        first, last = store.block_range(binary)
        for block in range(first, last):
            triplets, api = decode_block(view.sequence(block), store.kinds_of(block))
            want += [('i', t) for t in triplets] + [('a', a) for a in api]
            seen_instructions += len(triplets)
            seen_api += len(api)
        got = instruction_stream(view, binary)
        assert got.ids.shape == (len(want), 3) and got.kind.shape == (len(want),)
        assert as_rows(got) == want
        assert len(want) > 0
    assert seen_instructions > 0 and seen_api > 0                                  # both kinds of position were compared


def test_regroup_every_binary_has_one_position_per_instruction_and_api_token(export):
    store = export.store
    view = FoldView(store, 0, 'all')
    for binary in range(len(store.binaries)):
        first, last = store.block_range(binary)
        kinds = np.asarray(store.tok_kind[store.blk_tok_ptr[first]:store.blk_tok_ptr[last]])
        stream = instruction_stream(view, binary)
        assert len(stream.ids) == int((kinds == 0).sum() + (kinds == 3).sum())
        assert int((stream.kind == 1).sum()) == int((kinds == 3).sum())
        assert (stream.ids[stream.kind == 1][:, 1:] == 0).all()
        assert (stream.ids[stream.kind == 0] > 3).all()                            # a fold id, never a special


def test_regroup_hand_made_blocks_with_truncated_tails():
    """A block cut by the export rule (70 instructions and 3 API references keep the first 53 and last 9 instructions
    and all 3 API references), an empty block, a block that holds only API references and one of two instructions."""
    head_ins, tail_ins, head_api, tail_api, cut = (int(x[0]) for x in be.truncation_plan([70], [3]))
    assert (head_ins, tail_ins, head_api, tail_api, cut) == (53, 9, 0, 3, 1)
    keep = list(range(head_ins)) + list(range(70 - tail_ins, 70))
    api = [5000, 5001, 5002]
    block_a = np.concatenate([1000 + np.array(keep), 2000 + np.array(keep), 3000 + np.array(keep), api])
    kind_a = [0] * len(keep) + [1] * len(keep) + [2] * len(keep) + [3] * len(api)
    tokens = np.concatenate([block_a, [], [7, 8, 9, 10], [11, 12, 13, 14, 15, 16]])
    kinds = np.array(kind_a + [] + [3] * 4 + [0, 0, 1, 1, 2, 2])
    ptr = np.cumsum([0, len(block_a), 0, 4, 6])                                       # block 1 is empty
    got = as_rows(regroup(tokens, kinds, ptr))
    want = [('i', (1000 + k, 2000 + k, 3000 + k)) for k in keep] + [('a', a) for a in api]
    want += [('a', 7), ('a', 8), ('a', 9), ('a', 10), ('i', (11, 13, 15)), ('i', (12, 14, 16))]
    assert got == want


def test_regroup_refuses_a_block_that_is_not_kind_major():
    with pytest.raises(ValueError, match='block 0'):
        regroup([1, 2, 3, 4], [0, 1, 1, 2], [0, 4])                                 # 1 mn, 2 sig, 1 beh
    with pytest.raises(ValueError, match='laid out'):
        regroup([1, 2, 3, 4], [0, 3, 1, 2], [0, 4])                                 # api inside the instruction part
    assert len(regroup([], [], [0]).ids) == 0 and len(regroup([], [], [0, 0, 0]).ids) == 0


def synthetic(n):
    """A stream whose slot 0 of position p is p + 1 (unique), to read a window's start back."""
    ids = np.zeros((n, 3), dtype=np.int64)
    ids[:, 0] = np.arange(n) + 1
    return Stream(ids, (np.arange(n) % 3 == 0).astype(np.int64))


@pytest.mark.parametrize('n,L', [(0, 8), (1, 8), (7, 8), (8, 8), (9, 8), (100, 8), (100, 512)])
def test_eval_windows_cover_the_stream_in_order(n, L):
    stream = synthetic(n)
    ids, kind, pad = windows(stream, L)
    assert ids.shape == (-(-n // L), L, 3) and kind.shape == pad.shape == (len(ids), L)
    assert np.array_equal(ids[~pad].numpy(), stream.ids) and np.array_equal(kind[~pad].numpy(), stream.kind)
    assert (ids[pad] == 0).all() and (kind[pad] == 0).all()
    again = windows(stream, L)
    assert all(torch.equal(a, b) for a, b in zip((ids, kind, pad), again))
    if n:
        assert not pad[:-1].any()                                                   # only the last window is padded


def test_eval_windows_of_a_real_binary_are_deterministic(export):
    stream = instruction_stream(FoldView(export.store, 0, 'all'), 3)
    first, second = windows(stream, 16), windows(stream, 16)
    assert len(first[0]) == -(-len(stream.ids) // 16)
    assert all(torch.equal(a, b) for a, b in zip(first, second))


def test_train_windows_are_bounded_in_range_and_reproducible():
    L, n, cap = 16, 1000, 5
    stream = synthetic(n)
    ids, kind, pad = windows(stream, L, train=True, max_windows=cap, rng=np.random.default_rng(7))
    assert 0 < len(ids) <= cap
    starts = ids[:, 0, 0].numpy() - 1
    assert len(set(starts % L)) == 1 and 0 <= starts[0] % L < L                    # one offset for all windows
    assert (np.diff(starts) > 0).all() and starts.min() >= 0 and starts.max() < n
    for w, a in enumerate(starts):
        live = ~pad[w]
        assert int(live.sum()) == min(L, n - a)
        assert np.array_equal(ids[w][live].numpy(), stream.ids[a:a + int(live.sum())])
    again = windows(stream, L, train=True, max_windows=cap, rng=np.random.default_rng(7))
    assert all(torch.equal(a, b) for a, b in zip((ids, kind, pad), again))
    other = windows(stream, L, train=True, max_windows=cap, rng=np.random.default_rng(8))
    assert not torch.equal(ids, other[0])
    few = windows(synthetic(20), L, train=True, max_windows=cap, rng=np.random.default_rng(0))
    assert 1 <= len(few[0]) <= 2                                                    # fewer windows than the cap
    assert len(windows(synthetic(0), L, train=True, rng=np.random.default_rng(0))[0]) == 0


def make_batch(export, split='val', L=16, n=3):
    data = FoldSequences(export.root, 0, split, L=L)
    return data, collate([data[i] for i in range(n)])


def test_collate_shapes_and_window_to_binary_index(export):
    data, batch = make_batch(export, n=4)
    W = len(batch['ids'])
    assert batch['ids'].shape == (W, 16, 3) and batch['kind'].shape == batch['pad_mask'].shape == (W, 16)
    assert batch['win2bin'].shape == (W,) and batch['win2bin'].tolist() == sorted(batch['win2bin'].tolist())
    assert batch['labels'].shape == (4,) and batch['sample_ids'] == data.sample_ids()[:4]
    assert batch['labels'].tolist() == data.labels()[:4]
    for i in range(4):
        assert int((~batch['pad_mask'][batch['win2bin'] == i]).sum()) == len(data.stream(i).ids)
    assert batch['ids'].dtype == torch.long and batch['pad_mask'].dtype == torch.bool
    assert int(batch['ids'].max()) < data.n_ids


def small_model(data, **kw):
    return Model1TokenSequence(n_ids=data.n_ids, d_model=16, nhead=2, num_layers=1, max_seq_len=64, **kw)


def test_padding_does_not_change_the_pooled_vector(export):
    torch.manual_seed(0)
    data, batch = make_batch(export, n=3)
    net = BinaryTokenSequence(small_model(data)).eval()
    ids, kind, pad, w2b = batch['ids'], batch['kind'], batch['pad_mask'], batch['win2bin']
    with torch.no_grad():
        base_vec, base_logits = net.encode(ids, kind, pad, w2b), net(ids, kind, pad, w2b)
        extra = 4                                                                  # all-padding windows, for binaries 0 and 2
        ids2 = torch.cat([ids, torch.zeros(extra, 16, 3, dtype=torch.long)])
        kind2 = torch.cat([kind, torch.zeros(extra, 16, dtype=torch.long)])
        pad2 = torch.cat([pad, torch.ones(extra, 16, dtype=torch.bool)])
        w2b2 = torch.cat([w2b, torch.tensor([0, 2, 0, 1])])
        vec2 = net.encode(ids2, kind2, pad2, w2b2, n_binaries=3)
        assert (vec2 - base_vec).abs().max() < 1e-6
        # more padding positions on every window (wider than L, still within max_seq_len)
        ids3 = torch.cat([ids, torch.zeros(len(ids), 9, 3, dtype=torch.long)], 1)
        kind3 = torch.cat([kind, torch.zeros(len(ids), 9, dtype=torch.long)], 1)
        pad3 = torch.cat([pad, torch.ones(len(ids), 9, dtype=torch.bool)], 1)
        assert (net.encode(ids3, kind3, pad3, w2b) - base_vec).abs().max() < 1e-6
        assert (net(ids3, kind3, pad3, w2b) - base_logits).abs().max() < 1e-6
        # and both at once, for a model whose window pooling is also checked per window
        win = net.model.encode(ids, kind, pad)
        win3 = net.model.encode(ids3, kind3, pad3)
        assert (win - win3).abs().max() < 1e-6 and torch.isfinite(win).all()
        assert (net.model.encode(ids2, kind2, pad2)[-extra:] == 0).all()


def test_binary_without_any_position_pools_to_zeros_and_is_finite(export):
    data, batch = make_batch(export, n=2)
    net = BinaryTokenSequence(small_model(data)).eval()
    with torch.no_grad():
        vec = net.encode(batch['ids'], batch['kind'], batch['pad_mask'], batch['win2bin'], n_binaries=3)
        assert vec.shape == (3, 16) and (vec[2] == 0).all() and torch.isfinite(net(
            batch['ids'], batch['kind'], batch['pad_mask'], batch['win2bin'], n_binaries=3)).all()


def test_model_shapes_gradients_and_max_seq_len(export):
    torch.manual_seed(0)
    data, batch = make_batch(export, n=3)
    ids, kind, pad, w2b = batch['ids'], batch['kind'], batch['pad_mask'], batch['win2bin']
    model = small_model(data)
    net = BinaryTokenSequence(model)
    logits = net(ids, kind, pad, w2b)
    assert logits.shape == (3, 2)
    assert model.encode(ids, kind, pad).shape == (len(ids), 16) and model(ids, kind, pad).shape == (len(ids), 2)
    first = torch.stack([(w2b == i).nonzero()[0, 0] for i in range(3)])            # one window per binary: [n_binaries, 2]
    one = model(ids[first], kind[first], pad[first])
    assert one.shape == (3, 2)
    torch.nn.functional.cross_entropy(logits, batch['labels']).backward()
    for param in (model.embedder.embedding.weight, model.embedder.kind_embedding.weight, model.pos_encoder):
        assert param.grad is not None and param.grad.abs().sum() > 0
    assert (model.embedder.embedding.weight.grad[0] == 0).all()                      # PAD is not trained
    with pytest.raises(ValueError, match='max_seq_len'):
        Model1TokenSequence(n_ids=data.n_ids, d_model=16, nhead=2, num_layers=1, max_seq_len=8)(ids, kind, pad)
    with pytest.raises(ValueError, match='max_seq_len'):
        BinaryTokenSequence(Model1TokenSequence(n_ids=data.n_ids, max_seq_len=8))(ids, kind, pad, w2b)


def test_defaults_and_instruction_embedding():
    model = Model1TokenSequence(n_ids=100)
    assert (model.max_seq_len, model.d_model, model.pos_encoder.shape) == (512, 64, (1, 512, 64))
    assert len(model.transformer_encoder.layers) == 2 and model.transformer_encoder.layers[0].self_attn.num_heads == 4
    assert model.transformer_encoder.layers[0].dropout.p == 0.1
    emb = InstructionEmbedder(10, 8)
    ids = torch.tensor([[[2, 3, 4], [5, 0, 0], [0, 0, 0]]])
    kind = torch.tensor([[0, 1, 0]])
    out = emb(ids, kind)
    table, kinds = emb.embedding.weight, emb.kind_embedding.weight
    assert torch.allclose(out[0, 0], table[2] + table[3] + table[4] + kinds[0])
    assert torch.allclose(out[0, 1], table[5] + kinds[1])
    assert torch.allclose(out[0, 2], kinds[0]) and (table[0] == 0).all()


def test_same_seed_gives_identical_logits(export):
    data, batch = make_batch(export, n=3)
    logits = []
    for seed in (5, 5, 6):
        torch.manual_seed(seed)
        net = BinaryTokenSequence(small_model(data)).eval()
        with torch.no_grad():
            logits.append(net(batch['ids'], batch['kind'], batch['pad_mask'], batch['win2bin']))
    assert torch.equal(logits[0], logits[1]) and not torch.equal(logits[0], logits[2])


def test_train_split_never_yields_a_validation_or_test_binary(export):
    manifest = json.loads((export.root / 'fold0' / 'manifest.json').read_text(encoding='utf-8'))
    sid = lambda ids: {export.store.binaries[i]['sample_id'] for i in ids}
    train, val, test = sid(manifest['train']), sid(manifest['val']), sid(manifest['test'])
    assert train and val and test and not (train & val) and not (train & test) and not (val & test)
    assert len(train | val | test) == len(export.store.binaries)
    for split, allowed in (('train', train), ('val', val), ('test', test)):
        data = FoldSequences(export.root, 0, split, L=16)
        yielded = [item['sample_id'] for item in data]
        assert set(yielded) == allowed and len(yielded) == len(allowed) == len(data)
        assert data.view.split == split
    with pytest.raises(ValueError, match='split'):
        FoldSequences(export.root, 0, 'all')
    with pytest.raises(IndexError):
        FoldSequences(export.root, 0, 'train')[10 ** 6]


def test_fold_sequences_train_windows_depend_on_seed_and_epoch_and_eval_does_not(export):
    train = FoldSequences(export.root, 0, 'train', L=8, max_windows=3, seed=1)
    a, b = train[0], train[0]
    assert torch.equal(a['ids'], b['ids']) and len(a['ids']) <= 3
    train.set_epoch(1)
    assert not all(torch.equal(train[i]['ids'], FoldSequences(export.root, 0, 'train', L=8, max_windows=3, seed=1)[i]['ids'])
                   for i in range(len(train)))
    val = FoldSequences(export.root, 0, 'val', L=8, max_windows=3)
    first = [val[i]['ids'] for i in range(len(val))]
    val.set_epoch(4)
    assert all(torch.equal(x, val[i]['ids']) for i, x in enumerate(first))
    assert all(len(first[i]) == -(-len(val.stream(i).ids) // 8) for i in range(len(val)))
