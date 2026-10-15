"""python -m pytest test_gnn_model.py: HierGNN pieces and arms on synthetic batches that follow the batch contract."""
import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from config import RunConfig
from model_gnn import DirGINELayer, FixedReadout, GatedReadout, HierGNN, segment_softmax

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
N_TOKENS = 40
D = 32


def cfg_for(**kw):
    return RunConfig(model='gnn', d_model=D, dropout=0.0, **kw)


def make_fn(rng, n_blocks):
    """One function: blocks (token ids, weights, 6 features) and typed edges with loops and self-loops."""
    blocks = []
    for _ in range(n_blocks):
        n_tok = int(rng.integers(1, 6))
        ids = rng.integers(0, N_TOKENS, n_tok)
        blocks.append({'ids': ids, 'w': rng.random(n_tok).astype(np.float32) + 0.1,
                       'feat': rng.normal(size=6).astype(np.float32)})
    edges = [(i, i + 1, int(rng.integers(0, 12))) for i in range(n_blocks - 1)]
    edges += [(int(rng.integers(0, n_blocks)), int(rng.integers(0, n_blocks)), int(rng.integers(0, 12)))
              for _ in range(n_blocks)]
    if n_blocks > 2:
        edges += [(n_blocks - 1, 0, 1), (1, 1, 3)]
    return {'blocks': blocks, 'edges': edges, 'lib': float(rng.integers(0, 2))}


def make_bins(seed=0, sizes=((4, 1, 5, 3), (2, 6, 1), (3, 3, 2, 1, 4))):
    """Binaries as python structures; collate() turns them into a contract batch. Binary 0 has an empty-token block."""
    rng = np.random.default_rng(seed)
    bins = []
    for b, fn_sizes in enumerate(sizes):
        fns = [make_fn(rng, n) for n in fn_sizes]
        calls = sorted({(int(rng.integers(0, len(fns))), int(rng.integers(0, len(fns)))) for _ in range(2 * len(fns))})
        calls = [(a, c) for a, c in calls if a != c]
        bins.append({'fns': fns, 'calls': calls, 'bin_feat': rng.random(5).astype(np.float32) * 5,
                     'label': float(b % 2)})
    bins[0]['fns'][0]['blocks'][0].update(ids=np.zeros(0, dtype=np.int64), w=np.zeros(0, np.float32),
                                          feat=np.zeros(6, np.float32))
    set_degrees(bins)
    return bins


def set_degrees(bins):
    for b in bins:
        for fn in b['fns']:
            deg = np.zeros((len(fn['blocks']), 2), np.float32)
            for s, d, _ in fn['edges']:
                deg[d, 0] += 1
                deg[s, 1] += 1
            for blk, row in zip(fn['blocks'], deg):
                blk['feat'][4:6] = np.log1p(row)


def collate(bins, device=DEV, stack=False):
    t = lambda x, dt=torch.long: torch.as_tensor(np.asarray(x), dtype=dt, device=device)
    out = {k: [] for k in ('tokens', 'tok_w', 'tok_off', 'blk_feat', 'blk_seg', 'blk_pos', 'seg_len', 'seg_feat',
                           'seg_bin', 'edge_seg', 'edge_src', 'edge_dst', 'edge_type', 'call_src', 'call_dst')}
    n_tok = n_seg = 0
    for bi, b in enumerate(bins):
        seg0 = n_seg
        for fn in b['fns']:
            for pos, blk in enumerate(fn['blocks']):
                out['tok_off'].append(n_tok)
                out['tokens'] += list(blk['ids'])
                out['tok_w'] += list(blk['w'])
                n_tok += len(blk['ids'])
                out['blk_feat'].append(blk['feat'])
                out['blk_seg'].append(n_seg)
                out['blk_pos'].append(pos)
            out['seg_len'].append(len(fn['blocks']))
            out['seg_feat'].append([np.log1p(len(fn['blocks'])), fn['lib']])
            out['seg_bin'].append(bi)
            for s, d, ty in fn['edges']:
                out['edge_seg'].append(n_seg)
                out['edge_src'].append(s)
                out['edge_dst'].append(d)
                out['edge_type'].append(ty)
            n_seg += 1
        out['call_src'] += [seg0 + a for a, _ in b['calls']]
        out['call_dst'] += [seg0 + c for _, c in b['calls']]
    batch = {
        'tokens': t(out['tokens']), 'tok_offsets': t(out['tok_off']), 'tok_weights': t(out['tok_w'], torch.float32),
        'blk_feat': t(out['blk_feat'], torch.float32), 'blk_seg': t(out['blk_seg']), 'blk_pos': t(out['blk_pos']),
        'seg_len': t(out['seg_len']), 'seg_feat': t(out['seg_feat'], torch.float32), 'seg_fn': t(np.arange(n_seg)),
        'seg_bin': t(out['seg_bin']), 'n_fn': n_seg, 'edge_seg': t(out['edge_seg']), 'edge_src': t(out['edge_src']),
        'edge_dst': t(out['edge_dst']), 'edge_type': t(out['edge_type']), 'call_src': t(out['call_src']),
        'call_dst': t(out['call_dst']), 'bin_feat': t([b['bin_feat'] for b in bins], torch.float32),
        'label': t([b['label'] for b in bins], torch.float32),
    }
    batch['deg_in'] = batch['deg_out'] = torch.zeros(len(out['blk_seg']), dtype=torch.long, device=device)
    if stack:
        batch['stack_logit'] = torch.linspace(-1, 1, len(bins), device=device)
    return batch


def model_for(**kw):
    torch.manual_seed(0)
    return HierGNN(cfg_for(**kw), N_TOKENS).to(DEV).eval()


def logits(model, bins, **kw):
    with torch.no_grad():
        return model(collate(bins, **kw)).cpu()


def permute_blocks(bins, seed):
    """Relabel the blocks of every function and shuffle each function's edge list."""
    rng = np.random.default_rng(seed)
    out = copy.deepcopy(bins)
    for b in out:
        for fn in b['fns']:
            n = len(fn['blocks'])
            perm = rng.permutation(n)            # old block i moves to position perm[i]
            blocks = [None] * n
            for old, new in enumerate(perm):
                blocks[new] = fn['blocks'][old]
            edges = [(int(perm[s]), int(perm[d]), ty) for s, d, ty in fn['edges']]
            fn['blocks'], fn['edges'] = blocks, [edges[i] for i in rng.permutation(len(edges))]
    return out


def permute_functions(bins, seed):
    rng = np.random.default_rng(seed)
    out = copy.deepcopy(bins)
    for b in out:
        perm = rng.permutation(len(b['fns']))
        fns = [None] * len(perm)
        for old, new in enumerate(perm):
            fns[new] = b['fns'][old]
        b['fns'] = fns
        calls = [(int(perm[a]), int(perm[c])) for a, c in b['calls']]
        b['calls'] = [calls[i] for i in rng.permutation(len(calls))]
    return out


def differs(a, b, tol=1e-4):
    return (a - b).abs().max().item() > tol


def test_batch_follows_contract():
    batch = collate(make_bins())
    n_blk, n_seg = len(batch['blk_seg']), len(batch['seg_len'])
    assert batch['n_fn'] == n_seg and int(batch['seg_len'].sum()) == n_blk
    assert len(batch['tok_offsets']) == n_blk and len(batch['tokens']) == len(batch['tok_weights'])
    assert (batch['tok_offsets'][1:] - batch['tok_offsets'][:-1] == 0).any()          # the empty-token block
    assert int(batch['edge_type'].max()) < 12 and (batch['call_src'] != batch['call_dst']).all()


def test_segment_softmax():
    gen = torch.Generator().manual_seed(0)
    lengths = [1, 3, 7, 2, 1]
    index = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths))
    score = torch.randn(len(index), generator=gen) * 5
    got = segment_softmax(score, index, len(lengths))
    padded = torch.full((len(lengths), max(lengths)), float('-inf'))
    start = 0
    for i, n in enumerate(lengths):
        padded[i, :n] = score[start:start + n]
        start += n
    want = torch.softmax(padded, -1)
    start = 0
    for i, n in enumerate(lengths):
        assert torch.allclose(got[start:start + n], want[i, :n], atol=1e-6, rtol=0)
        start += n
    for i in (0, 4):
        assert got[index == i].item() == 1.0
    unused = segment_softmax(score, index, len(lengths) + 3)
    assert torch.isfinite(unused).all() and torch.equal(unused, got)
    big = segment_softmax(torch.tensor([1e4, 1e4 - 1.0]), torch.zeros(2, dtype=torch.long), 1)
    assert torch.isfinite(big).all() and abs(big.sum().item() - 1) < 1e-6


def test_readouts():
    torch.manual_seed(0)
    h = torch.randn(7, D)
    index = torch.tensor([0, 0, 0, 1, 2, 2, 2])
    fixed = FixedReadout(D)
    got = fixed(h, index, 4)                                      # segment 3 is empty
    assert got.shape == (4, D) and torch.isfinite(got).all()
    for s in range(3):
        x = h[index == s]
        want = F.gelu(fixed.proj(torch.cat([x.sum(0), x.mean(0), x.max(0).values])))
        assert torch.allclose(got[s], want, atol=1e-5)
    gated = GatedReadout(D)
    got = gated(h, index, 4)
    assert got.shape == (4, D)
    for s in range(3):
        x = h[index == s]
        alpha = torch.softmax(gated.w(torch.tanh(gated.v(x))).squeeze(-1), 0)
        assert torch.allclose(got[s], (alpha[:, None] * x).sum(0), atol=1e-5)
    assert torch.equal(got[3], torch.zeros(D))


def test_dirgine_layer():
    torch.manual_seed(0)
    layer = DirGINELayer(D, 12, 0.0).eval()
    with torch.no_grad():
        layer.eps.fill_(0.3)
    n = 6
    h = torch.randn(n, D)
    src, dst, ty = torch.tensor([0, 1, 1, 4, 2]), torch.tensor([1, 2, 1, 0, 5]), torch.tensor([0, 3, 11, 5, 3])
    with torch.no_grad():
        m_in, m_out = torch.zeros(n, D), torch.zeros(n, D)
        for s, d, t in zip(src, dst, ty):
            m_in[d] += F.relu(h[s] + layer.e_in.weight[t])
            m_out[s] += F.relu(h[d] + layer.e_out.weight[t])
        z = layer.norm(1.3 * h + layer.mix(torch.cat([m_in, m_out], 1)))
        want = h + layer.mlp(z)
        assert torch.allclose(layer(h, src, dst, ty), want, atol=1e-5)
        none = torch.zeros(0, dtype=torch.long)
        z0 = layer.norm(1.3 * h + layer.mix(torch.zeros(n, 2 * D)))
        assert torch.allclose(layer(h, none, none, none), h + layer.mlp(z0), atol=1e-5)
        out = layer.to(torch.bfloat16)(h.bfloat16(), src, dst, ty)
        assert out.dtype == torch.bfloat16 and torch.allclose(out.float(), want, atol=0.15)


@pytest.mark.skipif(DEV != 'cuda', reason='needs CUDA')
def test_layer_under_autocast():
    torch.manual_seed(0)
    layer = DirGINELayer(D, 12, 0.0).cuda().eval()
    h = torch.randn(5, D, device='cuda')
    src, dst, ty = (torch.tensor(v, device='cuda') for v in ([0, 1, 4], [1, 2, 0], [0, 7, 2]))
    with torch.no_grad():
        want = layer(h, src, dst, ty)
        with torch.autocast('cuda', torch.bfloat16):
            got = layer(h, src, dst, ty)
    assert got.dtype == h.dtype and torch.allclose(got, want, atol=0.1)


def test_forward_shape_and_head_input():
    bins = make_bins()
    for kw, extra in (({}, 0), ({'stack_tfidf': True}, 1), ({'readout': 'gated'}, 0)):
        model = model_for(**kw)
        out = model(collate(make_bins(), stack=bool(kw.get('stack_tfidf'))))
        assert out.shape == (3,) and out.dtype == torch.float32 and torch.isfinite(out).all()
        assert model.head[0].in_features == D + 5 + extra
    assert len(model.fn_layers) == 3 and len(model.bin_layers) == 2
    assert model.fn_layers[0].e_in.num_embeddings == 12 and model.bin_layers[0].e_in.num_embeddings == 1
    bad = collate(bins)
    bad['n_fn'] += 1
    with pytest.raises(AssertionError):
        model_for()(bad)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def test_param_counts():
    arms = {'default': {}, 'untyped': {'untyped': True}, 'no_edges': {'control': 'no_edges'},
            'no_call_mp': {'no_call_mp': True}, 'chain': {'control': 'chain'}, 'shuffle': {'control': 'shuffle'},
            'permute_types': {'control': 'permute_types'}}
    counts = {name: n_params(model_for(**kw)) for name, kw in arms.items()}
    assert len(set(counts.values())) == 1, counts
    stacked = n_params(model_for(stack_tfidf=True))
    assert stacked - counts['default'] == D
    assert n_params(model_for(readout='gated')) != counts['default']


def test_permutation_invariance():
    bins = make_bins()
    for kw in ({}, {'readout': 'gated'}):
        model = model_for(**kw)
        base = logits(model, bins)
        assert torch.allclose(base, logits(model, permute_blocks(bins, 1)), atol=1e-5)
        assert torch.allclose(base, logits(model, permute_functions(bins, 2)), atol=1e-5)
        assert torch.allclose(base, logits(model, permute_functions(permute_blocks(bins, 3), 4)), atol=1e-5)


def random_edges(bins, seed):
    rng = np.random.default_rng(seed)
    out = copy.deepcopy(bins)
    for b in out:
        for fn in b['fns']:
            n = len(fn['blocks'])
            fn['edges'] = [(int(rng.integers(0, n)), int(rng.integers(0, n)), int(rng.integers(0, 12)))
                           for _ in range(len(fn['edges']) + 3)]
    return out


def degree_columns_changed(bins):
    out = copy.deepcopy(bins)
    for b in out:
        for fn in b['fns']:
            for blk in fn['blocks']:
                blk['feat'][4:6] += 1.7
    return out


def calls_reversed(bins):
    out = copy.deepcopy(bins)
    for b in out:
        b['calls'] = [(c, a) for a, c in b['calls']]
    return out


def edges_reversed(bins):
    out = copy.deepcopy(bins)
    for b in out:
        for fn in b['fns']:
            fn['edges'] = [(d, s, ty) for s, d, ty in fn['edges']]
    return out


def types_permuted(bins):
    out = copy.deepcopy(bins)
    for b in out:
        for fn in b['fns']:
            fn['edges'] = [(s, d, (ty * 5 + 3) % 12) for s, d, ty in fn['edges']]
    return out


def test_controls_remove_structure():
    bins = make_bins()
    no_edges = model_for(control='no_edges')
    base = logits(no_edges, bins)
    for changed in (random_edges(bins, 5), degree_columns_changed(bins), calls_reversed(bins),
                    random_edges(degree_columns_changed(calls_reversed(bins)), 6)):
        assert torch.allclose(base, logits(no_edges, changed), atol=1e-6)

    untyped = model_for(untyped=True)
    base = logits(untyped, bins)
    assert torch.allclose(base, logits(untyped, types_permuted(bins)), atol=1e-6)
    assert differs(base, logits(untyped, edges_reversed(bins)))

    default = model_for()
    base = logits(default, bins)
    assert differs(base, logits(default, types_permuted(bins)))
    assert differs(base, logits(default, edges_reversed(bins)))
    assert differs(base, logits(default, calls_reversed(bins)))
    assert differs(base, logits(default, degree_columns_changed(bins)))
    assert differs(base, logits(model_for(control='no_edges'), bins))

    no_call = model_for(no_call_mp=True)
    base = logits(no_call, bins)
    emptied = copy.deepcopy(bins)
    for b in emptied:
        b['calls'] = []
    assert torch.allclose(base, logits(no_call, emptied), atol=1e-6)
    assert torch.allclose(base, logits(no_call, calls_reversed(bins)), atol=1e-6)
    assert differs(logits(default, bins), logits(default, emptied))


def test_collate_no_crosstalk():
    bins = make_bins()
    model = model_for()
    together = logits(model, bins)
    for i in range(len(bins)):
        assert torch.allclose(together[i:i + 1], logits(model, bins[i:i + 1]), atol=1e-5)
    reordered = logits(model, [bins[2], bins[0], bins[1]])
    assert torch.allclose(together[[2, 0, 1]], reordered, atol=1e-5)


def test_stack_logit_enters():
    model = model_for(stack_tfidf=True)
    batch = collate(make_bins(), stack=True)
    with torch.no_grad():
        a = model(batch)
        batch['stack_logit'] = batch['stack_logit'] + 2
        assert differs(a, model(batch))


@pytest.mark.skipif(DEV != 'cuda', reason='needs CUDA')
@pytest.mark.parametrize('kw', [{}, {'readout': 'gated'}, {'no_call_mp': True}, {'control': 'no_edges'}])
def test_bf16_forward_backward(kw):
    model = model_for(**kw).train()
    batch = collate(make_bins())
    with torch.autocast('cuda', torch.bfloat16):
        out = model(batch)
    assert out.dtype == torch.float32
    loss = F.binary_cross_entropy_with_logits(out, batch['label'])
    loss.backward()
    assert torch.isfinite(loss)
    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert grads and all(torch.isfinite(g).all() for g in grads.values())
    assert not kw.get('readout') == 'gated' or any('fn_readout.v' in n for n in grads)
    assert sum(g.abs().sum().item() for g in grads.values()) > 0
    if not kw:
        assert len(grads) == len(list(model.parameters()))
