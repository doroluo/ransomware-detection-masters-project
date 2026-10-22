"""python -m pytest tests/test_reader_wiring.py: the vocabulary option `vocab` ours | yanping | hybrid on the synthetic tiny cache
(tests/tiny_cache.py plus the operand-role side arrays of tests/yanping_side.py): names, the vocabularies cut on the training
binaries, block tokens, masking, the batch and network on CPU, and the baselines (`baselines --vocab yanping|hybrid`) with the
groups the gates read.

Ported from the 3-block-level-counting branch (tests/test_reader_wiring.py). The header-import half is test_imports.py; `noapi`
is test_noapi.py; the queue stages are test_queue.py and the plan files. The converter that writes side/yanping is
test_reader_side.py.
"""
import json

import numpy as np
import pytest
import torch

from common.config import RunConfig
from common.manifests import load_sample
from eval_synth import evaltiny  # noqa: F401  (a fixture)
from experiments.train import parse_config
from s02_preprocess.tokenizer.block_tokens import token_table
from s02_preprocess.tokenizer.vocab import VOCAB_KINDS, TokenVocab
from reference.gnn.batching import BinaryDataset, encode_binary, loader
from reference.gnn.spec import build_model
from s04_evaluation import baselines, gate_tables, stats
from tiny_cache import N_API, N_BEH, N_MN, tiny  # noqa: F401  (a fixture that tiny_yanping builds on)
from yanping_side import tiny_yanping  # noqa: F401  (a fixture)


def gnn(**kw):
    return RunConfig(**{'model': 'gnn', 'num_workers': 0, **kw})


def fold(f=0, **kw):
    return gnn(protocol='kfold', fold=f, **kw)


# ---- configuration ------------------------------------------------------------------------------------------------

def test_default_names_are_unchanged():
    assert RunConfig().name == 'bundle_x86_gnn_s0'
    assert RunConfig(protocol='kfold', fold=2, model='gnn').name == 'kfold2_x86_gnn_s0'
    assert fold(0, control='no_edges', mask_blocks=0.15).name == 'kfold0_x86_gnn-noedges-mblk15_s0'
    assert fold(0, vocab='ours').name == fold(0).name and RunConfig().vocab == 'ours'
    assert fold(0, mask_blocks=0.15).group == 'kfold_x86_gnn-mblk15'


def test_vocab_suffixes():
    assert fold(0, control='no_edges', mask_blocks=0.15, vocab='yanping').name == 'kfold0_x86_gnn-noedges-mblk15-voc-yanping_s0'
    assert fold(3, control='no_edges', mask_blocks=0.15, vocab='hybrid').name == 'kfold3_x86_gnn-noedges-mblk15-voc-hybrid_s0'
    assert fold(0, vocab='hybrid', imports='header').variant == 'gnn-voc-hybrid-imph'


def test_vocab_validation():
    for bad in (dict(vocab='bert'), dict(vocab='yanping', block_input='vec', block_vec='x')):
        with pytest.raises(ValueError):
            fold(0, **bad)


def test_cli_parses_the_vocab_option():
    cfg, _ = parse_config('--protocol kfold --fold 0 --model gnn --vocab hybrid --imports header'.split())
    assert (cfg.vocab, cfg.imports, cfg.name) == ('hybrid', 'header', 'kfold0_x86_gnn-voc-hybrid-imph_s0')
    cfg, _ = parse_config('--protocol kfold --fold 0 --model gnn --vocab yanping --mask-blocks 0.15'.split())
    assert (cfg.vocab, cfg.name) == ('yanping', 'kfold0_x86_gnn-mblk15-voc-yanping_s0')
    with pytest.raises(SystemExit):
        parse_config('--protocol kfold --fold 0 --model gnn --vocab nope'.split())


# ---- vocabularies -------------------------------------------------------------------------------------------------

def train_rows(tiny_yanping, n=30):
    return tiny_yanping.rows[:n]


def old_vocab(rows, cfg, cache):
    """The vocabulary code before the `vocab` option, as a reference for 'ours'."""
    sizes = {k: len(v) for k, v in json.loads((cache / 'vocab.json').read_text(encoding='utf-8')).items()}
    df = {k: np.zeros(sizes[k], dtype=np.int64) for k in ('mn', 'sig', 'api')}
    df_beh = np.zeros(sizes['beh'], dtype=np.int64)
    for r in rows:
        with np.load(cache / 'samples' / f"{r['sample_id']}.npz") as z:
            for k in df:
                df[k][z[f'{k}_u']] += 1
            df_beh[np.unique(z['beh'])] += 1
    kinds = ('mn', 'sig', 'beh', 'api')
    lut, size, kept_df = {}, {}, {}
    for k, counts in df.items():
        keep = counts >= cfg.min_df
        lut[k] = np.where(keep, np.cumsum(keep), 0).astype(np.int64)
        size[k] = int(keep.sum()) + 1
        kept_df[k] = counts[keep]
    lut['beh'], size['beh'] = np.arange(sizes['beh'], dtype=np.int64), sizes['beh']
    offset = dict(zip(kinds, np.cumsum([0] + [size[k] for k in kinds[:-1]]).tolist()))
    total = sum(size.values())
    idf = np.ones(total + 4, dtype=np.float32)
    kept_df['beh'] = df_beh
    for k, counts in kept_df.items():
        lo = offset[k] + (k != 'beh')
        idf[lo:lo + len(counts)] = np.log((1 + len(rows)) / (1 + counts)) + 1
    return lut, size, offset, total, idf, {k: total + j for j, k in enumerate(kinds)}


def test_ours_vocabulary_is_the_vocabulary_of_before(tiny_yanping):
    rows = train_rows(tiny_yanping)
    new = TokenVocab(rows, gnn(protocol='kfold', fold=0))
    lut, size, offset, total, idf, mask_id = old_vocab(rows, gnn(), tiny_yanping.root)
    assert new.kinds == ('mn', 'sig', 'beh', 'api') and new.insn_kinds == ('mn', 'sig', 'beh')
    assert all((new.lut[k] == lut[k]).all() for k in lut) and list(new.lut) == list(lut)
    assert new.size == size and list(new.size) == list(size) and new.offset == offset
    assert new.total == total + 4 and new.mask_id == mask_id
    assert new.idf.dtype == idf.dtype and (new.idf == idf).all()


def reader_unique(tiny_yanping, rows, slot):
    from experiments.side_arrays.import_reader_side import load_ytok
    return [np.unique(load_ytok(tiny_yanping.root, r['sample_id'])[:, slot]) for r in rows]


@pytest.mark.parametrize('name', ['yanping', 'hybrid'])
def test_reader_vocabularies_cut_every_kind_on_the_training_binaries(tiny_yanping, name):
    rows = train_rows(tiny_yanping)
    cfg = gnn(protocol='kfold', fold=0, vocab=name)
    v = TokenVocab(rows, cfg)
    assert v.kinds == VOCAB_KINDS[name] and v.insn_kinds == v.kinds[:-1] and v.kinds[-1] == 'api'
    assert v.kinds == {'yanping': ('y0', 'y1', 'y2', 'api'), 'hybrid': ('mn', 'beh', 'y1', 'y2', 'api')}[name]
    width = {'y0': 256, 'y1': 76, 'y2': 76, 'mn': N_MN, 'api': N_API}
    uniques = {'y0': reader_unique(tiny_yanping, rows, 0), 'y1': reader_unique(tiny_yanping, rows, 1), 'y2': reader_unique(tiny_yanping, rows, 2)}
    for k in v.kinds:
        if k == 'beh':
            assert (v.lut[k] == np.arange(N_BEH)).all() and v.size[k] == N_BEH         # kept whole, as before
            continue
        if k in uniques:
            per_sample = uniques[k]
        else:
            per_sample = []
            for r in rows:
                with np.load(tiny_yanping.root / 'samples' / f"{r['sample_id']}.npz") as z:
                    per_sample.append(z[f'{k}_u'])
        df = np.zeros(width[k], dtype=np.int64)
        for ids in per_sample:
            df[ids] += 1
        keep = df >= cfg.min_df
        assert (v.lut[k] == np.where(keep, np.cumsum(keep), 0)).all() and v.size[k] == keep.sum() + 1, k
        run = v.offset[k] + v.lut[k][keep]
        want = np.log((1 + len(rows)) / (1 + df[keep])) + 1
        assert np.allclose(v.idf[run], want) and v.idf[v.offset[k]] == 1              # UNK weighs 1
    assert list(v.offset.values()) == np.cumsum([0] + [v.size[k] for k in v.kinds[:-1]]).tolist()
    assert v.total == sum(v.size.values()) + len(v.kinds)
    assert v.mask_id == {k: sum(v.size.values()) + j for j, k in enumerate(v.kinds)}
    assert len(v.idf) == v.total and v.idf[-len(v.kinds):].tolist() == [1.0] * len(v.kinds)
    assert any(v.lut[k].min() == 0 for k in v.kinds if k != 'beh')                    # some id is cut to UNK somewhere


def test_reader_vocabulary_without_side_array_names_the_sample(tiny_yanping):
    victim = tiny_yanping.root / 'side' / 'yanping' / f"{tiny_yanping.ids[3]}.npz"
    moved = victim.with_suffix('.moved')
    victim.rename(moved)
    try:
        with pytest.raises(FileNotFoundError, match=tiny_yanping.ids[3]):
            TokenVocab(train_rows(tiny_yanping), gnn(protocol='kfold', fold=0, vocab='yanping'))
        with pytest.raises(FileNotFoundError, match=tiny_yanping.ids[3]):
            load_sample(tiny_yanping.ids[3], ytok=True)
        assert 'ytok' not in load_sample(tiny_yanping.ids[3])                                  # the default path never reads it
    finally:
        moved.rename(victim)


def expected_tokens(z, v, blocks):
    """[insn kind 0 of the block's instructions | kind 1 | ... | api] per block, as run ids; with each token's kind."""
    values = lambda k: z['ytok'][:, int(k[1])] if k.startswith('y') else z[k]
    tokens, kind, offsets = [], [], []
    for b in blocks:
        offsets.append(len(tokens))
        lo, hi = z['blk_insn_ptr'][b], z['blk_insn_ptr'][b + 1]
        for j, k in enumerate(v.insn_kinds):
            tokens += [int(v.lut[k][g]) + v.offset[k] for g in values(k)[lo:hi]]
            kind += [j] * (hi - lo)
        api = z['api'][z['blk_api_ptr'][b]:z['blk_api_ptr'][b + 1]]
        tokens += [int(v.lut['api'][g]) + v.offset['api'] for g in api]
        kind += [len(v.insn_kinds)] * len(api)
    return np.array(tokens), np.array(kind), np.array(offsets)


@pytest.mark.parametrize('name', ['ours', 'yanping', 'hybrid'])
def test_block_tokens_follow_the_vocabulary_kinds(tiny_yanping, name):
    cfg = gnn(protocol='kfold', fold=0, vocab=name)
    v = TokenVocab(train_rows(tiny_yanping), cfg)
    z = load_sample(tiny_yanping.ids[35], ytok=name != 'ours')
    blocks = np.arange(0, len(z['blk_insn_ptr']) - 1, 2)
    tokens, kind, tok_offsets, per_block, ins_pos, api_pos = token_table(z, v, blocks)
    want_tokens, want_kind, want_offsets = expected_tokens(z, v, blocks)
    assert (tokens == want_tokens).all() and (kind == want_kind).all() and (tok_offsets == want_offsets).all()
    assert len(ins_pos) == len(v.insn_kinds) and (per_block == np.diff(np.append(tok_offsets, len(tokens)))).all()
    assert tokens.max() < v.total - len(v.kinds)                                        # no mask id without masking


@pytest.mark.parametrize('name', ['yanping', 'hybrid'])
def test_masking_uses_the_mask_id_of_each_kind(tiny_yanping, name):
    v = TokenVocab(train_rows(tiny_yanping), gnn(protocol='kfold', fold=0, vocab=name))
    z = load_sample(tiny_yanping.ids[2], ytok=True)
    plain = encode_binary(z, v, gnn(protocol='kfold', fold=0, vocab=name), None, 's')
    for mask in (dict(mask_tokens=0.6), dict(mask_blocks=0.6)):
        cfg = gnn(protocol='kfold', fold=0, vocab=name, **mask)
        out = encode_binary(z, v, cfg, np.random.default_rng(5), 's')
        blocks = np.arange(len(z['blk_insn_ptr']) - 1)
        _, kind, *_ = token_table(z, v, blocks)
        masked = out['tokens'] >= v.total - len(v.kinds)
        assert masked.any() and (~masked).any()
        assert (out['tokens'][masked] == np.array([v.mask_id[k] for k in v.kinds])[kind[masked]]).all()
        assert (out['tokens'][~masked] == plain['tokens'][~masked]).all()


def test_dataset_items_and_default_runs_carry_nothing_extra(tiny_yanping):
    rows = tiny_yanping.rows[:6]
    cfg = gnn(protocol='kfold', fold=0, vocab='yanping')
    v = TokenVocab(train_rows(tiny_yanping), cfg)
    item = BinaryDataset(rows, v, cfg)[(1, None)]
    z = load_sample(rows[1]['sample_id'], ytok=True)
    want, _, offsets = expected_tokens(z, v, np.arange(len(z['blk_insn_ptr']) - 1))
    assert (item['tokens'] == want).all() and (item['tok_offsets'] == offsets).all()      # the side array reached the bag
    plain = BinaryDataset(rows, TokenVocab(train_rows(tiny_yanping), fold()), fold())[(1, None)]
    assert not {'imp_idx', 'imp_w', 'imp_feat', 'ytok'} & set(plain)
    assert np.isin(item['tokens'], np.arange(v.total)).all()




# ---- the batch and the network ------------------------------------------------------------------------------------

@pytest.mark.parametrize('kw', [dict(vocab='yanping'), dict(vocab='hybrid', control='no_edges'),
                                dict(vocab='yanping', mask_blocks=0.3, mask_tokens=0.2)])
def test_forward_and_backward_on_cpu(tiny_yanping, kw):
    cfg = fold(0, batch_size=8, **kw)
    v = TokenVocab(train_rows(tiny_yanping), cfg)
    batch = next(iter(loader(tiny_yanping.rows[10:18], v, cfg, train=False)))
    model = build_model(cfg, v, None, 0)
    logits = model(batch)
    assert logits.shape == (8,) and torch.isfinite(logits).all()
    logits.sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad and p.dim() > 1 and p.grad is not None)


# ---- baselines, groups and the gates ------------------------------------------------------------------------------

def run_baselines(ty, vocab):
    sizes = {k: len(v) for k, v in json.loads((ty.root / 'vocab.json').read_text(encoding='utf-8')).items()}
    for f in range(5):
        baselines.run(RunConfig(protocol='kfold', fold=f, arch='x86', seed=0), sizes, ty.runs / 'baselines', vocab)


@pytest.fixture(scope='module')
def built(tiny_yanping, evaltiny):  # noqa: F811
    run_baselines(evaltiny, 'yanping')
    run_baselines(evaltiny, 'hybrid')
    gate_tables.pooled.cache_clear()
    yield evaltiny
    gate_tables.pooled.cache_clear()


def test_vocab_baselines_write_their_own_directories(built):
    for name in ('yanping', 'hybrid'):
        for f in range(5):
            d = built.runs / 'baselines' / f'kfold{f}_x86_s0_voc-{name}'
            m = json.loads((d / 'metrics.json').read_text(encoding='utf-8'))
            assert list(m['methods']) == [f'tfidf_{name}_lr'] and m['vocab'] == name and m['tfidf_features'] > 0
            assert 'threshold_val' in m['methods'][f'tfidf_{name}_lr'] and 'floors' in m
    assert not (built.runs / 'baselines' / 'kfold0_x86_s0').exists()                   # nothing wrote into the plain directory


def test_the_gates_find_what_the_vocab_baselines_wrote(built):
    for group in ('tfidf-yanping', 'tfidf-hybrid'):
        folds = stats.folds_by_seed(group)
        assert sorted(folds[0]) == [0, 1, 2, 3, 4], group
        r = gate_tables.pooled(group, 'auc')
        assert not isinstance(r, str) and 0 <= r[0] <= 1, (group, r)
    assert gate_tables.pooled('tfidf-yanping', 'auc')[0] > 0.6                         # the planted opcode signal is found
    assert stats.folds_by_seed('tfidf') == {}                                          # the plain baseline was not touched


def test_a_vocab_baseline_without_the_side_array_names_the_sample(tiny_yanping, evaltiny, tmp_path):  # noqa: F811
    sizes = {k: len(v) for k, v in json.loads((evaltiny.root / 'vocab.json').read_text(encoding='utf-8')).items()}
    victim = evaltiny.root / 'side' / 'yanping' / f'{evaltiny.ids[5]}.npz'
    moved = victim.with_suffix('.moved')
    victim.rename(moved)
    try:
        with pytest.raises(FileNotFoundError, match=evaltiny.ids[5]):
            baselines.run(RunConfig(protocol='kfold', fold=0, arch='x86', seed=0), sizes, tmp_path, 'yanping')
    finally:
        moved.rename(victim)
