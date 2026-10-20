"""python -m pytest tests/test_reader_wiring.py: the vocabulary option (`vocab` ours | yanping | hybrid) and the import feature
(`imports` header; `unreferenced` was removed in step 25) on a small synthetic cache (tests/tiny_cache.py): names, vocabularies,
block tokens, masking, the import vocabulary, batches and the network, the baselines (`--vocab`, `--imports`) with the groups the
gates read, and the queue stages."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from common.config import RunConfig
from common.manifests import load_index, load_sample, make_manifest
from evaluation import baselines, gates, stats
from pipeline.batching import BinaryDataset, collate, encode_binary, loader
from pipeline.model import HierGNN
from pipeline.s03_tokenizer import vocab as vocab_module
from pipeline.s03_tokenizer.block_tokens import token_table
from pipeline.s03_tokenizer.vocab import VOCAB_KINDS, TokenVocab
from pipeline.s10_unused_import_features import imports_feat
from pipeline.s10_unused_import_features.imports_feat import IMPORT_DIM, ImportVocab
from tiny_cache import N_API, N_BEH, N_MN, tiny  # noqa: F401  (tiny is a fixture)
from training.train import build_model, parse_config

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which('bash')
needs_bash = pytest.mark.skipif(BASH is None, reason='needs bash')


def gnn(**kw):
    return RunConfig(**{'model': 'gnn', 'num_workers': 0, **kw})


def fold(f=0, **kw):
    return gnn(protocol='kfold', fold=f, **kw)


# ---- configuration ------------------------------------------------------------------------------------------------

def test_default_names_are_unchanged():
    assert RunConfig().name == 'bundle_x86_gnn_s0'
    assert RunConfig(protocol='kfold', fold=2, model='gnn').name == 'kfold2_x86_gnn_s0'
    assert fold(0, control='no_edges', mask_blocks=0.15).name == 'kfold0_x86_gnn-noedges-mblk15_s0'
    assert fold(0, vocab='ours', imports='').name == fold(0).name
    assert (RunConfig().vocab, RunConfig().imports) == ('ours', '')
    assert fold(0, mask_blocks=0.15).group == 'kfold_x86_gnn-mblk15'


def test_vocab_and_imports_suffixes():
    assert fold(0, control='no_edges', mask_blocks=0.15, vocab='yanping').name == 'kfold0_x86_gnn-noedges-mblk15-voc-yanping_s0'
    assert fold(3, control='no_edges', mask_blocks=0.15, vocab='hybrid').name == 'kfold3_x86_gnn-noedges-mblk15-voc-hybrid_s0'
    assert fold(0, mask_blocks=0.15, imports='header').name == 'kfold0_x86_gnn-mblk15-imph_s0'
    assert fold(0, mask_blocks=0.15, imports='header').group == 'kfold_x86_gnn-mblk15-imph'
    assert fold(0, vocab='hybrid', imports='header').variant == 'gnn-voc-hybrid-imph'


def test_vocab_and_imports_validation():
    for bad in (dict(vocab='bert'), dict(imports='all'), dict(imports='unreferenced'),
                dict(vocab='yanping', block_input='vec', block_vec='x')):
        with pytest.raises(ValueError):
            fold(0, **bad)


def test_cli_parses_vocab_and_imports():
    cfg, _ = parse_config('--protocol kfold --fold 0 --model gnn --vocab hybrid --imports header'.split())
    assert (cfg.vocab, cfg.imports, cfg.name) == ('hybrid', 'header', 'kfold0_x86_gnn-voc-hybrid-imph_s0')
    with pytest.raises(SystemExit):
        parse_config('--protocol kfold --fold 0 --model gnn --imports unreferenced'.split())
    cfg, _ = parse_config('--protocol kfold --fold 0 --model gnn --mask-blocks 0.15 --imports header'.split())
    assert (cfg.imports, cfg.name) == ('header', 'kfold0_x86_gnn-mblk15-imph_s0')
    assert parse_config('--protocol kfold --fold 0 --model gnn'.split())[0] == fold(0, num_workers=2)
    with pytest.raises(SystemExit):
        parse_config('--protocol kfold --fold 0 --model gnn --vocab nope'.split())


# ---- vocabularies -------------------------------------------------------------------------------------------------

def train_rows(tiny, n=30):  # noqa: F811
    return tiny.rows[:n]


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


def test_ours_vocabulary_is_the_vocabulary_of_before(tiny):  # noqa: F811
    rows = train_rows(tiny)
    new = TokenVocab(rows, gnn(protocol='kfold', fold=0))
    lut, size, offset, total, idf, mask_id = old_vocab(rows, gnn(), tiny.root)
    assert new.kinds == ('mn', 'sig', 'beh', 'api') and new.insn_kinds == ('mn', 'sig', 'beh')
    assert all((new.lut[k] == lut[k]).all() for k in lut) and list(new.lut) == list(lut)
    assert new.size == size and list(new.size) == list(size) and new.offset == offset
    assert new.total == total + 4 and new.mask_id == mask_id
    assert new.idf.dtype == idf.dtype and (new.idf == idf).all()


def reader_unique(tiny, rows, slot):  # noqa: F811
    from experiments.side_arrays.import_reader_side import load_ytok
    return [np.unique(load_ytok(tiny.root, r['sample_id'])[:, slot]) for r in rows]


@pytest.mark.parametrize('name', ['yanping', 'hybrid'])
def test_reader_vocabularies_cut_every_kind_on_the_training_binaries(tiny, name):  # noqa: F811
    rows = train_rows(tiny)
    cfg = gnn(protocol='kfold', fold=0, vocab=name)
    v = TokenVocab(rows, cfg)
    assert v.kinds == VOCAB_KINDS[name] and v.insn_kinds == v.kinds[:-1] and v.kinds[-1] == 'api'
    assert v.kinds == {'yanping': ('y0', 'y1', 'y2', 'api'), 'hybrid': ('mn', 'beh', 'y1', 'y2', 'api')}[name]
    width = {'y0': 256, 'y1': 76, 'y2': 76, 'mn': N_MN, 'api': N_API}
    uniques = {'y0': reader_unique(tiny, rows, 0), 'y1': reader_unique(tiny, rows, 1), 'y2': reader_unique(tiny, rows, 2)}
    for k in v.kinds:
        if k == 'beh':
            assert (v.lut[k] == np.arange(N_BEH)).all() and v.size[k] == N_BEH         # kept whole, as before
            continue
        if k in uniques:
            per_sample = uniques[k]
        else:
            per_sample = []
            for r in rows:
                with np.load(tiny.root / 'samples' / f"{r['sample_id']}.npz") as z:
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


def test_reader_vocabulary_without_side_array_names_the_sample(tiny):  # noqa: F811
    victim = tiny.root / 'side' / 'yanping' / f"{tiny.ids[3]}.npz"
    moved = victim.with_suffix('.moved')
    victim.rename(moved)
    try:
        with pytest.raises(FileNotFoundError, match=tiny.ids[3]):
            TokenVocab(train_rows(tiny), gnn(protocol='kfold', fold=0, vocab='yanping'))
        with pytest.raises(FileNotFoundError, match=tiny.ids[3]):
            load_sample(tiny.ids[3], ytok=True)
        assert 'ytok' not in load_sample(tiny.ids[3])                                  # the default path never reads it
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
def test_block_tokens_follow_the_vocabulary_kinds(tiny, name):  # noqa: F811
    cfg = gnn(protocol='kfold', fold=0, vocab=name)
    v = TokenVocab(train_rows(tiny), cfg)
    z = load_sample(tiny.ids[35], ytok=name != 'ours')
    blocks = np.arange(0, len(z['blk_insn_ptr']) - 1, 2)
    tokens, kind, tok_offsets, per_block, ins_pos, api_pos = token_table(z, v, blocks)
    want_tokens, want_kind, want_offsets = expected_tokens(z, v, blocks)
    assert (tokens == want_tokens).all() and (kind == want_kind).all() and (tok_offsets == want_offsets).all()
    assert len(ins_pos) == len(v.insn_kinds) and (per_block == np.diff(np.append(tok_offsets, len(tokens)))).all()
    assert tokens.max() < v.total - len(v.kinds)                                        # no mask id without masking


@pytest.mark.parametrize('name', ['yanping', 'hybrid'])
def test_masking_uses_the_mask_id_of_each_kind(tiny, name):  # noqa: F811
    v = TokenVocab(train_rows(tiny), gnn(protocol='kfold', fold=0, vocab=name))
    z = load_sample(tiny.ids[2], ytok=True)
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


def test_dataset_items_and_default_runs_carry_nothing_extra(tiny):  # noqa: F811
    rows = tiny.rows[:6]
    cfg = gnn(protocol='kfold', fold=0, vocab='yanping')
    v = TokenVocab(train_rows(tiny), cfg)
    item = BinaryDataset(rows, v, cfg)[(1, None)]
    z = load_sample(rows[1]['sample_id'], ytok=True)
    want, _, offsets = expected_tokens(z, v, np.arange(len(z['blk_insn_ptr']) - 1))
    assert (item['tokens'] == want).all() and (item['tok_offsets'] == offsets).all()      # the side array reached the bag
    plain = BinaryDataset(rows, TokenVocab(train_rows(tiny), fold()), fold())[(1, None)]
    assert not {'imp_idx', 'imp_w', 'imp_feat', 'ytok'} & set(plain)
    assert np.isin(item['tokens'], np.arange(v.total)).all()


# ---- header imports ------------------------------------------------------------------------------------------------

def header(tiny, row):  # noqa: F811
    return imports_feat.load_imports(tiny.root, row['sample_id'])['header_imports']


def reference_header_item(tiny, train, sample_id, min_df=2):  # noqa: F811
    """The header-import vocabulary and item written out again from the definition, as plain Python. The two log counts are
    log1p(#unreferenced) and log1p(#header), as in the finished `-imph` runs."""
    count = {}
    for r in train:
        for n in set(header(tiny, r)):
            count[n] = count.get(n, 0) + 1
    names = sorted(n for n, c in count.items() if c >= min_df)
    idf = np.array([np.log((1 + len(train)) / (1 + count[n])) + 1 for n in names], dtype=np.float32)
    got = imports_feat.load_imports(tiny.root, sample_id)
    idx = np.array(sorted(names.index(n) for n in set(got['header_imports']) if n in names), dtype=np.int64)
    w = idf[idx]
    if len(w):
        w = w / np.float32(np.sqrt((w.astype(np.float64) ** 2).sum()))
    feat = np.log1p([len(got['unreferenced_imports']), len(got['header_imports'])]).astype(np.float32)
    return names, idf, {'imp_idx': idx, 'imp_w': w.astype(np.float32), 'imp_feat': feat}


def test_header_imports_stay_bit_identical_to_the_definition(tiny):  # noqa: F811
    train = tiny.rows[:30]
    v = ImportVocab(train, 2)
    for sample in tiny.rows[30:]:
        names, idf, ref = reference_header_item(tiny, train, sample['sample_id'])
        assert v.names == names and v.idf.dtype == idf.dtype and np.array_equal(v.idf, idf)
        item = v.item(sample['sample_id'])
        assert set(item) == set(ref)
        for k in ref:
            assert item[k].dtype == ref[k].dtype and np.array_equal(item[k], ref[k]), k
    torch.manual_seed(0)
    a = HierGNN(fold(0, imports='header'), 100, n_imports=len(v))
    torch.manual_seed(0)
    b = HierGNN(fold(0, imports='header'), 100, n_imports=len(v))
    assert list(a.state_dict()) == list(b.state_dict())
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())


def test_header_vocabulary_is_fitted_on_the_training_binaries_only(tiny):  # noqa: F811
    train, other = tiny.rows[:24], tiny.rows[24:]
    v = ImportVocab(train, min_df=2)
    df = {}
    for r in train:
        for n in set(header(tiny, r)):
            df[n] = df.get(n, 0) + 1
    assert v.names == sorted(n for n, c in df.items() if c >= 2) and len(v) > 0
    assert (v.idf == np.array([np.log(25 / (1 + df[n])) + 1 for n in v.names], dtype=np.float32)).all()
    assert not ({n for r in other for n in header(tiny, r)} - set(df)) & set(v.names)
    assert ImportVocab(train, min_df=3).names == sorted(n for n, c in df.items() if c >= 3)


def test_header_item_is_a_normalised_idf_bag_and_the_same_two_log_counts(tiny):  # noqa: F811
    v = ImportVocab(tiny.rows[:30], min_df=2)
    for r in tiny.rows[30:]:
        got = imports_feat.load_imports(tiny.root, r['sample_id'])
        item = v.item(r['sample_id'])
        kept = [n for n in set(got['header_imports']) if n in v.index]
        assert item['imp_idx'].tolist() == sorted(v.index[n] for n in kept) and item['imp_idx'].dtype == np.int64
        np.testing.assert_allclose(item['imp_feat'], np.log1p([len(got['unreferenced_imports']), len(got['header_imports'])]),
                                   rtol=1e-6)
        assert item['imp_feat'].shape == (2,) and item['imp_w'].dtype == np.float32
        if kept:
            assert abs(float((item['imp_w'].astype(np.float64) ** 2).sum()) - 1) < 1e-5
            raw = v.idf[item['imp_idx']]
            np.testing.assert_allclose(item['imp_w'], raw / np.linalg.norm(raw), rtol=1e-5)
        else:
            assert len(item['imp_w']) == 0
    assert v.summary()['which'] == 'header'


def make_batch(tiny, cfg, rows):  # noqa: F811
    v = TokenVocab(train_rows(tiny), cfg)
    imports = ImportVocab(train_rows(tiny), cfg.min_df) if cfg.imports else None
    data = loader(rows, v, cfg, train=False, imports=imports)
    return v, imports, next(iter(data))


def test_collate_shifts_import_offsets_and_default_batches_have_none(tiny):  # noqa: F811
    cfg = fold(0, imports='header', batch_size=8)
    v, imports, batch = make_batch(tiny, cfg, tiny.rows[28:36])
    items = [imports.item(r['sample_id']) for r in tiny.rows[28:36]]
    lengths = [len(i['imp_idx']) for i in items]
    assert batch['imp_offsets'].tolist() == np.cumsum([0] + lengths[:-1]).tolist()
    assert batch['imp_idx'].tolist() == np.concatenate([i['imp_idx'] for i in items]).tolist()
    assert batch['imp_feat'].shape == (8, 2) and batch['imp_w'].dtype == torch.float32
    assert max(lengths) > 0
    _, _, plain = make_batch(tiny, fold(0, batch_size=8), tiny.rows[28:36])
    assert not {'imp_idx', 'imp_w', 'imp_offsets', 'imp_feat'} & set(plain)


def test_network_with_and_without_the_import_bag(tiny):  # noqa: F811
    base = HierGNN(fold(0), 100)
    with_imports = HierGNN(fold(0, imports='header'), 100, n_imports=17)
    assert not [k for k in base.state_dict() if k.startswith('imports')]               # a default checkpoint keeps its keys
    assert list(with_imports.state_dict()) == list(base.state_dict())[:-4] + ['imports.bag.weight'] + list(base.state_dict())[-4:]
    assert base.head[0].in_features == 128 + 5 and with_imports.head[0].in_features == 128 + 5 + IMPORT_DIM + 2
    assert with_imports.imports.bag.weight.shape == (17, IMPORT_DIM)
    assert HierGNN(fold(0, imports='header'), 100, n_imports=0).imports.bag.weight.shape[0] == 1   # no name survived


@pytest.mark.parametrize('kw', [dict(vocab='yanping'), dict(vocab='hybrid', control='no_edges'),
                                dict(imports='header'), dict(vocab='yanping', imports='header', mask_blocks=0.3, mask_tokens=0.2)])
def test_forward_and_backward_on_cpu(tiny, kw):  # noqa: F811
    cfg = fold(0, batch_size=8, **kw)
    v, imports, batch = make_batch(tiny, cfg, tiny.rows[10:18])
    model = build_model(cfg, v, None, len(imports) if imports is not None else 0)
    logits = model(batch)
    assert logits.shape == (8,) and torch.isfinite(logits).all()
    logits.sum().backward()
    if imports is not None:
        assert model.imports.bag.weight.grad is not None and model.imports.bag.weight.grad.abs().sum() > 0
        # the import bag moves the logit: the same batch without its names gives another one
        blank = {**batch, 'imp_idx': batch['imp_idx'][:0], 'imp_w': batch['imp_w'][:0],
                 'imp_offsets': torch.zeros(8, dtype=torch.long)}
        model.eval()
        assert not torch.allclose(model(batch), model(blank))


# ---- baselines, groups and the gate --------------------------------------------------------------------------------

def run_baselines(tiny, vocab='ours', imports=False, min_df=2):  # noqa: F811
    sizes = {k: len(v) for k, v in json.loads((tiny.root / 'vocab.json').read_text(encoding='utf-8')).items()}
    for f in range(5):
        cfg = RunConfig(protocol='kfold', fold=f, arch='x86', seed=0, min_df=min_df)
        baselines.run(cfg, sizes, tiny.runs / 'baselines', vocab, imports)


@pytest.fixture(scope='module')
def built(tiny):  # noqa: F811
    run_baselines(tiny, 'yanping')
    run_baselines(tiny, 'hybrid')
    run_baselines(tiny, imports=True)
    gates.pooled.cache_clear()
    yield tiny
    gates.pooled.cache_clear()


def test_vocab_baselines_write_their_own_directories(built):
    for name in ('yanping', 'hybrid'):
        for f in range(5):
            d = built.runs / 'baselines' / f'kfold{f}_x86_s0_voc-{name}'
            m = json.loads((d / 'metrics.json').read_text(encoding='utf-8'))
            assert list(m['methods']) == [f'tfidf_{name}_lr'] and m['vocab'] == name and m['tfidf_features'] > 0
            assert 'threshold_val' in m['methods'][f'tfidf_{name}_lr'] and 'floors' in m
    assert not (built.runs / 'baselines' / 'kfold0_x86_s0').exists()                   # nothing wrote into the plain directory


def test_import_baselines_methods_and_feature_counts(built):
    m = json.loads((built.runs / 'baselines' / 'kfold0_x86_s0_imp' / 'metrics.json').read_text(encoding='utf-8'))
    assert list(m['methods']) == ['imports_header_lr']
    assert m['imports']['features_per_method']['imports_header_lr'] > 0
    rows = (built.runs / 'baselines' / 'kfold0_x86_s0_imp' / 'predictions.csv').read_text(encoding='utf-8').splitlines()
    assert {r.split(',')[2] for r in rows[1:]} == set(m['methods'])


def test_the_gates_find_what_the_baselines_wrote(built):
    for group in ('tfidf-yanping', 'tfidf-hybrid', 'imports-header'):
        folds = stats.folds_by_seed(group)
        assert sorted(folds[0]) == [0, 1, 2, 3, 4], group
        r = gates.pooled(group, 'auc')
        assert not isinstance(r, str) and 0 <= r[0] <= 1, (group, r)
    assert stats.folds_by_seed('tfidf') == {}                                          # the plain baseline was not touched
    assert gates.pooled('imports-header', 'auc')[0] > 0.6                             # the planted signal is found
    result, info = stats.compare('tfidf-yanping', 'imports-header', 'auc', 50, seeds=[0])
    assert info['seeds'] == [0] and 'auc' in result


def test_the_unreferenced_imports_gate_and_its_option_are_gone(monkeypatch, capsys):
    assert not hasattr(gates, 'imports_gate') and not hasattr(gates, 'majority_floor_auc')
    assert set(stats.BASELINES) & {'imports-unref', 'tfidf-imp'} == set()
    monkeypatch.setattr(sys, 'argv', ['gates.py', '--stage', 'imports', '--verdict'])
    with pytest.raises(SystemExit) as stop:
        gates.main()
    assert stop.value.code == 2


def test_a_feature_space_with_no_column_gives_the_constant_half(tiny):  # noqa: F811
    sizes = {k: len(v) for k, v in json.loads((tiny.root / 'vocab.json').read_text(encoding='utf-8')).items()}
    out = tiny.runs / 'empty'
    baselines.run(RunConfig(protocol='kfold', fold=0, arch='x86', seed=0, min_df=1000), sizes, out, 'ours', True)
    m = json.loads((out / 'kfold0_x86_s0_imp' / 'metrics.json').read_text(encoding='utf-8'))
    assert m['imports']['features_per_method']['imports_header_lr'] == 0
    assert m['methods']['imports_header_lr']['test']['auc'] == 0.5 and 'constant 0.5' in m['imports']['note']


def test_default_baseline_run_is_unchanged(tiny):  # noqa: F811
    sizes = {k: len(v) for k, v in json.loads((tiny.root / 'vocab.json').read_text(encoding='utf-8')).items()}
    out = tiny.runs / 'plain'
    baselines.run(RunConfig(protocol='kfold', fold=1, arch='x86', seed=0), sizes, out)
    m = json.loads((out / 'kfold1_x86_s0' / 'metrics.json').read_text(encoding='utf-8'))
    assert list(m['methods']) == ['tfidf_sig_api_lr', 'structural_lr'] and 'vocab' not in m and 'imports' not in m


def test_stage_tables_print_missing_arms(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    gates.pooled.cache_clear()
    for stage, arms, pairs in ((gates.vocab, gates.VOC_ARMS, gates.VOC_PAIRS), (gates.imports, gates.IMP_ARMS, gates.IMP_PAIRS)):
        text = '\n'.join(stage(20))
        assert text.startswith('# Gate summary, stage')
        for arm, group in arms:
            assert f'| {arm} | {group} | missing' in text
        for a, b in pairs:
            assert f'| {a} | {b} | missing | missing |' in text
    gates.pooled.cache_clear()


def test_gate_groups_follow_the_run_configs():
    base = dict(model='gnn', protocol='kfold', fold=0, mask_blocks=0.15)
    arms = dict(gates.VOC_ARMS)
    assert arms['R2 no-edges, ours'] == RunConfig(**base, control='no_edges').group
    assert arms['R2 no-edges, yanping'] == RunConfig(**base, control='no_edges', vocab='yanping').group
    assert arms['R2 no-edges, hybrid'] == RunConfig(**base, control='no_edges', vocab='hybrid').group
    arms = dict(gates.IMP_ARMS)
    assert arms['R7 typed'] == RunConfig(**base).group
    assert arms['R7 + header imports'] == RunConfig(**base, imports='header').group
    pairs = set(gates.IMP_PAIRS)
    assert {('R7 typed', 'R7 + header imports'), ('header imports, LR', 'R7 + header imports'),
            ('TF-IDF', 'R7 + header imports')} <= pairs
    assert all(stats.baseline_method(g) for g in ('tfidf-yanping', 'tfidf-hybrid', 'imports-header'))


# ---- the queue stages ---------------------------------------------------------------------------------------------

def queue(*args, **env):
    r = subprocess.run([BASH, 'queue_gnn.sh', *args], cwd=ROOT, capture_output=True, text=True, env={**os.environ, **env})
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()


@needs_bash
def test_vocab_stage_dry_names_match_run_configs():
    lines = queue('--stage', 'vocab', '--dry')
    runs = [l.split()[1:] for l in lines if l.startswith('run ')]
    assert len(runs) == 10
    for name, *argv in runs:
        cfg, _ = parse_config(argv)
        assert cfg.name == name, (name, cfg.name)
        assert (cfg.protocol, cfg.arch, cfg.seed, cfg.model, cfg.control, cfg.mask_blocks) == ('kfold', 'x86', 0, 'gnn', 'no_edges', 0.15)
    arms = [parse_config(a)[0] for _, *a in runs]
    assert [c.vocab for c in arms] == ['yanping'] * 5 + ['hybrid'] * 5 and [c.fold for c in arms] == list(range(5)) * 2
    assert lines[:2] == ['cpu baselines --protocol kfold --fold 0 1 2 3 4 --arch x86 --seed 0 --vocab yanping',
                         'cpu baselines --protocol kfold --fold 0 1 2 3 4 --arch x86 --seed 0 --vocab hybrid']
    assert lines[-3:] == ['cpu report.py', 'cpu gates.py --stage vocab > RUNS/gates_vocab.md', 'cpu val_threshold.py > RUNS/val_threshold.md']
    assert {c.group for c in arms} == {dict(gates.VOC_ARMS)[k] for k in ('R2 no-edges, yanping', 'R2 no-edges, hybrid')}


@needs_bash
def test_imports_stage_dry_names_match_run_configs():
    lines = queue('--stage', 'imports', '--dry')
    assert lines == ['cpu baselines --protocol kfold --fold 0 1 2 3 4 --arch x86 --seed 0 --imports', 'cpu report.py',
                     'cpu gates.py --stage imports > RUNS/gates_imports.md', 'cpu val_threshold.py > RUNS/val_threshold.md']


@needs_bash
def test_imports_header_stage_dry_names_match_run_configs():
    lines = queue('--stage', 'imports_header', '--dry')
    runs = [l.split()[1:] for l in lines if l.startswith('run ')]
    assert len(runs) == 5
    for name, *argv in runs:
        cfg, _ = parse_config(argv)
        assert cfg.name == name, (name, cfg.name)
        assert (cfg.imports, cfg.mask_blocks, cfg.control, cfg.vocab, cfg.protocol, cfg.arch, cfg.seed, cfg.model) == (
            'header', 0.15, '', 'ours', 'kfold', 'x86', 0, 'gnn')
    assert [parse_config(a)[0].fold for _, *a in runs] == [0, 1, 2, 3, 4]
    assert not any('baselines' in l or 'verdict' in l for l in lines)                    # no baselines here
    assert lines[-3:] == ['cpu report.py', 'cpu gates.py --stage imports > RUNS/gates_imports_header.md',
                          'cpu val_threshold.py > RUNS/val_threshold.md']
    assert parse_config(runs[0][1:])[0].group == dict(gates.IMP_ARMS)['R7 + header imports']


@needs_bash
def test_imports_header_stage_runs_five_arms_and_the_imports_gates(tmp_path):
    calls, out = run_stage(tmp_path, 'imports_header', 3, 'GATE imports: FAIL. stub')       # a failing verdict does not matter
    trains = [c for c in calls if c.startswith('train.py')]
    assert len(trains) == 5 and all('--imports header' in c and '--mask-blocks 0.15' in c for c in trains)
    assert not any('verdict' in c or c.startswith('baselines.py') for c in calls)
    assert out[-1].startswith('queue done') and 'gates.py --stage imports' in calls[-2] and 'imports_header' not in calls[-2]
    assert (tmp_path / 'runs' / 'gates_imports_header.md').exists()


STUB = """#!/bin/bash
echo "$*" >> "$GT_RUNS/calls.txt"
case " $* " in *" gates.py --stage imports --verdict "*) echo "$VERDICT"; exit $VERDICT_RC;; esac
exit 0
"""


def run_stage(tmp_path, stage, rc, verdict='GATE imports: stub'):
    runs = tmp_path / 'runs'
    (runs / 'logs').mkdir(parents=True)
    stub = tmp_path / 'stub.sh'
    stub.write_text(STUB, newline='\n')
    stub.chmod(0o755)
    env = {**os.environ, 'GT_PY': stub.as_posix(), 'GT_RUNS': runs.as_posix(), 'GT_TRAIN_PATTERN': 'gtqueue_never_matches_train[.]py',
           'VERDICT': verdict, 'VERDICT_RC': str(rc)}
    r = subprocess.run([BASH, 'queue_gnn.sh', '--stage', stage], cwd=ROOT, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr + r.stdout
    return (runs / 'calls.txt').read_text().splitlines(), r.stdout.splitlines()


@needs_bash
def test_imports_stage_runs_the_baseline_and_no_arm(tmp_path):
    calls, out = run_stage(tmp_path, 'imports', 0)
    assert any(c.startswith('baselines.py') and c.endswith('--imports') for c in calls)
    assert not [c for c in calls if c.startswith('train.py') or 'verdict' in c]
    assert out[-1].startswith('queue done') and 'report.py' in calls[-3] and 'gates.py --stage imports' in calls[-2]


@needs_bash
def test_vocab_stage_runs_baselines_then_ten_arms(tmp_path):
    calls, out = run_stage(tmp_path, 'vocab', 0)
    assert [c.split()[0] for c in calls[:3]] == ['-c', 'baselines.py', 'baselines.py']
    assert calls[1].endswith('--vocab yanping') and calls[2].endswith('--vocab hybrid')
    trains = [c for c in calls if c.startswith('train.py')]
    assert len(trains) == 10 and all('--control no_edges' in c and '--mask-blocks 0.15' in c for c in trains)
    assert sum('--vocab yanping' in c for c in trains) == sum('--vocab hybrid' in c for c in trains) == 5


@needs_bash
def test_queue_usage_lists_the_new_stages():
    r = subprocess.run([BASH, 'queue_gnn.sh', '--stage', 'v4', '--dry'], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 2 and 'day1|day1b|v3|subgraph|paths|vocab|imports' in r.stdout
