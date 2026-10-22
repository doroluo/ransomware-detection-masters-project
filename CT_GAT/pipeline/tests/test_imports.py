"""The header-import feature of the GNN head (`--imports header`, run-name suffix `-imph`) on the synthetic tiny cache: the
vocabulary fitted on the training binaries, one binary's bag and log counts, the batch, the network with and without the bag.

Ported from the 3-block-level-counting branch (tests/test_reader_wiring.py, the header-import part). The converter of the reader
output to `side/imports` is tested in test_import_side.py.
"""
import numpy as np
import pytest
import torch

from common.config import RunConfig
from s02_preprocess.tokenizer.vocab import TokenVocab
from reference.gnn import spec as gnn_spec
from reference.gnn.batching import BinaryDataset, loader
from reference.gnn.model import HierGNN
from s04_evaluation import imports_feat
from s04_evaluation.imports_feat import IMPORT_DIM, ImportVocab
from tiny_cache import tiny  # noqa: F401  (a fixture)


def gnn(**kw):
    return RunConfig(**{'model': 'gnn', 'num_workers': 0, **kw})


def fold(f=0, **kw):
    return gnn(protocol='kfold', fold=f, **kw)


def train_rows(tiny, n=30):  # noqa: F811
    return tiny.rows[:n]


def test_the_imports_option_names_the_run_and_the_default_has_none():
    assert fold(0, mask_blocks=0.15, imports='header').name == 'kfold0_x86_gnn-mblk15-imph_s0'
    assert fold(0, mask_blocks=0.15, imports='header').group == 'kfold_x86_gnn-mblk15-imph'
    assert RunConfig().imports == '' and fold(0).name == 'kfold0_x86_gnn_s0'


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


def test_a_missing_side_array_names_the_sample_and_the_command(tiny):  # noqa: F811
    with pytest.raises(FileNotFoundError, match='s04_evaluation.import_side'):
        imports_feat.load_imports(tiny.root.parent / 'elsewhere', tiny.rows[0]['sample_id'])


def test_dataset_items_and_default_runs_carry_nothing_extra(tiny):  # noqa: F811
    rows = tiny.rows[:6]
    cfg = fold(0)
    plain = BinaryDataset(rows, TokenVocab(train_rows(tiny), cfg), cfg)[(1, None)]
    assert not {'imp_idx', 'imp_w', 'imp_feat', 'ytok'} & set(plain)
    cfg = fold(0, imports='header')
    imports = ImportVocab(train_rows(tiny), cfg.min_df)
    item = BinaryDataset(rows, TokenVocab(train_rows(tiny), cfg), cfg, imports=imports)[(1, None)]
    assert {'imp_idx', 'imp_w', 'imp_feat'} <= set(item)
    assert item['imp_feat'].tolist() == imports.item(rows[1]['sample_id'])['imp_feat'].tolist()


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


@pytest.mark.parametrize('kw', [dict(imports='header'), dict(imports='header', mask_blocks=0.3, mask_tokens=0.2),
                                dict(control='no_edges')])
def test_forward_and_backward_on_cpu(tiny, kw):  # noqa: F811
    cfg = fold(0, batch_size=8, **kw)
    v, imports, batch = make_batch(tiny, cfg, tiny.rows[10:18])
    model = gnn_spec.build_model(cfg, v, None, len(imports) if imports is not None else 0)
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


def test_the_spec_fits_the_import_vocabulary_on_the_training_manifest_and_writes_it(tiny, tmp_path):  # noqa: F811
    """`--imports header` reads the import side arrays of the run's cache. (The training script of the branch passed the option
    itself, 'header', where ImportVocab takes the cache directory, so `--imports header` stopped on a missing 'header/side/imports'
    directory since step 25a; the spec calls ImportVocab(train, min_df) as the baselines do.)"""
    from common.manifests import load_index, make_manifest
    from common.registry import RunInfo, spec_for
    cfg = fold(0, imports='header')
    manifest = make_manifest(cfg, load_index(tiny.root))
    spec = spec_for(cfg)
    model = spec.build(cfg, RunInfo(manifest, tmp_path))
    want = ImportVocab(manifest.train, cfg.min_df)
    assert model.imports.bag.weight.shape == (len(want), IMPORT_DIM) and spec.imports.names == want.names
    assert (tmp_path / 'imports_vocab.json').is_file() and (tmp_path / 'vocab.json').is_file()
    batch = next(iter(spec.loader(cfg, manifest.val[:4], False)))
    assert spec.forward(model, batch).shape == (len(batch['sample_id']),)


def test_the_imports_baseline_runs_through_the_lazy_imports_of_the_evaluation_stage(tiny, tmp_path, monkeypatch):  # noqa: F811
    """`s04_evaluation/baselines.py --imports` imports ImportVocab and load_imports from this stage when the option is on."""
    import json
    from s04_evaluation import baselines
    monkeypatch.setattr(baselines, 'CACHE', tiny.root)
    sizes = {k: len(v) for k, v in json.loads((tiny.root / 'vocab.json').read_text(encoding='utf-8')).items()}
    cfg = RunConfig(protocol='kfold', fold=0, arch='x86', seed=0, min_df=2)
    baselines.run(cfg, sizes, tmp_path, 'ours', True)
    (written,) = tmp_path.rglob('metrics.json')
    doc = json.loads(written.read_text(encoding='utf-8'))
    assert written.parent.name == 'kfold0_x86_s0_imp' and list(doc['methods']) == ['imports_header_lr']
    assert doc['imports']['features_per_method']['imports_header_lr'] > 0
