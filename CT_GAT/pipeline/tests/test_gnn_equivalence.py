"""The data path and one training step of the GNN equal the 3-block-level-counting branch's (3764cc7).

fixtures/gnn_equivalence.npz holds, for ten configurations on the tiny cache (tests/tiny_cache.py), the batches that
`batching.loader` yields (three in training mode, two in evaluation mode) and the logits, loss and every parameter gradient of one
forward/backward step of `HierGNN`, all computed by the branch's code (fixtures/make_gnn_equivalence.py). Integer tensors must
be equal; floating-point tensors equal to 1e-6 relative (they are equal bit for bit on the machine that recorded them). The
real-data counterpart, two epochs of R7 on the GPU, is test_train_equivalence.py.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from common.config import RunConfig
from common.manifests import load_index, make_manifest
from gnn_equivalence_cases import CASES, compute
from s02_preprocess.tokenizer.vocab import TokenVocab
from s03b_cfg_feature.gnn.batching import loader
from s03b_cfg_feature.gnn.model import HierGNN
from s03b_cfg_feature.gnn.s10_unused_import_features.imports_feat import ImportVocab
from tiny_cache import tiny  # noqa: F401  (a fixture)

FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'gnn_equivalence.npz'
API = SimpleNamespace(RunConfig=RunConfig, load_index=load_index, make_manifest=make_manifest, TokenVocab=TokenVocab,
                      ImportVocab=ImportVocab, loader=loader, HierGNN=HierGNN)


@pytest.fixture(scope='module')
def recorded():
    with np.load(FIXTURE, allow_pickle=False) as z:
        return {k: z[k] for k in z.files if k != 'meta/torch'}


@pytest.fixture(scope='module')
def computed(tiny):  # noqa: F811
    return compute(API, tiny.root)


def same(got, want, key):
    assert got.shape == want.shape and got.dtype == want.dtype, (key, got.shape, want.shape, got.dtype, want.dtype)
    if got.dtype.kind == 'f':
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-7, err_msg=key)
    else:
        assert np.array_equal(got, want), key


def test_the_fixture_covers_every_case_and_this_code_computes_the_same_keys(recorded, computed):
    assert {k.split('/')[0] for k in recorded} == set(CASES)
    assert set(computed) == set(recorded)


@pytest.mark.parametrize('case', CASES)
def test_batches_equal_the_branchs(recorded, computed, case):
    keys = [k for k in recorded if k.startswith(f'{case}/') and k.split('/')[1].startswith(('train', 'eval', 'ids', 'vocab'))]
    assert len(keys) > 40
    for key in keys:
        same(computed[key], recorded[key], key)


@pytest.mark.parametrize('case', CASES)
def test_forward_and_backward_step_equal_the_branchs(recorded, computed, case):
    keys = [k for k in recorded if k.startswith((f'{case}/logits', f'{case}/loss', f'{case}/grad/'))]
    assert f'{case}/logits' in keys and any('/grad/' in k for k in keys)
    for key in keys:
        same(computed[key], recorded[key], key)
