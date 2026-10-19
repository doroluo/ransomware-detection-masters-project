"""python -m pytest tests/test_codebert_encoder.py: step 17, the CodeBERT block encoder on the strings of the block export.

Needs the reference export RUNS/export/ours_cap0 (the text and regex tests) and, for the tests that run the model, the
local CodeBERT files under HF_HOME (the tests set it to C:/Users/chaoa/Downloads/hf_home when it is unset); each skips
cleanly without them. The model runs on CUDA when there is one, else on the CPU. The queue and gates tests need nothing.
"""
import os
import re
import subprocess
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ.setdefault('HF_HOME', 'C:/Users/chaoa/Downloads/hf_home')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

from common.config import RUNS, RunConfig
from evaluation import gates, stats
from training.train import parse_config
from transformer_paths import codebert_encoder as cb
from transformer_paths import encoder_contract as ec
from transformer_paths.reader import BlockStore, FoldView

ROOT = Path(__file__).resolve().parents[1]
EXPORT = RUNS / 'export' / 'ours_cap0'
BASH = shutil.which('bash')
needs_export = pytest.mark.skipif(not (EXPORT / 'store' / 'store.json').exists(), reason='needs RUNS/export/ours_cap0')
try:
    cb.weights_file(cb.MODEL_NAME)
    HAVE_WEIGHTS = True
except (FileNotFoundError, RuntimeError):
    HAVE_WEIGHTS = False
needs_model = pytest.mark.skipif(not HAVE_WEIGHTS, reason='needs the local microsoft/codebert-base files under HF_HOME')
needs_bash = pytest.mark.skipif(BASH is None, reason='needs bash')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
REGISTER = re.compile(r'\b(e?[abcd]x|[re]?[sb]p|[re]?[sd]i|r\d+[dwb]?|[abcd][lh]|xmm\d+)\b')
ADDRESS = re.compile(r'0x[0-9a-f]+')


@pytest.fixture(scope='module')
def store():
    return BlockStore(EXPORT)


@pytest.fixture(scope='module')
def pieces(store):
    return cb.fold_pieces(store, 0)


@pytest.fixture(scope='module')
def encoder():
    return cb.CodeBertBlockEncoder(**cb.init_kwargs_for(EXPORT, 0))


def fold_id(store, kind, text, fold=0):
    """Fold id of a signature or API string of the cache vocabulary."""
    import json
    strings = json.loads((Path(store.meta['cache']) / 'vocab.json').read_text(encoding='utf-8'))[kind]
    return int(store.fold(fold)['lut'][store.fold(fold)['vocab']['global_offsets'][kind] + strings.index(text)])


# the text rule -------------------------------------------------------------------------------------------------------


@needs_export
def test_text_of_a_hand_made_block(store, pieces):
    vocab = store.fold(0)['vocab']
    xor, mov = fold_id(store, 'sig', 'xor r,r'), fold_id(store, 'sig', 'mov r,m')
    api = next(i for i in range(vocab['kinds']['api']['offset'] + 1, vocab['n_ids']) if pieces[i])
    mn_ids = [vocab['kinds']['mn']['offset'] + 1, vocab['kinds']['mn']['offset'] + 2]
    beh = [vocab['kinds']['beh']['offset'] + 1, vocab['kinds']['beh']['offset'] + 2]
    sig_unk, api_unk = vocab['kinds']['sig']['unk'], vocab['kinds']['api']['unk']
    # layout of the export: [mn_1 mn_2 mn_3 | sig_1 sig_2 sig_3 | beh_1 beh_2 beh_3 | api_1 api_2]
    ids = mn_ids + [vocab['kinds']['mn']['unk']] + [xor, sig_unk, mov] + beh + [0] + [api, api_unk]
    expected = ' ; '.join(['xor r,r', '<unk>', 'mov r,m', pieces[api], '<unk>'])
    assert cb.block_text(pieces, ids) == expected
    assert 'xor r,r ; <unk> ; mov r,m ; ' in expected and expected.endswith(' ; <unk>')
    # PAD is dropped, the specials CLS / SEP / MASK are the placeholder, mn and beh never appear
    assert cb.block_text(pieces, [0, 0, xor, 0]) == 'xor r,r'
    assert cb.block_text(pieces, [1, xor, 2, 3]) == '<unk> ; xor r,r ; <unk> ; <unk>'
    assert cb.block_text(pieces, mn_ids + beh) == ''
    assert cb.block_text(pieces, []) == ''


@needs_export
def test_pieces_cover_exactly_the_kept_signature_and_api_ids(store, pieces):
    vocab = store.fold(0)['vocab']
    assert len(pieces) == vocab['n_ids'] and pieces[0] == ''
    assert [pieces[i] for i in (1, 2, 3)] == ['<unk>'] * 3
    for kind in ('mn', 'beh'):
        k = vocab['kinds'][kind]
        assert not any(pieces[k['offset']:k['offset'] + k['size']])
    for kind in ('sig', 'api'):
        k = vocab['kinds'][kind]
        chunk = pieces[k['offset']:k['offset'] + k['size']]
        assert chunk[0] == '<unk>' and all(chunk) and len(set(chunk[1:])) == k['size'] - 1


@needs_export
def test_generated_text_never_has_a_register_name_or_an_address(store, pieces):
    view = FoldView(store, 0, 'train')
    blocks, _ = view.distinct_blocks()
    pick = blocks[np.random.default_rng(0).choice(len(blocks), 2000, replace=False)]
    texts = [cb.block_text(pieces, view.sequence(int(b))) for b in pick]
    assert sum(bool(t) for t in texts) > 1500 and sum(len(t) for t in texts) > 100_000
    bad = [t for t in texts if REGISTER.search(t) or ADDRESS.search(t)]
    assert not bad, bad[:3]


# the model -----------------------------------------------------------------------------------------------------------


@needs_export
@needs_model
def test_identical_token_sequences_give_identical_vectors(store, encoder):
    view = FoldView(store, 0, 'train')
    blocks, _ = view.distinct_blocks()
    tokens, mask = view.padded(blocks[[5, 50_000, 300_000]], 192)
    # the same sequences again, in another order and with extra padding on the right
    order = [2, 0, 1, 0]
    padded = torch.cat([tokens, torch.zeros(3, 7, dtype=tokens.dtype)], 1)
    padded_mask = torch.cat([mask, torch.zeros(3, 7, dtype=torch.bool)], 1)
    a = encoder(tokens.to(DEVICE), mask.to(DEVICE))
    b = encoder(padded[order].to(DEVICE), padded_mask[order].to(DEVICE))
    assert a.shape == (3, 768) and a.dtype == torch.float32 and torch.isfinite(a).all()
    assert torch.equal(b[1], b[3])      # the same sequence twice in one batch: bit-identical
    # another batch shape: equal up to the fp16 noise of the GPU kernels (about 3e-4 on values of order 1)
    assert torch.allclose(b[1], a[0], rtol=0, atol=5e-3) and torch.allclose(b[0], a[2], rtol=0, atol=5e-3)
    assert torch.equal(a, encoder(tokens.to(DEVICE), mask.to(DEVICE)))
    assert not torch.allclose(a[0], a[1], atol=1e-3)
    empty = encoder(torch.zeros(1, 1, dtype=torch.long, device=DEVICE), torch.zeros(1, 1, dtype=torch.bool, device=DEVICE))
    assert empty.shape == (1, 768) and torch.isfinite(empty).all()


@needs_export
@needs_model
def test_the_codebert_weights_stay_out_of_the_state_dict(encoder):
    assert encoder.state_dict() == {} and list(encoder.parameters()) == []
    kwargs = cb.init_kwargs_for(EXPORT, 0)
    assert kwargs['model_name'] == 'microsoft/codebert-base' and len(kwargs['weights_sha256']) == 64
    assert kwargs['revision'] and kwargs['max_subword_len'] == 256 and kwargs['layer'] == -1 and kwargs['pooling'] == 'mean'
    with pytest.raises(ValueError, match='sha256'):
        cb.CodeBertBlockEncoder(**{**kwargs, 'weights_sha256': '0' * 64})


@needs_export
@needs_model
def test_checkpoint_is_small_and_passes_the_contract(tmp_path):
    path = cb.write_checkpoints(EXPORT, 'codebert_test', folds=[0], out_root=tmp_path)[0]
    assert path.stat().st_size < 10_000_000
    ckpt = ec.load_checkpoint(path)
    assert ckpt['state_dict'] == {} and ckpt['module'] == 'transformer_paths.codebert_encoder'
    assert (ckpt['d_out'], ckpt['max_len'], ckpt['fold']) == (768, 192, 0)
    passed = ec.check_checkpoint(path, EXPORT, 0)
    assert set(passed.values()) == {'ok'} and 'batch_independent' in passed and 'padding_independent' in passed


# the queue stage and the gates --------------------------------------------------------------------------------------


def queue(*args, **env):
    r = subprocess.run([BASH, 'queue_gnn.sh', *args], cwd=ROOT, capture_output=True, text=True, env={**os.environ, **env})
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()


@needs_bash
def test_queue_paths_codebert_dry_names_match_run_configs():
    lines = queue('--stage', 'paths_codebert', '--dry')
    runs = [l.split()[1:] for l in lines if l.startswith('run ')]
    assert len(runs) == 30
    seen = set()
    for name, *argv in runs:
        cfg, _ = parse_config(argv)
        assert cfg.name == name, (name, cfg.name)
        assert (cfg.protocol, cfg.arch, cfg.model, cfg.mask_blocks) == ('kfold', 'x86', 'gnn', 0.15)
        assert (cfg.block_input, cfg.block_vec, cfg.val_seed) == ('vec', 'ours_cap0_codebert', None)
        seen.add((cfg.control, cfg.seed, cfg.fold))
    assert seen == {(c, s, f) for c in ('no_edges', '') for s in (0, 1, 2) for f in range(5)}
    assert lines[-3:] == ['cpu report.py', 'cpu gates.py --stage paths_codebert > RUNS/gates_paths_codebert.md',
                          'cpu val_threshold.py > RUNS/val_threshold.md']
    assert [l.split()[1] for l in lines if l.startswith('cpu transformer_paths')] == [
        'transformer_paths.codebert_encoder', 'transformer_paths.precompute_blockvec']
    groups = {parse_config(a)[0].group for _, *a in runs}
    arms = dict(gates.CB_ARMS)
    assert groups == {arms['path 1: no-edges, CodeBERT vectors'], arms['path 2: typed, CodeBERT vectors']}
    # the token arms the gates pair them with are the existing R2 and R7 groups
    assert arms['R2 no-edges, tokens'] == RunConfig(model='gnn', protocol='kfold', fold=0, mask_blocks=0.15,
                                                    control='no_edges').group
    assert arms['R7 typed, tokens'] == RunConfig(model='gnn', protocol='kfold', fold=0, mask_blocks=0.15).group


@needs_bash
def test_queue_paths_codebert_budget_guard():
    lines = queue('--stage', 'paths_codebert', '--dry', GPU_BUDGET_MIN='290', DRY_RUN_MIN='100')
    assert sum(l.startswith('run ') for l in lines) == 3 and sum(l.startswith('skipped (budget) ') for l in lines) == 27


def test_gates_paths_codebert_prints_missing_arms(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    gates.pooled.cache_clear()
    text = '\n'.join(gates.paths_codebert(50))
    assert text.startswith('# Gate summary, stage paths_codebert')
    for arm, group in gates.CB_ARMS:
        assert f'| {arm} | {group} | missing' in text
    for a, b in gates.CB_PAIRS:
        assert f'| {a} | {b} | missing | missing |' in text
    assert gates.CB_PAIRS[0] == ('path 1: no-edges, CodeBERT vectors', 'path 2: typed, CodeBERT vectors')
    assert 'gnn-mblk15-bv-ours_cap0_codebert' in text and 'gnn-noedges-mblk15-bv-ours_cap0_codebert' in text
    gates.pooled.cache_clear()
