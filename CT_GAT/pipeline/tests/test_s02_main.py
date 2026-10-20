"""`preprocess_full.main` end to end on tiny collection folders (see s02_collection.py)."""
import json
import sys
from pathlib import Path

import pytest
import torch

import s02_collection as col

STAGE = Path(__file__).resolve().parents[1] / 's02_preprocess'
sys.path.insert(0, str(STAGE))
import preprocess_full as pf  # noqa: E402
import tokenize_asm_gz as tok  # noqa: E402


def sha(n):
    return f'{n:064x}'


def run(tmp_path, roots, *extra, out='out', vocab='vocab.json', seed=0):
    """Run `main` on the roots; returns (output folder, vocabulary file)."""
    out_dir, vocab_path = tmp_path / out, tmp_path / vocab
    argv = [arg for root in roots for arg in ('--root', str(root))]
    argv += ['--num-blocks', '8', '--seed', str(seed), '--output', str(out_dir), '--vocab', str(vocab_path), *extra]
    pf.main(argv)
    return out_dir, vocab_path


def api_column(out_dir, sample_id):
    payload = torch.load(out_dir / f'{sample_id}.pt', weights_only=False)
    return payload['instruction_ids'][:, 3].tolist()


# ---- 1. the vocabulary is fitted on training rows only -----------------------------------------------------------

def test_a_test_only_api_gets_unk_and_a_train_api_keeps_its_id(tmp_path):
    # root a holds only a test sample and is listed first; root b holds the training sample
    a = col.write_collection(tmp_path / 'a', [(sha(1), 'test', 1, ['CreateFileW', 'RegSetValueExW'])])
    b = col.write_collection(tmp_path / 'b', [(sha(2), 'train', 0, ['CreateFileW'])])
    out_dir, vocab_path = run(tmp_path, [a, b])
    assert json.loads(vocab_path.read_text(encoding='utf-8')) == ['CreateFileW']
    assert tok.FIRST_API_ID in api_column(out_dir, sha(2))
    test_apis = [value for value in api_column(out_dir, sha(1)) if value != tok.NO_API]
    assert sorted(test_apis) == [tok.UNK_API, tok.FIRST_API_ID]


def test_split_all_and_split_train_build_the_same_vocabulary(tmp_path):
    a = col.write_collection(tmp_path / 'a', [(sha(1), 'test', 1, ['RegSetValueExW']), (sha(3), 'train', 1, ['CloseHandle'])])
    _, vocab_all = run(tmp_path, [a], out='all', vocab='all.json')
    _, vocab_train = run(tmp_path, [a], '--split', 'train', out='train', vocab='train.json')
    assert vocab_all.read_text(encoding='utf-8') == vocab_train.read_text(encoding='utf-8')
    assert json.loads(vocab_all.read_text(encoding='utf-8')) == ['CloseHandle']


# ---- 2. the embedder is seeded -----------------------------------------------------------------------------------

def three_samples(tmp_path, name):
    return col.write_collection(tmp_path / name, [
        (sha(1), 'train', 1, ['CreateFileW']),
        (sha(2), 'train', 0, ['CreateFileW', 'RegSetValueExW']),
        (sha(3), 'test', 1, ['CloseHandle']),
    ])


def block_embeddings(out_dir):
    return {sample_id: torch.load(out_dir / f'{sample_id}.pt', weights_only=False)['block_embeddings']
            for sample_id in (sha(1), sha(2), sha(3))}


def test_two_fresh_runs_give_identical_block_embeddings(tmp_path):
    root = three_samples(tmp_path, 'c')
    first, _ = run(tmp_path, [root], out='first', vocab='v1.json')
    torch.manual_seed(999)                       # whatever the process did in between
    second, _ = run(tmp_path, [root], out='second', vocab='v2.json')
    a, b = block_embeddings(first), block_embeddings(second)
    for sample_id in a:
        assert a[sample_id].numel() and torch.equal(a[sample_id], b[sample_id])
    other, _ = run(tmp_path, [root], out='other', vocab='v3.json', seed=1)
    assert not torch.equal(block_embeddings(other)[sha(1)], a[sha(1)])


def test_new_api_rows_depend_on_the_seed_and_the_row_only():
    one = pf.FiveChannelEmbedder(tok.FIRST_API_ID, seed=7)
    one.expand_api(tok.FIRST_API_ID + 2)
    one.expand_api(tok.FIRST_API_ID + 5)
    torch.manual_seed(123)
    two = pf.FiveChannelEmbedder(tok.FIRST_API_ID, seed=7)
    two.expand_api(tok.FIRST_API_ID + 5)
    assert torch.equal(one.api_embedding.weight[tok.FIRST_API_ID:], two.api_embedding.weight[tok.FIRST_API_ID:])
    assert float(one.api_embedding.weight[0].detach().abs().sum()) == 0.0
    three = pf.FiveChannelEmbedder(tok.FIRST_API_ID, seed=8)
    three.expand_api(tok.FIRST_API_ID + 5)
    assert not torch.equal(one.api_embedding.weight[tok.FIRST_API_ID:], three.api_embedding.weight[tok.FIRST_API_ID:])
