"""The names and the sample file that other stages (the token-sequence model) read from `preprocess_full.py` and
`tokenize_asm_gz.py` stay as they were on main a53234a. New keys may be added; none may go or change meaning."""
import sys
from pathlib import Path

import torch

import s02_collection as col

STAGE = Path(__file__).resolve().parents[1] / 's02_preprocess'
sys.path.insert(0, str(STAGE))
import preprocess_full as pf  # noqa: E402
import tokenize_asm_gz as tok  # noqa: E402

# the keys of a `<id>.pt` on main a53234a (save_sample)
PT_KEYS = {
    'graph_format_version', 'file_id', 'label', 'dataset_metadata', 'random_seed', 'instruction_ids', 'insn_to_block',
    'instruction_matched', 'instruction_rvas', 'block_features', 'block_embeddings', 'block_embedding_dim', 'edge_index',
    'edge_type', 'edge_attr', 'edge_type_names', 'edge_back', 'edge_back_addr', 'block_types', 'block_external',
    'block_keys', 'model1_block_keys', 'model2_block_keys', 'model3_block_keys',
}


def test_public_names_and_values():
    assert pf.OPCODE_VOCAB_SIZE == 256
    assert pf.OPERAND_VOCAB_SIZE == tok.FIRST_API_ID == 76
    assert tok.UNK_API == 75 and tok.NO_API == 0
    assert callable(tok.load_api_vocab) and callable(tok.save_api_vocab) and callable(tok.collect_ida_imports)
    assert isinstance(tok.API_VOCAB_PATH, Path) and tok.API_VOCAB_PATH.name == 'api_vocab.json'
    assert isinstance(tok.PROJECT_ROOT, Path)
    for name in ('load_cfg', 'process_sample', 'save_sample', 'sample_connected_subgraph', 'sample_signal_subgraph',
                 'align_sampled_blocks', 'tokenize_selected_blocks', 'FiveChannelEmbedder', 'api_table_size'):
        assert hasattr(pf, name)


def test_a_sample_file_keeps_the_old_keys_and_shapes(tmp_path):
    root = col.write_collection(tmp_path / 'c', [('1' * 64, 'train', 1, ['CreateFileW', 'CloseHandle'])])
    assert pf.main(['--root', str(root), '--num-blocks', '8', '--output', str(tmp_path / 'out'),
                    '--vocab', str(tmp_path / 'vocab.json')]) == 0
    payload = torch.load(tmp_path / 'out' / f'{"1" * 64}.pt', weights_only=False)
    assert PT_KEYS <= set(payload)
    ids = payload['instruction_ids']
    assert ids.dim() == 2 and ids.shape[1] == 5 and ids.dtype == torch.long
    assert payload['label'] == 1 and isinstance(payload['label'], int)
    assert payload['insn_to_block'].shape == (ids.shape[0],)
    assert payload['instruction_matched'].shape == (ids.shape[0],)
    assert payload['block_embeddings'].shape == (len(payload['block_keys']), payload['block_embedding_dim'])
    assert payload['block_features'].shape[0] == len(payload['block_keys'])
    assert payload['block_features'].shape[1] == len(pf.HANDCRAFTED_FEATURE_NAMES)
    assert payload['edge_index'].shape[0] == 2
