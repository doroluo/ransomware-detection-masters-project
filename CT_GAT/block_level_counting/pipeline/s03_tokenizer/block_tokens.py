"""Run-id tokens of a block selection: [mn | sig | beh | api] per block (the kinds of `vocab.insn_kinds` and then api for
another vocabulary), with each token's kind."""
import numpy as np

from pipeline.s03_tokenizer.vocab import insn_values

OURS_INSN_KINDS = ('mn', 'sig', 'beh')


def ranges(starts, lengths):
    """Concatenation of arange(s, s + n) for each (s, n)."""
    lengths = np.asarray(lengths, dtype=np.int64)
    total = int(lengths.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    first = np.cumsum(lengths) - lengths
    return np.repeat(np.asarray(starts, dtype=np.int64) - first, lengths) + np.arange(total, dtype=np.int64)


def token_table(z, vocab, blk):
    """Run-id tokens of the blocks in the order [insn kind 0 | kind 1 | ... | api] within each block, with each token's
    kind (index into `vocab.kinds`). For the default vocabulary the kinds are mn, sig, beh, api; a vocabulary object
    without `insn_kinds` (the block export's identity vocabulary) is that one. `z['ytok']` serves the reader kinds."""
    kinds = getattr(vocab, 'insn_kinds', OURS_INSN_KINDS)
    width = len(kinds)
    insn_ptr, api_ptr = z['blk_insn_ptr'].astype(np.int64), z['blk_api_ptr'].astype(np.int64)
    n_ins, n_apis = insn_ptr[blk + 1] - insn_ptr[blk], api_ptr[blk + 1] - api_ptr[blk]
    ins, apis = ranges(insn_ptr[blk], n_ins), ranges(api_ptr[blk], n_apis)
    per_block = width * n_ins + n_apis
    tok_offsets = np.cumsum(per_block) - per_block
    ins_base = np.repeat(tok_offsets, n_ins) + np.arange(len(ins)) - np.repeat(np.cumsum(n_ins) - n_ins, n_ins)
    ins_len = np.repeat(n_ins, n_ins)
    api_pos = (np.repeat(tok_offsets + width * n_ins, n_apis) + np.arange(len(apis))
               - np.repeat(np.cumsum(n_apis) - n_apis, n_apis))
    tokens = np.empty(int(per_block.sum()), dtype=np.int64)
    kind = np.empty(len(tokens), dtype=np.int64)
    ins_pos = [ins_base + j * ins_len for j in range(width)]
    for j, k in enumerate(kinds):
        tokens[ins_pos[j]] = vocab.lut[k][insn_values(z, k)[ins]] + vocab.offset[k]
        kind[ins_pos[j]] = j
    tokens[api_pos] = vocab.lut['api'][z['api'][apis]] + vocab.offset['api']
    kind[api_pos] = width
    return tokens, kind, tok_offsets, per_block, ins_pos, api_pos
