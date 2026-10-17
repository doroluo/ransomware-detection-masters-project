"""Converter of the team's operand-role reader output to a cache-aligned side array.

python -m pipeline.s03_tokenizer.import_reader_side --cache C:/.../graph_transformer_cache_v2 [C:/.../graph_transformer_cache_combined]
    [--reader-root C:/Users/chaoa/Downloads/ida_features/out] [--limit N] [--sample-id ID ...] [--replace-stale]

Each `<reader root>/<sample_id>/operand_roles.jsonl.gz` (schema ida-operand-roles/1, CT_GAT/IDA_FEATURE_READERS.md) becomes
`<cache>/side/yanping/<sample_id>.npz`:

  ytok         int16 [n_insn, 3]  the reader's triplet [opcode_or_api, operand1, operand2] of each instruction, in the
                                  cache's instruction order (the order of `mn`, `sig`, `beh`)
  ytok_sizes   int16 [3]          exclusive upper bounds of the three slots (256, 76, 76), the reader's contract
  schema, sample_id, provenance_json (cache / reader / vocabulary / code hashes), reader_json (the reader's manifest)

Alignment against the cache is checked before anything is written; any mismatch fails that sample, writes nothing for
it and is listed at the end: the reader's complete footer and counts (functions and blocks from imports.json, instructions from the footer); the same functions, blocks and instructions
(`block_index` of every row is the cache block of that instruction, so the instruction count of every block agrees,
and `fn_index` is the function that owns the block, empty functions included); `instruction_in_block`; no orphans; slot
ranges; and, as a content check, one reader opcode id per cache mnemonic (calls may carry an API id instead). The raw
export is never read. Samples without reader output are reported as missing, not as failures.
"""
import gzip
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from common.side_io import (AlignmentError, atomic_npz, cache_arrays, cache_vocab, check_counts, main, read_json,
                            reader_dir, sha256_file, sha256_files)

SCHEMA = 'ida-yanping/1'
READER_SCHEMA = 'ida-operand-roles/1'
YTOK_SIZES = (256, 76, 76)
FILES = ('operand_roles.jsonl.gz', 'manifest.json', 'imports.json')
CALL = 'call'


def code_hash():
    return sha256_files(Path(__file__), Path(__file__).parents[2] / 'common' / 'side_io.py')


@lru_cache(maxsize=2)
def read_operand_rows(path):
    """(header, footer, columns) of an operand-role file. Columns are int64 arrays over the instruction rows:
    fn_index, block_index, instruction_in_block, instruction_index, orphan, tokens [n, 3]."""
    header = footer = None
    cols = {k: [] for k in ('fn_index', 'block_index', 'instruction_in_block', 'instruction_index', 'orphan', 'tokens')}
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            kind = row.get('record')
            if kind == 'instruction':
                cols['fn_index'].append(row['fn_index'])
                cols['block_index'].append(-1 if row['block_index'] is None else row['block_index'])
                cols['instruction_in_block'].append(row['instruction_in_block'])
                cols['instruction_index'].append(row['instruction_index'])
                cols['orphan'].append(bool(row['orphan']))
                cols['tokens'].append(row['tokens'])
            elif kind == 'header' and header is None:
                header = row
            elif kind == 'footer':
                footer = row
            else:
                raise AlignmentError(f'unexpected record {kind!r}')
    out = {k: np.array(v, dtype=np.int64) for k, v in cols.items() if k != 'tokens'}
    out['tokens'] = np.array(cols['tokens'], dtype=np.int64).reshape(-1, 3)
    return header, footer, out


def check_alignment(sample_id, header, footer, cols, manifest, z, mn_strings):
    """Raises AlignmentError unless the reader rows are the cache binary's instructions, in the cache's order."""
    if header is None or header.get('schema') != READER_SCHEMA:
        raise AlignmentError(f'reader schema {None if header is None else header.get("schema")!r}, expected {READER_SCHEMA}')
    if header.get('sample_id') != sample_id or manifest.get('sample_id') != sample_id:
        raise AlignmentError('reader output is for another sample')
    if header.get('include_orphans'):
        raise AlignmentError('reader output includes orphan instructions (the cache has none)')
    if not footer or not footer.get('complete'):
        raise AlignmentError('reader output has no complete footer')
    n = len(cols['instruction_index'])
    if footer['counts']['instructions'] != n or manifest['counts']['instructions'] != n:
        raise AlignmentError(f'{n} instruction rows but footer {footer["counts"]["instructions"]}, '
                             f'manifest {manifest["counts"]["instructions"]}')
    if cols['orphan'].any():
        raise AlignmentError('orphan instruction rows')
    insn_ptr, fn_ptr = z['blk_insn_ptr'].astype(np.int64), z['fn_blk_ptr'].astype(np.int64)
    if n != len(z['mn']) or n != int(insn_ptr[-1]):
        raise AlignmentError(f'{n} reader instructions, the cache has {len(z["mn"])}')
    n_blk = len(insn_ptr) - 1
    if len(fn_ptr) - 1 != int(z['num_functions']) or n_blk != int(z['num_blocks']):
        raise AlignmentError('cache arrays disagree with its own counts')
    if (cols['instruction_index'] != np.arange(n)).any():
        raise AlignmentError('instruction_index is not 0..n-1 in order')
    block_of = np.repeat(np.arange(n_blk, dtype=np.int64), np.diff(insn_ptr))
    if (cols['block_index'] != block_of).any():
        bad = int(np.flatnonzero(cols['block_index'] != block_of)[0])
        raise AlignmentError(f'instruction {bad} is in block {cols["block_index"][bad]} for the reader, '
                             f'{block_of[bad]} in the cache: instruction counts per block differ')
    if (cols['instruction_in_block'] != np.arange(n) - insn_ptr[block_of]).any():
        raise AlignmentError('instruction_in_block does not follow the cache block layout')
    fn_of_block = np.searchsorted(fn_ptr, np.arange(n_blk), side='right') - 1
    if (cols['fn_index'] != fn_of_block[block_of]).any():
        bad = int(np.flatnonzero(cols['fn_index'] != fn_of_block[block_of])[0])
        raise AlignmentError(f'instruction {bad} is in function {cols["fn_index"][bad]} for the reader, '
                             f'{fn_of_block[block_of][bad]} in the cache')
    tokens = cols['tokens']
    if (tokens < 0).any() or (tokens >= np.array(YTOK_SIZES)).any():
        raise AlignmentError(f'reader token outside the slot ranges {YTOK_SIZES}')
    mn = z['mn'].astype(np.int64)
    pairs, counts = np.unique(np.stack([mn, tokens[:, 0]], 1), axis=0, return_counts=True)
    per_mn = Counter(int(m) for m in pairs[:, 0])
    split = sorted(str(mn_strings[m]) for m, k in per_mn.items() if k > 1 and mn_strings[m] != CALL)
    if split:
        raise AlignmentError(f'mnemonics with more than one reader opcode id (instruction content differs): {split[:5]}')


def yanping_path(cache, sample_id):
    return Path(cache) / 'side' / 'yanping' / f'{sample_id}.npz'


def convert(cache, sample_id, root, replace_stale=False):
    """One sample of one cache: {'sample_id', 'status': ok | skipped | failed | missing, 'error'?}."""
    d = reader_dir(root, sample_id, FILES)
    if d is None:
        return {'sample_id': sample_id, 'status': 'missing'}
    cache, out = Path(cache), yanping_path(cache, sample_id)
    try:
        vocab, vocab_sha = cache_vocab(str(cache))
        provenance = {'schema': SCHEMA, 'sample_id': sample_id,
                      'cache_sha256': sha256_file(cache / 'samples' / f'{sample_id}.npz'),
                      'reader_operand_roles_sha256': sha256_file(d / FILES[0]),
                      'reader_manifest_sha256': sha256_file(d / FILES[1]),
                      'vocab_sha256': vocab_sha, 'code_sha256': code_hash()}
        if out.exists():
            with np.load(out, allow_pickle=False) as old:
                if json.loads(str(old['provenance_json'])) == provenance:
                    return {'sample_id': sample_id, 'status': 'skipped'}
            if not replace_stale:
                raise AlignmentError(f'stale side file {out}; use --replace-stale to replace it')
        manifest = read_json(d / FILES[1])
        header, footer, cols = read_operand_rows(str(d / FILES[0]))
        z = cache_arrays(cache, sample_id)
        check_counts(read_json(d / FILES[2])['counts'], z, FILES[2])   # functions, blocks and instructions of the export
        check_alignment(sample_id, header, footer, cols, manifest, z, vocab['mn'])
        arrays = dict(schema=np.array(SCHEMA), sample_id=np.array(sample_id), ytok=cols['tokens'].astype(np.int16),
                      ytok_sizes=np.array(YTOK_SIZES, dtype=np.int16), provenance_json=np.array(json.dumps(provenance, sort_keys=True)),
                      reader_json=np.array(json.dumps({k: manifest[k] for k in manifest if k != 'source'}, sort_keys=True)))
        atomic_npz(out, arrays)
        return {'sample_id': sample_id, 'status': 'ok', 'instructions': len(cols['tokens'])}
    except Exception as e:
        if replace_stale and out.exists():
            out.unlink()
        return {'sample_id': sample_id, 'status': 'failed', 'error': f'{type(e).__name__}: {e}'}


def load_ytok(cache, sample_id):
    """int16 [n_insn, 3] of a sample from `<cache>/side/yanping/`. A missing or foreign file raises, naming the sample."""
    path = yanping_path(cache, sample_id)
    if not path.exists():
        raise FileNotFoundError(f'{path}: no reader side array for {sample_id}; run '
                                f'python -m pipeline.s03_tokenizer.import_reader_side --cache {cache}')
    with np.load(path, allow_pickle=False) as z:
        if str(z['schema']) != SCHEMA or str(z['sample_id']) != sample_id:
            raise ValueError(f'{path}: schema {z["schema"]} / sample {z["sample_id"]}, expected {SCHEMA} / {sample_id}')
        return z['ytok']


if __name__ == '__main__':
    main(__doc__, convert, 'yanping')
