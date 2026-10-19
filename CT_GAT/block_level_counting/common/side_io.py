"""Shared plumbing of the two reader-side converters (`import_reader_side.py`, `import_side.py`): hashing, atomic
writes, the cache index, the reader-output layout and the command line. The reader output is the team's, read only:
`<reader root>/<sample_id>/{operand_roles.jsonl.gz, imports.json, manifest.json}` (CT_GAT/IDA_FEATURE_READERS.md)."""
import argparse
import csv
import hashlib
import json
import os
import sys
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np

from common.config import CACHE

READER_ROOT = Path('C:/Users/chaoa/Downloads/ida_features/out')
CACHE_KEYS = ('fn_blk_ptr', 'fn_flags', 'blk_insn_ptr', 'blk_api_ptr', 'mn', 'api', 'num_functions', 'num_blocks',
              'num_instructions')


class AlignmentError(ValueError):
    """The reader output does not describe the cache's binary."""


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha256_files(*paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def atomic_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def index_ids(cache):
    """Sample ids of the built rows (status ok) of a cache's index.csv, in index order."""
    with (Path(cache) / 'index.csv').open(encoding='utf-8', newline='') as f:
        return [r['sample_id'] for r in csv.DictReader(f) if r['status'] == 'ok']


def reader_dir(root, sample_id, names):
    """The reader output directory of a sample, or None when it is absent or lacks one of `names`
    (a `.pending-*` directory is never a sample directory)."""
    d = Path(root) / sample_id
    return d if all((d / n).is_file() for n in names) else None


@lru_cache(maxsize=4)
def cache_vocab(cache):
    """(vocab.json of a cache, its sha256)."""
    path = Path(cache) / 'vocab.json'
    return read_json(path), sha256_file(path)


def cache_arrays(cache, sample_id):
    with np.load(Path(cache) / 'samples' / f'{sample_id}.npz', allow_pickle=False) as z:
        return {k: z[k] for k in CACHE_KEYS}


def check_counts(counts, z, what):
    """The reader's function / block / instruction counts must be the cache's."""
    got = (counts['functions'], counts['blocks'], counts['instructions'])
    want = (int(z['num_functions']), int(z['num_blocks']), int(z['num_instructions']))
    if got != want:
        raise AlignmentError(f'{what}: functions, blocks, instructions {got} but the cache has {want}')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def build_summary(out_dir, results):
    """Counts of ok / failed / missing and the failure list; also written to `_build-<uuid>.json` in `out_dir`."""
    n = {s: sum(r['status'] == s for r in results) for s in ('ok', 'skipped', 'failed', 'missing')}
    summary = {'ok': n['ok'] + n['skipped'], 'built': n['ok'], 'skipped_up_to_date': n['skipped'],
               'failed': n['failed'], 'missing': n['missing'], 'total': len(results),
               'failures': [{'sample_id': r['sample_id'], 'error': r['error']} for r in results
                            if r['status'] == 'failed']}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / f'_build-{uuid.uuid4().hex}.json').write_text(
        json.dumps({**summary, 'results': results}, indent=1), encoding='utf-8')
    return summary


def main(description, convert, out_name, argv=None):
    """Command line of a converter. `convert(cache, sample_id, reader_dir, replace_stale)` returns a result dict with
    `status` ok / skipped (up to date); an AlignmentError or any other error fails that sample only."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument('--cache', type=Path, nargs='+', default=[CACHE], help='cache directories (default: GT_CACHE or v2)')
    ap.add_argument('--reader-root', type=Path, default=READER_ROOT)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--sample-id', nargs='+')
    ap.add_argument('--replace-stale', action='store_true', help='replace a stale side file (and drop one whose sample now fails)')
    ap.add_argument('--require-all', action='store_true', help='exit 1 when any sample is missing, too')
    args = ap.parse_args(argv)
    wanted = set(args.sample_id) if args.sample_id else None
    ids = {cache: [i for i in index_ids(cache) if wanted is None or i in wanted][:args.limit or None]
           for cache in args.cache}
    members = {cache: set(ids[cache]) for cache in args.cache}
    results = {cache: [] for cache in args.cache}
    # sample-major, so the reader output of a sample that two caches share is parsed once (the converters memoize it)
    for sid in dict.fromkeys(i for cache in args.cache for i in ids[cache]):
        for cache in args.cache:
            if sid in members[cache]:
                result = convert(cache, sid, args.reader_root, args.replace_stale)
                results[cache].append(result)
                if result['status'] == 'failed':
                    print(json.dumps(result), flush=True)
    code = 0
    for cache in args.cache:
        summary = build_summary(Path(cache) / 'side' / out_name, results[cache])
        print(json.dumps({'cache': str(cache), 'side': out_name, **{k: v for k, v in summary.items() if k != 'failures'}}),
              flush=True)
        for f in summary['failures']:
            print(f"FAILED {f['sample_id']}: {f['error']}", file=sys.stderr)
        if summary['failed'] or (args.require_all and summary['missing']):
            code = 1
    raise SystemExit(code)
