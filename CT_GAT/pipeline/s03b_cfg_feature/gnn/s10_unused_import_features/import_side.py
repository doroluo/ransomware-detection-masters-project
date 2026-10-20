"""Converter of the team's import-feature reader output to a cache-aligned side array.

python -m s03b_cfg_feature.gnn.s10_unused_import_features.import_side --cache <cache dir> [<another cache dir>]
    [--reader-root <reader output dir>] [--limit N] [--sample-id ID ...] [--replace-stale]

Each `<reader root>/<sample_id>/imports.json` (schema ida-import-features/1, CT_GAT/IDA_FEATURE_READERS.md) becomes
`<cache>/side/imports/<sample_id>.npz`:

  header_imports, referenced_imports, unreferenced_imports   Unicode [k]  distinct lowercased `dll!name` / `dll!#ordinal`
      identities exactly as the reader gives them: every header entry, those with an observed static reference (function
      and orphan instructions), and the header set minus the referenced set ("unreferenced" is not "unused at run time")
  n_entries  int   header entries including duplicates and delay-load entries
  schema, sample_id, provenance_json, reader_json (the reader's counts, scope and limitations)

Checked before anything is written; a failing sample writes nothing and is listed: schema and sample id, a complete export
(`export_complete_verified`), the reader's function / block / instruction counts against the cache's, the three sets
consistent with the entry list (header = entries, header = referenced + unreferenced, the two disjoint), and every import the
cache's instructions reference (`api`, lowercased) among the reader's referenced identities (the reader also scans orphan
instructions, which the cache omits, so it may see more). The raw export is never read. A sample without reader output is
missing, not failed.
"""
import json
from pathlib import Path

import numpy as np

from common.side_io import (AlignmentError, atomic_npz, cache_arrays, cache_vocab, check_counts, main, read_json,
                            reader_dir, sha256_file, sha256_files)

SCHEMA = 'ida-imports/1'
READER_SCHEMA = 'ida-import-features/1'
FILES = ('imports.json', 'manifest.json')
SETS = ('header_imports', 'referenced_imports', 'unreferenced_imports')


def code_hash():
    return sha256_files(Path(__file__), Path(__file__).parents[3] / 'common' / 'side_io.py')


def check_imports(sample_id, doc, manifest, z, api_strings):
    """Raises AlignmentError unless the reader's import doc is this sample's and is self-consistent."""
    if doc.get('schema') != READER_SCHEMA:
        raise AlignmentError(f'reader schema {doc.get("schema")!r}, expected {READER_SCHEMA}')
    if doc.get('sample_id') != sample_id or manifest.get('sample_id') != sample_id:
        raise AlignmentError('reader output is for another sample')
    if not doc.get('export_complete_verified'):
        raise AlignmentError('reader did not verify a complete export')
    check_counts(doc['counts'], z, 'imports.json')
    header, referenced, unreferenced = (set(doc[k]) for k in SETS)
    if any(len(doc[k]) != len(set(doc[k])) for k in SETS):
        raise AlignmentError('an import identity set has duplicates')
    if header != {e['key'] for e in doc['imports']}:
        raise AlignmentError('header_imports differs from the identities of the import entries')
    if referenced | unreferenced != header or referenced & unreferenced:
        raise AlignmentError('referenced and unreferenced imports do not partition the header imports')
    if (manifest['header_imports'], manifest['referenced_imports'], manifest['unreferenced_imports']) != (
            len(header), len(referenced), len(unreferenced)):
        raise AlignmentError('manifest.json counts differ from imports.json')
    seen = {api_strings[i].lower() for i in np.unique(z['api'])}
    if not seen <= referenced:
        raise AlignmentError(f'{len(seen - referenced)} imports referenced by cache instructions are not referenced for '
                             f'the reader, e.g. {sorted(seen - referenced)[:3]}')


def imports_path(cache, sample_id):
    return Path(cache) / 'side' / 'imports' / f'{sample_id}.npz'


def as_array(names):
    return np.array(list(names), dtype=str) if len(names) else np.zeros(0, dtype='<U1')


def convert(cache, sample_id, root, replace_stale=False):
    """One sample of one cache: {'sample_id', 'status': ok | skipped | failed | missing, 'error'?}."""
    d = reader_dir(root, sample_id, FILES)
    if d is None:
        return {'sample_id': sample_id, 'status': 'missing'}
    cache, out = Path(cache), imports_path(cache, sample_id)
    try:
        vocab, vocab_sha = cache_vocab(str(cache))
        provenance = {'schema': SCHEMA, 'sample_id': sample_id,
                      'cache_sha256': sha256_file(cache / 'samples' / f'{sample_id}.npz'),
                      'reader_imports_sha256': sha256_file(d / FILES[0]),
                      'reader_manifest_sha256': sha256_file(d / FILES[1]), 'vocab_sha256': vocab_sha,
                      'code_sha256': code_hash()}
        if out.exists():
            with np.load(out, allow_pickle=False) as old:
                if json.loads(str(old['provenance_json'])) == provenance:
                    return {'sample_id': sample_id, 'status': 'skipped'}
            if not replace_stale:
                raise AlignmentError(f'stale side file {out}; use --replace-stale to replace it')
        doc, manifest = read_json(d / FILES[0]), read_json(d / FILES[1])
        check_imports(sample_id, doc, manifest, cache_arrays(cache, sample_id), vocab['api'])
        arrays = {k: as_array(doc[k]) for k in SETS}
        arrays.update(schema=np.array(SCHEMA), sample_id=np.array(sample_id), n_entries=np.array(len(doc['imports'])),
                      provenance_json=np.array(json.dumps(provenance, sort_keys=True)),
                      reader_json=np.array(json.dumps({k: doc[k] for k in ('counts', 'reference_scope', 'limitations')},
                                                      sort_keys=True)))
        atomic_npz(out, arrays)
        return {'sample_id': sample_id, 'status': 'ok', 'header': len(doc['header_imports']),
                'unreferenced': len(doc['unreferenced_imports'])}
    except Exception as e:
        if replace_stale and out.exists():
            out.unlink()
        return {'sample_id': sample_id, 'status': 'failed', 'error': f'{type(e).__name__}: {e}'}


def load_imports(cache, sample_id):
    """{'header_imports', 'referenced_imports', 'unreferenced_imports': list[str], 'n_entries': int} of a sample from
    `<cache>/side/imports/`. A missing or foreign file raises, naming the sample."""
    path = imports_path(cache, sample_id)
    if not path.exists():
        raise FileNotFoundError(f'{path}: no import side array for {sample_id}; run '
                                f'python -m s03b_cfg_feature.gnn.s10_unused_import_features.import_side --cache {cache}')
    with np.load(path, allow_pickle=False) as z:
        if str(z['schema']) != SCHEMA or str(z['sample_id']) != sample_id:
            raise ValueError(f'{path}: schema {z["schema"]} / sample {z["sample_id"]}, expected {SCHEMA} / {sample_id}')
        return {**{k: z[k].tolist() for k in SETS}, 'n_entries': int(z['n_entries'])}


if __name__ == '__main__':
    main(__doc__, convert, 'imports')
