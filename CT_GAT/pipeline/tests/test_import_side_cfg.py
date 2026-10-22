"""python -m pytest tests/test_import_side_cfg.py: `import_side --from-cfg`, the header-import arrays built from the first record of
the sample's CFG export (`.jsonl.gz`, `pe.imports`) instead of the reader output. Synthetic cache and exports in `tmp_path`:
named, ordinal, duplicate and delay-load imports; the Mendeley data located as `build_cache` does (the bundle, or the two
collections); the written keys and values; a stale file of the other source; the reader mode untouched.
"""
import csv
import gzip
import json
import shutil

import numpy as np
import pytest

from common import side_io
from s04_evaluation import import_side as imp
from test_import_side import FIX, SID as READER_SID

SID = 'c' * 64
API = ['kernel32.dll!CreateFileW', 'kernel32.dll!WriteFile', 'advapi32.dll!CryptAcquireContextW', 'ws2_32.dll!#115']
ENTRIES = [  # dll, name, ordinal, delayed
    ('KERNEL32.dll', 'CreateFileW', None, False), ('KERNEL32.dll', 'WriteFile', None, False),
    ('KERNEL32.dll', 'WriteFile', None, False),                                      # a duplicate entry
    ('ADVAPI32.dll', 'CryptAcquireContextW', None, False), ('ADVAPI32.dll', 'RegCloseKey', None, False),
    ('WS2_32.dll', '', 115, False), ('COMCTL32.dll', '', 17, False),                  # imports by ordinal
    ('SHELL32.dll', 'ShellExecuteW', None, True), ('SHELL32.dll', 'ShellExecuteW', None, True)]   # delay-load, twice
HEADER = sorted({'kernel32.dll!createfilew', 'kernel32.dll!writefile', 'advapi32.dll!cryptacquirecontextw',
                 'advapi32.dll!regclosekey', 'ws2_32.dll!#115', 'comctl32.dll!#17', 'shell32.dll!shellexecutew'})
REFERENCED = ['advapi32.dll!cryptacquirecontextw', 'kernel32.dll!createfilew', 'kernel32.dll!writefile', 'ws2_32.dll!#115']


def write_export(path, sample_id=SID, entries=ENTRIES, body=b'not json at all\n'):
    """A gzipped jsonl: a header record, then `body`, which must never be parsed."""
    header = {'record': 'header', 'sha256': sample_id, 'schema': 'ida-cfg-research/2.1',
              'pe': {'imports': [{'dll': d, 'name': n, 'ordinal': o, 'rva': 4096 + 4 * i, 'delayed': delayed}
                                 for i, (d, n, o, delayed) in enumerate(entries)]}}
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wb') as f:
        f.write(json.dumps(header).encode() + b'\n' + body)


def write_cache(cache, api_ids=(0, 1, 1, 2, 3, 0), sample_id=SID, corpus=None):
    """A cache of one sample whose instructions name `api_ids` of API, with every array of `side_io.CACHE_KEYS`."""
    (cache / 'samples').mkdir(parents=True)
    arrays = {k: np.zeros(3, dtype=np.int32) for k in side_io.CACHE_KEYS}
    arrays.update(api=np.array(api_ids, dtype=np.int32), num_functions=np.array(2), num_blocks=np.array(3),
                  num_instructions=np.array(len(api_ids)))
    np.savez_compressed(cache / 'samples' / f'{sample_id}.npz', **arrays)
    (cache / 'vocab.json').write_text(json.dumps({'api': API}), encoding='utf-8')
    fields = ['sample_id', 'status'] + (['corpus'] if corpus else [])
    with (cache / 'index.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({'sample_id': sample_id, 'status': 'ok', **({'corpus': corpus} if corpus else {})})


def write_samples(folder, sample_id=SID, cfg_path='cfg/x.jsonl.gz'):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'samples.csv').write_text(f'sample_id,cfg_path\n{sample_id},{cfg_path}\n', encoding='utf-8')
    write_export(folder / cfg_path, sample_id)


@pytest.fixture()
def bundle(tmp_path, monkeypatch):
    """A cache and a Mendeley_GNN_v1-shaped bundle (GT_BUNDLE) holding the sample's export."""
    for var in ('GT_BUNDLE', 'GT_COLLECTIONS', 'GT_READER_ROOT'):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv('GT_DATASETS', str(tmp_path / 'no-datasets.json'))
    side_io.cache_vocab.cache_clear()
    imp.cfg_rows.cache_clear()
    imp.index_corpora.cache_clear()
    write_cache(tmp_path / 'cache')
    write_samples(tmp_path / 'bundle')
    monkeypatch.setenv('GT_BUNDLE', str(tmp_path / 'bundle'))
    return tmp_path


def load(cache, sid=SID):
    with np.load(cache / 'side' / 'imports' / f'{sid}.npz', allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def test_the_header_gives_the_reader_spelling_and_counts_every_entry(bundle):
    assert imp.convert_cfg(bundle / 'cache', SID)['status'] == 'ok'
    z = load(bundle / 'cache')
    assert sorted(z) == ['header_imports', 'n_entries', 'provenance_json', 'reader_json', 'referenced_imports', 'sample_id',
                         'schema', 'unreferenced_imports']
    assert z['header_imports'].tolist() == HEADER                                       # lowercased, ordinals as #n, distinct, sorted
    assert int(z['n_entries']) == len(ENTRIES) == 9                                      # duplicates and delay-load entries kept
    assert z['referenced_imports'].tolist() == REFERENCED
    assert z['unreferenced_imports'].tolist() == sorted(set(HEADER) - set(REFERENCED))
    assert all(z[k].dtype.kind == 'U' for k in imp.SETS) and z['n_entries'].dtype == np.int64
    assert (str(z['schema']), str(z['sample_id'])) == (imp.SCHEMA, SID)
    got = json.loads(str(z['provenance_json']))
    assert got['source'] == 'cfg-header' and got['code_sha256'] == imp.code_hash() and got['sample_id'] == SID
    assert json.loads(str(z['reader_json']))['counts'] == {'functions': 2, 'blocks': 3, 'instructions': 6}
    loaded = imp.load_imports(bundle / 'cache', SID)                                      # what the consumers read
    assert loaded['header_imports'] == HEADER and loaded['n_entries'] == 9
    assert loaded['referenced_imports'] == REFERENCED


def test_only_the_header_line_is_read(bundle):
    """The body of the export is not valid json: the sample converts, so nothing past the first line was parsed."""
    assert imp.convert_cfg(bundle / 'cache', SID)['status'] == 'ok'


def test_a_second_run_is_up_to_date_and_a_changed_cache_sample_is_stale(bundle):
    cache = bundle / 'cache'
    assert imp.convert_cfg(cache, SID)['status'] == 'ok'
    assert imp.convert_cfg(cache, SID)['status'] == 'skipped'
    with np.load(cache / 'samples' / f'{SID}.npz') as z:
        arrays = {k: z[k] for k in z.files}
    arrays['api'] = np.array([0], dtype=np.int32)
    np.savez_compressed(cache / 'samples' / f'{SID}.npz', **arrays)
    stale = imp.convert_cfg(cache, SID)
    assert stale['status'] == 'failed' and 'stale' in stale['error']
    assert imp.convert_cfg(cache, SID, replace_stale=True)['status'] == 'ok'
    assert imp.load_imports(cache, SID)['referenced_imports'] == ['kernel32.dll!createfilew']


def test_the_collections_route_locates_the_export_like_build_cache(bundle, monkeypatch):
    coll = bundle / 'coll'
    write_samples(coll / 'Mendeley_Ransomware', SID, 'samples/train/conti/x.jsonl.gz')
    write_samples(coll / 'Mendeley_Goodware', 'd' * 64, 'samples/test/y.jsonl.gz')
    monkeypatch.delenv('GT_BUNDLE')
    monkeypatch.setenv('GT_COLLECTIONS', str(coll))
    imp.cfg_rows.cache_clear()
    assert imp.convert_cfg(bundle / 'cache', SID)['status'] == 'ok'
    assert imp.load_imports(bundle / 'cache', SID)['header_imports'] == HEADER


def test_a_sample_of_another_corpus_is_looked_up_in_its_own(bundle, monkeypatch):
    write_cache(bundle / 'cache2', sample_id='e' * 64, corpus='balanced_goodware')
    write_samples(bundle / 'coll' / 'Balanced_Goodware', 'e' * 64, 'samples/e.jsonl.gz')
    monkeypatch.setenv('GT_COLLECTIONS', str(bundle / 'coll'))
    assert imp.convert_cfg(bundle / 'cache2', 'e' * 64)['status'] == 'ok'
    assert imp.load_imports(bundle / 'cache2', 'e' * 64)['n_entries'] == 9


def test_a_sample_absent_from_samples_csv_is_missing_not_failed(bundle):
    cache = bundle / 'cache'
    shutil.copyfile(cache / 'samples' / f'{SID}.npz', cache / 'samples' / ('f' * 64 + '.npz'))
    assert imp.convert_cfg(cache, 'f' * 64)['status'] == 'missing'
    assert not (cache / 'side' / 'imports' / ('f' * 64 + '.npz')).exists()


def test_an_export_that_is_not_this_samples_fails_and_writes_nothing(bundle):
    write_export(bundle / 'bundle' / 'cfg' / 'x.jsonl.gz', sample_id='a' * 64)
    result = imp.convert_cfg(bundle / 'cache', SID)
    assert result['status'] == 'failed' and 'header of this sample' in result['error']
    assert not (bundle / 'cache' / 'side' / 'imports' / f'{SID}.npz').exists()


def test_an_import_the_cache_references_must_be_in_the_header(bundle):
    write_export(bundle / 'bundle' / 'cfg' / 'x.jsonl.gz', entries=[e for e in ENTRIES if e[1] != 'WriteFile'])
    result = imp.convert_cfg(bundle / 'cache', SID)
    assert result['status'] == 'failed' and 'not in the header' in result['error'] and 'writefile' in result['error']
    assert not (bundle / 'cache' / 'side' / 'imports' / f'{SID}.npz').exists()


def test_a_header_without_imports_gives_empty_string_arrays(bundle):
    write_export(bundle / 'bundle' / 'cfg' / 'x.jsonl.gz', entries=[])
    write_cache(bundle / 'cache3', api_ids=())
    (bundle / 'cache3' / 'index.csv').write_text(f'sample_id,status\n{SID},ok\n', encoding='utf-8')
    assert imp.convert_cfg(bundle / 'cache3', SID)['status'] == 'ok'
    z = load(bundle / 'cache3')
    assert all(z[k].dtype.kind == 'U' and len(z[k]) == 0 for k in imp.SETS) and int(z['n_entries']) == 0


def test_the_command_line_needs_no_reader_output(bundle, capsys):
    argv = ['--cache', str(bundle / 'cache'), '--from-cfg']
    with pytest.raises(SystemExit) as stop:
        side_io.main('t', imp.convert, 'imports', argv, from_cfg=imp.convert_cfg)
    assert stop.value.code == 0
    summary = json.loads(next(l for l in capsys.readouterr().out.splitlines() if l.startswith('{"cache"')))
    assert (summary['ok'], summary['failed'], summary['missing']) == (1, 0, 0)
    assert imp.load_imports(bundle / 'cache', SID)['n_entries'] == 9
    with pytest.raises(SystemExit) as stop:                                              # the two sources are not mixed
        side_io.main('t', imp.convert, 'imports', argv + ['--reader-root', str(bundle)], from_cfg=imp.convert_cfg)
    assert stop.value.code == 2


def test_the_reader_mode_is_unchanged(tmp_path, monkeypatch):
    """Without --from-cfg the reader output is still required, and the reader's file converts as before."""
    monkeypatch.delenv('GT_READER_ROOT', raising=False)
    monkeypatch.setenv('GT_DATASETS', str(tmp_path / 'no-datasets.json'))
    side_io.cache_vocab.cache_clear()
    cache, reader = tmp_path / 'cache', tmp_path / 'reader'
    shutil.copytree(FIX / 'cache', cache)
    shutil.copytree(FIX / 'reader', reader)
    with pytest.raises(Exception, match='GT_READER_ROOT'):                               # no reader root, no --from-cfg
        side_io.main('t', imp.convert, 'imports', ['--cache', str(cache)], from_cfg=imp.convert_cfg)
    with pytest.raises(SystemExit) as stop:
        side_io.main('t', imp.convert, 'imports', ['--cache', str(cache), '--reader-root', str(reader)],
                     from_cfg=imp.convert_cfg)
    assert stop.value.code == 0
    provenance = json.loads(str(load(cache, READER_SID)['provenance_json']))
    assert 'source' not in provenance and 'reader_imports_sha256' in provenance
    assert imp.convert(cache, READER_SID, reader)['status'] == 'skipped'
