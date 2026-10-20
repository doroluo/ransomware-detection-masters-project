"""common/side_io.py: the plumbing the reader-side converters share (the converters themselves are experiments)."""
import csv
import hashlib
import json

import numpy as np
import pytest

from common import side_io
from common.paths import MissingSource


def test_hashes(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    a.write_bytes(b'abc')
    b.write_bytes(b'def')
    assert side_io.sha256_file(a) == hashlib.sha256(b'abc').hexdigest()
    assert side_io.sha256_files(a, b) == hashlib.sha256(b'abcdef').hexdigest()
    big = tmp_path / 'big'
    big.write_bytes(b'x' * ((1 << 20) + 5))
    assert side_io.sha256_file(big) == hashlib.sha256(b'x' * ((1 << 20) + 5)).hexdigest()


def test_atomic_npz_creates_parents_replaces_and_leaves_no_temporary(tmp_path):
    path = tmp_path / 'deep' / 'er' / 's.npz'
    side_io.atomic_npz(path, {'x': np.arange(3)})
    side_io.atomic_npz(path, {'x': np.arange(5)})
    with np.load(path) as z:
        assert z['x'].tolist() == [0, 1, 2, 3, 4]
    assert [p.name for p in path.parent.iterdir()] == ['s.npz']


def make_cache(root, rows):
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'index.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['sample_id', 'status'])
        w.writeheader()
        w.writerows(rows)


def test_index_ids_keep_the_built_rows_in_order(tmp_path):
    make_cache(tmp_path, [{'sample_id': 'b', 'status': 'ok'}, {'sample_id': 'x', 'status': 'failed'},
                          {'sample_id': 'a', 'status': 'ok'}])
    assert side_io.index_ids(tmp_path) == ['b', 'a']


def test_reader_dir_needs_every_named_file(tmp_path):
    (tmp_path / 's1').mkdir()
    (tmp_path / 's1' / 'one.json').write_text('{}')
    (tmp_path / 's1' / 'two.json').write_text('{}')
    (tmp_path / 's2').mkdir()
    (tmp_path / 's2' / 'one.json').write_text('{}')
    assert side_io.reader_dir(tmp_path, 's1', ('one.json', 'two.json')) == tmp_path / 's1'
    assert side_io.reader_dir(tmp_path, 's2', ('one.json', 'two.json')) is None
    assert side_io.reader_dir(tmp_path, 'absent', ('one.json',)) is None


def test_cache_arrays_cache_vocab_and_count_check(tmp_path):
    (tmp_path / 'samples').mkdir()
    arrays = {k: np.arange(2) for k in side_io.CACHE_KEYS}
    arrays.update(num_functions=np.array(1), num_blocks=np.array(2), num_instructions=np.array(3))
    np.savez(tmp_path / 'samples' / 's.npz', **arrays)
    (tmp_path / 'vocab.json').write_text(json.dumps({'mn': ['mov']}), encoding='utf-8')
    z = side_io.cache_arrays(tmp_path, 's')
    assert set(z) == set(side_io.CACHE_KEYS)
    vocab, sha = side_io.cache_vocab(tmp_path)
    assert vocab == {'mn': ['mov']} and sha == side_io.sha256_file(tmp_path / 'vocab.json')
    side_io.check_counts({'functions': 1, 'blocks': 2, 'instructions': 3}, z, 'ok')
    with pytest.raises(side_io.AlignmentError, match='what'):
        side_io.check_counts({'functions': 1, 'blocks': 2, 'instructions': 4}, z, 'what')
    assert issubclass(side_io.AlignmentError, ValueError)


def test_build_summary_counts_and_writes_its_record(tmp_path):
    results = [{'sample_id': 'a', 'status': 'ok'}, {'sample_id': 'b', 'status': 'skipped'},
               {'sample_id': 'c', 'status': 'failed', 'error': 'bad'}, {'sample_id': 'd', 'status': 'missing'}]
    summary = side_io.build_summary(tmp_path / 'side' / 'x', results)
    assert summary == {'ok': 2, 'built': 1, 'skipped_up_to_date': 1, 'failed': 1, 'missing': 1, 'total': 4,
                       'failures': [{'sample_id': 'c', 'error': 'bad'}]}
    written = list((tmp_path / 'side' / 'x').glob('_build-*.json'))
    assert len(written) == 1 and json.loads(written[0].read_text(encoding='utf-8'))['results'] == results


def test_main_runs_a_converter_per_sample_and_exits_by_the_outcome(tmp_path, capsys, monkeypatch):
    cache = tmp_path / 'cache'
    make_cache(cache, [{'sample_id': s, 'status': 'ok'} for s in ('a', 'b', 'c')])
    seen = []

    def convert(cache_dir, sid, root, replace_stale):
        seen.append((sid, root, replace_stale))
        return {'sample_id': sid, 'status': 'failed' if sid == 'b' else 'ok', 'error': 'boom'}

    monkeypatch.delenv('GT_READER_ROOT', raising=False)
    monkeypatch.setenv('GT_DATASETS', str(tmp_path / 'absent.json'))
    with pytest.raises(MissingSource, match='GT_READER_ROOT'):
        side_io.main('d', convert, 'out', ['--cache', str(cache)])                  # no reader root configured
    with pytest.raises(SystemExit) as stop:
        side_io.main('d', convert, 'out', ['--cache', str(cache), '--reader-root', str(tmp_path / 'reader'), '--limit', '2'])
    assert stop.value.code == 1 and seen == [('a', tmp_path / 'reader', False), ('b', tmp_path / 'reader', False)]
    out = capsys.readouterr()
    assert '"failed": 1' in out.out and 'FAILED b: boom' in out.err
    assert list((cache / 'side' / 'out').glob('_build-*.json'))
    with pytest.raises(SystemExit) as stop:
        side_io.main('d', convert, 'out', ['--cache', str(cache), '--reader-root', str(tmp_path), '--sample-id', 'a', 'c'])
    assert stop.value.code == 0
