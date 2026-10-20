"""python -m pytest tests/test_import_side.py: the converter of the team's import-feature reader output to `side/imports`
(`s03b_cfg_feature/gnn/s10_unused_import_features/import_side.py`), on one real reader output (tests/fixtures/reader_side: a
binary with 29 unreferenced imports, its reader files copied byte for byte, and its cache sample with the vocabulary shrunk) and
deliberately inconsistent copies of it.

Ported from the 3-block-level-counting branch (tests/test_reader_side.py, the imports half; the operand-role half belongs to
experiments/side_arrays). The fixture directory is the same.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from common import side_io
from s03b_cfg_feature.gnn.s10_unused_import_features import import_side as imp

FIX = Path(__file__).parent / 'fixtures' / 'reader_side'
SID = next((FIX / 'reader').iterdir()).name


class Box:
    """A scratch copy of the fixture: `cache`, `reader` (the reader root) and helpers to derive an inconsistent sample."""

    def __init__(self, tmp):
        self.cache, self.reader = tmp / 'cache', tmp / 'reader'
        shutil.copytree(FIX / 'cache', self.cache)
        shutil.copytree(FIX / 'reader', self.reader)

    def edit_json(self, name, fn, sid=SID):
        path = self.reader / sid / name
        doc = json.loads(path.read_text(encoding='utf-8'))
        fn(doc)
        path.write_text(json.dumps(doc), encoding='utf-8')

    def clone(self, new_id):
        """The same cache sample and reader output under another sample id (headers rewritten), to damage separately."""
        shutil.copyfile(self.cache / 'samples' / f'{SID}.npz', self.cache / 'samples' / f'{new_id}.npz')
        shutil.copytree(self.reader / SID, self.reader / new_id)
        for name in ('imports.json', 'manifest.json'):
            self.edit_json(name, lambda d: d.update(sample_id=new_id), new_id)
        with (self.cache / 'index.csv').open('a', encoding='utf-8', newline='') as f:
            f.write(f'{new_id},ok\n')
        return new_id

    def convert_imports(self, sid=SID, **kw):
        return imp.convert(self.cache, sid, self.reader, **kw)


@pytest.fixture()
def box(tmp_path):
    side_io.cache_vocab.cache_clear()
    return Box(tmp_path)


def test_real_imports_convert_as_given_by_the_reader(box):
    assert box.convert_imports()['status'] == 'ok'
    doc = json.loads((box.reader / SID / 'imports.json').read_text(encoding='utf-8'))
    got = imp.load_imports(box.cache, SID)
    assert got['header_imports'] == doc['header_imports'] and len(doc['header_imports']) == 64
    assert got['referenced_imports'] == doc['referenced_imports']
    assert got['unreferenced_imports'] == doc['unreferenced_imports'] and len(got['unreferenced_imports']) == 29
    assert got['n_entries'] == len(doc['imports'])
    assert set(got['referenced_imports']) | set(got['unreferenced_imports']) == set(got['header_imports'])
    assert box.convert_imports()['status'] == 'skipped'


def test_a_binary_without_unreferenced_imports_gives_an_empty_string_array(box):
    def none_unreferenced(d):
        d['unreferenced_imports'], d['unreferenced_entry_indices'] = [], []
        d['referenced_imports'] = list(d['header_imports'])
    box.edit_json('imports.json', none_unreferenced)
    box.edit_json('manifest.json', lambda d: d.update(referenced_imports=64, unreferenced_imports=0))
    assert box.convert_imports()['status'] == 'ok'
    with np.load(box.cache / 'side' / 'imports' / f'{SID}.npz', allow_pickle=False) as z:
        assert z['unreferenced_imports'].dtype.kind == 'U' and len(z['unreferenced_imports']) == 0
    assert imp.load_imports(box.cache, SID)['unreferenced_imports'] == []


def not_a_partition(d):
    d['unreferenced_imports'] = d['unreferenced_imports'][1:]


def overlapping(d):
    d['unreferenced_imports'] = d['unreferenced_imports'] + d['referenced_imports'][:1]


def cache_reference_unseen(d):
    d['referenced_imports'] = d['referenced_imports'][:-5]
    d['unreferenced_imports'] = d['unreferenced_imports'] + d['referenced_imports'][-5:]


def foreign_identity(d):
    d['header_imports'] = d['header_imports'][:-1] + ['kernel32.dll!notanentry']


IMPORT_DAMAGE = [
    (lambda d: d.update(export_complete_verified=False), 'complete export'),
    (lambda d: d.update(sample_id='e' * 64), 'another sample'),
    (lambda d: d.update(schema='ida-import-features/2'), 'schema'),
    (lambda d: d['counts'].update(blocks=d['counts']['blocks'] + 1), 'functions, blocks, instructions'),
    (not_a_partition, 'partition'),
    (overlapping, 'partition'),
    (foreign_identity, 'identities of the import entries'),
]


@pytest.mark.parametrize('damage,expected', IMPORT_DAMAGE, ids=range(len(IMPORT_DAMAGE)))
def test_inconsistent_imports_fail_and_write_nothing(box, damage, expected):
    box.edit_json('imports.json', damage)
    result = box.convert_imports()
    assert result['status'] == 'failed' and expected in result['error'], result
    assert not (box.cache / 'side' / 'imports' / f'{SID}.npz').exists()


def test_imports_the_cache_references_must_be_referenced_for_the_reader(box):
    """The cache's instructions reference some imports; a reader output that calls them unreferenced contradicts it."""
    doc = json.loads((box.reader / SID / 'imports.json').read_text(encoding='utf-8'))
    with np.load(box.cache / 'samples' / f'{SID}.npz') as z:
        vocab = json.loads((box.cache / 'vocab.json').read_text(encoding='utf-8'))['api']
        seen = sorted({vocab[i].lower() for i in z['api']})
    victim = seen[0]

    def damage(d):
        d['referenced_imports'] = [n for n in d['referenced_imports'] if n != victim]
        d['unreferenced_imports'] = sorted(d['unreferenced_imports'] + [victim])
        d['imports'] = [{**e, 'observed_reference': e['observed_reference'] and e['key'] != victim} for e in d['imports']]
    box.edit_json('imports.json', damage)
    box.edit_json('manifest.json', lambda d: d.update(referenced_imports=d['referenced_imports'] - 1,
                                                       unreferenced_imports=d['unreferenced_imports'] + 1))
    result = box.convert_imports()
    assert result['status'] == 'failed' and 'referenced by cache instructions' in result['error'], result
    assert victim in doc['referenced_imports']


def test_imports_command_line_counts(box, capsys):
    box.clone('a' * 64)
    box.edit_json('imports.json', lambda d: d.update(export_complete_verified=False), 'a' * 64)
    with pytest.raises(SystemExit) as stop:
        side_io.main('t', imp.convert, 'imports', ['--cache', str(box.cache), '--reader-root', str(box.reader)])
    assert stop.value.code == 1
    summary = json.loads(next(l for l in capsys.readouterr().out.splitlines() if l.startswith('{"cache"')))
    assert (summary['ok'], summary['failed'], summary['missing']) == (1, 1, 0)
