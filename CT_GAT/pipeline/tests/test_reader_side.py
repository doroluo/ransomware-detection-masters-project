"""python -m pytest tests/test_reader_side.py: the converter of the team's operand-role reader output to `side/yanping`
(`experiments/side_arrays/import_reader_side.py`), on one real reader output (tests/fixtures/reader_side: a 641-instruction
binary, its reader files copied byte for byte, and its cache sample with the vocabulary shrunk) and deliberately misaligned
copies of it.

Ported from the 3-block-level-counting branch (tests/test_reader_side.py, the operand-role half; the imports half is
test_import_side.py). The fixture directory is the same."""
import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from common import side_io
from experiments.side_arrays import import_reader_side as yan

FIX = Path(__file__).parent / 'fixtures' / 'reader_side'
SID = next((FIX / 'reader').iterdir()).name


class Box:
    """A scratch copy of the fixture: `cache`, `reader` (the reader root) and helpers to derive a misaligned sample."""

    def __init__(self, tmp):
        self.cache, self.reader = tmp / 'cache', tmp / 'reader'
        shutil.copytree(FIX / 'cache', self.cache)
        shutil.copytree(FIX / 'reader', self.reader)

    def rows(self, sid=SID):
        with gzip.open(self.reader / sid / 'operand_roles.jsonl.gz', 'rt', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    def put_rows(self, rows, sid=SID):
        with gzip.open(self.reader / sid / 'operand_roles.jsonl.gz', 'wt', encoding='utf-8') as f:
            f.writelines(json.dumps(r) + '\n' for r in rows)

    def edit_json(self, name, fn, sid=SID):
        path = self.reader / sid / name
        doc = json.loads(path.read_text(encoding='utf-8'))
        fn(doc)
        path.write_text(json.dumps(doc), encoding='utf-8')

    def clone(self, new_id):
        """The same cache sample and reader output under another sample id (headers rewritten), to misalign separately."""
        shutil.copyfile(self.cache / 'samples' / f'{SID}.npz', self.cache / 'samples' / f'{new_id}.npz')
        shutil.copytree(self.reader / SID, self.reader / new_id)
        rows = self.rows(new_id)
        rows[0]['sample_id'] = new_id
        self.put_rows(rows, new_id)
        for name in ('imports.json', 'manifest.json'):
            self.edit_json(name, lambda d: d.update(sample_id=new_id), new_id)
        with (self.cache / 'index.csv').open('a', encoding='utf-8', newline='') as f:
            f.write(f'{new_id},ok\n')
        return new_id

    def convert_yanping(self, sid=SID, **kw):
        return yan.convert(self.cache, sid, self.reader, **kw)


@pytest.fixture()
def box(tmp_path):
    side_io.cache_vocab.cache_clear()
    yan.read_operand_rows.cache_clear()
    return Box(tmp_path)


def instruction_rows(rows):
    return [r for r in rows if r['record'] == 'instruction']


def with_counts(box, n):
    """Make the footer and manifest say `n` instructions, so only the alignment against the cache can object."""
    rows = box.rows()
    rows[-1]['counts']['instructions'] = n
    box.put_rows(rows)
    box.edit_json('manifest.json', lambda d: d['counts'].update(instructions=n))


# ---- operand roles -> side/yanping ---------------------------------------------------------------------------------

def test_real_reader_output_converts_to_cache_aligned_ytok(box):
    result = box.convert_yanping()
    assert result['status'] == 'ok'
    ytok = yan.load_ytok(box.cache, SID)
    with np.load(box.cache / 'samples' / f'{SID}.npz') as z:
        ptr, mn = z['blk_insn_ptr'], z['mn']
    reader = np.array([r['tokens'] for r in instruction_rows(box.rows())])
    assert ytok.dtype == np.int16 and ytok.shape == (len(mn), 3) == reader.shape
    assert (ytok == reader).all()
    rows = instruction_rows(box.rows())
    for b in (0, 7, len(ptr) - 2):                                                 # block b holds its own instructions
        assert [r['tokens'] for r in rows[ptr[b]:ptr[b + 1]]] == ytok[ptr[b]:ptr[b + 1]].tolist()
        assert {r['block_index'] for r in rows[ptr[b]:ptr[b + 1]]} <= {b}
    assert ytok[:, 0].max() < 256 and ytok[:, 1:].max() < 76
    with np.load(box.cache / 'side' / 'yanping' / f'{SID}.npz', allow_pickle=False) as z:
        assert str(z['schema']) == yan.SCHEMA and str(z['sample_id']) == SID and z['ytok_sizes'].tolist() == [256, 76, 76]
        prov = json.loads(str(z['provenance_json']))
        assert prov['reader_operand_roles_sha256'] == side_io.sha256_file(box.reader / SID / 'operand_roles.jsonl.gz')
        assert prov['cache_sha256'] == side_io.sha256_file(box.cache / 'samples' / f'{SID}.npz')
        assert json.loads(str(z['reader_json']))['counts']['instructions'] == len(mn)


def test_conversion_is_idempotent_and_refuses_stale_output(box):
    assert box.convert_yanping()['status'] == 'ok'
    assert box.convert_yanping()['status'] == 'skipped'
    rows = box.rows()
    rows[0]['token_contract'] = rows[0]['token_contract'] + ' '                    # any change of the reader file is a new input
    box.put_rows(rows)
    stale = box.convert_yanping()
    assert stale['status'] == 'failed' and 'stale' in stale['error']
    assert box.convert_yanping(replace_stale=True)['status'] == 'ok'


def test_missing_reader_output_is_missing_not_failed(box):
    assert box.convert_yanping('0' * 64)['status'] == 'missing'
    (box.reader / ('1' * 64 + '.pending-abc')).mkdir()
    assert box.convert_yanping('1' * 64)['status'] == 'missing'
    (box.reader / SID / 'imports.json').unlink()
    assert box.convert_yanping()['status'] == 'missing'


def drop_row(box):
    rows = box.rows()
    i = next(k for k, r in enumerate(rows) if r['record'] == 'instruction' and r['instruction_index'] == 100)
    del rows[i]
    box.put_rows(rows)


def drop_row_with_counts(box):
    drop_row(box)
    with_counts(box, len(instruction_rows(box.rows())))


def shift_block_boundary(box):
    """The last instruction of a block belongs to the next one: the same rows and counts overall, other blocks' sizes."""
    rows = box.rows()
    with np.load(box.cache / 'samples' / f'{SID}.npz') as z:
        ptr = z['blk_insn_ptr']
    b = next(k for k in range(len(ptr) - 2) if ptr[k + 1] - ptr[k] > 1 and ptr[k + 2] > ptr[k + 1])
    row = instruction_rows(rows)[ptr[b + 1] - 1]
    row['block_index'] = b + 1
    box.put_rows(rows)


def move_function(box):
    rows = box.rows()
    row = instruction_rows(rows)[300]
    row['fn_index'] += 1
    box.put_rows(rows)


def change_opcode(box):
    """Same counts and layout, but one instruction is another mnemonic for the reader than in the cache."""
    rows = box.rows()
    with np.load(box.cache / 'samples' / f'{SID}.npz') as z:
        mn = z['mn']
    vocab = json.loads((box.cache / 'vocab.json').read_text(encoding='utf-8'))['mn']
    ins = instruction_rows(rows)
    target = next(i for i, r in enumerate(ins) if vocab[mn[i]] != 'call' and sum(mn == mn[i]) > 3)
    ins[target]['tokens'][0] = (ins[target]['tokens'][0] + 1) % 256
    box.put_rows(rows)


def set_first(key, value):
    def fn(box):
        rows = box.rows()
        rows[0][key] = value
        box.put_rows(rows)
    return fn


def set_last(key, value):
    def fn(box):
        rows = box.rows()
        rows[-1][key] = value
        box.put_rows(rows)
    return fn


def big_operand(box):
    rows = box.rows()
    instruction_rows(rows)[5]['tokens'][1] = 99
    box.put_rows(rows)


def orphan_row(box):
    rows = box.rows()
    instruction_rows(rows)[5]['orphan'] = True
    box.put_rows(rows)


def other_function_count(box):
    box.edit_json('imports.json', lambda d: d['counts'].update(functions=d['counts']['functions'] + 1))


MISALIGNED = [
    (drop_row, 'footer'),                                     # a row lost: the footer no longer counts it
    (drop_row_with_counts, 'reader instructions'),            # the counts repaired: the cache has one more
    (shift_block_boundary, 'instruction counts per block differ'),
    (move_function, 'function'),
    (change_opcode, 'more than one reader opcode id'),
    (set_first('sample_id', 'f' * 64), 'another sample'),
    (set_first('include_orphans', True), 'orphan'),
    (set_first('schema', 'ida-operand-roles/2'), 'schema'),
    (set_last('complete', False), 'complete footer'),
    (big_operand, 'slot ranges'),
    (orphan_row, 'orphan'),
    (other_function_count, 'functions, blocks, instructions'),
]


@pytest.mark.parametrize('damage,expected', MISALIGNED, ids=[d.__name__ for d, _ in MISALIGNED])
def test_misaligned_copy_fails_and_writes_nothing(box, damage, expected):
    damage(box)                                                                    # the box is a copy of the real output
    result = box.convert_yanping()
    assert result['status'] == 'failed', result
    assert expected in result['error'], result['error']
    assert not (box.cache / 'side' / 'yanping' / f'{SID}.npz').exists()


def test_misalignment_is_listed_with_the_counts_by_the_command_line(box, capsys):
    bad = box.clone('c' * 64)
    rows = box.rows(bad)
    ins = instruction_rows(rows)
    ins[10]['tokens'][0], ins[11]['tokens'][0] = ins[11]['tokens'][0], ins[10]['tokens'][0]   # same counts, other content
    ins[10]['block_index'] += 1                                                                 # and a block boundary moved
    box.put_rows(rows, bad)
    gone = 'd' * 64
    with (box.cache / 'index.csv').open('a', encoding='utf-8', newline='') as f:
        f.write(f'{gone},ok\n')                                                    # in the cache index, no reader output
    with pytest.raises(SystemExit) as stop:
        side_io.main('t', yan.convert, 'yanping', ['--cache', str(box.cache), '--reader-root', str(box.reader)])
    assert stop.value.code == 1
    captured = capsys.readouterr()
    summary = json.loads(next(l for l in captured.out.splitlines() if l.startswith('{"cache"')))
    assert (summary['ok'], summary['failed'], summary['missing'], summary['total']) == (1, 1, 1, 3)
    assert f'FAILED {bad}' in captured.err and bad in captured.out
    audit = json.loads(next((box.cache / 'side' / 'yanping').glob('_build-*.json')).read_text(encoding='utf-8'))
    assert [f['sample_id'] for f in audit['failures']] == [bad] and audit['missing'] == 1
    assert (box.cache / 'side' / 'yanping' / f'{SID}.npz').exists() and not (box.cache / 'side' / 'yanping' / f'{bad}.npz').exists()


def test_loader_names_the_sample_when_the_side_array_is_missing(box):
    with pytest.raises(FileNotFoundError, match=SID):
        yan.load_ytok(box.cache, SID)
