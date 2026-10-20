"""The operand-role side arrays (`side/yanping`) of the tiny cache, for the tests of `--vocab yanping|hybrid`. They are written in
a separate step, from a generator of their own, so that the cache arrays of tests/tiny_cache.py are the same with or without them.
Not a test module: a test module that uses `tiny_yanping` imports `tiny` from tiny_cache as well (fixtures are looked up there)."""
import numpy as np
import pytest

from common.side_io import atomic_npz
from experiments.side_arrays import import_reader_side as yan


def add_yanping(root, rows, seed=3):
    """Write `<root>/side/yanping/<sample_id>.npz` for every row of the tiny cache: ytok [n_insn, 3] int16, opcode ids that
    separate the classes, operand ids below 70 except one id (75) seen in the first binary only."""
    for i, row in enumerate(rows):
        rng = np.random.default_rng([seed, 2, i])
        n_ins, label = int(row['num_instructions']), int(row['label'])
        ytok = np.stack([rng.integers(0, 256, n_ins), rng.integers(0, 76, n_ins), rng.integers(0, 76, n_ins)], 1)
        ytok[:, 1] = ytok[:, 1] % 70
        ytok[:3, 1] = 75 if i == 0 else ytok[:3, 1]                                     # a reader id seen in one binary only
        ytok[:, 0] = np.where(label == 1, ytok[:, 0] % 40, 100 + ytok[:, 0] % 100)      # opcode ids separate the classes
        atomic_npz(yan.yanping_path(root, row['sample_id']),
                   dict(schema=np.array(yan.SCHEMA), sample_id=np.array(row['sample_id']), ytok=ytok.astype(np.int16)))


@pytest.fixture(scope='module')
def tiny_yanping(tiny):
    """The tiny cache of this module with its side/yanping arrays."""
    add_yanping(tiny.root, tiny.rows)
    return tiny
