"""family_holdout/run_imports_baseline.py: the hashing must be the transformer's
own, the name documents must be one token per import, and the config list must
be the three the docstring promises."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from family_holdout import run_imports_baseline as B  # noqa: E402


def test_hash_matches_the_sequence_model_when_torch_is_present():
    try:
        from seq_model import data as D
    except Exception:
        pytest.skip("seq_model.data needs torch")
    names = ["kernel32.dll!createfilew", "ADVAPI32.dll!CryptEncrypt ", "user32.dll!#123"]
    assert np.array_equal(B.hash_imports(names), D.hash_imports(names, B.HASH_DIM))
    assert np.array_equal(B.hash_imports([]), np.zeros(B.HASH_DIM, dtype=np.float32))


def test_hash_is_case_insensitive_unit_norm_and_signed():
    a = B.hash_imports(["Kernel32.dll!CreateFileW"])
    b = B.hash_imports(["kernel32.dll!createfilew"])
    assert np.array_equal(a, b)
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-6
    assert set(np.unique(a)).issubset({-1.0, 0.0, 1.0})


def test_names_text_is_one_token_per_import():
    assert B.names_text(["a.dll!x", "b.dll!y z"]).split() == ["a.dll!x", "b.dll!y_z"]
    assert B.names_text([]) == ""


def test_configs_are_the_three_documented():
    assert B.CONFIGS == ("hashed2048", "names_tfidf", "mnem+names")
    assert B.PIPELINE == "imports_baseline"
