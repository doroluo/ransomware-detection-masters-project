"""The fixed-rule ensemble runner (family_holdout/run_ensemble.py).

Three things are worth a test: the two rules do what the docstring says, a
member ensembled with itself reproduces that member's own argmax decision
(so alignment and the 0.5 decision are right), and every row of a written
ensemble directory carries exactly the members' held-out rows in the fold
file's order (so no prediction was silently re-indexed).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from family_holdout import run_ensemble as E  # noqa: E402
from family_holdout.common import Folds  # noqa: E402

OUT = REPO / "results" / "family_holdout"
HAVE_MEMBERS = all((OUT / ds / p / m / "predictions.csv").is_file()
                   for ds in ("mendeley", "balanced") for p, m in E.MEMBERS)


def test_rules_are_mean_and_max():
    s = np.array([[0.2, 0.9, 0.5], [0.6, 0.1, 0.5]])
    assert np.allclose(E.combine(s, "mean"), [0.4, 0.5, 0.5])
    assert np.allclose(E.combine(s, "max"), [0.6, 0.9, 0.5])
    with pytest.raises(ValueError):
        E.combine(s, "median")
    with pytest.raises(ValueError):
        E.combine(np.array([0.1, 0.2]), "mean")


def test_decision_is_fixed_at_half():
    assert E.decide(np.array([0.49, 0.5, 0.51])).tolist() == [0, 1, 1]
    assert E.THRESHOLD == 0.5 and E.PRIMARY == "mean"


@pytest.mark.skipif(not HAVE_MEMBERS, reason="member result directories absent")
@pytest.mark.parametrize("ds", ["mendeley", "balanced"])
@pytest.mark.parametrize("member", list(E.MEMBERS))
def test_member_loads_aligned_and_self_ensemble_reproduces_it(ds, member):
    folds = Folds(ds)
    p, m = member
    score, lofo = E.load_member(folds, OUT, p, m)
    assert score.shape == (folds.n,)
    assert sum(len(v) for v in lofo.values()) == int((folds.y == 1).sum())
    own_kf, own_lofo = E._pred_of_member(folds, OUT, p, m)
    # both members decide at 0.5 on P(ransomware), so mean(x, x) must give
    # back their own pred column bit for bit
    self_pred = E.decide(E.combine(np.vstack([score, score]), "mean"))
    assert (self_pred == own_kf).all(), f"{ds} {p}/{m}: K-fold self-ensemble drifts"
    for fam, sc in lofo.items():
        assert (E.decide(E.combine(np.vstack([sc, sc]), "max")) == own_lofo[fam]).all()


@pytest.mark.skipif(not HAVE_MEMBERS, reason="member result directories absent")
@pytest.mark.parametrize("ds", ["mendeley", "balanced"])
def test_written_ensemble_rows_are_the_members_rows(ds):
    tag = "+".join(f"{p}_{m}".lower() for p, m in E.MEMBERS)
    d = OUT / ds / E.PIPELINE / f"{tag}_{E.PRIMARY}"
    if not (d / "predictions.csv").is_file():
        pytest.skip("run python family_holdout/run_ensemble.py first")
    folds = Folds(ds)
    members = [E.load_member(folds, OUT, p, m)[0] for p, m in E.MEMBERS]
    expect = E.combine(np.vstack(members), E.PRIMARY)
    with (d / "predictions.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["sha256"] for r in rows] == list(folds.sha)
    got = np.array([float(r["score"]) for r in rows])
    assert np.allclose(got, expect, atol=1e-9)
    assert [int(r["pred"]) for r in rows] == E.decide(expect).tolist()
