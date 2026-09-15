"""Fold-handling invariants of the family-holdout RUNNERS.

`test_family_holdout_folds.py` checks the fold FILE. This file checks what the
three runners in `family_holdout/` do with it: that the index sets they hand a
classifier are disjoint in every way the study claims, that the LOFO test set
is exactly one family with all goodware still in train, and that the two
places where a runner recomputes something the pipelines already do (the
graph2vec min_df column filter, the model-directory writer) agree with the
originals.

Everything heavy is imported lazily or skipped, so the file runs under the
system 3.14 interpreter even though the tokenization runner itself needs the
3.12 venv (gensim).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from family_holdout import common  # noqa: E402
from family_holdout.common import Folds, check_kfold, check_lofo  # noqa: E402

OUT = REPO / "results" / "family_holdout"
DATASETS = ["mendeley", "balanced"]
pytestmark = pytest.mark.skipif(
    not (OUT / "folds_mendeley.csv").is_file(),
    reason="run python family_holdout/folds.py --out results/family_holdout first")


@pytest.fixture(scope="module")
def folds():
    return {ds: Folds(ds) for ds in DATASETS}


# ---------------------------------------------------------------------------
# K-fold
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ds", DATASETS)
def test_kfold_partitions_every_file_exactly_once(folds, ds):
    f = folds[ds]
    seen = np.zeros(f.n, dtype=int)
    for _, tr, te in f.kfold():
        seen[te] += 1
        assert len(tr) + len(te) == f.n
    assert (seen == 1).all(), "a file is held out twice or never"


@pytest.mark.parametrize("ds", DATASETS)
def test_kfold_train_test_disjoint_by_sha_family_and_group(folds, ds):
    f = folds[ds]
    for fold, tr, te in f.kfold():
        assert not set(f.sha[tr]) & set(f.sha[te])
        fam_tr = {x for x, l in zip(f.family[tr], f.y[tr]) if l == 1}
        fam_te = {x for x, l in zip(f.family[te], f.y[te]) if l == 1}
        assert not fam_tr & fam_te, f"fold {fold}: family in both halves"
        g_tr = {x for x, l in zip(f.group[tr], f.y[tr]) if l == 0}
        g_te = {x for x, l in zip(f.group[te], f.y[te]) if l == 0}
        assert not g_tr & g_te, f"fold {fold}: goodware group split"


@pytest.mark.parametrize("ds", DATASETS)
def test_check_kfold_helper_agrees(folds, ds):
    rep = check_kfold(folds[ds])
    assert set(rep) == set(range(common.K))
    assert all(v["sha_overlap"] == 0 and not v["family_overlap"]
               and v["goodware_group_overlap"] == 0 for v in rep.values())


@pytest.mark.parametrize("ds", DATASETS)
def test_every_fold_has_both_classes_and_several_families(folds, ds):
    f = folds[ds]
    for fold, _, te in f.kfold():
        assert (f.y[te] == 0).sum() > 0 and (f.y[te] == 1).sum() > 0
        assert len({x for x, l in zip(f.family[te], f.y[te]) if l == 1}) >= 5


# ---------------------------------------------------------------------------
# LOFO
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ds", DATASETS)
def test_lofo_test_set_is_exactly_the_family(folds, ds):
    f = folds[ds]
    n_good = int((f.y == 0).sum())
    fams = set()
    for fam, tr, te in f.lofo():
        fams.add(fam)
        assert set(f.family[te]) == {fam}
        assert set(f.y[te]) == {1}
        assert len(te) == int(((f.family == fam) & (f.y == 1)).sum())
        assert fam not in set(f.family[tr])
        assert int((f.y[tr] == 0).sum()) == n_good, "goodware dropped from train"
        assert not set(f.sha[tr]) & set(f.sha[te])
        assert len(tr) + len(te) == f.n
    assert fams == set(f.families) and len(fams) == 38


@pytest.mark.parametrize("ds", DATASETS)
def test_check_lofo_helper_agrees(folds, ds):
    rep = check_lofo(folds[ds])
    assert len(rep) == 38
    assert sum(v["n_test"] for v in rep.values()) == int((folds[ds].y == 1).sum())


def test_lofo_families_identical_across_datasets(folds):
    assert folds["mendeley"].families == folds["balanced"].families


# ---------------------------------------------------------------------------
# the unsupervised-state reuse the tokenization and graph2vec runners rely on
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ds", DATASETS)
def test_family_lives_in_exactly_one_fold(folds, ds):
    """Both runners reuse fold f's tokenizer / Word2Vec / threshold for the
    LOFO run of a family in fold f. That is only sound if the family is wholly
    inside one fold, so that fold's training rows never contained it."""
    f = folds[ds]
    fold_of = f.fold_of_family()
    assert set(fold_of) == set(f.families)
    for fam, fold in fold_of.items():
        rows = np.flatnonzero((f.family == fam) & (f.y == 1))
        assert set(f.fold[rows]) == {fold}
        _, tr, _ = f.kfold()[fold]
        assert fam not in set(f.family[tr]), \
            "fold-f training rows contain a family assigned to fold f"


@pytest.mark.parametrize("ds", DATASETS)
def test_lofo_train_is_a_superset_of_that_folds_kfold_train(folds, ds):
    f = folds[ds]
    fold_of = f.fold_of_family()
    lofo = dict((fam, tr) for fam, tr, _ in f.lofo())
    for fam, fold in fold_of.items():
        _, ktr, _ = f.kfold()[fold]
        assert set(f.sha[ktr]) <= set(f.sha[lofo[fam]])


# ---------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------
def test_write_model_dir_round_trips(tmp_path, folds):
    """A synthetic perfect-on-x86 classifier, written and read back."""
    import json
    f = folds["mendeley"]
    rng = np.random.default_rng(0)
    score = np.where(f.arch == "x86", 0.9, 0.1) + rng.normal(0, 0.01, f.n)
    pred = (score >= 0.5).astype(int)
    lofo = {}
    for fam in f.families:
        idx = np.flatnonzero((f.family == fam) & (f.y == 1))
        lofo[fam] = (score[idx], pred[idx])
    d = tmp_path / "m"
    common.write_model_dir(d, f, "unit", "x86_rule", score, pred, f.fold,
                           lofo, {"note": "synthetic"}, 1.0)

    with (d / "predictions.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == f.n
    assert {r["sha256"] for r in rows} == set(f.sha)
    assert {int(r["fold"]) for r in rows} == set(range(common.K))

    with (d / "fold_metrics.csv").open(encoding="utf-8", newline="") as fh:
        fm = list(csv.DictReader(fh))
    assert [r["fold"] for r in fm] == [str(i) for i in range(common.K)]
    assert list(fm[0]) == common.FOLD_METRIC_COLS
    assert sum(int(r["n_test"]) for r in fm) == f.n
    # this classifier IS the x86 rule, so it must hit the x86-rule floor exactly
    for r in fm:
        assert abs(float(r["accuracy"]) - float(r["x86_rule_floor"])) < 1e-4

    with (d / "lofo_predictions.csv").open(encoding="utf-8", newline="") as fh:
        lp = list(csv.DictReader(fh))
    assert len(lp) == int((f.y == 1).sum())
    assert {r["family"] for r in lp} == set(f.families)

    with (d / "per_family.csv").open(encoding="utf-8", newline="") as fh:
        pf = list(csv.DictReader(fh))
    assert len(pf) == 38
    assert sum(int(r["n"]) for r in pf) == int((f.y == 1).sum())
    assert sum(int(r["n_x64"]) for r in pf) == int(((f.y == 1)
                                                    & (f.arch == "x64")).sum())

    doc = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    res = doc["results"][0]
    assert doc["n_configs"] == 1 and res["n_configs"] == 1
    assert set(res["fold_mean"]) == set(common.FOLD_MEAN_KEYS)
    assert set(res["fold_sd"]) == set(common.FOLD_MEAN_KEYS)
    assert "per_family_recall" in res and "per_arch" in res
    assert 0.0 <= res["lofo_mean_recall"] <= 1.0
    assert (d / "config_used.yaml").is_file()
    # the x86 rule detects every x86 ransomware file and no x64 one
    assert res["per_arch"]["x86"]["recall_ransomware"] == 1.0
    assert res["per_arch"]["x64"]["recall_ransomware"] == 0.0


def test_floor_blocks_match_the_trivial_rules(folds):
    f = folds["mendeley"]
    fl = common.floor_blocks(f.y, f.arch)
    maj = max((f.y == 0).mean(), (f.y == 1).mean())
    assert abs(fl["majority"]["accuracy"] - maj) < 1e-4
    x86 = ((f.arch == "x86").astype(int) == f.y).mean()
    assert abs(fl["x86_rule"]["accuracy"] - x86) < 1e-4


# ---------------------------------------------------------------------------
# runner-specific helpers
# ---------------------------------------------------------------------------
def test_tfidf_runner_settings_match_the_rules_baseline():
    from family_holdout import run_tfidf
    src = (REPO / "rules_pipeline" / "train_eval.py").read_text(encoding="utf-8")
    assert run_tfidf.MAX_MNEMS == 30_000 and "MAX_MNEMS = 30_000" in src
    assert run_tfidf.NGRAM == (1, 3) and "ngram_range=(1, 3)" in src
    assert run_tfidf.MIN_DF == 5 and "min_df=5" in src
    assert run_tfidf.MAX_FEATURES == 300_000 and "max_features=300_000" in src
    assert run_tfidf.C_GRID == [0.1, 1, 10] and '"C": [0.1, 1, 10]' in src
    v = run_tfidf.make_vec()
    assert v.sublinear_tf and v.token_pattern == r"\S+"
    assert v.get_params()["class_weight"] is None if hasattr(v, "class_weight") \
        else True


def test_tokenization_runner_settings_match_the_config():
    import yaml
    from family_holdout import run_tokenization as R
    cfg = yaml.safe_load(
        (REPO / "llm_features_pipeline" / "config.yaml").read_text(
            encoding="utf-8"))
    assert R.MAX_INSTRUCTIONS == cfg["tokenization"]["max_instructions"]
    assert R.VOCAB_SIZE == cfg["tokenization"]["vocab_size"]
    assert R.CV == cfg["classification"]["cv"]
    assert R.SEED == cfg["classification"]["seed"]
    w = cfg["embedding"]["w2v"]
    for k in ("vector_size", "window", "epochs", "seed", "workers"):
        assert R.W2V[k] == w[k], k
    assert R.W2V["workers"] == 1, "workers must stay 1 or the run is not seeded"
    assert {t for _, t in R.COMBOS} <= set(cfg["tokenization"]["methods"])


def test_graph2vec_runner_loads_both_fixed_configs():
    from family_holdout import run_graph2vec as R
    got = {name: (c, r, m) for name, c, r, m, _ in R.load_configs()}
    assert set(got) == set(R.WANT)
    c, r, m = got["baseline_h2_cap5000"]
    assert (c.max_blocks, c.windows, c.extern, c.cross_section) == (5000, 1, False, False)
    assert (r.kind, r.h, r.labelling, r.min_df) == ("wl_tfidf", 2, "class", 5)
    assert m.name == "LR" and dict(m.params)["C"] == 1.0
    c, r, m = got["tuned_best_sparse_histogram"]
    assert (c.max_blocks, c.extern) == (5000, True)
    assert (r.kind, r.h, r.labelling, r.min_df) == ("wl_tfidf", 4, "lenbucket", 3)
    assert dict(m.params) == {"C": 10.0, "class_weight": "balanced"}


def test_graph2vec_restrict_matches_the_pipeline_column_filter():
    """`restrict` exists only to avoid rebuilding the 2**28-wide count matrix
    per fold; it must select exactly the columns wl.layer_matrix would."""
    from scipy import sparse
    from family_holdout.run_graph2vec import restrict
    from graph2vec_pipeline import wl
    rng = np.random.default_rng(7)
    n_graphs, n_nodes = 40, 4000
    gid = np.sort(rng.integers(0, n_graphs, n_nodes)).astype(np.int32)
    lab = rng.integers(0, 500, n_nodes).astype(np.uint64)
    train = np.zeros(n_graphs, dtype=bool)
    train[: n_graphs // 2] = True
    for min_df in (1, 3, 5):
        ref = wl.layer_matrix(gid, lab, n_graphs, train, min_df)
        got = restrict(wl.layer_matrix(gid, lab, n_graphs, None, 0), train,
                       min_df)
        assert got.shape == ref.shape
        assert (got != ref).nnz == 0


# ---------------------------------------------------------------------------
# committed outputs, when they are there
# ---------------------------------------------------------------------------
def _model_dirs():
    out = []
    for ds in DATASETS:
        for pipe in ("tfidf", "tokenization", "graph2vec"):
            p = OUT / ds / pipe
            if p.is_dir():
                out += [(ds, pipe, d) for d in sorted(p.iterdir()) if d.is_dir()]
    return out


@pytest.mark.parametrize("case", _model_dirs(),
                         ids=lambda c: f"{c[0]}/{c[1]}/{c[2].name}")
def test_written_model_dir_is_complete_and_consistent(case, folds):
    import json
    ds, pipe, d = case
    f = folds[ds]
    for name in ("predictions.csv", "fold_metrics.csv", "per_family.csv",
                 "lofo_predictions.csv", "metrics.json", "config_used.yaml"):
        assert (d / name).is_file(), f"{d} missing {name}"
    with (d / "predictions.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == f.n
    sha2fold = dict(zip(f.sha, f.fold))
    sha2lab = dict(zip(f.sha, f.y))
    for r in rows:
        assert int(r["fold"]) == sha2fold[r["sha256"]]
        assert int(r["label"]) == sha2lab[r["sha256"]]
    with (d / "lofo_predictions.csv").open(encoding="utf-8", newline="") as fh:
        lp = list(csv.DictReader(fh))
    assert len(lp) == int((f.y == 1).sum())
    doc = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    assert doc["n_configs"] == 1
    assert len(doc["results"]) == 1
    assert "fold_mean" in doc["results"][0]
