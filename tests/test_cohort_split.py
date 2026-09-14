"""Tests for the shared cohort/split logic in cnn_vit_pipeline/cohort.py.

Both evaluation harnesses (EMBER and CNN-ViT) score on the same samples and
the same split, so the guarantees that make the numbers comparable are
asserted here rather than trusted:

  * family disjointness between train and test (the paper's protocol)
  * no sha256 in more than one fold
  * group disjointness once val is carved out of train
  * the counts actually match the cohort CSVs and expB's membership
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnn_vit_pipeline import cohort  # noqa: E402

pytestmark = pytest.mark.skipif(
    not cohort.COHORT_MENDELEY.exists() or not cohort.COHORT_BALANCED.exists(),
    reason="cohort CSVs not present (set RANSOM_SHARED_DIR)")


@pytest.fixture(scope="module")
def mendeley():
    return cohort.load_split("mendeley")


@pytest.fixture(scope="module")
def balanced():
    return cohort.load_split("balanced")


# ---------------------------------------------------------------- shape ----
@pytest.mark.parametrize("dataset", ["mendeley", "balanced"])
def test_columns_and_types(dataset):
    df = cohort.load_split(dataset)
    assert list(df.columns) == ["sha256", "label", "family_or_group", "split",
                                "arch", "source", "filename", "family"]
    assert set(df["split"]) == {"train", "test"}
    assert set(df["label"]) <= {0, 1}
    assert set(df["arch"]) <= {"x86", "x64"}
    assert df["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()


@pytest.mark.parametrize("dataset", ["mendeley", "balanced"])
def test_no_sha_in_both_splits(dataset):
    df = cohort.load_split(dataset)
    assert not df["sha256"].duplicated().any()
    train = set(df.loc[df["split"] == "train", "sha256"])
    test = set(df.loc[df["split"] == "test", "sha256"])
    assert train & test == set()


# --------------------------------------------------------- cohort filter ----
@pytest.mark.parametrize("dataset", ["mendeley", "balanced"])
def test_only_in_cohort_rows_survive(dataset):
    df = cohort.load_split(dataset)
    allowed = set(cohort.load_cohort("mendeley")["sha256"])
    if dataset == "balanced":
        allowed |= set(cohort.load_cohort("balanced")["sha256"])
    assert set(df["sha256"]) <= allowed


def test_excluded_tags_are_really_gone():
    """Excluded samples must not reappear in the split.

    `tag:dup` is the one exclude_reason that can repeat a sha256 that is also
    kept: the corpus holds the same bytes under several filenames, so the
    cohort marks one row in_cohort=1 and the rest tag:dup. Those shas are
    legitimately present exactly once. Every other exclusion is absolute.
    """
    raw = cohort._read_cohort(cohort.COHORT_MENDELEY)
    kept = set(raw.loc[raw["in_cohort"] == "1", "sha256"])
    hard_excluded = set(raw.loc[(raw["in_cohort"] != "1") &
                                (raw["exclude_reason"] != "tag:dup"), "sha256"])
    assert hard_excluded, "expected some excluded rows in the cohort CSV"
    assert hard_excluded & kept == set()

    df = cohort.load_split("mendeley")
    assert set(df["sha256"]) & hard_excluded == set()
    # and a dup sha appears once, not twice
    dup_shas = set(raw.loc[raw["exclude_reason"] == "tag:dup", "sha256"]) & kept
    assert dup_shas, "expected tag:dup rows sharing a sha with a kept row"
    for sha in dup_shas:
        assert int((df["sha256"] == sha).sum()) == 1


def test_no_thanos_anywhere():
    for dataset in cohort.DATASETS:
        df = cohort.load_split(dataset)
        assert "thanos" not in {str(f).lower() for f in df["family"]}


# --------------------------------------------------- family disjointness ----
@pytest.mark.parametrize("dataset", ["mendeley", "balanced"])
def test_ransomware_families_are_disjoint(dataset):
    df = cohort.load_split(dataset)
    mal = df[df["label"] == 1]
    train_fams = set(mal.loc[mal["split"] == "train", "family"])
    test_fams = set(mal.loc[mal["split"] == "test", "family"])
    assert train_fams & test_fams == set()
    # 25/15 in the paper; Night Sky and Thanos vanish under the cohort filter
    assert len(train_fams) == 24
    assert len(test_fams) == 14


def test_both_datasets_share_the_same_ransomware(mendeley, balanced):
    a = mendeley[mendeley["label"] == 1].set_index("sha256")["split"].sort_index()
    b = balanced[balanced["label"] == 1].set_index("sha256")["split"].sort_index()
    assert a.equals(b), "ransomware side must be identical across datasets"


# ------------------------------------------------------ mendeley counts ----
def test_mendeley_counts_match_cohort_csv(mendeley):
    raw = cohort.load_cohort("mendeley")
    for set_name, split, label in [("good_train", "train", 0), ("mal_train", "train", 1),
                                   ("good_test", "test", 0), ("mal_test", "test", 1)]:
        expected = int((raw["set"] == set_name).sum())
        got = int(((mendeley["split"] == split) & (mendeley["label"] == label)).sum())
        assert got == expected, f"{set_name}: {got} != {expected}"
    assert len(mendeley) == len(raw) == 2509


# ------------------------------------------------------ balanced counts ----
def test_balanced_goodware_comes_from_expb(balanced):
    audit = cohort.balanced_goodware_audit()
    assert audit["unmapped_to_sha256"] == [], "every expB row must map to a sha256"
    assert audit["expb_train"] == 1116 and audit["expb_test"] == 131
    good = balanced[balanced["label"] == 0]
    assert int((good["split"] == "train").sum()) == audit["survived_train"]
    assert int((good["split"] == "test").sum()) == audit["survived_test"]
    # attrition is small but real, and must be reported, not silently absorbed
    assert audit["survived_train"] < 1116 or audit["survived_test"] < 131
    assert audit["dropped_by_cohort"] == (1116 - audit["survived_train"]) + \
                                         (131 - audit["survived_test"])


def test_balanced_goodware_groups_are_disjoint(balanced):
    good = balanced[balanced["label"] == 0]
    train_g = set(good.loc[good["split"] == "train", "family_or_group"])
    test_g = set(good.loc[good["split"] == "test", "family_or_group"])
    assert train_g & test_g == set(), "a source project straddles train/test"


def test_balanced_goodware_sizes_track_mendeley(balanced, mendeley):
    """Sized to match dataset 1's goodware, up to cohort attrition."""
    mg = mendeley[mendeley["label"] == 0]
    bg = balanced[balanced["label"] == 0]
    for split in ("train", "test"):
        m = int((mg["split"] == split).sum())
        b = int((bg["split"] == split).sum())
        assert abs(m - b) <= 5, f"{split}: mendeley {m} vs balanced {b}"


# ------------------------------------------------------------- val fold ----
@pytest.mark.parametrize("dataset", ["mendeley", "balanced"])
def test_val_fold_is_group_aware_and_stratified(dataset):
    df = cohort.add_val_fold(cohort.load_split(dataset))
    assert set(df["fold"]) == {"train", "val", "test"}

    # test rows are untouched
    assert (df.loc[df["split"] == "test", "fold"] == "test").all()
    assert (df.loc[df["fold"] == "val", "split"] == "train").all()

    groups = {f: set(df.loc[df["fold"] == f, "family_or_group"]) for f in
              ("train", "val", "test")}
    assert groups["train"] & groups["val"] == set()
    assert groups["train"] & groups["test"] == set()
    assert groups["val"] & groups["test"] == set()

    # both classes present in val, and val is roughly the requested size
    val = df[df["fold"] == "val"]
    assert set(val["label"]) == {0, 1}
    n_train_total = int((df["split"] == "train").sum())
    assert 0.05 <= len(val) / n_train_total <= 0.20


@pytest.mark.parametrize("dataset", ["mendeley", "balanced"])
def test_val_fold_is_deterministic(dataset):
    df = cohort.load_split(dataset)
    a = cohort.add_val_fold(df)["fold"]
    b = cohort.add_val_fold(df)["fold"]
    assert a.equals(b)
    c = cohort.add_val_fold(df, seed=cohort.VAL_SEED + 1)["fold"]
    assert not a.equals(c), "a different seed should give a different val fold"


# -------------------------------------------------------------- metrics ----
def test_core_metrics_against_a_known_confusion_matrix():
    y_true = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
    m = cohort._core_metrics(y_true, y_pred)
    assert m["confusion_matrix"] == {"tn": 3, "fp": 1, "fn": 2, "tp": 4}
    assert m["accuracy"] == pytest.approx(0.7)
    assert m["recall_ransomware"] == pytest.approx(4 / 6)
    assert m["recall_goodware"] == pytest.approx(3 / 4)
    assert m["false_positive_rate"] == pytest.approx(1 / 4)
    assert m["balanced_accuracy"] == pytest.approx((3 / 4 + 4 / 6) / 2)
    assert m["majority_class_accuracy"] == pytest.approx(0.6)


# Keys in results/expA/metrics.json that belong to the tokenization pipeline
# alone, so this harness is not expected to produce them. Everything else in
# expA's result block is a generic metric the three pipelines must share -
# if this test starts failing, expA grew a metric and cohort.build_result
# needs to grow it too, rather than the list below getting longer.
EXPA_PIPELINE_SPECIFIC = {"best_params", "model", "tokenizer", "embedding",
                          "mask_rate", "n_train", "n_test"}


def test_build_result_has_the_expA_keys_plus_breakdowns():
    expA = REPO_ROOT / "results" / "expA" / "metrics.json"
    y_true = [0, 0, 1, 1, 1, 1]
    y_pred = [0, 1, 1, 1, 0, 1]
    score = [0.1, 0.7, 0.9, 0.8, 0.2, 0.95]
    arch = ["x86", "x64", "x86", "x64", "x86", "x64"]
    fam = ["root", "root", "conti", "conti", "hive", "hive"]
    res = cohort.build_result(y_true, y_pred, score, arch, fam, model="test")

    if expA.exists():
        import json
        required = set(json.loads(expA.read_text())["results"][0]) \
            - EXPA_PIPELINE_SPECIFIC
        missing = sorted(required - set(res))
        assert not missing, (
            f"results/expA/metrics.json carries generic metric(s) {missing} "
            f"that cohort.build_result does not emit; the pipelines' metrics "
            f"would no longer be directly comparable.")

    assert set(res["per_architecture"]) == {"x86", "x64"}
    assert res["per_architecture"]["x86"]["n"] == 3
    assert set(res["per_family_recall"]) == {"conti", "hive"}
    assert res["per_family_recall"]["conti"] == {"recall": 1.0, "correct": 2,
                                                 "support": 2}
    assert res["per_family_recall"]["hive"] == {"recall": 0.5, "correct": 1,
                                                "support": 1 + 1}
    assert res["roc_auc"] is not None


def test_split_summary_reports_no_group_overlap():
    df = cohort.add_val_fold(cohort.load_split("mendeley"))
    s = cohort.split_summary(df, "fold")
    assert all(v == [] for v in s["group_overlap"].values())
    assert s["total"] == s["train"]["n"] + s["val"]["n"] + s["test"]["n"]
