"""Invariants of the CNN-ViT family-holdout runner (family_holdout/run_cnn_vit.py).

Two halves:

* the real fold definition in results/family_holdout/folds_*.csv is put through
  plan_kfold / plan_lofo and every split invariant the study rests on is
  asserted - no sha256 and no ransomware family in two of train/val/test, no
  goodware group straddling train and test, every sample held out exactly once
  across the five folds, and a LOFO test set that is exactly the held-out
  family;
* a miniature synthetic dataset (24 ransomware files in 6 families, 30 goodware
  files in 10 groups, with stand-in PNG/mask pairs) exercises the parts that
  touch disk: the hard-linked tree, the manifest, resumability, and the shape
  of every file written under results/.

Training itself is stubbed out: the recipe is train_eval.py's and is tested
there. torch is only needed by the one test that opens the materialised tree
with the original dataset class, which is skipped when torch is absent.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

pd = pytest.importorskip("pandas")
from family_holdout import run_cnn_vit as R  # noqa: E402

REAL = REPO / "results" / "family_holdout"
needs_folds = pytest.mark.skipif(
    not (REAL / "folds_mendeley.csv").is_file(),
    reason="run python family_holdout/folds.py first")
DATASETS = ["mendeley", "balanced"]


# ---------------------------------------------------------------------------
# the real fold definition
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_tables():
    if not (REAL / "folds_mendeley.csv").is_file():
        pytest.skip("no folds_*.csv")
    return {ds: R.load_fold_table(REAL, ds) for ds in DATASETS}


def _parts(plan):
    return {f: plan[plan["fold"] == f] for f in ("train", "val", "test")}


@needs_folds
@pytest.mark.parametrize("ds", DATASETS)
def test_kfold_train_val_test_are_disjoint_by_sha_and_family(real_tables, ds):
    tbl = real_tables[ds]
    for k in sorted(tbl["kfold"].unique()):
        plan = R.plan_kfold(tbl, k)
        p = _parts(plan)
        # every row lands in exactly one of the three
        assert len(plan) == len(tbl)
        assert sum(len(v) for v in p.values()) == len(tbl)
        shas = {f: set(v["sha256"]) for f, v in p.items()}
        assert shas["train"] & shas["val"] == set()
        assert shas["train"] & shas["test"] == set()
        assert shas["val"] & shas["test"] == set()
        fams = {f: set(v.loc[v["label"] == 1, "family"]) for f, v in p.items()}
        assert fams["train"] & fams["test"] == set()
        assert fams["val"] & fams["test"] == set()
        assert fams["train"] & fams["val"] == set()
        # the test fold is exactly the families folds.py assigned to it
        assert fams["test"] == set(tbl.loc[(tbl["kfold"] == k) & (tbl["label"] == 1),
                                           "family"])


@needs_folds
@pytest.mark.parametrize("ds", DATASETS)
def test_goodware_groups_never_straddle_train_and_test(real_tables, ds):
    tbl = real_tables[ds]
    for k in sorted(tbl["kfold"].unique()):
        p = _parts(R.plan_kfold(tbl, k))
        grp = {f: set(v.loc[v["label"] == 0, "group"]) for f, v in p.items()}
        assert grp["train"] & grp["test"] == set()
        assert grp["val"] & grp["test"] == set()
        assert grp["train"] & grp["val"] == set()
        # and the val fold really is carved out of train, not out of test
        assert len(p["test"]) == int((tbl["kfold"] == k).sum())


@needs_folds
@pytest.mark.parametrize("ds", DATASETS)
def test_every_sample_is_held_out_exactly_once_over_the_five_folds(real_tables, ds):
    tbl = real_tables[ds]
    seen = []
    for k in sorted(tbl["kfold"].unique()):
        seen.extend(_parts(R.plan_kfold(tbl, k))["test"]["sha256"].tolist())
    assert len(seen) == len(tbl)
    assert set(seen) == set(tbl["sha256"])


@needs_folds
@pytest.mark.parametrize("ds", DATASETS)
def test_lofo_test_set_is_exactly_the_held_out_family(real_tables, ds):
    tbl = real_tables[ds]
    families = sorted(tbl.loc[tbl["label"] == 1, "family"].unique())
    assert len(families) == 38
    goodware = set(tbl.loc[tbl["label"] == 0, "sha256"])
    for fam in families:
        p = _parts(R.plan_lofo(tbl, fam))
        want = set(tbl.loc[(tbl["label"] == 1) & (tbl["family"] == fam), "sha256"])
        assert set(p["test"]["sha256"]) == want
        assert set(p["test"]["family"]) == {fam}
        assert (p["test"]["label"] == 1).all()          # no goodware in a LOFO test
        # training sees every other family, and all goodware minus the val carve
        trained = set(p["train"]["family"]) | set(p["val"]["family"])
        assert trained >= set(families) - {fam}
        assert fam not in trained
        assert set(p["train"]["sha256"]) | set(p["val"]["sha256"]) >= goodware - set()
        assert goodware & set(p["test"]["sha256"]) == set()


@needs_folds
def test_check_plan_catches_a_leak(real_tables):
    tbl = real_tables["mendeley"]
    plan = R.plan_kfold(tbl, 0)
    R.check_plan(plan, "kfold", 0)                      # the honest plan passes
    leaked = plan.copy()
    i = leaked.index[leaked["fold"] == "test"][0]
    leaked.loc[i, "fold"] = "train"                     # same sha in train and test
    leaked = pd.concat([leaked, plan.loc[[i]]], ignore_index=True)
    with pytest.raises(AssertionError):
        R.check_plan(leaked, "kfold", 0)


@needs_folds
def test_floors_are_the_documented_rules(real_tables):
    tbl = real_tables["mendeley"]
    part = tbl[tbl["kfold"] == 0]
    fl = R.floors(part["label"].values, part["arch"].values)
    n_maj = max(int((part["label"] == 0).sum()), int((part["label"] == 1).sum()))
    assert fl["majority_accuracy"] == pytest.approx(n_maj / len(part))
    hit = ((part["arch"] == "x86") == (part["label"] == 1)).sum()
    assert fl["x86_rule_accuracy"] == pytest.approx(hit / len(part))


# ---------------------------------------------------------------------------
# the miniature dataset
# ---------------------------------------------------------------------------
def _sha(i: int) -> str:
    return hashlib.sha256(str(i).encode()).hexdigest()


@pytest.fixture
def mini(tmp_path):
    """6 ransomware families x 4 files, 10 goodware groups x 3 files, 5 folds."""
    folds_dir = tmp_path / "folds"
    folds_dir.mkdir()
    images_root = tmp_path / "images"
    tree = images_root / "unified_mendeley"
    rows, n = [], 0
    for i in range(6):
        for j in range(4):
            n += 1
            rows.append({"dataset": "mendeley", "sha256": _sha(n), "label": 1,
                         "family": f"fam{i}", "group": f"fam{i}",
                         "arch": "x86" if j % 2 else "x64",
                         "orig_set": "mal_train", "fold": i % 5})
    for g in range(10):
        for j in range(3):
            n += 1
            rows.append({"dataset": "mendeley", "sha256": _sha(n), "label": 0,
                         "family": "goodware", "group": f"mn:{g}",
                         "arch": "x86" if j % 2 else "x64",
                         "orig_set": "good_train", "fold": g % 5})
    with (folds_dir / "folds_mendeley.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        cls = R.CLASS_DIRS[r["label"]]
        d = tree / cls
        d.mkdir(parents=True, exist_ok=True)
        stem = f"{r['family']}__{r['sha256']}"          # the real trees' shape
        (d / f"{stem}.png").write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(32))
        (d / f"{stem}_vit_mask.npy").write_bytes(b"\x93NUMPY" + bytes(16))
    return SimpleNamespace(rows=rows, folds_dir=folds_dir, images_root=images_root,
                           tree=tree, tmp=tmp_path)


def _args(mini, **kw):
    base = dict(dataset="mendeley", stage="all", device="cpu", epochs=2,
                batch_size=4, kfold_seeds=(1, 2), lofo_seeds=(1,),
                folds_dir=str(mini.folds_dir), out_root=str(mini.tmp / "results"),
                images_root=str(mini.images_root), data_root=str(mini.tmp / "data"),
                models_root=str(mini.tmp / "models"), folds=None, families=None,
                max_samples=0, keep_weights=False, keep_lofo_trees=False,
                report_only=False, no_summary=False)
    base.update(kw)
    return argparse.Namespace(**base)


def fake_score(sha: str, seed: int) -> float:
    """Deterministic stand-in for the model's ransomware probability."""
    return ((int(sha[:8], 16) ^ (seed * 2654435761)) % 1000) / 1000.0


def make_fake_train_one(calls):
    def _train_one(mt, data_dir, ckpt_dir, seed, device, epochs, batch_size,
                   max_samples, keep_weights):
        man = pd.read_csv(Path(data_dir) / "manifest.csv", dtype=str)
        test = man[man["fold"] == "test"]
        ids = list(test["sha256"])
        scores = [fake_score(s, seed) for s in ids]
        doc = {"ids": ids, "y_true": [int(v) for v in test["label"]],
               "y_pred": [int(s >= 0.5) for s in scores], "y_score": scores,
               "class_names": list(R.CLASS_DIRS.values()), "epochs_run": 7,
               "epoch_seconds_mean": 1.0, "epoch_seconds": [1.0] * 7,
               "best_val_loss": 0.5, "n_train": int((man["fold"] == "train").sum()),
               "n_val": int((man["fold"] == "val").sum()), "n_test": len(ids),
               "seed": seed, "device": device, "device_name": "stub",
               "wall_seconds": 7.0}
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        (Path(ckpt_dir) / "run.json").write_text(json.dumps(doc), encoding="utf-8")
        calls.append((str(data_dir), seed))
        return doc
    return _train_one


def test_materialise_hard_links_the_tree_and_writes_a_manifest(mini):
    tbl = R.load_fold_table(mini.folds_dir, "mendeley")
    index = R.image_index(mini.images_root, R.IMAGE_TREES["mendeley"])
    assert len(index) == len(tbl)
    plan = R.plan_kfold(tbl, 0)
    R.check_plan(plan, "kfold", 0)
    out = mini.tmp / "data" / "fold0"
    counts = R.materialise(plan, index, out)
    assert counts.get("mask_missing", 0) == 0
    assert counts["link"] + counts.get("copy", 0) == len(tbl)

    for fold in ("train", "val", "test"):
        for cls in R.CLASS_DIRS.values():
            assert (out / fold / cls).is_dir()
    for _, r in plan.iterrows():
        png = out / r["fold"] / R.CLASS_DIRS[int(r["label"])] / f"{r['sha256']}.png"
        assert png.is_file(), png
        # the stem is the sha256, so predictions join back to the fold table
        assert png.stem == r["sha256"]
        assert png.with_name(png.stem + "_vit_mask.npy").is_file()
        assert png.stat().st_nlink >= 2                 # hard link, not a copy

    man = pd.read_csv(out / "manifest.csv", dtype=str)
    assert set(man["fold"]) == {"train", "val", "test"}
    assert {"sha256", "asm_id", "label", "fold", "family", "arch", "kfold"} <= set(man)
    got = {p.stem for p in (out / "test").rglob("*.png")}
    assert got == set(plan.loc[plan["fold"] == "test", "sha256"])


def test_materialise_refuses_an_image_whose_class_folder_disagrees(mini):
    tbl = R.load_fold_table(mini.folds_dir, "mendeley")
    index = R.image_index(mini.images_root, R.IMAGE_TREES["mendeley"])
    victim = tbl.loc[tbl["label"] == 1, "sha256"].iloc[0]
    index[victim] = dict(index[victim], tree_label=0)
    with pytest.raises(AssertionError):
        R.materialise(R.plan_kfold(tbl, 0), index, mini.tmp / "data" / "bad")


def test_lofo_test_tree_holds_only_the_family(mini):
    tbl = R.load_fold_table(mini.folds_dir, "mendeley")
    index = R.image_index(mini.images_root, R.IMAGE_TREES["mendeley"])
    plan = R.plan_lofo(tbl, "fam3")
    R.check_plan(plan, "lofo", "fam3")
    out = mini.tmp / "data" / "lofo_fam3"
    R.materialise(plan, index, out)
    assert list((out / "test" / "Class_0_Goodware").glob("*.png")) == []
    got = {p.stem for p in (out / "test" / "Class_1_Ransomware").glob("*.png")}
    assert got == set(tbl.loc[tbl["family"] == "fam3", "sha256"])


def test_sweep_is_resumable(mini, monkeypatch):
    calls = []
    monkeypatch.setattr(R, "train_one", make_fake_train_one(calls))
    a = _args(mini)
    R.sweep("mendeley", a, mt=None)
    expected = 5 * len(a.kfold_seeds) + 6 * len(a.lofo_seeds)   # 5 folds, 6 families
    assert len(calls) == expected

    calls.clear()
    R.sweep("mendeley", a, mt=None)                 # everything is on disk already
    assert calls == []

    # one run removed -> exactly that one is redone
    gone = R.run_dir(a.models_root, "mendeley", "kfold", 2, 1) / "run.json"
    assert gone.is_file()
    gone.unlink()
    R.sweep("mendeley", a, mt=None)
    assert len(calls) == 1

    # a run.json that no longer matches the plan's test set is not trusted
    calls.clear()
    stale = R.run_dir(a.models_root, "mendeley", "lofo", "fam1", 1) / "run.json"
    doc = json.loads(stale.read_text())
    doc["ids"] = doc["ids"][:-1]
    stale.write_text(json.dumps(doc), encoding="utf-8")
    R.sweep("mendeley", a, mt=None)
    assert len(calls) == 1


def test_outputs_have_the_promised_shape(mini, monkeypatch):
    monkeypatch.setattr(R, "train_one", make_fake_train_one([]))
    a = _args(mini)
    R.sweep("mendeley", a, mt=None)
    res = R.score_dataset("mendeley", a)
    assert res is not None and res["complete"]
    out = Path(a.out_root) / "mendeley" / "cnn_vit" / R.MODEL_NAME
    R.write_summary([res], Path(a.out_root))

    for name in ("predictions.csv", "fold_metrics.csv", "per_family.csv",
                 "lofo_predictions.csv", "metrics.json", "config_used.yaml"):
        assert (out / name).is_file(), name
    assert (Path(a.out_root) / "summary_cnn_vit.md").is_file()

    preds = pd.read_csv(out / "predictions.csv", dtype={"sha256": str})
    assert list(preds.columns) == ["sha256", "label", "family", "arch", "fold",
                                   "score", "pred"]
    assert len(preds) == len(mini.rows)                 # every sample held out once
    assert preds["sha256"].is_unique

    # the pooled score is the mean over seeds, and pred is its argmax
    row = preds.iloc[0]
    want = sum(fake_score(row["sha256"], s) for s in a.kfold_seeds) / len(a.kfold_seeds)
    assert row["score"] == pytest.approx(want, abs=1e-6)
    assert row["pred"] == int(want >= 0.5)

    fm = pd.read_csv(out / "fold_metrics.csv")
    assert list(fm.columns) == [
        "fold", "n_test", "n_good", "n_rans", "accuracy", "balanced_accuracy",
        "macro_f1", "roc_auc", "recall_ransomware", "recall_goodware", "fpr",
        "x86_recall_ransomware", "x86_recall_goodware", "x64_recall_ransomware",
        "x64_recall_goodware", "majority_floor", "x86_rule_floor",
        "macro_f1_sd_over_seeds", "epochs_run_mean"]
    assert len(fm) == 5
    assert (fm["n_test"] == fm["n_good"] + fm["n_rans"]).all()
    assert fm["n_test"].sum() == len(mini.rows)

    pf = pd.read_csv(out / "per_family.csv")
    assert list(pf.columns) == ["family", "n", "n_x64", "fold", "recall_kfold",
                                "recall_lofo"]
    assert len(pf) == 6 and pf["recall_lofo"].notna().all()
    rans = preds[preds["label"] == 1]
    want = rans[rans["family"] == "fam0"]["pred"].mean()
    assert pf.loc[pf["family"] == "fam0", "recall_kfold"].iloc[0] == pytest.approx(want)

    lofo = pd.read_csv(out / "lofo_predictions.csv", dtype={"sha256": str})
    assert list(lofo.columns) == ["sha256", "family", "arch", "score", "pred"]
    assert len(lofo) == 24 and lofo["sha256"].is_unique
    assert set(lofo["family"]) == {f"fam{i}" for i in range(6)}

    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    for key in ("results", "fold_mean", "fold_sd", "lofo_mean_recall", "device",
                "seconds_per_epoch_mean", "train_seconds_total", "config"):
        assert key in m, key
    assert set(m["fold_mean"]) == {"accuracy", "balanced_accuracy", "macro_f1",
                                   "roc_auc"}
    assert m["samples"]["every_sample_held_out_once"] is True
    assert m["lofo_mean_recall"] == pytest.approx(
        float(pf["recall_lofo"].mean()))
    assert m["results"][0]["macro_f1"] == pytest.approx(
        __import__("cnn_vit_pipeline.cohort", fromlist=["x"])._core_metrics(
            preds["label"].values, preds["pred"].values)["macro_f1"])


def test_report_only_needs_no_training(mini, monkeypatch):
    monkeypatch.setattr(R, "train_one", make_fake_train_one([]))
    a = _args(mini)
    R.sweep("mendeley", a, mt=None)
    first = (Path(a.out_root) / "mendeley" / "cnn_vit" / R.MODEL_NAME)
    R.score_dataset("mendeley", a)
    before = (first / "predictions.csv").read_text()

    def explode(*args, **kw):                            # must not be reached
        raise AssertionError("report-only must not train")
    monkeypatch.setattr(R, "train_one", explode)
    R.main(["--dataset", "mendeley", "--report-only",
            "--folds-dir", str(mini.folds_dir), "--out-root", str(a.out_root),
            "--images-root", str(mini.images_root), "--data-root", str(a.data_root),
            "--models-root", str(a.models_root), "--kfold-seeds", "1,2",
            "--lofo-seeds", "1"])
    assert (first / "predictions.csv").read_text() == before


def test_recipe_constants_match_the_untuned_pipeline():
    """The study must measure train_eval.py's recipe, not a copy of it."""
    from cnn_vit_pipeline import train_eval as T
    assert R.T is T
    assert T.BATCH_SIZE == 16 and T.WARMUP_EPOCHS == 5
    assert T.BASE_LEARNING_RATE == 3e-4 and T.LABEL_SMOOTHING == 0.15
    assert R.MAX_EPOCHS == 80
    src = (REPO / "family_holdout" / "run_cnn_vit.py").read_text(encoding="utf-8")
    assert "T.train(" in src and "T.predict(" in src   # imported, never reimplemented


def test_the_dataset_class_reads_the_materialised_tree(mini):
    """The original MalwareMaskedDataset must see Class_0 as label 0."""
    pytest.importorskip("torch")
    pytest.importorskip("PIL")
    import numpy as np
    from PIL import Image
    from cnn_vit_pipeline import train_eval as T

    tbl = R.load_fold_table(mini.folds_dir, "mendeley")
    for p in mini.tree.rglob("*.png"):                  # real images this time
        Image.fromarray(np.zeros((256, 256), dtype=np.uint8)).save(p)
    for p in mini.tree.rglob("*_vit_mask.npy"):
        np.save(p, np.ones((16, 16), dtype=np.float32))
    index = R.image_index(mini.images_root, R.IMAGE_TREES["mendeley"])
    out = mini.tmp / "data" / "torchtree"
    plan = R.plan_kfold(tbl, 0)
    R.materialise(plan, index, out)

    mt = T.import_model_train()
    ds = mt.MalwareMaskedDataset(base_folder=str(out / "test"))
    assert list(ds.class_names) == ["Class_0_Goodware", "Class_1_Ransomware"]
    want = dict(zip(plan["sha256"], plan["label"]))
    for path, label in ds.all_samples:
        assert want[Path(path).stem] == label
    img, mask, label = ds[0]
    assert img.shape == (1, 256, 256) and tuple(mask.shape) == (16, 16)
