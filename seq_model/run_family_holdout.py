#!/usr/bin/env python3
"""
run_family_holdout.py - family-holdout evaluation of the mnemonic sequence model.

The whole study, in the order it was run. Every phase is resumable and a
re-invocation of any of them trains nothing that is already on disk, so this
is also the recovery procedure after an interruption:

    PY="C:/Users/chaoa/Downloads/cnn_vit_venv/Scripts/python.exe"
    $PY seq_model/data.py                       # the token cache, once
    $PY seq_model/pretrain.py --minutes 45       # the unsupervised pass, once
    $PY seq_model/lr_sanity.py                   # the one pre-registered check
    $PY seq_model/run_family_holdout.py --dataset both --scheme kfold
    $PY seq_model/run_family_holdout.py --dataset both
    $PY seq_model/run_family_holdout.py --dataset mendeley --ablations all \
        --only-ablations
    python seq_model/make_summary.py

`--scheme kfold` is the K-fold half on its own: it trains and caches the runs
but does NOT write the model directory, so a half-finished study never lands
in `results/`. The plain `--dataset both` invocation then finds those runs
done, trains only LOFO, and writes both directories. `--report-only` rebuilds
every CSV and metrics.json from the cached runs without touching the GPU.

The two schemes are `family_holdout/folds.py`'s, read from the same
`results/family_holdout/folds_<dataset>.csv` every other runner reads, and the
output directory is written by `family_holdout/common.write_model_dir`, so
`results/family_holdout/<dataset>/seq_transformer/<model>/` is byte-for-byte
the shape the coordinator's aggregator already knows how to read.

  K-fold  train on four folds, score the fifth. A group-aware 10% val fold is
          carved out of the four training folds by
          `cnn_vit_pipeline.cohort.add_val_fold`, so no family and no goodware
          duplicate-stream / source-project group straddles train and val
          either. The val fold is the early-stopping signal and nothing else.
          Five seeds per fold; a file's pooled score is the MEAN of its five
          seed scores, and the headline is the mean +/- sd over the five folds
          of the macro-F1 of that seed-mean score.
  LOFO    train on the other 37 families plus ALL goodware (again minus a val
          fold), score the held-out family alone. One seed. No goodware in the
          test set, so recall only - never quote an FPR from it.

Resumability. Every (scheme, key, seed) writes
`seq_models/runs/<dataset>/<model>/<scheme>_<key>_seed<n>.json` - the per-file
scores plus the training history - the moment it finishes, and its weights to
the sibling `.pt`. A second invocation reads those and trains nothing. The
fingerprint of the configuration is stored in each run file; a run whose
fingerprint does not match the current config is re-trained rather than
silently reused.

Nothing here reads a test fold before it is scored, and no threshold is ever
moved: the decision is argmax at 0.5 on the seed-mean P(ransomware).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Set before torch creates a CUDA context. A sweep trains ~120 models in one
# process; with the default allocator the reserved pool fragments run after run
# until it fills the 16 GB card, and on Windows WDDM a full pool spills into
# system memory instead of failing - which showed up as runs degrading from
# 80 s to 336 s partway through the K-fold pass. Expandable segments keep the
# pool compact. This changes allocation strategy only, never a result, so it is
# not part of a run's fingerprint.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from family_holdout.common import (DATASETS, Folds, OUT_ROOT,  # noqa: E402
                                   check_kfold, check_lofo, write_model_dir)
from seq_model import config as CFG                               # noqa: E402
from seq_model import data as D                                   # noqa: E402

PIPELINE = "seq_transformer"
MODEL = "seq_transformer"

# --max-runs: train at most this many models, then exit with MORE_WORK so an
# outer loop can re-invoke. Long single-process sweeps on this box accumulate
# host memory no matter how carefully the per-run objects are released, and a
# bounded process is the only leak-proof answer. Every run is already cached on
# disk, so a re-invocation costs one startup and skips everything finished.
MORE_WORK = 7
_TRAINED = 0
_MAX_RUNS = None


class _BudgetReached(Exception):
    pass


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
class Tee:
    """Progress goes to stdout AND to a log file, as the brief asks."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = path.open("a", encoding="utf-8")

    def __call__(self, *msg):
        s = " ".join(str(m) for m in msg)
        print(s, flush=True)
        self.fh.write(s + "\n")
        self.fh.flush()


def fingerprint(cfg: dict) -> str:
    """Everything that changes what a run computes, and nothing else."""
    keep = {k: cfg[k] for k in ("model", "windows", "vocabulary", "sampler",
                                "imports", "pretrain", "training", "evaluation",
                                "stream")
            if k in cfg}
    keep = json.loads(json.dumps(keep, sort_keys=True))
    for k in ("num_workers", "eval_workers"):
        keep.get("training", {}).pop(k, None)
    keep["ablation"] = cfg.get("ablation")
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:16]


def val_split(folds: Folds, train_idx: np.ndarray, test_idx: np.ndarray):
    """The group-aware val carve, via cnn_vit_pipeline.cohort.add_val_fold."""
    import pandas as pd

    from cnn_vit_pipeline import cohort as C
    df = pd.DataFrame({
        "sha256": folds.sha, "label": folds.y,
        "family_or_group": np.where(folds.y == 1, folds.family, folds.group),
        "split": "train"})
    df.loc[test_idx, "split"] = "test"
    drop = np.setdiff1d(np.arange(folds.n), np.union1d(train_idx, test_idx))
    if len(drop):
        df = df.drop(index=drop)
    out = C.add_val_fold(df)
    tr = np.asarray(out.index[out["fold"] == "train"], dtype=int)
    va = np.asarray(out.index[out["fold"] == "val"], dtype=int)
    te = np.asarray(out.index[out["fold"] == "test"], dtype=int)
    # the invariants add_val_fold is supposed to guarantee, asserted here so a
    # bad carve cannot quietly produce an optimistic early-stopping signal
    gtr = set(out.loc[tr, "family_or_group"])
    gva = set(out.loc[va, "family_or_group"])
    gte = set(out.loc[te, "family_or_group"])
    assert not gtr & gva, f"val carve leaks groups: {sorted(gtr & gva)[:3]}"
    assert not gtr & gte and not gva & gte, "test group in train/val"
    assert len(va) and len(tr), "empty train or val"
    return tr, va, te


# ---------------------------------------------------------------------------
# one training run
# ---------------------------------------------------------------------------
def build_model(cfg: dict, vocab: D.Vocab, log):
    import torch

    from seq_model.model import SeqTransformer, transfer_pretrained
    m = cfg["model"]
    imports_dim = cfg["imports"]["hash_dim"] if cfg["imports"]["enabled"] else 0
    model = SeqTransformer(
        vocab.size, window=cfg["windows"]["window"], emb_dim=m["emb_dim"],
        dim=m["dim"], depth=m["depth"], heads=m["heads"], mlp_dim=m["mlp_dim"],
        dropout=m["dropout"], drop_path_rate=m["drop_path_rate"],
        pooling=m["pooling"], mil=m["mil"], imports_dim=imports_dim)
    info = None
    if cfg["pretrain"]["enabled"]:
        p = Path(cfg["pretrain"]["checkpoint"])
        if not p.exists():
            raise SystemExit(
                f"pretrain.enabled is true but {p} is missing; run "
                f"seq_model/pretrain.py or use the no_pretrain ablation")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        if int(ck.get("window", 0)) != int(cfg["windows"]["window"]):
            # the relative-position table is sized by window; everything else
            # transfers. Loading is by-name and shape-checked, so the table is
            # simply left at its initialisation.
            log(f"    note: pretrained window {ck.get('window')} != "
                f"{cfg['windows']['window']}; the relative position-bias table "
                f"is not transferred")
        info = transfer_pretrained(model, ck, vocab.tokens)
        info["checkpoint_step"] = int(ck.get("step", -1))
    return model, info


def run_one(cfg, folds: Folds, scheme: str, key, seed: int, tr, va, te,
            vocab: D.Vocab, imports, run_dir: Path, log, device) -> dict:
    """Train one model and score `te`. Returns the run record (also on disk)."""
    import torch

    from seq_model.train import FileWindowDataset, loader, predict, seed_everything, train_model
    tag = f"[{folds.dataset}/{scheme}/{key}/seed{seed}] "
    rec_path = run_dir / f"{scheme}_{_slug(key)}_seed{seed}.json"
    fp = fingerprint(cfg)
    if rec_path.exists():
        try:
            rec = json.loads(rec_path.read_text(encoding="utf-8"))
            same_rows = rec.get("test_sha") == list(folds.sha[te])
            if rec.get("fingerprint") == fp and len(rec["scores"]) == len(te) \
                    and same_rows:
                log(f"{tag}already done (val macroF1 "
                    f"{rec.get('best_val_macro_f1')}), skipping")
                return rec
            log(f"{tag}stale run record "
                f"({'test rows changed' if not same_rows else 'config changed'});"
                f" retraining")
        except json.JSONDecodeError:
            log(f"{tag}unreadable run record; retraining")

    t0 = time.time()
    seed_everything(seed)
    W, N = cfg["windows"]["window"], cfg["windows"]["n_windows"]
    cache = Path(cfg["paths"]["token_cache"])
    imports_dim = cfg["imports"]["hash_dim"] if cfg["imports"]["enabled"] else 0
    mk = lambda idx: FileWindowDataset(  # noqa: E731
        folds.sha[idx], folds.y[idx], vocab, W, N, cache, imports, imports_dim)
    ds_tr, ds_va, ds_te = mk(tr), mk(va), mk(te)

    model, pre = build_model(cfg, vocab, log)
    if pre:
        log(f"{tag}pretrained init: {pre['embedding_rows_transferred']}/"
            f"{pre['embedding_rows_total']} embedding rows, "
            f"{pre['encoder_tensors_loaded']} encoder tensors "
            f"(step {pre['checkpoint_step']})")
    state, hist = train_model(model, ds_tr, ds_va, cfg, device, seed,
                              folds.y[tr], folds.arch[tr], log=log, tag=tag)

    # one pass over the test fold: spawning workers costs more than it saves
    dl_te = loader(ds_te, cfg["training"]["eval_batch_size"], workers=0)
    score = predict(model, dl_te, device, len(ds_te))
    assert not np.isnan(score).any(), "a test file got no prediction"

    run_dir.mkdir(parents=True, exist_ok=True)
    if state is not None:
        torch.save({"state_dict": state, "tokens": vocab.tokens,
                    "config": cfg, "scheme": scheme, "key": key, "seed": seed},
                   run_dir / f"{scheme}_{_slug(key)}_seed{seed}.pt")
    rec = {"scheme": scheme, "key": key, "seed": seed, "dataset": folds.dataset,
           "fingerprint": fp, "test_sha": list(folds.sha[te]),
           "scores": [float(x) for x in score],
           "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
           "vocab_size": int(vocab.size), "seconds": round(time.time() - t0, 1),
           "pretrained": pre, **hist}
    tmp = rec_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec), encoding="utf-8")
    tmp.replace(rec_path)
    log(f"{tag}done in {time.time()-t0:.0f}s "
        f"(best val macroF1 {hist['best_val_macro_f1']})")
    del model, state, ds_tr, ds_va, ds_te, dl_te
    if device.type == "cuda":
        torch.cuda.empty_cache()
    global _TRAINED
    _TRAINED += 1
    if _MAX_RUNS and _TRAINED >= _MAX_RUNS:
        raise _BudgetReached(f"{_TRAINED} runs trained in this process")
    return rec


def _slug(key) -> str:
    return str(key).replace(" ", "-").replace("/", "-")


def strip_lofo(model_dir: Path, why: str) -> None:
    """Blank the LOFO half of a model dir that did not run LOFO.

    The ablation study is a mendeley K-fold study - 3 ablations x 38 families
    of leave-one-family-out would cost more GPU time than the entire main run.
    `write_model_dir` always writes the six files, so rather than fork it the
    LOFO-derived fields are emptied afterwards and the reason is recorded in
    metrics.json. An empty column is honest; a zero is not.
    """
    import csv
    (model_dir / "lofo_predictions.csv").write_text(
        "sha256,family,arch,score,pred\n", encoding="utf-8")
    p = model_dir / "per_family.csv"
    rows = list(csv.DictReader(p.open(encoding="utf-8", newline="")))
    for r in rows:
        r["recall_lofo"] = ""
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["family", "n", "n_x64", "fold",
                                           "recall_kfold", "recall_lofo"])
        w.writeheader()
        w.writerows(rows)
    mp = model_dir / "metrics.json"
    doc = json.loads(mp.read_text(encoding="utf-8"))
    for d in [doc] + list(doc.get("results", [])):
        for k in ("lofo_mean_recall", "lofo_weighted_recall"):
            if k in d:
                d[k] = None
        if "lofo_recall_by_family" in d:
            d["lofo_recall_by_family"] = {}
    doc["lofo_not_run"] = why
    mp.write_text(json.dumps(doc, indent=1), encoding="utf-8")


# ---------------------------------------------------------------------------
# the whole study for one dataset x one configuration
# ---------------------------------------------------------------------------
def run_dataset(cfg: dict, dataset: str, out_root: Path, device, log,
                do_kfold=True, do_lofo=True, report_only=False,
                imports=None, write_dir=True) -> Path:
    folds = Folds(dataset)
    check_kfold(folds)
    check_lofo(folds)
    abl = cfg.get("ablation")
    # an imports run gets its own model and run directories, so it can never
    # overwrite the committed no-imports study or reuse its cached runs
    name = ((f"ablation_{abl}" if abl else MODEL)
            + (f"_{D.STREAM}" if D.STREAM != "mn" else "")
            + ("_imports" if imports is not None else ""))
    model_dir = out_root / dataset / PIPELINE / name
    run_dir = Path(cfg["paths"]["weights"]) / "runs" / dataset / name
    run_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(cfg["paths"]["token_cache"])
    t_start = time.time()

    kseeds = list(cfg["seeds"]["kfold"])
    lseeds = list(cfg["seeds"]["lofo"])
    n = folds.n

    # -- K-fold -----------------------------------------------------------
    seed_score = {s: np.full(n, np.nan) for s in kseeds}
    for f, tr_all, te in folds.kfold():
        tr, va, _ = val_split(folds, tr_all, te)
        vocab = D.Vocab.fit(folds.sha[np.concatenate([tr, va])], cache,
                            cfg["vocabulary"]["min_count"])
        for s in kseeds:
            if report_only:
                p = run_dir / f"kfold_{f}_seed{s}.json"
                if not p.exists():
                    raise SystemExit(f"--report-only but {p} is missing")
                rec = json.loads(p.read_text(encoding="utf-8"))
            elif do_kfold:
                rec = run_one(cfg, folds, "kfold", f, s, tr, va, te, vocab,
                              imports, run_dir, log, device)
            else:
                continue
            seed_score[s][te] = np.asarray(rec["scores"], dtype=float)

    # -- LOFO -------------------------------------------------------------
    lofo = {}
    if do_lofo:
        for fam, tr_all, te in folds.lofo():
            tr, va, _ = val_split(folds, tr_all, te)
            vocab = D.Vocab.fit(folds.sha[np.concatenate([tr, va])], cache,
                                cfg["vocabulary"]["min_count"])
            acc = []
            for s in lseeds:
                if report_only:
                    p = run_dir / f"lofo_{_slug(fam)}_seed{s}.json"
                    if not p.exists():
                        raise SystemExit(f"--report-only but {p} is missing")
                    rec = json.loads(p.read_text(encoding="utf-8"))
                else:
                    rec = run_one(cfg, folds, "lofo", fam, s, tr, va, te, vocab,
                                  imports, run_dir, log, device)
                acc.append(np.asarray(rec["scores"], dtype=float))
            sc = np.mean(acc, axis=0)
            lofo[fam] = (sc, (sc >= 0.5).astype(int))
    else:
        # placeholder rows; strip_lofo() blanks them once the dir is written
        for fam in folds.families:
            k = int(((folds.family == fam) & (folds.y == 1)).sum())
            lofo[fam] = (np.zeros(k), np.full(k, -1, dtype=int))

    if not (do_kfold and write_dir):
        log(f"[{dataset}] partial pass (scheme filter in effect): "
            f"{time.time()-t_start:.0f}s, model dir not written")
        return model_dir

    # -- assemble ---------------------------------------------------------
    stack = np.stack([seed_score[s] for s in kseeds], axis=0)
    assert not np.isnan(stack).any(), "a file has no K-fold prediction"
    kf_score = stack.mean(axis=0)
    kf_pred = (kf_score >= 0.5).astype(int)

    per_seed = {}
    from seq_model.train import macro_f1
    for i, s in enumerate(kseeds):
        per_seed[str(s)] = {
            str(f): round(macro_f1(folds.y[folds.fold == f],
                                   (stack[i][folds.fold == f] >= 0.5).astype(int)), 6)
            for f in range(5)}

    cfg_used = json.loads(json.dumps(cfg))
    cfg_used["fingerprint"] = fingerprint(cfg)
    cfg_used["device"] = str(device)
    cfg_used["per_seed_fold_macro_f1"] = per_seed
    cfg_used["runs_dir"] = str(run_dir)
    cfg_used["token_cache"] = {
        "path": str(cache), "global_vocabulary": len(D.global_vocab(cache)),
        "note": "global ids are a property of the corpus; the model vocabulary "
                "is refitted on the training folds of every split"}
    desc = ("mnemonic sequence transformer (embedding -> 1-D conv stem -> the "
            "HierarchicalMalwareNet ViT blocks -> masked mean pooling -> MIL "
            "over up to {n} windows of {w} tokens), family-holdout on {d}"
            .format(n=cfg["windows"]["n_windows"], w=cfg["windows"]["window"],
                    d=dataset))
    if abl:
        desc += f" [ablation: {abl}]"
    write_model_dir(model_dir, folds, PIPELINE,
                    name,
                    kf_score, kf_pred, folds.fold, lofo, cfg_used,
                    time.time() - t_start, description=desc)
    if not do_lofo:
        strip_lofo(model_dir, "ablations are a mendeley K-fold study only; "
                              "38 LOFO runs per ablation are outside the "
                              "compute budget. See the main model dir for LOFO.")
    log(f"wrote {model_dir} ({time.time()-t_start:.0f}s)")
    return model_dir


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=(*DATASETS, "both"), default="both")
    ap.add_argument("--config", default=str(CFG.CONFIG_PATH))
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ablations", default="",
                    help="comma-separated ablation names, or 'all'")
    ap.add_argument("--only-ablations", action="store_true",
                    help="skip the main configuration")
    ap.add_argument("--scheme", choices=("kfold", "lofo", "both"), default="both")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--imports", default="",
                    help="JSON {sha256: [import names]}; enables the imports "
                         "side-input. NOT used for the committed run.")
    ap.add_argument("--no-pretrain", action="store_true",
                    help="train the encoder from scratch (overrides the config)")
    ap.add_argument("--stream", default="",
                    help="token tree to read (mn, mn_api_top); must equal the "
                         "RANSOM_SEQ_STREAM the process was started with. A "
                         "non-default stream uses paths.token_cache_<stream> and "
                         "pretrain.checkpoint_<stream> and writes to "
                         "<model>_<stream>/")
    ap.add_argument("--log", default="")
    ap.add_argument("--max-runs", type=int, default=0,
                    help=f"exit with status {MORE_WORK} after this many "
                         f"training runs so an outer loop can re-invoke; 0 "
                         f"means no limit")
    a = ap.parse_args()
    global _MAX_RUNS
    _MAX_RUNS = a.max_runs or None

    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available())
                          else "cpu")

    cfg = CFG.load(a.config)
    if a.stream and a.stream != D.STREAM:
        raise SystemExit(f"--stream {a.stream} but RANSOM_SEQ_STREAM={D.STREAM}; "
                         f"set the environment variable before starting")
    if D.STREAM != "mn":
        cfg["paths"]["token_cache"] = str(D.CACHE_ROOT)
        ck = Path(cfg["pretrain"]["checkpoint"])
        cfg["pretrain"]["checkpoint"] = str(ck.with_name(f"{ck.stem}_{D.STREAM}{ck.suffix}"))
        cfg["stream"] = D.STREAM
    if a.no_pretrain:
        cfg["pretrain"]["enabled"] = False
    imports = None
    if a.imports:
        cfg["imports"]["enabled"] = True
        imports = D.load_imports(a.imports, cfg["imports"]["hash_dim"])
        print(f"imports side-input: {len(imports)} files hashed to "
              f"{cfg['imports']['hash_dim']} dims")

    log = Tee(Path(a.log) if a.log
              else Path(cfg["paths"]["weights"]) / "logs" / "family_holdout.log")
    log(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} device={device} "
        f"config={a.config} ===")

    names = (list(CFG.ABLATIONS) if a.ablations == "all"
             else [x for x in a.ablations.split(",") if x])
    configs = ([] if a.only_ablations else [cfg]) + [CFG.ablation(cfg, x) for x in names]
    datasets = DATASETS if a.dataset == "both" else (a.dataset,)
    for c in configs:
        for ds in datasets:
            if c.get("ablation") and ds != "mendeley":
                continue          # ablations are a mendeley K-fold study only
            log(f"--- dataset={ds} ablation={c.get('ablation')} ---")
            is_abl = bool(c.get("ablation"))
            # a scheme filter means "train this half now"; only a complete
            # pass (or an ablation, which is K-fold by definition) writes the
            # model directory, so a half-finished study never lands in results/
            try:
                run_dataset(c, ds, Path(a.out), device, log,
                            do_kfold=a.scheme in ("kfold", "both"),
                            do_lofo=(a.scheme in ("lofo", "both") and not is_abl),
                            report_only=a.report_only, imports=imports,
                            write_dir=(a.scheme == "both" or is_abl))
            except _BudgetReached as e:
                log(f"--- stopping this process: {e}; re-invoke to continue ---")
                return MORE_WORK
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
