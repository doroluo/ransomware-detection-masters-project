#!/usr/bin/env python3
"""The one and only test evaluation of the tuned CFG / graph2vec models.

Reads `results/graph2vec/tuned/<dataset>/chosen.json` -- written by `tune.py`
from a group-CV search that never loaded a test row -- refits each chosen
configuration on the whole TRAIN split, and scores TEST once.

    python graph2vec_pipeline/final_eval.py --dataset both

Writes, per dataset, into results/graph2vec/tuned/<dataset>/:
    metrics.json      shared schema (cohort.build_result), per-arch and
                      per-family blocks, plus the majority and x86-rule floors
    predictions.csv   one row per test sample per configuration scored
    config_used.yaml  exactly what was fitted, and the CV score it was picked on

Two decision rules are reported for every configuration, as in the untuned
metrics.json: `argmax` and a threshold taken from the TRAIN out-of-fold scores
of the same group-CV folds. The threshold is a train-side quantity; test is
never consulted to pick it.

Two things sit *beside* the committed results rather than inside them:

`posthoc_top5`   the five highest-CV configurations of the whole search, each
                 scored on test. Only `tuned_best` was committed to before the
                 test set was opened; the other four are here to size the
                 CV->test gap, and every row carries `posthoc: true`. They are
                 not a second selection pass -- nothing downstream is allowed
                 to pick among them by test score.
`reproducibility` written by a separate `--repro-check` pass: the chosen
                 configuration is refitted from scratch in a fresh process and
                 compared against `predictions.csv` sample by sample. It is a
                 separate pass on purpose -- a second fit inside the run that
                 produced the first shares an interpreter, a loaded graph
                 cache and a warm allocator, and so tests less.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cnn_vit_pipeline.cohort import (build_result, split_summary,  # noqa: E402
                                     write_metrics)
from graph2vec_pipeline import wl  # noqa: E402
from graph2vec_pipeline.tune import (SEED, Construction, Features,  # noqa: E402
                                     Model, N_FOLDS, Rep, SUPPORTS_WEIGHT,
                                     _fit_fold, _nbytes, _prep,
                                     arch_sample_weight, best_threshold,
                                     load_obj, obj_from_row)
from graph2vec_pipeline.graph_cache import DEFAULT_CACHE  # noqa: E402


def floors(y_true, arch) -> dict:
    """The two numbers every row has to be read against.

    majority   always predict the larger test class
    x86_rule   predict ransomware iff the sample is x86. The ransomware half of
               the split is 80% x86 in test while the goodware half is not, so
               this is the confound the CFG features are competing with.
    """
    y_true = np.asarray(y_true).astype(int)
    arch = np.asarray(arch, dtype=object)
    from cnn_vit_pipeline.cohort import _core_metrics
    maj = int(np.bincount(y_true).argmax())
    out = {"majority": _core_metrics(y_true, np.full(len(y_true), maj)),
           "x86_rule": _core_metrics(y_true, (arch == "x86").astype(int))}
    for v in out.values():
        v.pop("confusion_matrix", None)
        v.pop("roc_auc", None)
    return {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                for kk, vv in v.items()
                if kk in ("accuracy", "balanced_accuracy", "macro_f1",
                          "recall_goodware", "recall_ransomware",
                          "false_positive_rate")}
            for k, v in out.items()}


_SIZE_COLS = ("n_nodes", "n_edges", "n_blocks", "insns")


class _Scorer:
    """Fit one configuration on TRAIN, score TEST once.

    Everything a configuration needs that does not depend on the classifier --
    the feature matrices, the inner CV folds used only for the threshold, the
    per-sample metadata -- lives here so the committed picks and the post-hoc
    top-5 go through exactly the same code path.
    """

    def __init__(self, dataset: str, cache_dir: Path):
        from sklearn.model_selection import StratifiedGroupKFold
        self.F = F = Features(dataset, cache_dir, rows_mask="all")
        self.dataset = dataset
        self.y = F.y
        self.tr = np.flatnonzero(F.is_train)
        self.te = np.flatnonzero(~F.is_train)
        self.arch, self.fam = F.arch, F.family
        self.folds = list(StratifiedGroupKFold(
            n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(
                np.zeros(len(self.tr)), self.y[self.tr], F.groups[self.tr]))
        self._prep: dict = {}          # (c, r) -> {fold key: (A, B)}
        self._build_cache: dict = {}
        self._prep_bytes = 0
        self.no_cache = False

    # Cache only what is cheap to hold and expensive to rebuild.
    #
    # The expensive thing in this harness is TruncatedSVD over the WL
    # histogram: on `mendeley` that histogram is 1.05M columns, so the
    # randomized range finder allocates ~2.6GB per fit and there are six fits
    # per configuration. What it *produces* is tiny -- 2018x300 float64, under
    # 5MB -- and it is shared by the embedding pick, the classifier grid's
    # winner and all five post-hoc top-CV rows. So the dense results are worth
    # keeping indefinitely.
    #
    # The in-fold TF-IDF pairs are the opposite: hundreds of MB each, and
    # refitting a TfidfTransformer is seconds. Holding them is actively
    # harmful, because the memory they occupy is the headroom the next SVD
    # needs; caching them is what turned a 306s configuration into a
    # 100-minute one on a box that is also running the other harnesses.
    # A small per-entry cap keeps the dense bases and refuses the sparse ones,
    # and no count-based eviction is then needed -- the picks interleave
    # representations, so LRU would drop exactly the basis about to be reused.
    _KEEP_REPS = 32
    _ENTRY_CAP = 64_000_000
    _CACHE_BUDGET = 256_000_000

    def _prepped(self, rep_key, key, Xpair, a, b, needs_scaling, infold_tfidf,
                 svd_dim):
        """In-fold IDF / SVD / scaling, memoised per (construction, rep).

        It does not depend on the classifier, and several configurations here
        share one representation -- the five top-CV rows on `mendeley` are one
        SVD basis and five forests. A TruncatedSVD(300) over 300k columns,
        six times, is minutes; the forests it feeds are seconds.

        Two representations are kept, not one, because the picks interleave:
        the `size_only` control sits between the embedding rows and the
        post-hoc top-5 that reuse the very same SVD basis.
        """
        group = self._prep.setdefault(rep_key, {})
        hit = group.get(key)
        if hit is not None:
            return hit
        out = _prep(Xpair, a, b, needs_scaling, infold_tfidf, svd_dim)
        if self.no_cache:
            return out
        n = _nbytes(out[0]) + _nbytes(out[1])
        if n < self._ENTRY_CAP and self._prep_bytes + n < self._CACHE_BUDGET:
            group[key] = out
            self._prep_bytes += n
        return out

    def _built(self, c: Construction, r: Rep):
        """`Features.build` memoised per (construction, rep).

        `Features` caches the WL layer matrices but not the horizontal stack
        of them, and `sparse.hstack` over five 2509-row blocks with ~25M
        stored values goes through COO and back every time it is called --
        minutes, repeated once per configuration, for a matrix that is
        identical across every classifier sharing a representation.
        """
        hit = self._build_cache.get((c, r))
        if hit is None:
            hit = self.F.build(c, r)
            self._build_cache = {(c, r): hit}      # one at a time: it is big
        return hit

    def _evict(self, rep_key):
        """Keep the most recent `_KEEP_REPS` representations."""
        if rep_key in self._prep:
            self._prep[rep_key] = self._prep.pop(rep_key)   # mark as newest
        while len(self._prep) > self._KEEP_REPS:
            dropped = self._prep.pop(next(iter(self._prep)))
            self._prep_bytes -= sum(_nbytes(A) + _nbytes(B)
                                    for A, B in dropped.values())
            self._prep_bytes = max(0, self._prep_bytes)

    def run(self, name, c: Construction, r: Rep, m: Model, cv: dict,
            posthoc: bool = False) -> dict:
        F, y, tr, te = self.F, self.y, self.tr, self.te
        t0 = time.time()
        Xpair, needs_scaling, infold_tfidf = self._built(c, r)
        svd_dim = r.dim if r.kind == "wl_svd" else 0
        w = arch_sample_weight(y, self.arch) if m.arch_weight else None
        rep_key = (c, r)
        self._evict(rep_key)

        # --- threshold: TRAIN out-of-fold only --------------------------
        oof = np.zeros(len(tr), dtype=float)
        for fi, (a_idx, b_idx) in enumerate(self.folds):
            A, B = self._prepped(rep_key, fi, Xpair, tr[a_idx], tr[b_idx],
                                 needs_scaling, infold_tfidf, svd_dim)
            _, s = _fit_fold(m, A, y[tr[a_idx]], B,
                             None if w is None else w[tr[a_idx]])
            oof[b_idx] = s
        thr, oof_f1 = best_threshold(y[tr], oof, "macro_f1")
        thr_ba, oof_ba = best_threshold(y[tr], oof, "balanced_accuracy")

        # --- the single test evaluation ---------------------------------
        A, B = self._prepped(rep_key, "train|test", Xpair, tr, te,
                             needs_scaling, infold_tfidf, svd_dim)
        pred, score = _fit_fold(m, A, y[tr], B, None if w is None else w[tr])
        n_feat = int(A.shape[1])

        common = dict(config_name=name, representation=r.kind, model=m.name,
                      construction=c.key(), rep_spec=r.key(),
                      model_spec=m.key(), n_features=n_feat,
                      cv_macro_f1=round(cv["cv_macro_f1"], 4),
                      cv_balanced_acc=round(cv["cv_balanced_acc"], 4),
                      cv_auc=round(cv["cv_auc"], 4),
                      fit_seconds=round(time.time() - t0, 1))
        if posthoc:
            common["posthoc"] = True
        res_a = build_result(y[te], pred, score, self.arch[te], self.fam[te],
                             decision="argmax (default 0.5)", **common)
        res_t = build_result(y[te], (score >= thr).astype(int), score,
                             self.arch[te], self.fam[te],
                             decision="threshold from train out-of-fold scores",
                             threshold=round(float(thr), 6),
                             oof_macro_f1=round(float(oof_f1), 4),
                             threshold_balacc=round(float(thr_ba), 6),
                             oof_balanced_acc=round(float(oof_ba), 4), **common)
        for res in (res_a, res_t):
            res["cv_to_test_macro_f1_gap"] = round(
                cv["cv_macro_f1"] - res["macro_f1"], 4)

        # graph size for the rows actually scored, under *this* construction
        S = F.size(c)[te]
        sz = {k: S[:, wl.SIZE_FEATURES.index(k)] for k in _SIZE_COLS}
        sub = F.split.iloc[te]
        preds = pd.DataFrame({
            "config_name": name, "posthoc": bool(posthoc),
            "sha256": sub["sha256"].to_numpy(), "label": y[te],
            "arch": self.arch[te], "family": self.fam[te],
            "source": sub["source"].to_numpy(),
            "family_or_group": sub["family_or_group"].to_numpy(),
            "score": score, "pred_argmax": pred,
            "pred_threshold": (score >= thr).astype(int),
            **{k: v.astype(np.int64) for k, v in sz.items()},
            "at_block_cap": (sz["n_blocks"] >= c.max_blocks).astype(int)})
        print(f"  {name:<28s} CV={cv['cv_macro_f1']:.4f} "
              f"test argmax={res_a['macro_f1']:.4f} "
              f"thr={res_t['macro_f1']:.4f} auc={res_a['roc_auc']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        return dict(argmax=res_a, threshold=res_t, preds=preds,
                    thr=float(thr), n_features=n_feat, score=score)


def _cv_of(d: dict) -> dict:
    return {k: float(d[k]) for k in
            ("cv_macro_f1", "cv_balanced_acc", "cv_auc")}


def run_dataset(dataset: str, cache_dir: Path, out_dir: Path) -> dict:
    t_start = time.time()
    chosen = json.loads((out_dir / "chosen.json").read_text(encoding="utf-8"))

    sc = _Scorer(dataset, cache_dir)
    y, te = sc.y, sc.te
    print(f"[{dataset}] {len(y)} graphs, {len(sc.tr)} train / {len(te)} test",
          flush=True)

    results, pred_rows, used, by_key = [], [], [], {}
    for pick in chosen["picks"]:
        c, r, m = load_obj(pick)
        got = sc.run(pick["name"], c, r, m, _cv_of(pick))
        results.extend([got["argmax"], got["threshold"]])
        pred_rows.append(got["preds"])
        used.append({**pick, "threshold": got["thr"],
                     "n_features": got["n_features"]})
        by_key[(c.key(), r.key(), m.key())] = pick["name"]

    # --- post hoc: the top-5 CV configurations, for the gap table -------
    # Labelled, kept out of `results`, and never used to choose anything.
    print(f"[{dataset}] post-hoc: top-5 CV configurations on test", flush=True)
    posthoc, seen = [], set()
    for rank, row in enumerate(chosen.get("top5", []), 1):
        c, r, m = obj_from_row(row)
        k = (c.key(), r.key(), m.key())
        if k in seen:
            continue
        seen.add(k)
        name = f"cv_rank{rank}"
        if k in by_key:
            name += f" (= {by_key[k]})"
        got = sc.run(name, c, r, m, _cv_of(row), posthoc=True)
        for res in (got["argmax"], got["threshold"]):
            res["cv_rank"] = rank
        posthoc.extend([got["argmax"], got["threshold"]])
        if k not in by_key:
            pred_rows.append(got["preds"])

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(pred_rows, ignore_index=True).to_csv(
        out_dir / "predictions.csv", index=False)
    _write_yaml(out_dir / "config_used.yaml", dataset, chosen, used)

    write_metrics(
        out_dir / "metrics.json",
        experiment=f"graph2vec/tuned/{dataset}",
        description=(
            "Tuned CFG basic-block graphs from linear-sweep disassembly. "
            "Graph construction, node-label granularity, WL depth, histogram "
            "normalisation, embedding and classifier all selected by "
            "StratifiedGroupKFold(5) over family_or_group on TRAIN only; one "
            "test evaluation per selected configuration."),
        samples=split_summary(sc.F.split, "split"),
        results=results,
        elapsed_seconds=time.time() - t_start,
        floors=floors(y[te], sc.arch[te]),
        posthoc_top5=posthoc,
        reproducibility=None,        # filled in by --repro-check, see below
        selection=dict(protocol=chosen["selection"],
                       n_configurations_searched=chosen["n_configurations"],
                       cv_folds=N_FOLDS, seed=SEED,
                       cv_search_csv="cv_search.csv"),
        caveats=[
            "PV-DBOW (the graph2vec rows) is fitted once on the whole TRAIN "
            "split rather than per CV fold. It never sees a label, but a "
            "held-out fold's graphs did contribute to its label vectors, so "
            "its CV score is mildly optimistic relative to the sparse rows.",
            "The WL vocabulary (min_df) is taken over the whole TRAIN split; "
            "IDF, SVD, scaling and the classifier are refitted per fold.",
            "WL relabelling here is the hashed variant (order-independent sum "
            "of mixed neighbour hashes) rather than the sorted-tuple hash used "
            "by graph2vec_pipeline/cfg.wl_labels; same feature map up to "
            "64-bit collisions, ~100x faster on 20M nodes.",
            "`posthoc_top5` scores the five highest-CV configurations of the "
            "search on test. Only the ones that also appear in `results` were "
            "committed to before the test set was opened; the rest exist to "
            "size the CV->test gap and must not be read as a result.",
        ],
    )
    print(f"  wrote {out_dir/'metrics.json'}")
    return {"dataset": dataset, "results": results}


def repro_check(dataset: str, cache_dir: Path, out_dir: Path) -> dict:
    """Re-run the chosen configuration in a fresh process and diff.

    A second fit inside the run that produced the first one shares an
    interpreter, a loaded graph cache and a warm allocator, so it is the weaker
    of the two checks available. This one starts from the files: it reads back
    `predictions.csv`, rebuilds the graphs, refits `tuned_best` from scratch
    and compares the score of every test sample. The result is written into
    the existing metrics.json as `reproducibility` -- nothing else in the file
    is touched.

        python graph2vec_pipeline/final_eval.py --repro-check --dataset both
    """
    chosen = json.loads((out_dir / "chosen.json").read_text(encoding="utf-8"))
    best = next(p for p in chosen["picks"] if p["name"] == "tuned_best")
    prev = pd.read_csv(out_dir / "predictions.csv")
    prev = prev[prev["config_name"] == "tuned_best"].set_index("sha256")

    sc = _Scorer(dataset, cache_dir)
    sc.no_cache = True
    c, r, m = load_obj(best)
    print(f"[{dataset}] reproducibility: refitting {best['name']} from "
          f"scratch", flush=True)
    two = sc.run("tuned_best (repeat)", c, r, m, _cv_of(best))
    now = two["preds"].set_index("sha256").reindex(prev.index)

    d = float(np.abs(now["score"].to_numpy() - prev["score"].to_numpy()).max())
    same = {k: bool((now[k].to_numpy() == prev[k].to_numpy()).all())
            for k in ("pred_argmax", "pred_threshold")}
    doc_path = out_dir / "metrics.json"
    doc = json.loads(doc_path.read_text(encoding="utf-8"))
    first = next(x for x in doc["results"] if x["config_name"] == "tuned_best"
                 and x["decision"].startswith("argmax"))
    rep = {
        "config": "tuned_best",
        "protocol": ("the whole configuration -- graph assembly, WL "
                     "relabelling, min_df vocabulary, IDF, SVD, scaler and "
                     "classifier -- refitted in a separate process and "
                     "compared per test sample against predictions.csv"),
        "n_test_samples": int(len(prev)),
        "max_abs_score_difference": d,
        "identical_scores": d == 0.0,
        "identical_predictions_argmax": same["pred_argmax"],
        "identical_predictions_threshold": same["pred_threshold"],
        "macro_f1_run1": first["macro_f1"],
        "macro_f1_run2": two["argmax"]["macro_f1"],
        "threshold_run1": next(
            x["threshold"] for x in doc["results"]
            if x["config_name"] == "tuned_best" and "threshold" in x),
        "threshold_run2": two["thr"],
    }
    doc["reproducibility"] = rep
    doc_path.write_text(json.dumps(doc, indent=2, default=str),
                        encoding="utf-8")
    print(f"  max |score1-score2| = {d:.3e}; identical predictions "
          f"argmax={same['pred_argmax']} thr={same['pred_threshold']}; "
          f"updated {doc_path}", flush=True)
    return rep


def _write_yaml(path: Path, dataset: str, chosen: dict, used: list) -> None:
    def emit(v, ind):
        pad = " " * ind
        if isinstance(v, dict):
            return "\n".join(f"{pad}{k}:" + (
                "\n" + emit(x, ind + 2) if isinstance(x, (dict, list))
                else f" {_scalar(x)}") for k, x in v.items())
        if isinstance(v, list):
            out = []
            for x in v:
                if isinstance(x, (dict, list)):
                    body = emit(x, ind + 2)
                    out.append(f"{pad}-\n{body}")
                else:
                    out.append(f"{pad}- {_scalar(x)}")
            return "\n".join(out)
        return f"{pad}{_scalar(v)}"

    doc = {"dataset": dataset,
           "selection": chosen["selection"],
           "n_configurations_searched": chosen["n_configurations"],
           "cv_folds": N_FOLDS, "seed": SEED,
           "graph_cache": "graph2vec_pipeline/graph_cache.py "
                          "--max-blocks 20000 --windows {1,4}",
           "reproduce": [
               "python graph2vec_pipeline/graph_cache.py --max-blocks 20000",
               "python graph2vec_pipeline/graph_cache.py --max-blocks 20000 "
               "--windows 4",
               f"python graph2vec_pipeline/tune.py --dataset {dataset}",
               f"python graph2vec_pipeline/final_eval.py --dataset {dataset}",
               f"python graph2vec_pipeline/final_eval.py --repro-check "
               f"--dataset {dataset}"],
           "configurations": used}
    path.write_text(emit(doc, 0) + "\n", encoding="utf-8")


def _scalar(x):
    if x is None:
        return "null"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, str):
        return x if (x and all(ch.isalnum() or ch in "._-/+" for ch in x)) \
            else json.dumps(x)
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mendeley", "balanced", "both"],
                    default="both")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--out-dir",
                    default=str(REPO / "results" / "graph2vec" / "tuned"))
    ap.add_argument("--repro-check", action="store_true",
                    help="do not evaluate: refit the chosen configuration in "
                         "this fresh process, diff it against the existing "
                         "predictions.csv, and write the result into "
                         "metrics.json as `reproducibility`")
    a = ap.parse_args()
    dsets = ["mendeley", "balanced"] if a.dataset == "both" else [a.dataset]
    out = Path(a.out_dir)
    for ds in dsets:
        if a.repro_check:
            repro_check(ds, Path(a.cache_dir), out / ds)
        else:
            run_dataset(ds, Path(a.cache_dir), out / ds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
