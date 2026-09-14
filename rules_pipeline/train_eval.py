#!/usr/bin/env python3
"""Rule-based ransomware detection prototypes, on the shared cohort split.

    python rules_pipeline/train_eval.py --dataset both

(a) mined mnemonic n-gram rules        rules_pipeline/ngram_rules.py
(b) hand-written behaviour signatures  rules_pipeline/behaviour_rules.py
(c) one calibration baseline: mnemonic 1-3-gram TF-IDF + linear SVM / logistic
    regression. This is NOT a rule method; it is here so the rule numbers can
    be read against the cheapest strong classical model on the same features
    and the same split.

Everything -- vocabulary, supports, rule selection, thresholds, the decision
threshold -- is fitted on TRAIN. The val fold from cohort.add_val_fold picks
the score threshold for the weighted-vote classifiers, so that choice does not
come from the same rows the rules were mined on.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from cnn_vit_pipeline.cohort import (add_val_fold, build_result, load_split,  # noqa: E402
                                     split_summary, write_metrics)
from rules_pipeline import behaviour_rules as B  # noqa: E402
from rules_pipeline import ngram_rules as N  # noqa: E402

SHARED = Path(os.environ.get(
    "RANSOM_SHARED_DIR", REPO.parent / "asm and mm" / "Shared"))
TREES = {"mendeley": SHARED / "Extract",
         "balanced_goodware": SHARED / "Extract_Goodware_Balanced"}

SEED = 42
MAX_MNEMS = 30_000      # mnemonics read per file for n-gram mining
MAX_INSNS = 80_000      # instruction lines read per file for behaviour rules
POOL_PER_SIDE = 2000    # candidates kept per direction before rule selection
ALPHA = 0.5


def _tree(source: str) -> str:
    return "balanced_goodware" if source == "balanced_goodware" else "mendeley"


# ---------------------------------------------------------------------------
# mnemonic streams, stored as int16 ids (a raw list-of-str corpus is ~1GB)
# ---------------------------------------------------------------------------
class MnemCorpus:
    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.inv: dict[int, str] = {N.OOV: "<oov>"}
        self.cache: dict[str, np.ndarray] = {}

    def _id(self, tok: str) -> int:
        i = self.vocab.get(tok)
        if i is None:
            i = len(self.vocab) + 1
            if i >= N.BASE:            # would break the int64 4-gram code
                return N.OOV
            self.vocab[tok] = i
            self.inv[i] = tok
        return i

    def get(self, sha: str, path) -> np.ndarray:
        a = self.cache.get(sha)
        if a is None:
            toks = N.read_mnemonics(path, MAX_MNEMS)
            a = np.fromiter((self._id(t) for t in toks), dtype=np.int16,
                            count=len(toks))
            self.cache[sha] = a
        return a

    def text(self, a: np.ndarray) -> str:
        inv = self.inv
        return " ".join([inv[int(i)] for i in a])


def load_encoded(split: pd.DataFrame, corpus: MnemCorpus) -> list[np.ndarray]:
    return [corpus.get(r.sha256,
                       TREES[_tree(r.source)] / "mn" / f"{r.sha256}.txt")
            for r in split.itertuples()]


def load_behaviour(split: pd.DataFrame, cache: dict) -> list[dict]:
    feats = []
    for k, r in enumerate(split.itertuples(), 1):
        key = r.sha256
        if key not in cache:
            cache[key] = B.extract(
                TREES[_tree(r.source)] / "asm" / f"{key}.asm", MAX_INSNS)
        feats.append(cache[key])
        if k % 500 == 0:
            print(f"    behaviour features {k}/{len(split)}", flush=True)
    return feats


# ---------------------------------------------------------------------------
def _prf(y_true, pred):
    y_true = np.asarray(y_true)
    pred = np.asarray(pred, dtype=bool)
    tp = int(((y_true == 1) & pred).sum())
    fp = int(((y_true == 0) & pred).sum())
    fn = int(((y_true == 1) & ~pred).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(precision=round(prec, 4), recall=round(rec, 4),
                f1=round(f1, 4), fired=int(pred.sum()), tp=tp, fp=fp)


def _pick_threshold(scores, y, idx):
    from sklearn.metrics import f1_score
    cands = np.unique(np.quantile(scores[idx], np.linspace(0.01, 0.99, 99)))
    best, best_f1 = 0.0, -1.0
    for t in cands:
        f1 = f1_score(y[idx], (scores[idx] >= t).astype(int), average="macro")
        if f1 > best_f1:
            best, best_f1 = float(t), float(f1)
    return best, best_f1


# ---------------------------------------------------------------------------
def run(dataset: str, out_dir: Path, corpus: MnemCorpus, bh_cache: dict,
        min_support: int, top_k: int, max_n: int) -> dict:
    t_start = time.time()
    split = add_val_fold(load_split(dataset))
    y = split["label"].to_numpy()
    fold = split["fold"].to_numpy()
    arch = split["arch"].to_numpy()
    fam = split["family"].to_numpy()
    fname = split["filename"].to_numpy()
    is_train = (split["split"] == "train").to_numpy()
    tr_idx = np.flatnonzero(is_train)
    fit_idx = np.flatnonzero(fold == "train")    # train minus val
    val_idx = np.flatnonzero(fold == "val")
    te_idx = np.flatnonzero(~is_train)
    print(f"[{dataset}] {len(split)} samples, {len(tr_idx)} train "
          f"({len(fit_idx)} fit / {len(val_idx)} val), {len(te_idx)} test")

    results = []
    report: dict = {"dataset": dataset}

    # =====================================================================
    # (a) mined mnemonic n-gram rules
    # =====================================================================
    print("  (a) mnemonic n-gram rules")
    enc = load_encoded(split, corpus)
    print(f"    vocabulary {len(corpus.vocab)} mnemonics, "
          f"{np.mean([len(a) for a in enc]):.0f} mnemonics/file "
          f"(cap {MAX_MNEMS})")

    mined = N.mine(enc, y, fit_idx, min_support=min_support, max_n=max_n)
    n_r = int((y[fit_idx] == 1).sum())
    n_g = int((y[fit_idx] == 0).sum())

    rows = []
    for k in sorted(mined):
        codes, a, b = mined[k]
        sc = N.score_rules(a, b, n_r, n_g)
        for j in range(len(codes)):
            rows.append(dict(k=k, code=int(codes[j]), a=int(a[j]), b=int(b[j]),
                             **{s: float(sc[s][j]) for s in sc}))
    rules_df = pd.DataFrame(rows)
    if rules_df.empty:
        raise SystemExit("no n-gram reached the support threshold")
    print(f"    {len(rules_df):,} candidate n-grams above support")

    rules_df["direction"] = np.where(rules_df["f1_r"] >= rules_df["f1_g"],
                                     "ransomware", "goodware")
    # Keep a bounded pool: a dense hit matrix over every candidate would be GBs.
    pool = set(rules_df.sort_values("f1_r", ascending=False)
               .head(POOL_PER_SIDE).index.tolist())
    pool |= set(rules_df.sort_values("f1_g", ascending=False)
                .head(POOL_PER_SIDE).index.tolist())
    pool_df = rules_df.loc[sorted(pool)].sort_values(["k", "code"])
    pool_df = pool_df.reset_index(drop=True)
    codes_by_k = {int(k): np.sort(g["code"].to_numpy())
                  for k, g in pool_df.groupby("k")}
    hits = N.hit_matrix(enc, codes_by_k)     # columns follow (k asc, code asc)
    assert hits.shape[1] == len(pool_df), (hits.shape, len(pool_df))
    print(f"    hit matrix {hits.shape}")

    r_pool = pool_df.index[pool_df["direction"] == "ransomware"].to_numpy()
    g_pool = pool_df.index[pool_df["direction"] == "goodware"].to_numpy()
    r_order = r_pool[np.argsort(-pool_df.loc[r_pool, "f1_r"].to_numpy())]
    g_order = g_pool[np.argsort(-pool_df.loc[g_pool, "f1_g"].to_numpy())]

    sel_r, covered, total_pos = N.greedy_set_cover(
        hits, y, fit_idx, r_order, top_k, min_precision=0.80)
    sel_g, cov_g, tot_g = N.greedy_set_cover(
        hits, 1 - y, fit_idx, g_order, top_k, min_precision=0.80)
    print(f"    selected {len(sel_r)} ransomware rules (train coverage "
          f"{covered}/{total_pos}) and {len(sel_g)} goodware rules "
          f"({cov_g}/{tot_g})")

    def _emit(pred, score, model, **extra):
        r = build_result(y[te_idx], np.asarray(pred, dtype=int),
                         np.asarray(score, dtype=float), arch[te_idx],
                         fam[te_idx], track="ngram_rules", model=model, **extra)
        results.append(r)
        return r

    any_hit = (hits[:, sel_r].any(axis=1) if sel_r
               else np.zeros(len(y), dtype=bool))
    _emit(any_hit[te_idx], any_hit[te_idx], "any_hit", n_rules=len(sel_r),
          train_metrics=_prf(y[fit_idx], any_hit[fit_idx]))

    votes = (hits[:, sel_r].sum(axis=1) / max(len(sel_r), 1)
             - hits[:, sel_g].sum(axis=1) / max(len(sel_g), 1))
    t_v, f1_v = _pick_threshold(votes, y, val_idx if len(val_idx) else fit_idx)
    _emit(votes[te_idx] >= t_v, votes[te_idx], "rule_votes",
          n_rules=len(sel_r) + len(sel_g), threshold=round(t_v, 4),
          threshold_source="val fold", val_macro_f1=round(f1_v, 4))

    w = np.zeros(len(pool_df))
    lo = np.log(pool_df["odds"].to_numpy())
    w[sel_r] = lo[sel_r]
    w[sel_g] = lo[sel_g]
    wscore = hits.astype(np.float64) @ w
    t_w, f1_w = _pick_threshold(wscore, y, val_idx if len(val_idx) else fit_idx)
    _emit(wscore[te_idx] >= t_w, wscore[te_idx], "weighted_logodds",
          n_rules=len(sel_r) + len(sel_g), threshold=round(t_w, 4),
          threshold_source="val fold", val_macro_f1=round(f1_w, 4))

    def _rule_row(j, direction):
        r = pool_df.loc[j]
        ransom = direction == "ransomware"
        return dict(
            rule=N.decode(int(r["code"]), int(r["k"]), corpus.inv),
            n=int(r["k"]), direction=direction,
            train_ransomware=int(r["a"]), train_goodware=int(r["b"]),
            lift=round(float(r["lift"]), 2), odds=round(float(r["odds"]), 2),
            train_precision=round(float(r["precision_r"] if ransom
                                        else r["precision_g"]), 3),
            train_recall=round(float(r["recall_r"] if ransom
                                     else r["recall_g"]), 3),
            test=_prf(y[te_idx] if ransom else 1 - y[te_idx], hits[te_idx, j]),
        )

    top_lift = pool_df.sort_values("lift", ascending=False).head(15).index
    report["ngram"] = dict(
        candidates=int(len(rules_df)), pooled=int(len(pool_df)),
        min_support=min_support, max_n=max_n, max_mnemonics=MAX_MNEMS,
        vocabulary=len(corpus.vocab),
        per_level={int(k): int(len(v[0])) for k, v in mined.items()},
        selected_ransomware=[_rule_row(j, "ransomware") for j in sel_r],
        selected_goodware=[_rule_row(j, "goodware") for j in sel_g],
        top_by_lift=[_rule_row(j, "ransomware") for j in top_lift],
        train_coverage=dict(covered=int(covered), positives=int(total_pos)),
    )

    # =====================================================================
    # (b) hand-written behaviour rules
    # =====================================================================
    print("  (b) behaviour signatures")
    feats = load_behaviour(split, bh_cache)
    rules = B.fit_thresholds(B.RULES, feats, y, fit_idx)
    H = B.rule_hits(rules, feats)

    good_te = (y[te_idx] == 0)
    rtab = []
    for i, r in enumerate(rules):
        fp_names = fname[te_idx][good_te & H[te_idx, i]]
        rtab.append(dict(
            id=r.rid, name=r.name, kind=r.kind, description=r.description,
            threshold=(None if r.threshold is None else round(r.threshold, 4)),
            train=_prf(y[fit_idx], H[fit_idx, i]),
            test=_prf(y[te_idx], H[te_idx, i]),
            goodware_fire_rate_train=round(
                float(H[fit_idx, i][y[fit_idx] == 0].mean()), 4),
            goodware_fire_rate_test=round(float(H[te_idx, i][good_te].mean()), 4),
            ransomware_fire_rate_test=round(
                float(H[te_idx, i][y[te_idx] == 1].mean()), 4),
            goodware_false_positives_test=[str(s) for s in fp_names[:8]],
        ))
    report["behaviour"] = dict(max_insns=MAX_INSNS, rules=rtab)

    crypto_ids = ["R01", "R02", "R03", "R05", "R06"]
    ci = [i for i, r in enumerate(rules) if r.rid in crypto_ids]
    chit = H[:, ci].any(axis=1)
    results.append(build_result(
        y[te_idx], chit[te_idx].astype(int), chit[te_idx].astype(float),
        arch[te_idx], fam[te_idx], track="behaviour_rules",
        model="crypto_signature_any_hit", n_rules=len(ci), rules=crypto_ids,
        train_metrics=_prf(y[fit_idx], chit[fit_idx])))

    a_cnt = H[fit_idx][y[fit_idx] == 1].sum(axis=0).astype(float)
    b_cnt = H[fit_idx][y[fit_idx] == 0].sum(axis=0).astype(float)
    pr = (a_cnt + ALPHA) / (n_r + 2 * ALPHA)
    pg = (b_cnt + ALPHA) / (n_g + 2 * ALPHA)
    wb = np.log((pr / (1 - pr)) / (pg / (1 - pg)))
    bscore = H.astype(np.float64) @ wb
    t_b, f1_b = _pick_threshold(bscore, y, val_idx if len(val_idx) else fit_idx)
    results.append(build_result(
        y[te_idx], (bscore[te_idx] >= t_b).astype(int), bscore[te_idx],
        arch[te_idx], fam[te_idx], track="behaviour_rules",
        model="weighted_logodds", n_rules=len(rules), threshold=round(t_b, 4),
        threshold_source="val fold", val_macro_f1=round(f1_b, 4)))
    report["behaviour"]["weights"] = {r.rid: round(float(wb[i]), 3)
                                      for i, r in enumerate(rules)}

    # combined: behaviour rules + selected n-gram rules, one logistic model
    from sklearn.linear_model import LogisticRegression
    Xc = np.hstack([H, hits[:, sorted(set(sel_r) | set(sel_g))]]).astype(float)
    lr = LogisticRegression(max_iter=4000, random_state=SEED).fit(
        Xc[tr_idx], y[tr_idx])
    results.append(build_result(
        y[te_idx], lr.predict(Xc[te_idx]), lr.predict_proba(Xc[te_idx])[:, 1],
        arch[te_idx], fam[te_idx], track="combined",
        model="behaviour+ngram_rules_logreg", n_features=int(Xc.shape[1])))

    # =====================================================================
    # (c) calibration baseline
    # =====================================================================
    print("  (c) calibration baseline: mnemonic TF-IDF + LinearSVC / LogReg")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.svm import LinearSVC

    vec = TfidfVectorizer(analyzer="word", token_pattern=r"\S+",
                          ngram_range=(1, 3), min_df=5, sublinear_tf=True,
                          max_features=300_000, dtype=np.float32)
    Xtr_t = vec.fit_transform(corpus.text(enc[i]) for i in tr_idx)
    Xte_t = vec.transform(corpus.text(enc[i]) for i in te_idx)
    folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
    for name, clf, grid in (
            ("LinearSVC", LinearSVC(random_state=SEED, max_iter=5000),
             {"C": [0.1, 1, 10]}),
            ("LogReg", LogisticRegression(max_iter=4000, random_state=SEED),
             {"C": [0.1, 1, 10]})):
        gs = GridSearchCV(clf, grid, cv=folds, scoring="f1_macro", n_jobs=-1)
        gs.fit(Xtr_t, y[tr_idx])
        est = gs.best_estimator_
        pred = est.predict(Xte_t)
        sc = (est.decision_function(Xte_t)
              if hasattr(est, "decision_function") else pred)
        results.append(build_result(
            y[te_idx], pred, sc, arch[te_idx], fam[te_idx],
            track="calibration_baseline", model=f"mnemonic_tfidf_1_3+{name}",
            n_features=int(Xtr_t.shape[1]),
            best_params={k: str(v) for k, v in gs.best_params_.items()},
            cv_best_f1_macro=round(float(gs.best_score_), 4)))

    for r in results:
        print(f"    {r.get('track','?'):>22s} {r['model']:30s} "
              f"macroF1={r['macro_f1']:.3f} acc={r['accuracy']:.3f} "
              f"FPR={r['false_positive_rate']:.3f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_metrics(
        out_dir / "metrics.json",
        experiment=f"rules/{dataset}",
        description=("mined mnemonic n-gram rules + hand-written behaviour "
                     "signatures, learned on train only, on the shared cohort "
                     "split"),
        samples=split_summary(split, "fold"),
        results=results,
        elapsed_seconds=time.time() - t_start,
        config=dict(min_support=min_support, top_k=top_k, max_n=max_n,
                    pool_per_side=POOL_PER_SIDE, max_mnemonics=MAX_MNEMS,
                    max_insns=MAX_INSNS, seed=SEED),
        rule_report=report,
    )
    (out_dir / "rules.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"  wrote {out_dir/'metrics.json'} and rules.json")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mendeley", "balanced", "both"],
                    default="both")
    ap.add_argument("--min-support", type=int, default=40)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-n", type=int, default=4)
    ap.add_argument("--results-dir", default=str(REPO / "results" / "rules"))
    a = ap.parse_args()

    corpus = MnemCorpus()
    bh_cache: dict = {}
    for ds in (["mendeley", "balanced"] if a.dataset == "both" else [a.dataset]):
        run(ds, Path(a.results_dir) / ds, corpus, bh_cache,
            a.min_support, a.top_k, a.max_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
