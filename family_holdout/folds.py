#!/usr/bin/env python3
"""
folds.py - fold definition for family-holdout evaluation over every cohort family.

Why: the fixed Mendeley split (24 train / 14 test families) gave every pipeline
one number, and the tuning study showed that group cross-validation inside the
24 training families does not predict the 14 held-out ones. Evaluating on
every family in turn removes the dependence on which families happened to
land on which side.

Pool
    ransomware  every in-cohort file from mal_train and mal_test: 1,266 files,
                38 families (Thanos is .NET, Night Sky is 100% packed; neither
                can enter the opcode track, so "all 40" is 38 here).
                Extra ransomware corpora (--ransomware NAME=cohort.csv) are
                pooled with Mendeley: same family name means the same family,
                held out together.
    goodware    dataset "mendeley": in-cohort good_train + good_test (1,243)
                dataset "balanced": in-cohort Goodware_Balanced (1,337)

Folds (K = 5, deterministic)
    ransomware  families assigned whole to folds by greedy count balancing
                (largest family first, into the currently smallest fold).
                The assignment is shared by both datasets. With --arch-aware
                the balancer keys on (files, x64 files) so x64 ransomware is
                spread over the folds instead of concentrating in one.
    goodware    assigned by GROUP, never by file, so that duplicates and
                same-project files never straddle a fold:
                  mendeley  group = SHA-256 of the mnemonic stream (60/47/35-file
                            duplicate groups exist; 1,024 distinct streams)
                  balanced  group = source project (entry_id, 103 projects)
                greedy count balancing as above.

Cross-corpus rules
    family names are normalised (lower case, alphanumerics only) and mapped
    through ALIASES onto the Mendeley spelling; a sha256 seen in an earlier
    corpus is dropped from a later one (Mendeley wins); a sha256 that appears
    with both labels aborts the build. With --pool-small, families with fewer
    than SMALL_FAMILY files are assigned to folds as one `_small` group; they
    keep their own family name for LOFO. --arch-aware and --pool-small change
    the assignment, so the committed fold files (and every result under them)
    were built without either flag.

Two evaluation schemes are derived from the same file:
    K-fold        train on 4 folds, test on the 5th; report mean +/- sd over
                  folds and pooled per-family / per-arch metrics (every file
                  gets exactly one held-out prediction).
    LOFO          leave one family out: train on all other ransomware families
                  plus ALL goodware, test on the held-out family alone; gives
                  per-family recall with maximal training data. FPR is not
                  defined for LOFO (no goodware in the test set), so LOFO is a
                  recall study only; use the K-fold numbers for everything else.

Output (--out DIR)
    folds_mendeley.csv, folds_balanced.csv   dataset, sha256, label, family,
                                             group, arch, orig_set, fold, corpus
    folds_all.csv (with --goodware / --all)  every ransomware corpus against
                                             every goodware source, goodware
                                             groups balanced jointly
    families.csv                             family, n, n_x64, fold, orig_set, corpus

Usage
    python family_holdout/folds.py --out results/family_holdout
    python family_holdout/folds.py --out results/family_holdout \
        --ransomware vs=C:/.../Shared/cohort_vs.csv --arch-aware

Paths: RANSOM_SHARED_DIR (the `Shared` folder) and RANSOM_BALANCED_INDEX (the
Goodware_Balanced corpus_index.csv) override the defaults below.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import os
import re
from pathlib import Path

K = 5
SHARED = Path(os.environ.get("RANSOM_SHARED_DIR", r"C:/Users/chaoa/Downloads/asm and mm/Shared"))
COHORT = {"mendeley": SHARED / "cohort_mendeley.csv", "balanced": SHARED / "cohort_balanced.csv"}
UNIFIED_MANIFEST = SHARED / "Extract" / "manifest.csv"
CORPUS_INDEX = Path(os.environ.get("RANSOM_BALANCED_INDEX",
                                   r"C:/Users/chaoa/Downloads/Goodware_Balanced/corpus_index.csv"))
FIELDS = ["dataset", "sha256", "label", "family", "group", "arch", "orig_set", "fold", "corpus"]
SMALL_FAMILY = 5

# other corpora's names for a family Mendeley already has, mapped onto the
# Mendeley spelling so that existing results, tests and summaries keep their keys
ALIASES = {
    "sodinokibi": "revil", "djvu": "stop", "stopdjvu": "stop", "alphv": "blackcat",
    "noberus": "blackcat", "crysis": "dharma", "mespinoza": "pysa",
    "mailto": "netwalker", "play": "playcrypt",
    "agenda": "qilin", "cl0p": "clop", "basta": "blackbasta", "defray777": "ransomexx",
    # NOT aliases, kept distinct on purpose: MedusaLocker (2019) vs Medusa (2021),
    # Nemty vs Nefilim (a successor, not the same code base).
}


def normalise_family(name: str) -> str:
    n = re.sub(r"[^a-z0-9]", "", str(name).lower())
    return ALIASES.get(n, n)


def _read(path: Path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def assign_greedy(counts: dict, k: int = K) -> dict:
    """Largest group first into the currently lightest fold. Ties broken by name,
    so the result is a pure function of the input.

    A count is an int (files) or a (x64 files, files) pair. With pairs the
    fold load is the sum of the two shares, x64 / total_x64 + files /
    total_files, so the scarce architecture and the file count are balanced
    together; groups are placed in order of x64 count, then size. With ints
    the behaviour is the original count balancing."""
    pairs = {n: ((c, 0) if isinstance(c, int) else tuple(c)) for n, c in counts.items()}
    weighted = any(not isinstance(c, int) for c in counts.values())
    tot0 = sum(p[0] for p in pairs.values()) or 1
    tot1 = sum(p[1] for p in pairs.values()) or 1
    folds = [(0, 0)] * k
    out = {}

    def load(i):
        a, b = folds[i]
        return (a / tot0 + b / tot1, i) if weighted else (a, i)

    for name in sorted(pairs, key=lambda n: (-pairs[n][0], -pairs[n][1], n)):
        f = min(range(k), key=load)
        out[name] = f
        folds[f] = (folds[f][0] + pairs[name][0], folds[f][1] + pairs[name][1])
    return out


def ransomware_rows(extra: dict | None = None, arch_aware: bool = False, pool_small: bool = False):
    """All in-cohort ransomware rows of Mendeley plus every extra corpus, with
    families pooled across corpora, sha256 de-duplicated (first corpus wins) and
    a fold per family. Returns (rows, family -> fold)."""
    sources = [("mendeley", COHORT["mendeley"])] + list((extra or {}).items())
    rows, seen, dropped = [], {}, collections.Counter()
    for corpus, path in sources:
        for r in _read(Path(path)):
            if r["in_cohort"] != "1" or r["label"] != "1":
                continue
            sha = r["sha256"].lower()
            if sha in seen:
                dropped[corpus] += 1
                continue
            seen[sha] = corpus
            rows.append({"sha256": sha, "label": 1, "family": normalise_family(r["family"]),
                         "arch": r["arch"], "orig_set": r["set"], "corpus": corpus})
    if dropped:
        print("ransomware sha256 already seen in an earlier corpus, dropped: "
              + ", ".join(f"{c}={n}" for c, n in sorted(dropped.items())))
    fam_n = collections.Counter(r["family"] for r in rows)
    unit = {f: (f if fam_n[f] >= SMALL_FAMILY or not pool_small else "_small") for f in fam_n}
    counts = collections.Counter(unit[r["family"]] for r in rows)
    if arch_aware:
        # (x64 files, files): the scarce architecture is balanced first, file
        # counts second, so x64 ransomware cannot pile into one fold
        x64 = collections.Counter(unit[r["family"]] for r in rows if r["arch"] == "x64")
        counts = {u: (x64.get(u, 0), counts[u]) for u in counts}
    unit_fold = assign_greedy(counts)
    fam_fold = {f: unit_fold[unit[f]] for f in fam_n}
    for r in rows:
        r["group"] = r["family"]
        r["fold"] = fam_fold[r["family"]]
    return rows, fam_fold


def mendeley_goodware_rows():
    rows = [r for r in _read(COHORT["mendeley"]) if r["in_cohort"] == "1" and r["label"] == "0"]
    mn_of = {}
    for m in _read(UNIFIED_MANIFEST):
        if m["mn_file"]:
            mn_of.setdefault(m["sha256"], m["mn_file"])
    stream = {}
    for r in rows:
        if r["sha256"] not in mn_of:
            raise KeyError(f"{r['sha256']} is in the cohort but has no mn_file in {UNIFIED_MANIFEST}")
        p = SHARED / "Extract" / mn_of[r["sha256"]]
        stream[r["sha256"]] = "mn:" + hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    grp_fold = assign_greedy(collections.Counter(stream.values()))
    return [{"sha256": r["sha256"], "label": 0, "family": "goodware", "group": stream[r["sha256"]],
             "arch": r["arch"], "orig_set": r["set"], "fold": grp_fold[stream[r["sha256"]]],
             "corpus": "mendeley"} for r in rows]


def balanced_goodware_rows():
    rows = [r for r in _read(COHORT["balanced"]) if r["in_cohort"] == "1" and r["label"] == "0"]
    entry = {r["sha256"]: r["entry_id"] for r in _read(CORPUS_INDEX)}
    groups = {r["sha256"]: "proj:" + (entry.get(r["sha256"]) or "unknown:" + r["sha256"][:8]) for r in rows}
    grp_fold = assign_greedy(collections.Counter(groups.values()))
    return [{"sha256": r["sha256"], "label": 0, "family": "goodware", "group": groups[r["sha256"]],
             "arch": r["arch"], "orig_set": r["family"], "fold": grp_fold[groups[r["sha256"]]],
             "corpus": "balanced"} for r in rows]


def extra_goodware_rows(name: str, path: Path) -> list:
    """In-cohort goodware rows of one extra cohort file. The file must carry a
    `group` column (program / project identity); files of one group never
    straddle a fold. Folds are assigned here per corpus; all_goodware_rows()
    re-assigns them jointly."""
    rows = [r for r in _read(Path(path)) if r["in_cohort"] == "1" and r["label"] == "0"]
    if rows and "group" not in rows[0]:
        raise ValueError(f"{path}: an extra goodware cohort needs a `group` column "
                         "(e.g. from tools/build_cohort.py --index)")
    groups = {r["sha256"].lower(): (r["group"] or "file:" + r["sha256"][:8]) for r in rows}
    grp_fold = assign_greedy(collections.Counter(groups.values()))
    return [{"sha256": r["sha256"].lower(), "label": 0, "family": "goodware", "group": groups[r["sha256"].lower()],
             "arch": r["arch"], "orig_set": r["set"], "fold": grp_fold[groups[r["sha256"].lower()]],
             "corpus": name} for r in rows]


def all_goodware_rows(extra: dict | None = None) -> list:
    """Every goodware source in one pool (Mendeley, Goodware_Balanced, extras),
    sha256 de-duplicated in that order, groups assigned to folds jointly so the
    five folds are balanced over the whole pool."""
    rows, seen = [], set()
    sources = [mendeley_goodware_rows(), balanced_goodware_rows()]
    sources += [extra_goodware_rows(n, p) for n, p in (extra or {}).items()]
    for src in sources:
        for r in src:
            if r["sha256"] in seen:
                continue
            seen.add(r["sha256"])
            rows.append(dict(r))
    grp_fold = assign_greedy(collections.Counter(r["group"] for r in rows))
    for r in rows:
        r["fold"] = grp_fold[r["group"]]
    return rows


def check_dataset(ds: str, rows: list) -> None:
    """A sha256 appears once, and never with both labels; arch is x86 or x64."""
    by_sha = collections.defaultdict(set)
    for r in rows:
        by_sha[r["sha256"]].add(r["label"])
    dup = [s for s, labels in by_sha.items() if len(labels) > 1]
    if dup:
        raise ValueError(f"{ds}: {len(dup)} sha256 appear as both goodware and ransomware, e.g. {dup[:3]}")
    n = collections.Counter(r["sha256"] for r in rows)
    twice = [s for s, c in n.items() if c > 1]
    if twice:
        raise ValueError(f"{ds}: {len(twice)} sha256 listed twice, e.g. {twice[:3]}")
    bad = sorted({r["arch"] for r in rows} - {"x86", "x64"})
    if bad:
        raise ValueError(f"{ds}: unexpected arch values {bad}; the x86-rule floor and per-arch "
                         "metrics assume x86/x64 only")


def build(out: Path, extra: dict | None = None, arch_aware: bool = False,
          pool_small: bool = False, extra_goodware: dict | None = None,
          build_all: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rans, fam_fold = ransomware_rows(extra, arch_aware, pool_small)
    datasets = {"mendeley": rans + mendeley_goodware_rows(), "balanced": rans + balanced_goodware_rows()}
    if extra_goodware or build_all:
        # the joint dataset: every ransomware corpus against every goodware source
        datasets["all"] = rans + all_goodware_rows(extra_goodware)
    for ds, rows in datasets.items():
        check_dataset(ds, rows)
        with (out / f"folds_{ds}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            for r in sorted(rows, key=lambda r: (r["fold"], r["label"], r["family"], r["sha256"])):
                w.writerow({"dataset": ds, **r})
    fam_n = collections.Counter(r["family"] for r in rans)
    fam_x64 = collections.Counter(r["family"] for r in rans if r["arch"] == "x64")
    with (out / "families.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "n", "n_x64", "fold", "orig_set", "corpus"])
        orig = {r["family"]: r["orig_set"] for r in rans}
        corp = collections.defaultdict(set)
        for r in rans:
            corp[r["family"]].add(r["corpus"])
        for f in sorted(fam_n, key=lambda f: (fam_fold[f], f)):
            w.writerow([f, fam_n[f], fam_x64[f], fam_fold[f], orig[f], "+".join(sorted(corp[f]))])
    return datasets


def load(ds: str, out: Path):
    return _read(out / f"folds_{ds}.csv")


def summary(datasets: dict) -> str:
    L = []
    for ds, rows in datasets.items():
        L.append(f"{ds}: {len(rows)} files")
        for f in range(K):
            fr = [r for r in rows if r["fold"] == f]
            g = sum(r["label"] == 0 for r in fr); m = len(fr) - g
            fams = sorted({r["family"] for r in fr if r["label"] == 1})
            x64 = sum(r["arch"] == "x64" for r in fr if r["label"] == 1)
            L.append(f"  fold {f}: {len(fr):4d} files  goodware {g:4d}  ransomware {m:4d} (x64 {x64:3d})  families {len(fams):2d}: {', '.join(fams)}")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/family_holdout")
    ap.add_argument("--ransomware", action="append", default=[], metavar="NAME=COHORT.csv",
                    help="extra ransomware cohort file(s), pooled with Mendeley by family")
    ap.add_argument("--arch-aware", action="store_true",
                    help="balance x64 ransomware across folds as well as file counts")
    ap.add_argument("--pool-small", action="store_true",
                    help=f"assign families with fewer than {SMALL_FAMILY} files to folds as one group")
    ap.add_argument("--goodware", action="append", default=[], metavar="NAME=COHORT.csv",
                    help="extra goodware cohort file(s) with a `group` column; implies --all")
    ap.add_argument("--all", action="store_true",
                    help="also write folds_all.csv: all ransomware corpora vs all goodware sources")
    a = ap.parse_args()

    def specs(items, flag):
        out = {}
        for spec in items:
            name, _, path = spec.partition("=")
            if not path:
                ap.error(f"{flag} expects NAME=PATH, got {spec!r}")
            out[name] = Path(path)
        return out
    extra = specs(a.ransomware, "--ransomware")
    extra_good = specs(a.goodware, "--goodware")
    datasets = build(Path(a.out), extra, a.arch_aware, a.pool_small, extra_good, a.all)
    print(summary(datasets))
    print("wrote " + ", ".join(f"folds_{ds}.csv" for ds in datasets) + f", families.csv under {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
