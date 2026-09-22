"""
data.py - sample discovery, grouping and splitting for the binary
ransomware/goodware task.

Labels are binary everywhere: 0 = goodware, 1 = ransomware.

The ransomware filenames carry a family prefix (`avaddon_<sha256>.txt`). That
prefix is used ONLY as a split group. It is never a target: the upstream repos
classify 9 BIG-2015 families, this project answers one yes/no question.
"""

from __future__ import annotations

import collections
import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

# "avaddon_05af0cf4...f2.txt" -> family "avaddon"
FAMILY_RE = re.compile(r"^(?P<family>.+?)_[0-9a-f]{32,}\.txt$", re.IGNORECASE)


@dataclass
class Sample:
    path: Path
    label: int                 # 0 goodware, 1 ransomware
    source: str                # mendeley_ransomware | mendeley_goodware | balanced_goodware
    group: str                 # nothing in one group may straddle train/test
    split: str = ""            # train | test
    meta: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.path.name


def _family_of(name: str) -> str:
    m = FAMILY_RE.match(name)
    if not m:
        # every positive would otherwise share one group and one CV fold
        raise ValueError(f"ransomware file {name!r} does not match <family>_<sha>.txt; "
                         "the family is the split group and cannot be guessed")
    return m.group("family").lower()


def load_mendeley(root: Path, which: str) -> list[Sample]:
    """`which` is 'ransomware' or 'goodware'. Uses the release's own
    train/test folders; see config.yaml for why they are not re-split."""
    if which == "ransomware":
        folders, label, source = {"mal_train": "train", "mal_test": "test"}, 1, "mendeley_ransomware"
    else:
        folders, label, source = {"good_train": "train", "good_test": "test"}, 0, "mendeley_goodware"

    out: list[Sample] = []
    for folder, split in folders.items():
        d = root / folder
        if not d.is_dir():
            raise FileNotFoundError(f"missing {d}")
        # sorted(), not iterdir(): row order has to be reproducible because
        # the embedding matrix and the label vector are matched by position.
        for p in sorted(d.glob("*.txt")):
            group = _family_of(p.name) if label == 1 else f"file:{p.name}"
            out.append(Sample(p, label, source, group, split,
                              {"family": _family_of(p.name) if label == 1 else "",
                               # the release folder this file came out of; the
                               # cohort CSV's `set` column uses the same names,
                               # so the join can be scoped to one folder.
                               "cohort_set": folder}))
    return out


def load_balanced(pool_dir: Path, manifest: Path, group_field: str,
                  ungrouped: list[str]) -> list[Sample]:
    """Goodware_Balanced as an unsplit pool, with grouping metadata attached."""
    meta_by_txt: dict[str, dict] = {}
    if not manifest.is_file():
        raise FileNotFoundError(f"{manifest}: the balanced manifest is what groups and "
                                "de-duplicates the pool; without it the split is ungrouped")
    with manifest.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            meta_by_txt[row["txt_file"]] = row

    out = []
    for p in sorted(pool_dir.glob("*.txt")):
        meta = meta_by_txt.get(p.name, {})
        gid = meta.get(group_field) or ""
        # `system_local` is not a project: it is "whatever was copied out of
        # System32". Those binaries ship independently, so treating all 286 as
        # one atomic group would force the entire system bucket onto one side
        # of the split and leave the test set with no OS binaries at all.
        if not gid or gid in ungrouped:
            gid = f"file:{p.name}"
        out.append(Sample(p, 0, "balanced_goodware", gid, "", meta))
    return out


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dedup_goodware_sources(pool: list[Sample], mendeley_goodware: list[Sample],
                           manifest: Path, mendeley_sha_json: Path) -> tuple[list[Sample], dict]:
    """Drop any Goodware_Balanced sample that is also Mendeley goodware.

    Enforced at load time, not measured once and trusted. Two independent
    identities are checked, because either one alone can miss a duplicate:

    * **source-binary sha256** - `opcode_manifest.csv` records the sha256 of the
      PE each feature file came from, and `mendeley_goodware_sha256.json` lists
      the sha256 of every Mendeley goodware binary. This catches the same
      executable shipped under two different filenames.
    * **feature-file content hash** - the sha256 of the opcode text itself,
      against the Mendeley goodware feature files actually in play. This catches
      the case the first check cannot see: two *different* binaries whose
      disassembly is byte-identical (installer stubs, §2.1 of the audit), which
      would make an Exp B "unseen" negative a verbatim copy of an Exp A one.

    Returns the surviving pool and a report of what was removed. Measured today
    the removal count is 0 in both channels; the point of the function is that a
    future corpus refresh cannot silently reintroduce an overlap.
    """
    report = {"pool_in": len(pool), "removed_by_source_sha256": [],
              "removed_by_content_hash": [], "manifest_rows_without_sha256": 0}

    if not mendeley_sha_json.is_file():
        raise FileNotFoundError(f"{mendeley_sha_json}: needed for the source-sha256 dedup channel")
    loaded = json.loads(mendeley_sha_json.read_text(encoding="utf-8"))
    mendeley_bin_sha: set[str] = set(loaded if isinstance(loaded, list) else loaded.keys())
    report["mendeley_binary_sha256_known"] = len(mendeley_bin_sha)

    sha_by_txt: dict[str, str] = {}
    if not manifest.is_file():
        raise FileNotFoundError(f"{manifest}: needed for the source-sha256 dedup channel")
    with manifest.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            sha_by_txt[row["txt_file"]] = (row.get("sha256") or "").strip().lower()

    mendeley_content = {file_sha256(s.path) for s in mendeley_goodware}
    report["mendeley_goodware_feature_files"] = len(mendeley_goodware)

    kept = []
    for s in pool:
        src_sha = sha_by_txt.get(s.name, "")
        if not src_sha:
            report["manifest_rows_without_sha256"] += 1
        if src_sha and src_sha in mendeley_bin_sha:
            report["removed_by_source_sha256"].append(s.name)
            continue
        if file_sha256(s.path) in mendeley_content:
            report["removed_by_content_hash"].append(s.name)
            continue
        kept.append(s)

    report["pool_out"] = len(kept)
    report["removed_total"] = len(pool) - len(kept)
    return kept, report


def _apportion(strata: collections.Counter, n: int, total: int) -> dict:
    """Split n across strata proportionally, largest remainder first, so the
    parts sum to exactly n."""
    exact = {k: v * n / total for k, v in strata.items()}
    out = {k: int(v) for k, v in exact.items()}
    short = n - sum(out.values())
    for k in sorted(exact, key=lambda k: -(exact[k] - out[k]))[:short]:
        out[k] += 1
    return out


def group_split(samples: list[Sample], n_train: int, n_test: int,
                seed: int, strat_field: str = "bucket") -> list[Sample]:
    """Assign whole groups to train or test, preserving `strat_field`
    proportions and hitting the requested sizes as nearly as group granularity
    allows. Returns only the selected samples, `split` filled in.

    Split is done independently *within* each stratum. A single global greedy
    pass does not work here: it places large groups first, so by the time the
    286 one-file `system_local` groups come up the test quota is already full
    and the test set ends up with zero OS binaries - exactly the category the
    corpus was balanced to include. Per-stratum quotas make that impossible.

    Every group lies in exactly one stratum by construction (an entry_id is
    downloaded into one bucket, and per-file groups are single files), so no
    group is ever torn across strata.
    """
    rng = random.Random(seed)
    by_group: dict[str, list[Sample]] = collections.defaultdict(list)
    for s in samples:
        by_group[s.group].append(s)

    total = len(samples)
    if n_train + n_test > total:
        raise ValueError(f"asked for {n_train + n_test} samples, pool has {total}")

    strata = collections.Counter(s.meta.get(strat_field, "") for s in samples)
    groups_by_strat: dict[str, list[list[Sample]]] = collections.defaultdict(list)
    for members in by_group.values():
        # dominant stratum, though in practice a group is homogeneous
        key = collections.Counter(m.meta.get(strat_field, "") for m in members).most_common(1)[0][0]
        groups_by_strat[key].append(members)

    # Largest-remainder apportionment so the per-stratum quotas sum to exactly
    # n_train and n_test. Plain round() drifts by a sample or two per stratum,
    # which is enough to break the "A and B have identical class sizes" claim
    # the whole comparison rests on.
    want = {"train": _apportion(strata, n_train, total),
            "test": _apportion(strata, n_test, total)}

    chosen: list[Sample] = []
    for strat, size in strata.items():
        want_train = want["train"][strat]
        want_test = want["test"][strat]
        groups = groups_by_strat[strat]
        rng.shuffle(groups)
        groups.sort(key=lambda g: -len(g))

        have_train = have_test = 0
        for members in groups:
            k = len(members)
            fits_train = have_train + k <= want_train
            fits_test = have_test + k <= want_test
            d_train = want_train - have_train
            d_test = want_test - have_test
            if fits_test and (not fits_train or d_test > d_train):
                side = "test"
            elif fits_train:
                side = "train"
            elif fits_test:
                side = "test"
            else:
                continue  # both quotas full; group sits out this experiment
            for m in members:
                m.split = side
            chosen.extend(members)
            if side == "train":
                have_train += k
            else:
                have_test += k

    return chosen


def summarize(samples: list[Sample]) -> dict:
    out: dict = {"total": len(samples)}
    for split in ("train", "test"):
        sub = [s for s in samples if s.split == split]
        out[split] = {
            "n": len(sub),
            "goodware": sum(1 for s in sub if s.label == 0),
            "ransomware": sum(1 for s in sub if s.label == 1),
            "sources": dict(collections.Counter(s.source for s in sub)),
            "groups": len({s.group for s in sub}),
        }
    tr = {s.group for s in samples if s.split == "train"}
    te = {s.group for s in samples if s.split == "test"}
    out["group_overlap"] = sorted(tr & te)
    return out


def content_leak(samples: list[Sample]) -> dict:
    """How much of the test set is an exact copy of something in train?

    Grouping stops *related* samples straddling the split; this catches the
    stronger case where the opcode streams are byte-identical. It happens a
    lot: extract.py disassembles the executable section of whatever it is
    handed, so one NSIS/Inno installer stub yields one stream no matter what
    payload it wraps. In LLM_Features that makes 64/131 = 48.9% of
    good_test a verbatim copy of some training file (62 of a good_train file,
    2 of a mal_train file), and two installer-wrapped samples carry one
    identical stream under both labels.

    Reported, not repaired: dropping the duplicates would change the corpus the
    baseline was published on. It belongs in every results table as a ceiling
    on how much the goodware score means.
    """
    digest: dict[Path, str] = {s.path: file_sha256(s.path) for s in samples}

    train_hashes = {digest[s.path] for s in samples if s.split == "train"}
    by_hash: dict[str, set] = collections.defaultdict(set)
    for s in samples:
        by_hash[digest[s.path]].add(s.label)

    out = {}
    for label, tag in ((0, "goodware"), (1, "ransomware")):
        test = [s for s in samples if s.split == "test" and s.label == label]
        hit = [s for s in test if digest[s.path] in train_hashes]
        out[f"test_{tag}_duplicated_in_train"] = len(hit)
        out[f"test_{tag}_n"] = len(test)
        out[f"test_{tag}_leak_rate"] = round(len(hit) / len(test), 4) if test else None

    out["unique_streams"] = len({digest[s.path] for s in samples})
    out["total_files"] = len(samples)
    contradictions = [h for h, labels in by_hash.items() if len(labels) > 1]
    out["streams_labelled_both_classes"] = len(contradictions)
    return out


# ---------------------------------------------------------------------------
# Cohort metadata (architecture, packing tag, family) and the cohort filter
# ---------------------------------------------------------------------------
#
# `cohort_mendeley.csv` / `cohort_balanced.csv` are produced by the unified
# extractor's profiling pass. One row per input binary, whether it survived or
# not, with columns:
#
#     corpus sha256 set label family filename arch tag in_cohort exclude_reason
#
# They are the first source in this project that carries ARCHITECTURE for every
# file of every set - including `good_test` and the whole ransomware side, which
# the earlier audit could not obtain (the architecture note in
# results/summary.md said so explicitly; that limitation is now lifted).
#
# JOIN KEY. Both feature writers (`extract.py` and `asm_tool/mn_to_features.py`)
# name their output `<family>_<filename>.txt`, so a row's own `family` +
# `filename` reconstructs the feature filename exactly. That is the key used
# here.
#
# Joining ransomware on sha256 instead looks natural - the Mendeley ransomware
# filenames ARE sha256 strings - but it is wrong for 25 of 1,408 rows whose
# recorded sha256 differs from the sha256 in their filename. Two of those
# collide: `darkside_4d9432e8....txt` and `darkside_ec368752....txt` are two
# different samples, and a sha256-first join maps BOTH onto the single row whose
# filename is 4d9432e8 and whose sha256 is ec368752. Measured, that mis-join
# inflates the traditional cohort by 2 files in mal_train and 3 in mal_test and
# silently double-counts one row. The filename join is a clean bijection:
# 0 unmatched files and 0 rows claimed twice across all six folders.
#
# `tag == "dup"` rows share a sha256 with a kept row and are always
# `in_cohort == 0`; where a lookup is ambiguous the in-cohort row wins.


class Cohort:
    """Index over one cohort CSV, keyed by the feature filename it implies."""

    def __init__(self, rows, path=None):
        self.path = path
        self.rows = rows
        self._by_set_name = {}
        self._by_name = {}
        for r in rows:
            name = f"{r['family']}_{r['filename']}.txt".lower()
            for key, idx in (((r["set"], name), self._by_set_name),
                             (name, self._by_name)):
                prev = idx.get(key)
                # An in-cohort row always beats a `dup`/excluded one.
                if prev is None or (prev.get("in_cohort") != "1"
                                    and r.get("in_cohort") == "1"):
                    idx[key] = r

    def __len__(self):
        return len(self.rows)

    def lookup(self, name, set_=""):
        n = name.lower()
        if set_:
            hit = self._by_set_name.get((set_, n))
            if hit is not None:
                return hit
        return self._by_name.get(n)


def load_cohort(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"cohort csv not found: {path}")
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    need = {"set", "label", "family", "filename", "arch", "tag", "in_cohort"}
    missing = need - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return Cohort(rows, path)


def annotate_cohort(samples, cohort, set_field="cohort_set"):
    """Attach `arch`, `cohort_tag`, `in_cohort` and the cohort's `family` to
    every sample's meta. Never drops anything - see `filter_cohort` for that.

    Annotation is deliberately separate from filtering so the *unfiltered*
    experiments (Exp A, Exp B) can be reported per architecture as well, which
    is what closes the open question in results/summary.md.
    """
    report = {"cohort_csv": str(cohort.path), "cohort_rows": len(cohort),
              "annotated": 0, "unmatched_examples": [], "in_cohort": 0}
    unmatched = 0
    for s in samples:
        row = cohort.lookup(s.name, s.meta.get(set_field, ""))
        if row is None:
            unmatched += 1
            if len(report["unmatched_examples"]) < 10:
                report["unmatched_examples"].append(s.name)
            s.meta.setdefault("arch", "")
            s.meta["cohort_tag"] = ""
            s.meta["in_cohort"] = False
            continue
        report["annotated"] += 1
        # The cohort CSV's arch is authoritative: it is read from the PE header
        # by the profiling pass, and it exists for every row, while
        # `opcode_manifest.csv` only covers Goodware_Balanced.
        s.meta["arch"] = row["arch"] or ""
        s.meta["cohort_tag"] = row["tag"]
        s.meta["cohort_sha256"] = row.get("sha256", "")
        s.meta["exclude_reason"] = row.get("exclude_reason", "")
        s.meta["in_cohort"] = row["in_cohort"] == "1"
        if s.label == 1 and row["family"]:
            s.meta["family"] = row["family"]
        if s.meta["in_cohort"]:
            report["in_cohort"] += 1
    report["unmatched"] = unmatched
    return report


def filter_cohort(samples):
    """Keep only samples the cohort marks `in_cohort == 1`.

    A sample with no cohort row at all is DROPPED, not kept: an unmatched file
    is a file whose packing status and architecture are unknown, and the point
    of the cohort variant is that every member has been profiled.
    """
    kept, dropped = [], collections.Counter()
    for s in samples:
        if s.meta.get("in_cohort"):
            kept.append(s)
        else:
            dropped[s.meta.get("exclude_reason") or "no_cohort_row"] += 1
    return kept, {"in": len(samples), "out": len(kept),
                  "dropped": len(samples) - len(kept),
                  "dropped_by_reason": dict(sorted(dropped.items()))}


def reuse_split(pool, splits_csv, source):
    """Assign `split` from an already-committed splits.csv instead of re-splitting.

    Exp B's goodware split was drawn once, with a seed, by `group_split`, and
    committed. Exp B_cohort and Exp D must use *that* membership rather than
    re-drawing it: re-running `group_split` on a pool the cohort filter has
    changed would move whole projects between train and test, and the resulting
    numbers would then differ from Exp B for two reasons at once instead of one.

    Files in the pool that the committed split never selected are excluded, and
    files the split names but the pool no longer contains are reported.
    """
    splits_csv = Path(splits_csv)
    if not splits_csv.is_file():
        raise FileNotFoundError(f"cannot reuse split, missing {splits_csv}")
    want = {}
    with splits_csv.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["source"] == source:
                want[row["file"]] = (row["split"], row.get("group", ""))

    chosen, regrouped = [], 0
    for s in pool:
        hit = want.get(s.name)
        if not hit:
            continue
        s.split, group = hit
        # The committed splits.csv is the authoritative record of the split
        # being reused, groups included. Taking the group from it matters for
        # the REVISED Goodware_Balanced pool, whose manifest carries no
        # `entry_id`: without this the group would fall back to `file:<name>`
        # and the train/test group-overlap assertion would become vacuous
        # exactly where the project-disjointness has to be proven.
        if group and group != s.group:
            regrouped += 1
            s.group = group
        chosen.append(s)
    got = collections.Counter(s.split for s in chosen)
    absent = sorted(set(want) - {s.name for s in pool})
    return chosen, {
        "splits_csv": str(splits_csv), "source": source,
        "named_by_split": len(want), "pool": len(pool), "selected": len(chosen),
        "selected_train": got.get("train", 0), "selected_test": got.get("test", 0),
        "groups_taken_from_splits_csv": regrouped,
        "named_but_absent_from_pool": len(absent),
        "named_but_absent_examples": absent[:10],
    }


def arch_breakdown(samples):
    """x86/x64 counts per split and class - the confound, made countable."""
    out = {}
    for split in ("train", "test"):
        for label, tag in ((0, "goodware"), (1, "ransomware")):
            c = collections.Counter(
                s.meta.get("arch") or "unknown"
                for s in samples if s.split == split and s.label == label)
            if c:
                out[f"{split}_{tag}"] = dict(sorted(c.items()))
    return out


def _floor(tn, fp, fn, tp):
    """Metrics for one fixed rule, in the same shape the classifiers report."""
    def prf(hit, over, miss):
        p = hit / (hit + over) if hit + over else 0.0
        r = hit / (hit + miss) if hit + miss else 0.0
        return (2 * p * r / (p + r)) if p + r else 0.0, r

    f_good, r_good = prf(tn, fn, fp)
    f_ran, r_ran = prf(tp, fp, fn)
    n = tn + fp + fn + tp
    return {"accuracy": round((tn + tp) / n, 6) if n else 0.0,
            "balanced_accuracy": round((r_good + r_ran) / 2, 6),
            "macro_f1": round((f_good + f_ran) / 2, 6),
            "recall_goodware": round(r_good, 6),
            "recall_ransomware": round(r_ran, 6),
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}


def baselines(arch_block: dict) -> dict:
    """Two label-free floors for the test set, so no score is read on its own.

    Both are computed from `arch_breakdown`'s test counts alone, which is why
    they can also be recovered from an already-committed `metrics.json` without
    re-running anything.

    `majority_class` predicts the larger TEST class for every file. That is the
    floor `majority_class_accuracy` already reports per result, restated as
    macro-F1 and balanced accuracy so it sits in the same units as the tables.
    It peeks at the test label distribution - deliberately, because a floor
    should be the most generous trivial rule, not the fairest one.

    `x86_is_ransomware` predicts ransomware for every x86 file and goodware for
    everything else, reading ONLY the architecture recorded in
    cohort_mendeley.csv / cohort_balanced.csv - never an opcode. Ransomware here
    is ~80% x86 on the test side and Goodware_Balanced is ~18-35% x86, so this
    rule is not weak, and any model that does not clear it has not been shown to
    use the code at all. Files whose architecture is `unknown` count as not-x86,
    i.e. predicted goodware.
    """
    good = dict(arch_block.get("test_goodware", {}))
    ran = dict(arch_block.get("test_ransomware", {}))
    n_good, n_ran = sum(good.values()), sum(ran.values())
    out = {}

    if n_ran >= n_good:
        out["majority_class"] = dict(predicts="ransomware",
                                     **_floor(0, n_good, 0, n_ran))
    else:
        out["majority_class"] = dict(predicts="goodware",
                                     **_floor(n_good, 0, n_ran, 0))

    out["x86_is_ransomware"] = dict(
        rule="predict ransomware iff arch == x86, from the cohort CSV",
        test_x86=good.get("x86", 0) + ran.get("x86", 0),
        test_unknown_arch=good.get("unknown", 0) + ran.get("unknown", 0),
        **_floor(tn=n_good - good.get("x86", 0), fp=good.get("x86", 0),
                 fn=n_ran - ran.get("x86", 0), tp=ran.get("x86", 0)))
    return out


def family_breakdown(samples):
    """Ransomware family sizes per split. Families are split groups, not
    labels; the counts exist so a per-family recall table has denominators."""
    out = {}
    for split in ("train", "test"):
        c = collections.Counter(s.meta.get("family") or "unknown"
                                for s in samples if s.split == split and s.label == 1)
        out[split] = dict(sorted(c.items()))
    return out
