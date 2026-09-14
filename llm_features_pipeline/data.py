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
    return m.group("family").lower() if m else "unknown"


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
                              {"family": _family_of(p.name) if label == 1 else ""}))
    return out


def load_balanced(pool_dir: Path, manifest: Path, group_field: str,
                  ungrouped: list[str]) -> list[Sample]:
    """Goodware_Balanced as an unsplit pool, with grouping metadata attached."""
    meta_by_txt: dict[str, dict] = {}
    if manifest.is_file():
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

    mendeley_bin_sha: set[str] = set()
    if mendeley_sha_json.is_file():
        loaded = json.loads(mendeley_sha_json.read_text(encoding="utf-8"))
        mendeley_bin_sha = set(loaded if isinstance(loaded, list) else loaded.keys())
    report["mendeley_binary_sha256_known"] = len(mendeley_bin_sha)

    sha_by_txt: dict[str, str] = {}
    if manifest.is_file():
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
