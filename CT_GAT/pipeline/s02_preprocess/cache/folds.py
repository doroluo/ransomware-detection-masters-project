"""Fold CSV of the 5-fold family-holdout evaluation of a dataset.

python -m s02_preprocess.cache.folds [--corpus mendeley] [--out DIR] [--groups CSV]

The assignment is the one of `family_holdout/folds.py` in the LLM_Classification history (results/family_holdout/
folds_mendeley.csv was written by it), restated for one corpus. It is deterministic and uses no random generator:

  ransomware  families are assigned whole to folds, largest family first into the currently smallest fold;
  goodware    identity groups are assigned whole to folds the same way, so a duplicate or a same-project file never
              stands on both sides of a fold;
  ties        broken by name, so the result is a pure function of the (family, count) and (group, count) tables.

Family names are lower-cased and reduced to letters and digits, then mapped through ALIASES onto the Mendeley spelling.

What reproduces and what does not. Given the rows of folds_mendeley.csv (its `family` and `group` columns), `assign_folds`
returns the committed `fold` column on all 2,509 rows (tests/test_folds.py). Rebuilding that CSV from the bundle's
samples.csv does NOT reproduce it, because two inputs of the original are not in the bundle:
  * the goodware group of the original is `mn:` plus the SHA-256 prefix of the file's mnemonic stream (1,024 groups for
    1,243 files, with duplicate groups of 60, 47 and 35 files). The bundle's `identity_group` is one group per file
    (1,169 groups for its 1,169 eligible goodware files), so duplicates are no longer kept together. `--groups CSV` (columns
    sample_id, group) replaces the group of a sample, for a table that has the stream hash.
  * the pool: the original read `cohort_mendeley.csv` (2,509 files); the bundle's eligible rows are 2,411.
On the Mendeley bundle this build joins 2,411 of the CSV's rows and puts 2,085 of them in another fold. A cache built from
folds written here keeps ransomware families whole and goodware identity groups whole, but is not the cache that the
published results used. The joint all-corpora build (folds_all.csv, goodware groups balanced across corpora) is not ported.
"""
import argparse
import collections
import csv
import re
from pathlib import Path

from common.paths import MissingSource, source_for
from s02_preprocess.cache.build_cache import corpus_root, corpus_samples, exclusion, read_aliases

K = 5
FIELDS = ['dataset', 'sha256', 'label', 'family', 'group', 'arch', 'orig_set', 'fold', 'corpus']

# other corpora's names for a family Mendeley already has, mapped onto the Mendeley spelling
ALIASES = {
    "sodinokibi": "revil", "djvu": "stop", "stopdjvu": "stop", "alphv": "blackcat",
    "noberus": "blackcat", "crysis": "dharma", "mespinoza": "pysa",
    "mailto": "netwalker", "play": "playcrypt",
    "agenda": "qilin", "cl0p": "clop", "basta": "blackbasta", "defray777": "ransomexx",
    # NOT aliases, kept distinct on purpose: MedusaLocker (2019) vs Medusa (2021),
    # Nemty vs Nefilim (a successor, not the same code base).
}


def normalise_family(name):
    n = re.sub(r'[^a-z0-9]', '', str(name).lower())
    return ALIASES.get(n, n)


def assign_greedy(counts, k=K):
    """{name: fold}: largest group first into the currently lightest fold (lowest index on a tie); equal sizes by name."""
    loads = [0] * k
    out = {}
    for name in sorted(counts, key=lambda n: (-counts[n], n)):
        fold = min(range(k), key=lambda i: (loads[i], i))
        out[name] = fold
        loads[fold] += counts[name]
    return out


def assign_folds(rows, k=K):
    """Rows (dicts with sha256, label, family, group, arch, orig_set, corpus) with a `fold` added, sorted as the CSV is:
    by fold, label, family, sha256. Ransomware rows go by family, goodware rows by group."""
    family_fold = assign_greedy(collections.Counter(r['family'] for r in rows if r['label'] == 1), k)
    group_fold = assign_greedy(collections.Counter(r['group'] for r in rows if r['label'] == 0), k)
    out = [{**r, 'fold': family_fold[r['family']] if r['label'] == 1 else group_fold[r['group']]} for r in rows]
    return sorted(out, key=lambda r: (r['fold'], r['label'], r['family'], r['sha256']))


def check_rows(rows):
    """A sha256 appears once and with one label; arch is x86 or x64 (the per-arch counts assume nothing else)."""
    labels = collections.defaultdict(set)
    seen = collections.Counter(r['sha256'] for r in rows)
    for r in rows:
        labels[r['sha256']].add(r['label'])
    both = [s for s, v in labels.items() if len(v) > 1]
    twice = [s for s, n in seen.items() if n > 1]
    bad = sorted({r['arch'] for r in rows} - {'x86', 'x64'})
    if both or twice or bad:
        raise ValueError(f'{len(both)} sha256 with both labels, {len(twice)} listed twice, arch values {bad}')


def pool_rows(corpus='mendeley', groups=None):
    """The fold pool of one corpus: the rows its cache holds (`build_cache.exclusion` leaves out a missing split, a UPX stub,
    an ineligible or non-x86/x64 row). `groups` maps sample_id -> goodware group; without it the bundle's identity_group
    stands (see the module docstring for what that changes)."""
    aliases = read_aliases(corpus_root(corpus))
    rows = []
    for r in corpus_samples(corpus):
        if exclusion(r, aliases.get(r['sample_id'], {})):
            continue
        label = int(r['label'])
        family = normalise_family(r['family']) if label else 'goodware'
        group = family if label else (groups or {}).get(r['sample_id']) or r['identity_group']
        rows.append({'sha256': r['sample_id'], 'label': label, 'family': family, 'group': group,
                     'arch': r['architecture'], 'orig_set': r.get('original_split', ''), 'corpus': corpus})
    check_rows(rows)
    return rows


def report(rows, k=K):
    """Per fold: files, goodware and ransomware counts, each by architecture, and the ransomware families."""
    lines = [f'{len(rows)} files']
    for f in range(k):
        part = [r for r in rows if r['fold'] == f]
        n = collections.Counter((r['label'], r['arch']) for r in part)
        families = sorted({r['family'] for r in part if r['label'] == 1})
        lines.append(f"  fold {f}: {len(part):4d} files  goodware {n[(0, 'x86')] + n[(0, 'x64')]:4d} "
                     f"(x86 {n[(0, 'x86')]}, x64 {n[(0, 'x64')]})  ransomware {n[(1, 'x86')] + n[(1, 'x64')]:4d} "
                     f"(x86 {n[(1, 'x86')]}, x64 {n[(1, 'x64')]})  families {len(families)}: {', '.join(families)}")
    return '\n'.join(lines)


def write_csv(path, rows, dataset):
    with Path(path).open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({'dataset': dataset, **r})


def read_groups(path):
    """sample_id -> group from a CSV with columns sample_id and group."""
    with Path(path).open(encoding='utf-8', newline='') as fh:
        return {r['sample_id']: r['group'] for r in csv.DictReader(fh)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--corpus', default='mendeley', choices=['mendeley', 'ransomware_balanced', 'balanced_goodware'])
    ap.add_argument('--out', type=Path, help='output folder (default: the folder of the configured folds CSV)')
    ap.add_argument('--groups', type=Path, help='CSV (sample_id, group) replacing the goodware identity groups')
    args = ap.parse_args(argv)
    out = args.out
    if out is None:
        try:
            out = source_for('folds').parent
        except MissingSource as exc:
            raise SystemExit(f'give --out: {exc}')
    out.mkdir(parents=True, exist_ok=True)
    rows = assign_folds(pool_rows(args.corpus, read_groups(args.groups) if args.groups else None))
    write_csv(out / f'folds_{args.corpus}.csv', rows, args.corpus)
    print(report(rows))
    print(f'wrote {out / f"folds_{args.corpus}.csv"}')


if __name__ == '__main__':
    main()
