# Committed folds

The five-fold family-holdout assignments that every cache and run of the pipeline reads (`build_cache` joins them to the
samples by sha256 and writes the fold into `index.csv`). They are inputs, not generated here: `common/catalog.py` points the
`folds` and `folds_v3` sources at this folder, and `GT_FOLDS` / `GT_FOLDS_V3` or `datasets.json` replace them.
`.gitattributes` pins the bytes (`*.csv -text`), so a checkout with `core.autocrlf=true` still has the files below.

The files are copies of the LLM_Classification history (local clone `rdmp-llm`, branch `origin/LLM_Classification`), the
same ones `family_holdout_folds.zip` on Drive holds.

| file | what it is | sha256 |
|---|---|---|
| `folds_mendeley.csv` | the Mendeley cohort, 2,509 files, 38 ransomware families, the `mendeley` protocols (`kfold`) | `2463fc24fe538a69b5d5198b9f66d7fe33f4c0c5f55b8d53fe3882bcdf4c15f9` |
| `v3/folds_all.csv` | dataset v3, 7,243 files, 97 ransomware families: Mendeley rows plus `balanced` (the balanced goodware), `hostgood` (host goodware) and `vs` (MalwareBazaar ransomware) rows, the `kfold_v3` protocol | `b2c69eab59768e18297552c0660711f37b2dbdcd16a61146ebba52353e58e133` |
| `v3/arch_matched.csv` | `sha256,keep` over the same 7,243 files; `keep = 1` on the 6,402 that give both classes identical x86 and x64 counts (2,454 / 747), the `--arch matched` selection | `d26a75a74b8c182d9f107d52a20e3a5fe439d7f2e99cf4bbb91d244160e4d0bf` |

Columns of the two fold files: `dataset,sha256,label,family,group,arch,orig_set,fold,corpus`. Ransomware families go whole
into one fold; goodware identity groups go whole into one fold.

## Provenance

- `folds_mendeley.csv`: `family_holdout/folds.py` of the LLM_Classification history. Written first at 0c72287 (2026-09-14);
  this file is the default build of the script at 95caf5f (2026-09-21, `python family_holdout/folds.py`), the same
  assignment with the `corpus` column added (95caf5f rewrote the committed file).
- `v3/folds_all.csv`: `family_holdout/folds.py` at 0bc6cbb (2026-09-22, "Dataset v3: the completed MalwareBazaar pull"),
  run as in `docs/DATASET_PLAN.md` section 6 and 6b: `python family_holdout/folds.py --out results/family_holdout_v3
  --ransomware vs=manifests/cohort/cohort_vs.csv --goodware hostgood=manifests/cohort/cohort_hostgood.csv --arch-aware
  --pool-small` (`--goodware` implies `--all`, which writes `folds_all.csv`). The cohort files are `manifests/cohort/` of that
  commit.
- `v3/arch_matched.csv`: `tools/arch_balance.py` at 0bc6cbb (first written by d543539), `--match` over the four cohort files
  of `manifests/cohort/`, seed 1 (the script's default). The history records neither the command line nor the seed of the
  run that wrote the file, only the tool and its documented usage, so the seed is the default, not a recorded value.

## Why `s02_preprocess.cache.folds` does not reproduce them

It applies the same assignment rule to the bundle's rows, but the original goodware groups were mnemonic-stream hashes of
files that are not in the bundle and the original cohort had 2,509 files against the 2,411 eligible ones, so the folds it
writes differ (see its docstring and `tests/test_folds.py`).
