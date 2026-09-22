# Dataset plan, version 2 (2026-09-21)

What the next corpus looks like, what gets excluded and why, and what every
model in this repository and on the teammates' branches needs from it. Written
against the measured cohort numbers, the six code audits in
[CODE_AUDIT.md](CODE_AUDIT.md), and the MalwareBazaar fetch that is running on
the SEED VM as this is written.

## 0. The problem the new corpus has to fix

The headline evaluation (`results/family_holdout/summary.md`) is a 5-fold
family holdout plus leave-one-family-out over 38 Mendeley ransomware families.
Three properties of that corpus limit what the numbers can mean:

| fact | measured | consequence |
|---|---|---|
| ransomware is x86 | 1,153 x86 / 114 x64 in the cohort; 26 of 38 families have **zero** x64; Hive alone holds 43 of the 114 | "x64 means goodware" is a free 0.66 (mendeley) / 0.84 (balanced) accuracy; x64 ransomware recall collapses for CNN-ViT (0.39 / 0.13) and graph2vec (0.63 / 0.47) |
| goodware is x64 | Mendeley goodware 632 x86 / 482 x64; Goodware_Balanced 308 x86 / 1,029 x64 | the confound is two-sided: adding x64 goodware made it worse |
| the x64 ransomware sits in one fold | 59 of 114 x64 files land in fold 2 | fold sd partly measures architecture mix, not family difficulty |
| .NET and packed files cannot be disassembled | 38 + 84 ransomware, 15 + 4 goodware excluded (`dotnet`, `packed_other`) | Thanos and Night Sky drop out entirely; every opcode-track model shares this blind spot |
| Mendeley goodware is duplicated | 48% of goodware test rows are byte-copies of a training row (installer stubs) | fixed by grouping duplicate streams in the fold builder; still inflates any fixed-split number |
| file type is a class signal | Mendeley ransomware has no extension and is all EXE; Goodware_Balanced is 686 DLL / 651 EXE | "DLL means goodware" is a second free shortcut nobody has floored yet |

The new corpus exists to raise the x64 ransomware count into the hundreds
across many families, so that architecture stops being a usable shortcut and
so that per-architecture metrics stop being computed on 12 files per fold.

## 1. Corpora

| corpus | role | source | state |
|---|---|---|---|
| `mendeley` ransomware | existing, 38 families | Mendeley yzhcvn7sj5 (VirusShare + Hybrid-Analysis) | text outputs on host under `Shared/Extract/`; **raw binaries are on no machine any more** (old VM gone). Re-download from Mendeley if a re-extraction is ever needed. |
| `vs` ransomware (new) | x64 lift, newer families | MalwareBazaar public CSV dump, family = MB `signature`, downloaded on the VM | 1,239 candidates, 44 families, 60 per family cap, newest first, none overlapping the 4,160 known sha256s. Pools for later: GandCrab 5.9k, STOP 2.3k, REvil 280, BlackMatter 240, Hive 170, LockBit 170, Akira 145. |
| `mendeley` goodware | existing | Mendeley cv3v9szdn7 | 1,243 in cohort, x86-leaning, heavy duplication (grouped) |
| `balanced` goodware | existing | Goodware_Balanced (curated, DLL-heavy, x64-heavy) | 1,337 in cohort |

Everything stays keyed by sha256. The corpus name becomes a first-class column
(`corpus`) in the cohort file and in the fold file; today it is lost at
`family_holdout/folds.py:60`.

## 2. Inclusion rules (cohort definition, version 2)

### 2.1 Tags, unchanged from `extract_unified.py`

| tag | rule | opcode track | combined / byte track |
|---|---|---|---|
| `plain` | native x86/x64 code, executable-section entropy <= 7.2 | keep | keep |
| `upx_unpacked` | was UPX, unpacked to a temp copy, disassembled from that | keep (record the tag) | keep |
| `upx` | UPX but unpacking failed | exclude | keep |
| `packed_other` | entropy > 7.2, not UPX (packer stub only) | exclude | keep |
| `dotnet` | IL-only managed assembly (CIL is not x86) | exclude | keep |
| `no_exec` | no executable section with data | exclude | exclude |
| `arch_unsupported` | machine not 0x14c / 0x8664 (ARM64 etc.) | exclude | exclude |
| `bad_pe` | header does not parse | exclude | exclude |
| `dup` | same sha256 already seen | exclude | exclude |

No generic unpacking is attempted. `decoded_ratio` is not a packing test and is
never used as one; entropy is. The "combined / byte track" column exists only
for EMBER-style byte and header features, which can consume packed and .NET
files; nothing else in the repository uses it.

### 2.2 Architecture

- Keep x86 and x64 only. ARM64 is out (`arch_unsupported`).
- Never filter by architecture for training. Always report per-architecture
  recall, the x86-rule floor, and per-architecture macro-F1 with an explicit
  "undefined" when one class has zero support (today `_safe_div` prints 0.0).
- Composition targets for the merged ransomware set: at least 25% x64 overall,
  and for every family that ships any x64 build at least 10 x64 files. Families
  that are x86-only in the wild (GandCrab, Phobos, Dharma, STOP) stay x86-only;
  the point is not to fake a distribution but to stop it from being decisive.
- Fold assignment becomes architecture-aware: the greedy balancer keys on
  `(family_count, x64_count)` so that x64 ransomware is spread over the five
  folds instead of concentrated in one.

### 2.3 File type

- Keep EXE and DLL on both sides. Record `is_dll` (from the PE characteristics
  flag, not the filename, since Mendeley ransomware carries no extension) in
  the manifest.
- Add a `dll_rule` floor next to `x86_rule`: predict goodware iff DLL. If a
  model does not clearly beat it, the model has learned the file type.
- The `vs` candidates include DLLs where MalwareBazaar lists them; keep them.

### 2.4 Cross-corpus rules (new)

1. **Dedup by sha256 across every corpus** before fold assignment, keeping the
   Mendeley row (it carries the release's family label). Today `folds.py:115`
   concatenates without a check; `common.Folds` never checks either.
2. **Dedup by mnemonic-stream hash across corpora**, the same grouping already
   used for Mendeley goodware, so that a repacked or re-signed copy of the same
   sample cannot straddle train and test.
3. **Normalise family names**: lowercase, strip non-alphanumerics, then apply
   one alias table (`sodinokibi -> revil`, `djvu -> stop`, `alphv -> blackcat`,
   `medusalocker -> medusa`, `agenda -> qilin`, `mailto -> netwalker`,
   `crysis -> dharma`, `mespinoza -> pysa`, `nemty -> nefilim`, `playcrypt ->
   play`). The same family name from two corpora is **one** family and is held
   out together. Nothing in the fold builder does this today.
4. **Small families**: a family with fewer than 5 usable files is assigned to
   folds as part of one `_small` group (so its files never split across
   folds) and is excluded from LOFO. LOFO reports only families with at least
   10 files.
5. **Goodware family column is the literal `goodware`** and the fold builder
   must filter on `label == 0` when it reads a goodware manifest. Today
   `folds.py:104-108` hard-codes label 0 for every row it reads from the
   balanced cohort file; appending ransomware there would silently relabel it.

### 2.5 Length

- `asm/` keeps the 100,000-instruction cap; `mn/` is uncapped.
- The tokenizer pipeline reads only the first 5,000 lines of a file whose
  median length is 144,000 mnemonics (CRT start-up code). Raise its cap to the
  30,000-line window the seq/TF-IDF runners already use, or use the strided
  sampler, before comparing it with anything else.

## 3. What each model needs

| model | input it consumes | split/label source | what the `vs` corpus must add | change needed in code |
|---|---|---|---|---|
| TF-IDF + LogReg (`family_holdout/run_tfidf.py`) | `Extract*/mn/<sha>.txt` (also `mn_api`, `mn_opclass`) | `folds_<ds>.csv` | `mn/` tree, cohort rows | tree routing at `run_tfidf.py:76-78` maps `(dataset,label)` to a tree; a third ransomware tree needs a `corpus -> tree` map |
| imports baseline | `manifests/imports/imports_flat.json` | fold file | `imports_vs.json` merged in | none beyond the merge |
| sequence transformer (`seq_model/`) | `mn/` (or `mn_api_top`), token cache, optional imports | fold file, val carve | `mn/` tree, imports JSON, re-run vocab and pretrain cache | tree list at `seq_model/data.py:77-78`; env var name mismatch (`RANSOM_SHARED` vs `RANSOM_SHARED_DIR`) |
| graph2vec / WL (`graph2vec_pipeline/`) | `asm/<sha>.asm` | fold file | `asm/` tree | tree routing in `graph_cache.py`, extra-cache name in `run_graph2vec.py:196` |
| tokenizer + word2vec (`llm_features_pipeline/`) | mnemonic text named `<family>_<sha>.txt` in set folders | Mendeley release split or cohort CSV | a folder per set with the naming rule, cohort rows, raise the 5,000-line cap | filename regex at `data.py:24` collapses any other naming into one family |
| CNN-ViT (`cnn_vit_pipeline/`) | `asm/` -> uint8 token cache -> 256x256 canvas | cohort CSV + fold file | `asm/` tree, cache rebuild | four hard-coded tree tables (`build_dataset.py`, `tuned_train.py`, `run_cnn_vit.py`); cache has no fingerprint so it must be deleted by hand |
| EMBER-style LightGBM (`ember_pipeline/`) | raw PE bytes -> 537 hand-rolled features | cohort CSV | **raw PE on the VM**; extract goodware there too | never run; extractor is not real EMBER; goodware on host vs ransomware on VM is a label-correlated environment confound; pin LIEF and record its version |
| GIN over opcode transitions (Dorothy, `dorothy-work`) | its own PE -> mnemonic extractor | its own random or 4-family split | nothing if it reads `mn/` + the fold file | replace `extract_opcodes.py` + `split_dataset.py` (875 lines) with a 40-line adapter over `mn/` and `folds_*.csv`; carry `arch` through to the metrics |
| CT_GAT transformer (Yanping, `yanping`) | its own PE -> asm -> `[opcode, op1, op2]` triplets | random 70/15/15, not family-disjoint | nothing if it reads `mn_api/` | its tokenizer diverges from the CNN-ViT one in three behaviours; the GAT half is two empty files |

The recurring theme: every pipeline can consume the VM's `asm/`, `mn/`,
`manifest.csv` and `imports` outputs plus one fold file. The new corpus should
be produced in exactly that shape and nothing else.

## 4. Production steps

On the VM (`ssh seedvm`, everything under `~/work`):

1. `vs_fetch.py fetch candidates.csv` (running; MalwareBazaar first, VirusShare fallback; log `vs_log.csv`).
2. `extract_unified.py --out out/Extract_VS --mal-train samples/vs --disasm-packed --resume` (queued behind the fetch; family = folder name).
3. `extract_imports.py --in samples/vs --out out/imports_vs.json --manifest out/imports_vs.csv` (queued behind the extraction).
4. Rewrite the token streams on the host with `asm_tool/rewrite_streams.py` and `cap_api_vocab.py` once `imports_vs.json` is back, because `mn_api*` streams need the IAT map.

On the host:

5. `scp -r seedvm:~/work/out/Extract_VS/{manifest.csv,asm,mn}` and `out/imports_vs.*` into `C:/Users/chaoa/Downloads/asm and mm/Shared/Extract_VS/`. Binaries never leave the VM.
6. Build `cohort_vs.csv` from the manifest with the 2.1 to 2.4 rules, `corpus = vs`, `set = mal_train` for every row (family holdout ignores `set`).
7. Extend `family_holdout/folds.py` to read a list of cohort files, dedup across them, normalise families, key the balancer on `(count, x64)`, and write `corpus` into the fold file.
8. Merge `imports_vs.json` into `imports_flat.json` with `imports/merge_imports.py`.
9. Re-run the runners in the order TF-IDF, imports baseline, graph2vec, seq transformer (with pretrain), CNN-ViT; regenerate `summary.md` with one sd definition.

## 5. Evaluation protocol changes that come with the corpus

- One definition of the fold sd everywhere (sample sd, ddof=1); today three
  aggregators disagree by 12%.
- The majority floor is computed from the training folds, not from the test
  fold's own labels.
- Per-corpus ransomware recall in every summary row (`mendeley` families vs
  `vs` families), so that a model that only recognises 2020-era Mendeley
  toolchains is visible.
- Per-architecture and per-file-type recall for both classes, and the three
  floors: majority, x86 rule, DLL rule.
- Pooled AUC stays reported but the per-fold AUC is the one to quote; the pooled
  number mixes five separately fitted score scales.

## 6. Decisions still open

1. Whether to also pull extra x86 samples for families that Mendeley covers
   thinly (Maui 3, Thanos 1, HolyGhost 4), or to leave those as LOFO-only.
2. Whether EMBER is worth the VM-side goodware extraction it needs, or is
   dropped from the comparison. It has no result today and its features are
   not EMBER's.
3. How much of the MalwareBazaar pool to take beyond the first 1,239: the
   x64-heavy families (Akira, BlackMatter, Hive, LockBit, BlackCat, Medusa,
   Trigona) are the ones worth deepening.
