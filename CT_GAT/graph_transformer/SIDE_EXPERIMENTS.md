# IDA side arrays and WL experiment plan

Implementation starts at `b6c0d95` on `side-arrays` in `rdmp-gt-side`.
The active `rdmp-gt` queue and `rdmp-gt-s5` are outside this change. No model,
training configuration, existing variant names, or main cache arrays are modified.
This delivery supplies side artifacts and a CPU baseline; the owner will wire the
side features into HierGNN and run the evaluation after the current queue finishes.

## 1. Side artifact contract

`build_side.py --cache <cache>` writes `<cache>/side/<sample_id>.npz`. Both v2 and
combined caches use exactly the same schema, `ida-side/1`. Load with
`np.load(path, allow_pickle=False)`. Every array is numeric or Unicode, never object.

| Array | Shape / dtype | Meaning |
|---|---|---|
| `fn_tags` | `[F,11] float32` | Direct static import-reference evidence, binary presence |
| `fn_tags_callee` | `[F,11] float32` | Union of direct tags of immediate cached callees |
| `tag_names` | `[11] Unicode` | Feature columns, stable taxonomy in `side_tags.py` |
| `df_edges` | `[E,2] int32` | Definition block -> use block, **global cache block IDs** |
| `df_kind` | `[E] int8` | Local kind 0 = explicit-register may-def-use |
| `df_edges_shuffled` | `[E,2] int32` | Seeded degree-preserving null; same `df_kind` alignment |
| `df_kind_names` | `[1] Unicode` | Edge-kind dictionary |
| `fn_blk_ptr`, `fn_flags` | Same as main cache | Alignment checks; retain empty-function rows |
| `coverage_json` | Scalar Unicode JSON | Supported instructions, barriers, omitted address operands, fixed-point work |
| `df_shuffle_changed` | Scalar integer | Number of destination entries changed by the null |
| `provenance_json` | Scalar Unicode JSON | Input sample/cache/export/vocab/code hashes and control seed |
| `schema`, `sample_id`, `control_seed` | Scalars | Artifact identity |

Function order is export order, as in `build_cache.py`; blocks are sorted by start
RVA **inside each function**. Export identity, schema, processor, function flags,
block ranges, instruction counts, mnemonic stream and CFG endpoints are checked
against the main cache before publication. Missing exports or mismatches fail the
sample and make the command exit nonzero. Nothing fabricates empty successful data.
Publication is atomic; source and cache hashes are checked again before writing.
Each invocation writes a separate `side/build-<uuid>.json` audit (including failures).

Matching provenance skips existing output. Changed code/input/seed refuses stale
output unless `--replace-stale` explicitly authorizes replacing **side files only**.
The stored code fingerprint covers all four side-builder modules.

### Behavior evidence and controls

The taxonomy covers file enumeration/open/read/write/mutation, crypto transforms,
key operations, registry, process, service and network APIs. Exact API membership
normalizes DLL decoration and A/W suffixes; no substring heuristics, label-based
mining, strings, family names or paths. Cached import references can include taking
an import's address, not just calling it. Statically linked, dynamically resolved
and unresolved indirect APIs are missed. Tags indicate association, **not observed
execution, encryption, or malicious behavior**. No tags is absence of evidence.

Primary arm uses `fn_tags` only. Optional one-hop callee evidence is a separate
ablation; it must not be smuggled into tags-off or tag-permuted arms. For controls,
call `side_common.tag_control(tags, 'real'|'off'|'permute', rng)` on the full
cache-order function matrix **before** dropping/capping functions, then gather by
the same retained function indices as `fn_flags`. Permute whole rows within the
binary; preserve each column's binary-wide prevalence. Use
`sample_rng(sample_id, seed)` for repeatable evaluation. If testing callee features,
compute them from permuted direct tags on the unchanged call graph for that control.

### Data-flow scope and controls

The current IDA export has operands but lacks complete read/write and implicit
effect metadata. This implementation therefore extracts **partial explicit general
register may-def-use**, using a documented mnemonic allowlist. It does not claim
complete machine-level data flow. Memory aliasing, flags, implicit stack effects,
address-register reads, SIMD and interprocedural effects are omitted. Calls/system
instructions and unsupported instructions kill propagated state. The analysis can
miss genuine flow across these barriers; this is intentional and measured.

Registers are canonicalized internally; full writes kill prior definitions and
partial writes are weak updates. Conditional moves and variable/zero-count shifts
retain possible earlier definitions. Zero idioms do not consume the old value.
Reaching definitions are solved to a fixed point over each function CFG, including
joins and loops. Edges collapse multiple instruction/register links into unique
block pairs and exclude same-block links. Fixed-point/edge budgets raise errors
rather than publish truncated successful results. Register identities and widths
are never emitted as model/baseline features. Topology/coverage can still correlate
with architecture, so the x86 headline and matched/weighted checks remain essential.

Suggested integration after merge: map `df_kind` to new edge type `12 + df_kind`
(expand embedding cardinality accordingly). Do not apply CFG `kind*2+back` encoding
to these edges. Keep CFG edges and call graph identical across real/no-DF/shuffled-DF
arms. Filter/remap def-use endpoints through the existing block retention map, then
offset at collation. The saved null preserves node in/out degree **within function
and kind**, including edge count; self-loops and parallel edges are allowed. Never
deduplicate the null afterward. This is a directed multigraph configuration null,
matching the existing GNN destination-permutation control, not a simple-graph swap.
Report changed-endpoint fraction; tiny functions can remain unchanged.

New HierGNN switches should follow the existing frozen RunConfig + default-off CLI
+ variant suffix pattern. That wiring is deliberately left to the owner's next
step, so no additional fields are introduced into the active model configuration.

## 2. CPU WL baseline

`wl_baseline.py` fits all three pre-specified arms on identical manifests:

| stats.py name | metrics.json method | Representation |
|---|---|---|
| `wl0` | `wl_h0_lr` | Initial block labels only |
| `wl2` | `wl_h2_lr` | Directed CFG WL subtree counts, levels 0, 1, 2 |
| `wl2-shuffle` | `wl_h2_shuffle_lr` | Same WL over degree-preserving shuffled CFGs |

Initial labels hash sorted signature + API multisets in each block. Signatures are
mnemonics and operand categories already in the cache; no operand widths/register
names, RVAs, function flags, node degrees or function IDs. IDs are resolved through
the vocabulary to strings, making labels independent of cache vocabulary numbering.
At each refinement, incoming and outgoing neighbor-label multisets are separately
sorted and hashed. Graphs are untyped for both real and null arms. `h=0` still uses
basic-block boundaries; it is a bag of block contents, not the flat TF-IDF baseline.

The hash map is fixed (262,144 nonnegative bins), with possible collisions shared by
all arms; no learned corpus-wide vocabulary. Training-only min-DF and IDF, sublinear
TF, L2 normalization and fixed `C=1` liblinear LR follow. Class balance or
`--arch-balance` uses existing harness weights. No validation/test feature selection.
All blocks are retained; feature hashing streams one function at a time to avoid a
whole-binary dictionary/adjacency expansion for multi-million-block files. Full
combined-corpus runtime and peak memory still need measurement before queueing it.

Thresholds use the existing validation macro-F1 grid and tie rule. Output contains
`manifest.json`, baseline-compatible `metrics.json` with `methods{}` and per-method
`threshold_val`, and `predictions.csv` with val/test rows. `pred` is at 0.5, as in
`baselines.py`; stats selection `auc` applies the saved validation threshold.
Float probabilities retain full precision. A completed directory is published only
after all methods finish; `.pending-*` directories preserve diagnostics after a
failure and are ignored by the report. Existing output is never silently replaced.

Outputs default to `graph_transformer_runs/baselines_wl/<existing baseline run name>`.
`report.py` reads that root in addition to `baselines`; `stats.py` recognizes the
three new aliases with existing `-tfNN` and `-ab` suffixes. Existing baselines remain
untouched. A custom `--out-root` is for isolated smoke/debug output and is not
automatically discovered by the shared report.

## 3. Execution and acceptance

Run in `C:/Users/chaoa/Downloads/rdmp-gt-side/CT_GAT/graph_transformer` with
`C:/Users/chaoa/Downloads/gt_venv/Scripts/python.exe` (abbreviated `python` below).
All new commands are CPU-only. On PowerShell use `$env:CUDA_VISIBLE_DEVICES='-1'`
for regression tests; an empty value is not reliable on this host.

```powershell
python build_side.py --cache C:/Users/chaoa/Downloads/graph_transformer_cache_v2
python build_side.py --cache C:/Users/chaoa/Downloads/graph_transformer_cache_combined

python wl_baseline.py --protocol kfold --fold 0 1 2 3 4 --arch x86 --seed 0 1 2
python wl_baseline.py --protocol kfold_v3 --fold 0 1 2 3 4 --arch all --arch-balance --seed 0 1 2
python wl_baseline.py --protocol kfold_v3 --fold 0 1 2 3 4 --arch matched --seed 0 1 2
python report.py
python stats.py --a wl0 --b wl2 --selection auc
python stats.py --a wl2-shuffle --b wl2 --selection auc
python stats.py --a tfidf --b wl2 --selection auc
python stats.py --a wl2-shuffle-ab --b wl2-ab --selection auc --protocol kfold_v3 --arch all --balanced
```

`GT_CACHE` overrides the cache; otherwise `--protocol kfold_v3` chooses combined,
and kfold/bundle choose v2. `build_side.py` uses explicit `--cache` and supports
`--corpus-root name=path`, `--sample-id ...`, `--limit N`, and `--seed` for the null.

Milestones and acceptance:

1. Build side artifacts: no main cache mutations; alignment and degree invariants;
   coverage and tag prevalence stratified by corpus/architecture/class. Review a
   sample of API matches and def-use links against exported instructions. Do not
   interpret coverage/barrier counts as input features.
2. Owner integrates new edges/tags after the GPU queue: real/none/shuffled DF with
   other structure fixed; real/off/permuted tags with identical function retention.
   Keep direct-only and optional callee-augmented tag ablations separate.
3. Run WL x86 first on all 3 seeds x 5 folds. Compare real WL with both `h=0` and
   shuffled WL before comparing it with TF-IDF. Old Capstone/graph2vec scores are
   context only and are not pooled with this IDA experiment.
4. Repeat the comparisons on combined all-architecture with training and scoring
   cell balance, and on the existing architecture-matched manifests. Report test
   sample counts, exact split/cache hashes, architecture recall and coverage.
5. Use the existing paired family-cluster bootstrap. Apply the pre-registered
   structure rule (delta AUC >= .01, delta macro-F1 >= .02, both 95% intervals above
   zero) only after all 3 seeds x 5 folds are complete. A five-fold synthetic test
   proves integration; it is not research evidence. No headline performance claim
   is made by this implementation delivery.

## Validation record

See `SIDE_VALIDATION.json` for real smoke sample IDs, coverage, array equality,
hash checks, WL feature checks and test outcomes from this implementation session.
Only smoke side files are built during delivery; corpus-wide generation, GNN
integration and scored research runs remain the next evaluation phase.
