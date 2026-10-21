# Plan: edge-typed hierarchical MPNN with a control ladder and masking

Draft 2026-10-15, grounded in the literature review
`Downloads/Ransomware Combined Structural Feature Dataset/reports/CFG GNN pooling and masking.md`
(notes in `research_notes/CFG GNN pooling and masking/`). Evidence tiers follow the review: A measured on
malware code/CFGs, B generic graph/ML benchmarks, C the notes' inference, D the review's own inference;
"team" = measurements already in PLAN.md/RESULTS.md. Paths are relative to `CT_GAT/block_level_counting/`; `RUNS` =
`C:/Users/chaoa/Downloads/graph_transformer_runs`. Since step 9 the code is grouped by pipeline step (see `../README.md`);
the sections below name modules as they were when written: `data.py` is now `common/manifests.py`, `pipeline/s03_tokenizer/`,
`pipeline/s04_select_subgraph/` and `pipeline/batching.py`, `model_gnn.py` is `pipeline/model.py` and the stage packages,
`model.py` was the graph transformer (removed in step 25, with `--model transformer`), and the scripts keep their names (root shims call
`training/`, `evaluation/` and `pipeline/s02_build_cfg_cache/`).

## 0. What changed and why

The question is now "does CFG wiring carry ransomware signal that block contents do not", not "does a GNN
beat the transformer". Consequences:

1. The no-edge arm must reach TF-IDF parity before any edge claim means anything.
2. The structure claim must beat a degree-preserving shuffled twin trained from scratch.
3. Every claim needs 3 seeds x 5 folds and a paired bootstrap over families.
4. Masking enters as augmentation arms on the best arm, and as inference-time curves that show which units
   the model depends on.

Kept from the first draft: cache v2 by full rebuild (~5 min), whole-function message passing in plain torch,
dual checkpoint selection with the validation carve unchanged, a step-0 snapshot, finishing the transformer's
kfold seed 2, a resumable one-process-per-run queue on `gt_venv`, the same manifests/folds/floors/report
tables, and no register names or operand widths as inputs.

Verified facts the plan rests on:
- Raw blocks carry `start_rva, end_rva, type` (IDA `fc_block_type_t`); block lists are address-sorted.
- Mnemonics are plain (18 `j*` + `loop*`, no prefixes).
- Edge survey on 8 binaries: cond not-taken fall-through 17.7k, cond taken fwd 15.4k, plain fall-through
  7.7k, `jmp` fwd 3.6k, cond back 2.3k, after-`call` fall-through 1.2k, `jmp` back 1.0k, `jmp` to extern
  0.9k, switch fan-out >= 3 ~0.5k, residue ~0.2k.
- Cache v1 built in 245 s + 40 s, vocab sha `4ded77e0...`; 25.4M blocks / 34.1M edges in total; the largest
  binary has 235k blocks / 285k edges / 1.2M instructions.
- Transformer runs took 5-16 min (median ~7.5), peak GPU 6.3-8.4 GB.
- Transformer kfold seed 2 is incomplete: `kfold3_x86_nograph_s2` died at epoch 11, `kfold4_*_s2` never ran.
  The committed `RESULTS.md` predates the seed-1 kfold runs, so it is stale.
- Goodware identity groups are all singletons.

## 1. Decisions and the findings that drive them

| # | Decision | Driving finding | Tier |
|---|---|---|---|
| 1 | Primary protocol: kfold x86, 3 seeds x 5 folds, pooled OOF; bundle x86 and all-arch rows are secondary tables | Team confound and carve instability; paired bootstrap precedent in arXiv 2609.24280 | A (team), B |
| 2 | Encoder gate before any edge claim: the hierarchical no-edge arm (R2) must come within a pre-registered margin of TF-IDF (R0) | Both deep arms sit 0.03-0.10 below TF-IDF while sharing one block encoder; Errica "structure not exploited"; FSE'21 MAGIC < opcode sequences | D on A/B |
| 3 | Block input = per-block L2-normalised sublinear-tf x IDF token bag, projected linearly (`EmbeddingBag(mode='sum', per_sample_weights)`), IDF fitted on the run's training binaries; the old mean-of-embeddings kept as `--encoder mean` for the gate | Marcelli (200-opcode bag matched engineered features); Maier (16-dim embeddings suffice) | B, D |
| 4 | Typed, directed MPNN: GINE-style type embedding added to the sender state; in-edges and out-edges aggregated separately with their own type banks and a learned mix (Dir-GNN); shared weights, no per-type matrices | Tönshoff (tuned MPNNs match GPS); Allamanis (typed edges matter); Rossi (direction) | B |
| 5 | 3 block-CFG layers, 2 call-graph layers; depth sweep {1,2,4} only after the controls | MalGraph 2 > 1; ProGraML 6 shared steps; Errica selected 1-5 | A, B |
| 6 | Readouts at both levels: fixed sum, mean, max concatenation; gated attention only as an opt-in arm | Mesquita (random cluster assignments lose nothing); 2024 pooling benchmark (no method wins); MalGraph used max | B, A |
| 7 | Identical 2-layer MLP head and identical log-size features (blocks, functions, instructions, CFG edges, call edges) on every arm, controls included | Tönshoff: head depth explained most of the MPNN-GPS gap; size shortcut | B, C |
| 8 | Edge taxonomy: kind in {fall-through incl. cond not-taken, cond taken, unconditional jump, indirect/switch, extern, other} x DFS back-edge bit = 12 ids | Types are mostly readable from the terminator; what edges add is wiring and loop membership; ProGraML defines loops by DFS back edges | C, B |
| 9 | Ladder core R0, R2, R4, R5, R7; R3, R6, R7b, R8 gate-conditioned; R1 and the Erdős-Rényi variant cut | The decision rule needs only R7/R5 vs R2 and R4; MalNet: degree alone is informative, so the null must preserve degree; R6 ≈ R7 predicted | D, A |
| 10 | R3/R4/R6 controls applied in training and evaluation: redrawn per epoch in training, seeded per sample in evaluation; R3 and R4 untyped so R5 vs R4 differs only in wiring | Bechler-Speicher: a model can read uninformative structure | B, D |
| 11 | No truncation: every block and function kept; `batch_size 4` with a block budget that cuts a batch short (the largest binaries run alone) | Team: the 512-segment cap cost 0.20 macro-F1 | A (team), C |
| 12 | Augmentation arms in evidence order: token masking 0.15; function dropping 0.15 uniform plus a library-biased variant; block-feature masking 0.15; token+function pair last; no DropEdge | McLaughlin (opcode blanking best at 0.2, < 1 F1 point); GraphCL (node/subgraph dropping helps, edge perturbation hurts NCI1); AIA (DropEdge -23.6 OOD, environment-only masking beats ERM) | A (Android), B, C |
| 13 | Inference-time curves on trained checkpoints: deletion for unaugmented arms, `[MASK]` replacement only for arms trained with it; targeted vs random matched on the number of functions dropped; degree-preserving edge rewiring 0 to 1 | ROAR/DropGNN protocol and its train/test-mismatch caveat; TF-IDF is rewiring-invariant, so a flat R7 curve means wiring is ignored | C, D |
| 14 | Auxiliary masked-block reconstruction (0.3) + edge-type prediction (15%) only as the last arm, in the loss; no two-stage pretraining | Hu (node-level SSL alone limited); Maier (end-to-end best at this label count); MalwarePT trailed EMBER | B, A |
| 15 | AUC and PR-AUC beside macro-F1; gates use pooled OOF AUC first, macro-F1 at the validation-chosen threshold second | Rankings flip between protocols | D (team) |
| 16 | Thresholds: validation carve with dual checkpoints (AUC-selected) for deep arms; inner family-grouped CV thresholds for TF-IDF and the R8 stacking logit only | Inner CV for deep arms would multiply GPU cost by 5 | D |
| 17 | Paired cluster bootstrap (ransomware families; singleton goodware) with pre-registered minimum effects | 2609.24280 precedent | D |
| 18 | Plain torch (`index_add_`, `scatter_reduce` with `include_self=False`), fp32 accumulation, several seeds because CUDA scatter is nondeterministic | Notes on `scatter_reduce_` | C |

## 2. Edge typing (cache v2)

| array | dtype | values |
|---|---|---|
| `edge_kind` | int8 [E] | 0 fallthrough (`dst.start_rva == src.end_rva`; includes cond not-taken and after `call`), 1 cond_taken (source's last mnemonic `j*` except `jmp`, or `loop*`, and not fallthrough), 2 jmp (`jmp`, source not indjump), 3 indirect (`jmp` from an IDA `fcb_indjump` block), 4 extern (destination type `fcb_extern`), 5 other |
| `edge_back` | int8 [E] | 1 if the edge targets a node on the stack of an iterative DFS over the function's blocks, started at the entry block and restarted at the lowest-address unvisited block; self-loops are back edges |
| `edge_back_addr` | int8 [E] | `dst.start_rva <= src.start_rva` (address heuristic; checks and a histogram against `edge_back` only) |
| `blk_type` | int8 [B] | IDA block type |
| `blk_rva` | uint32 [B, 2] | start and end RVA; verification only; never returned by `data.load_sample` (tested) |
| `cache_version` | int | 2 |

Model type id = `edge_kind * 2 + edge_back` (12 ids). Untyped arms (R3, R4, R5) use id 0 everywhere; types
enter only at R6 and R7. Under any shuffle or rewiring the type stays with the source edge slot.

## 3. Architecture (`pipeline/model.py`)

```
h0_b = EmbeddingBag_sum(tokens_b, per_sample_weights = w_t) + Linear(6->d)(blk_feat_b)
   --encoder tfidf: w_t = idf[t] * (1 + log c_{t,b}) / c_{t,b} / ||tfidf_b||_2
   --encoder mean : w_t = 1 / n_b                      (the transformer's block encoder)
   --encoder raw : w_t = 1 (a sum of counts)   presence : 1 on the first occurrence of each distinct token, 0 on repeats
   --encoder rawlog : w_t = log1p(c_{t,b}) / c_{t,b}   (the count weightings of the step 11 arms)
DirGINE layer x gnn_fn_layers (3), 12 types:
   m_in_i  = sum_{j->i} relu(h_j + E_in[type_ji]);  m_out_i = sum_{i->j} relu(h_j + E_out[type_ij])  (fp32)
   h_i <- h_i + MLP(LN((1 + eps) h_i + W [m_in_i ; m_out_i]))
function vector f = Readout_fn(h over the function's blocks) + Linear(2->d)(log len, library)
   --readout fixed: GELU(Linear(3d->d)([sum/sqrt(n) ; mean ; max]))   --readout gated: w·tanh(Vh) segment softmax
   (the `gated` readout is softmax(w·tanh(Vh)) attention pooling, Ilse et al. Eq. 8, not a gated variant; the config
   value stays `gated` because run names depend on it)
DirGINE layer x gnn_bin_layers (2), 1 type (caller->callee / callee->caller) over the call graph
g = Readout_bin(f over functions);  z = [g ; log-size feats (5) ; TF-IDF logit if --stack-tfidf]
logit = Linear(d->1)(Dropout(GELU(Linear(|z|->d)(z))))      (same head on every arm)
```

| rung | switch | effect |
|---|---|---|
| R0 TF-IDF | `baselines.py` | lexical floor |
| R2 hierarchy, no edges | `--control no_edges` | empty CFG and call edge lists; `blk_feat[:, 4:6]` zeroed |
| R3 address chain | `--control chain` | CFG edges replaced by `i -> i+1` within each function; untyped |
| R4 degree-preserving shuffle | `--control shuffle` | edge destinations permuted within each function (out-degree exact, in-degree multiset preserved); untyped; call graph kept |
| R5 real, untyped | `--untyped` | all type ids 0; call graph on |
| R6 types permuted | `--control permute_types` | type ids permuted among the function's edges |
| R7 full | default for `--model gnn` | typed, directed CFG + call graph |
| R7b no call MP | `--no-call-mp` | call-graph layers skipped (still allocated, so parameter counts match) |
| R8 stacking | `--stack-tfidf` | OOF TF-IDF logit appended to `z` |

## 4. Masking

Training augmentation (train mode only, drawn per item per epoch):

| flag | unit | arm rate | mechanism |
|---|---|---|---|
| `--mask-tokens p` | instruction tokens (`mn, sig, beh`) and API tokens | 0.15 | replaced by a per-kind `[MASK]` id |
| `--drop-functions p` | kept functions, uniform | 0.15 | removed from blocks, edges, call graph and pooling; at least one function kept |
| `--drop-library p` | FLIRT/thunk functions (`fn_flags & 3`) | 0.4, with `--drop-functions 0.05` on the rest | targeted: mask the "environment" heavily, the candidate payload lightly (AIA prior) |
| `--mask-blocks p` | block content | 0.15 | block tokens replaced by `[MASK]`, `blk_feat` zeroed; wiring intact |

No edge dropping as augmentation.

Inference-time curves (`mask_curves.py`, no retraining):
- Units: `tokens_drop`, `blocks_drop`, `functions` (random), `functions_library` (targeted), `edges`
  (degree-preserving rewiring); arms trained with masking also get `tokens_mask` / `blocks_mask`.
- Targeted and random runs are matched on the number of functions dropped.
- Rates {0, 0.1, 0.2, 0.3, 0.5} (edges {0, 0.25, 0.5, 1.0}); 3 masks per binary.
- Each point reports macro-F1 at the run's validation threshold, AUC, PR-AUC and per-family recall. Each curve
  is summarised as the area under macro-F1 vs rate and the rate at which macro-F1 falls 5 points.
- TF-IDF gets the token and function deletion curves on masked count vectors.

Optional auxiliary loss (`--aux-recon 0.3 --aux-edge 0.15 --aux-weight 0.3`): masked-block bag-of-sig
reconstruction and 12-way edge-type prediction from `[h_src ; h_dst]`. Its edge-type accuracy from content
doubles as the diagnostic for R6 ≈ R7.

## 5. Statistics, thresholds and gates

- `metrics.scores` gains `pr_auc`; every report table gets a PR-AUC column.
- `stats.py --a <group> --b <group>` runs a paired cluster bootstrap over OOF predictions matched by
  `sample_id`. Clusters are ransomware families and single goodware binaries. It draws 2,000 resamples, each
  applied to all seeds, and reports the mean difference and 95% CI for pooled AUC, PR-AUC and macro-F1.
- **Pre-registered rule (fixed before stage B):** minimum effect ΔAUC >= 0.01 and Δmacro-F1 >= 0.02 pooled,
  CI excluding zero, 3 seeds x 5 folds. The stricter bar (Δ >= fold sd ≈ 0.076) is labelled "strong" when met.
- **Gate A (encoder parity):** R2 `--encoder tfidf`, seed 0. Passes if pooled OOF AUC >= R0 − 0.01 and
  macro-F1 at the validation threshold >= R0 − 0.02; re-assessed at 3 seeds once stage B is complete.
  - On a fail, one bounded fallback round runs (`--readout gated`, then `--d-model 256`, 5 runs each).
  - The model then proceeds with the best variant, and PLAN.md records that the "beats TF-IDF" claim is
    foreclosed.
- **Gate B (structure):**

  | outcome | verdict | then run |
  |---|---|---|
  | R7 or R5 beats both R2 and R4 | structure claim | R3, R7b, R8; R6 only if R7 beats R5 by the minimum effect |
  | R5 ≈ R4 > R2 | message passing acts as smoothing | |
  | R7 ≈ R4 ≈ R2 | structure not exploited | R8 only if gate A passed |
- **Gate C (augmentation):** each masking arm vs the unaugmented best arm under the same rule; the
  token+function pair runs only if one of them passes.

## 6. Stages, run matrix and GPU budget (~6 min per run until calibrated)

| stage | content | runs | GPU |
|---|---|---|---|
| 0 | snapshot; cache v2; transformer kfold seed-2 completion; baselines re-run (val split, PR-AUC, inner-CV thresholds, OOF logits); memory smoke | 3 GPU + CPU | ~0.5 h |
| A | encoder gate: R2 `--encoder mean` (5) and R2 `--encoder tfidf` seed 0 (5, reused by B); fallback if needed | 10 (+10) | 1-2 h |
| B core | R2, R4, R5, R7 on kfold x86 seeds 0-2; R2/R5/R7 on bundle x86 and bundle all, seeds 0-2 | 73 | ~7.5 h |
| B tail | gate-conditioned R3, R7b, R8, R6 (kfold x86, 3 seeds each) | 0-60 | 0-6 h |
| C | masking on the best arm (kfold x86, 3 seeds): M1 tokens 0.15, M2 functions 0.15, M2t library-biased; conditional M3 blocks 0.15, M4 tokens+functions; optional aux-loss arm | 45 (+30, +15) | 4.5 h (+3, +2.5) |
| D | curves on seed-0 checkpoints of R0, R2, R4, R7 and the best C arm | inference | 3-5 h |

### Budget: 5 GPU hours on day 1 (2026-10-15), more from day 2

Implementation (steps 0-6) uses no GPU time beyond smoke tests. Day 1 spends its GPU on the runs that decide
whether day 2 is worth running in full. Each run's time is checked after the first three; if a run takes more
than 8 min, the day-1 list is cut from the bottom.

| day 1 order | runs | est. GPU at 6 min/run (10 min/run) |
|---|---|---|
| memory smoke + 2-epoch smoke of R7 | - | 0.2 h |
| stage A: R2 `--encoder tfidf` seed 0, kfold x86 folds 0-4 | 5 | 0.5 h (0.8 h) |
| stage A: R2 `--encoder mean` seed 0 | 5 | 0.5 h (0.8 h) |
| B preview: R7, R5, R4 seed 0, kfold x86 folds 0-4 | 15 | 1.5 h (2.5 h) |
| transformer kfold seed-2 completion (3 runs) | 3 | 0.4 h |
| total | 28 | ~3.1 h (~4.7 h) |

The day-1 result is a seed-0 read of gate A and of the ladder (R2 vs R4 vs R5 vs R7). With one seed it
**cannot** pass the pre-registered rule, which needs 3 seeds; it only decides what day 2 spends time on.
The gate A fallback round (`--readout gated`, `--d-model 256`) is deferred to day 2.

Day 2 and later, in this order:
1. Seeds 1-2 of R2, R4, R5 and R7 on kfold x86 (40 runs, ~4 h). This completes the pre-registered test.
2. Bundle x86 and bundle all rows (18 runs, ~2 h).
3. The gated B tail.
4. Stage C masking arms on the best arm.
5. Stage D curves.

## 7. Implementation steps

| step | content | files | depends on |
|---|---|---|---|
| 0 | snapshot `report.py` / `val_threshold.py` output; identify live python processes | none | - |
| 1 | cache v2: edge kinds, DFS back edges, checks, synthetic tests | `build_cache.py`, `check_cache.py`, `config.py` (CACHE), `test_gnn.py` | - |
| 2 | config fields, IDF vocab, whole-function data path, controls, masking, block-budget batching, tests | `config.py`, `data.py`, `test_gnn.py` | real-data checks after 1 |
| 3 | `model_gnn.py` (DirGINE, readouts, HierGNN arms, aux heads) and behavioural tests | `model_gnn.py`, `test_gnn.py` | - |
| 4 | harness: dual checkpoints, PR-AUC, stacking, inner-CV baselines, `stats.py`, report tables, memory smoke | `train.py`, `baselines.py`, `metrics.py`, `report.py`, `stats.py`, `smoke_gnn_memory.py` | 2, 3 |
| 5 | `mask_curves.py` and TF-IDF curves | `mask_curves.py`, `report.py`, `baselines.py` | 4 |
| 6 | staged resumable queue and gate checker | `queue_gnn.sh`, `gates.py` | 4 |
| 7 | docs and results | `PLAN.md`, `RESULTS.md` (generated) | runs |

Steps 1, 2 and 3 can run concurrently. Steps 5 and 6 run concurrently after step 4.

## 8. Risks

- **Run time:** the per-run time is unknown until stage A. At 10 min per run the day-1 list fills the 5 h
  budget, B core takes ~12 h over later days, and C core shrinks to M1 + M2t.
- **Encoder parity:** the tfidf encoder may not reach TF-IDF parity. The fallback is bounded, and the claim is
  then scoped to the ladder.
- **Shuffle control:** R4 redrawn per epoch may train noisier than R7. If it differs from a fixed per-sample
  shuffle by more than the minimum effect, both are reported.
- **Nondeterminism:** CUDA scatter is nondeterministic, so single-seed differences under ~0.01 are noise.
- **Curves:** deletion curves show sensitivity, not necessity (the ROAR caveat).
- **Low-FPR metrics:** TPR at 0.1% FPR cannot be estimated with 115 x86 goodware test binaries, so it is not
  claimed.

## Day-1 results (2026-10-15, seed 0 only; not the pre-registered test)

144 GPU minutes, 28 runs, all rc=0. Pooled 5-fold family-holdout, x86 (1,810 binaries, 712 clusters).
Full tables: `RESULTS.md`, `RUNS/gates_day1.md`.

| arm | AUC-selected, validation threshold: AUC / PR-AUC / macro-F1 | F1-selected, 0.5: AUC / macro-F1 |
|---|---|---|
| R0 TF-IDF LR | 0.982 / 0.990 / 0.935 | 0.982 / 0.939 |
| R2 GNN no edges (tfidf encoder) | 0.976 / 0.981 / 0.924 | 0.926 / 0.896 |
| R2 GNN no edges (mean encoder) | 0.975 / 0.984 / 0.917 | 0.961 / 0.910 |
| R4 GNN shuffled edges | 0.975 / 0.984 / 0.929 | 0.940 / 0.901 |
| R5 GNN real edges, untyped | 0.945 / 0.969 / 0.898 | 0.959 / 0.880 |
| R7 GNN real edges, typed | 0.964 / 0.974 / 0.905 | 0.972 / 0.880 |
| graph transformer (3 seeds) | n/a (no AUC checkpoint) | 0.946 ± 0.007 / 0.906 ± 0.021 |
| no-graph transformer (3 seeds) | n/a | 0.967 ± 0.010 / 0.916 ± 0.016 |

1. **Gate A passes on the AUC-selected rule.** The no-edge GNN is within 0.006 AUC and 0.011 macro-F1 of
   TF-IDF. On the F1-selected rule it fails, because F1 checkpoints land on very early epochs (best epoch
   1-4 on some folds).
2. **No structure gain at seed 0.** Typed real edges minus no edges is ΔAUC -0.012 [-0.034, +0.002];
   typed minus shuffled is -0.011 [-0.044, +0.025]. This is the "structure not exploited" outcome, pending
   seeds 1-2.
3. **Edge types beat untyped edges:** ΔAUC +0.013 [+0.003, +0.026] under F1 selection, and +0.019
   [-0.033, +0.106] under AUC selection. Types carry some signal, but not enough to beat dropping the edges.
4. **The checkpoint rule reorders the arms,** so one seed decides nothing. Day 2 starts with seeds 1-2 of
   R2, R4, R5 and R7.

Day-2 data change, agreed with the user: the Oct-8 collection zips `Mendeley_Ransomware` and
`Mendeley_Goodware` are byte-identical to Mendeley_GNN_v1 (checked on all 2,670 sha256s, the metadata and
the CFG file hashes). `Ransomware_Balanced` (2,326) and `Balanced_Goodware` (1,527) are new and become the
all-architecture evaluation.

## Combined cache for the all-architecture evaluation (2026-10-15)

`Downloads/graph_transformer_cache_combined` (18.7 GB, built on CPU) holds Mendeley, `Ransomware_Balanced`
and `Balanced_Goodware`. Point the code at it with `GT_CACHE`; `config.CACHE` still defaults to
`graph_transformer_cache_v2`, which is unchanged.

```
GT_CACHE=C:/Users/chaoa/Downloads/graph_transformer_cache_combined python build_cache.py \
    --corpora mendeley ransomware_balanced balanced_goodware --base-cache C:/Users/chaoa/Downloads/graph_transformer_cache_v2
GT_CACHE=... python check_cache.py --compare-v1 200 --compare-base C:/Users/chaoa/Downloads/graph_transformer_cache_v2
```

- **Inputs.** Both zips matched their `.zip.sha256` and were extracted to `Downloads/ida_collections_balanced/`.
  Every CFG file matches `cfg_sha256` and `SHA256SUMS.txt`. Every eligible export is `ida-cfg-research/2.1`,
  with header sha256 equal to `sample_id` and a complete footer (2,107 + 1,346).
- **Filters.** Mendeley keeps its v2 rule (recommended split, not `upx_packed`); the selected rows are identical
  to v2's 2,411. The collections keep `eligible_for_gnn == 1` with an x86/x64 architecture. UPX stubs are
  dropped: 6 `Ransomware_Balanced` rows whose identity alias is a Mendeley `upx_packed` stub (the same sha256s
  are in the Mendeley bundle), and 19 `Balanced_Goodware` rows under `_upx_packed`. One babuk sample was
  exported with IDA's CLI processor (a managed binary whose functions are CIL methods with negative RVAs). The
  builder now rejects any export whose processor is not `metapc`; it is the only one in the three corpora.
  Built: Mendeley 2,411, Ransomware_Balanced 2,100, Balanced_Goodware 1,327 (5,838).
- **Byte identity.** Mendeley npz are copied from v2 and stay byte-identical, and so do v2's index.csv values
  (checked on all 2,411). The vocab is append-only: v2's `vocab.json` is a prefix of the combined vocab, new
  tokens follow it in sorted order, and `vocab_generations.json` records each vocab's sha and sizes, so Mendeley npz keep
  v2's `vocab_sha`. Combined vocab: mn 1,321, sig 3,379, api 88,913 (v2: 609 / 1,337 / 14,509).
- **index.csv** adds `corpus`, `fold_v3`, `fold_v3_sha256` and `arch_matched` after the v2 columns. `fold_v3`
  joins `rdmp-llm/results/family_holdout_v3/folds_all.csv` by sha256 (all joins were direct; none needed an
  identity alias). Label and arch agree with the fold file on every joined row. Joined collection rows take
  `family` and `identity_group` from the fold file (`play` becomes `playcrypt`; goodware groups such as
  `proj:system_local` replace the per-sha256 group), so the validation carve keeps related goodware
  together. Collection rows have empty `split`, `fold` and `fold_sha256`. `arch_matched` is `keep` from
  `arch_matched.csv`.
- **Protocol.** `RunConfig(protocol='kfold_v3', fold=k)` tests on v3 fold k (run names `v3kfold{k}_...`). Rows
  without a v3 fold are left out of train, validation and test alike. On every fold, test families and goodware
  groups are disjoint from train+val. The v3 folds are a different partition from the day-1 Mendeley folds
  (1,655 of 2,509 Mendeley rows share a fold number), so v3 numbers do not compare fold-by-fold with day 1.
  `folds_all.csv` also has 1,462 `hostgood` x86 goodware with no IDA export; they are not in this cache.
- **Wiring (step 6).** `train.py` and `baselines.py` take `--protocol kfold_v3 --fold k --arch all|matched`. With
  `GT_CACHE` unset, a kfold_v3 process reads the combined cache (an explicit `GT_CACHE` wins), and `config.json`
  records the path in `cache`. `arch matched` keeps the rows with `arch_matched = 1`, any architecture. `report.py`,
  `stats.py`, `val_threshold.py` and `gates.py` pool kfold_v3 like kfold; `stats.py` resolves the `tfidf` /
  `structural` baseline groups to the other group's protocol and arch (`--protocol` / `--arch` override).
  `gates.py --stage v3` and `queue_gnn.sh --stage v3` (CPU baselines, then R2, R7, R4, R5 for all and for matched,
  folds 0-4, seed 0) run the v3 protocol. Run names are `v3kfold{k}_{all|matched}_{variant}_s{seed}`; baselines are
  `baselines/v3kfold{k}_{arch}_s{seed}`.
- **Memory risk.** 71 combined binaries exceed v2's largest (235,331 blocks); 68 of them have a v3 fold, 33 of those
  are in the matched subset. The largest five by blocks are 7,991,293, 7,190,784, 4,438,416, 3,002,869 and 2,729,833
  (all `x64` goodware from `Balanced_Goodware`; the fifth has no v3 fold). `smoke_gnn_memory.py` with `GT_CACHE` set must
  pass on the GPU before the v3 queue starts. Step 6b caps the gnn data path at `data.MAX_BINARY_BLOCKS` = 500,000
  blocks per binary: a binary over the cap keeps whole functions (evaluation: non-library first, API references,
  size; training: the same weighted random keys, redrawn each epoch), `bin_feat` still counts the full binary, and
  `predictions*.csv` gain `n_blocks_kept` / `n_blocks_total` (`metrics.json`: `truncated_test`). No v2 binary
  reaches the cap, so existing gnn runs are unchanged.
- **TF-IDF baselines (seed 0, pooled over folds 0-4).** Logistic regression on sig+api counts; macro-F1 at 0.5 /
  at each fold's validation threshold.

| arch | n | AUC | PR-AUC | macro-F1 @0.5 | macro-F1 @val thr | x86-rule macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| all | 5652 | 0.973 | 0.979 | 0.917 | 0.885 | 0.690 |
| matched | 4850 | 0.970 | 0.984 | 0.905 | 0.894 | 0.606 |

Built rows per v3 fold (R ransomware, G goodware). The x86 rule predicts ransomware iff x86; its macro-F1 is the
floor every all-arch result must beat. The arch-matched subset is the 4,850 rows with `arch_matched = 1`.

| fold | R x86 | R x64 | G x86 | G x64 | total | x86-rule mF1 | arch-matched n | arch-matched x86-rule mF1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 318 | 182 | 226 | 313 | 1039 | 0.607 | 884 | 0.535 |
| 1 | 412 | 157 | 193 | 354 | 1116 | 0.686 | 924 | 0.598 |
| 2 | 520 | 143 | 194 | 269 | 1126 | 0.685 | 974 | 0.588 |
| 3 | 602 | 126 | 150 | 258 | 1136 | 0.733 | 1007 | 0.660 |
| 4 | 569 | 133 | 205 | 328 | 1235 | 0.715 | 1061 | 0.628 |
| all | 2421 | 741 | 968 | 1522 | 5652 | 0.690 | 4850 | 0.606 |

| fold | mendeley R | mendeley G | ransomware_balanced R | balanced_goodware G |
|---|---:|---:|---:|---:|
| 0 | 191 | 202 | 309 | 337 |
| 1 | 205 | 262 | 364 | 285 |
| 2 | 233 | 262 | 430 | 201 |
| 3 | 261 | 197 | 467 | 211 |
| 4 | 352 | 246 | 350 | 287 |

Rows without a v3 fold: 186.

- balanced_goodware G x64: 5 (hard_negative 4, system 1)
- balanced_goodware G x86: 1 (everyday 1)
- ransomware_balanced R x64: 44 (magniber 12, lockbit 6, akira 4, hive 4, rook 4, stop 3, pandora 3, conti 2)
- ransomware_balanced R x86: 136 (stop 32, lockbit 29, revil 14, teslacrypt 9, legionlocker 6, astralocker 5, petya 4, avaddon 3)

These 186 rows are built but carry no v3 fold, so every kfold_v3 manifest leaves them out. They are not assigned
by family, because folds_all.csv did not place them (stop, lockbit and revil all have folds).

## Arch-balanced training and cell-balanced metrics (step 7, 2026-10-15)

Goal: keep every binary and make architecture worth nothing. Ransomware is 77% x86 and goodware 39% x86 in the v3
all-arch test pool (R x86 2,421, R x64 741, G x86 968, G x64 1,522), so the x86 rule already scores 0.690 macro-F1.

- **Training.** `RunConfig.arch_balance` (`--arch-balance`, both models) gives each training sample the loss weight
  `N / (C * n_cell)`, C the number of non-empty (label, arch) cells of the training manifest. It replaces the class
  weight; the mean weight is 1 and every non-empty cell carries the same total weight. Variant suffix `-ab` (after
  the train-fraction suffix, before `-enc-mean` / `-gated`): `gnn-ab`, `gnn-noedges-ab`, `graph-ab`, `nograph-ab`.
  Without the flag the loss and every name are unchanged. `baselines.py --arch-balance` passes the same weights as
  `sample_weight` to both logistic regressions instead of `class_weight='balanced'`; output dirs end in `_ab`
  (`baselines/v3kfold0_all_s0_ab`); `stats.py` names them `tfidf-ab` / `structural-ab`.
- **Metrics.** `metrics.cell_balanced(y, prob, pred, arch)` returns accuracy, macro-F1, AUC and PR-AUC with sample
  weight `1 / (C_test * n_cell)` and the cell counts (`n_cells`, `n_R_x86`, ...). `scores()` adds it as the
  `cell_balanced` block, so every `test`, `selection.*.test`, baseline `test` / `test_at_val_threshold` and floor carries it.
  An empty cell (an x86-only manifest) is left out of C and its count is 0. Under it the x86 rule scores exactly 0.5
  (accuracy, macro-F1, AUC) when all four cells are non-empty.
- **Reports.** `report.py` adds "Cell-balanced (each class x arch weighted equally)" (kfold_v3 rows, both selection
  rules, floors). `gates.py --stage v3` adds cell-balanced AUC / macro-F1 columns, the `-ab` arms and the paired
  cell-balanced differences (R0-R2, R2-R7, R4-R7, R5-R7 for the plain and for the `-ab` arms, and each `-ab` arm
  against its unweighted twin). `stats.py --balanced` bootstraps with the cell weights, fixed per test sample from the
  pooled test set.
- **Queue.** `queue_gnn.sh --stage v3`: CPU baselines all, matched, all `--arch-balance`; the `-ab` arms R2, R7, R4,
  R5 on arch all (the main result); the unweighted arch-all arms; the arch-matched arms; then report, gates and
  val_threshold. Matched has no `-ab` arms in the queue; it keeps all ransomware (77% x86) and 56% x86 of its goodware, so it is
  less skewed than arch all but not cell-balanced.
- **TF-IDF baselines, kfold_v3 arch all, seed 0, pooled over folds 0-4** (n = 5,652; accuracy / macro-F1 / AUC / PR-AUC):

| TF-IDF LR | threshold | normal | cell-balanced |
|---|---|---|---|
| class weight (existing) | 0.5 | 0.918 / 0.917 / 0.974 / 0.979 | 0.902 / 0.902 / 0.960 / 0.959 |
| class weight (existing) | validation | 0.886 / 0.885 / 0.974 / 0.979 | 0.847 / 0.846 / 0.960 / 0.959 |
| arch-balanced | 0.5 | 0.910 / 0.909 / 0.969 / 0.976 | 0.909 / 0.909 / 0.967 / 0.968 |
| arch-balanced | validation | 0.868 / 0.868 / 0.969 / 0.976 | 0.848 / 0.847 / 0.967 / 0.968 |
| majority floor | - | 0.559 / 0.359 / 0.500 / 0.559 | 0.500 / 0.333 / 0.500 / 0.500 |
| x86-rule floor | - | 0.698 / 0.690 / 0.688 / 0.678 | 0.500 / 0.500 / 0.500 / 0.500 |

## Day-1b results (2026-10-15/16, seed 0 only; not the pre-registered test)

307 GPU minutes, 65 runs (two OOMs on a shared GPU, rerun). Every arm is the typed GNN (R7) with one change,
on kfold x86, AUC-selected checkpoint at the validation threshold. The paired difference against R7 comes from
1,000 cluster resamples. Full tables: `RUNS/gates_day1b.md`, `RESULTS.md`.

| arm | AUC | PR-AUC | macro-F1 | dAUC vs R7 [95% CI] |
|---|---|---|---|---|
| R7 typed GNN | 0.964 | 0.974 | 0.905 | |
| block masking 15% | 0.986 | 0.990 | 0.935 | +0.022 [+0.005, +0.054] |
| edge flip 20% | 0.986 | 0.990 | 0.888 | +0.022 [+0.003, +0.050] |
| DropEdge 20% | 0.982 | 0.987 | 0.908 | +0.018 [+0.004, +0.038] |
| backward only | 0.980 | 0.984 | 0.887 | +0.016 [+0.000, +0.040] |
| 8 layers | 0.978 | 0.982 | 0.913 | +0.014 [-0.004, +0.041] |
| function dropping 15% | 0.972 | 0.974 | 0.921 | +0.007 [-0.018, +0.041] |
| token masking 50% | 0.963 | 0.979 | 0.915 | -0.001 [-0.014, +0.009] |
| token masking 15% | 0.957 | 0.977 | 0.910 | -0.007 [-0.048, +0.016] |
| library dropping 40% (+5% others) | 0.944 | 0.972 | 0.921 | -0.020 [-0.107, +0.039] |
| forward only | 0.930 | 0.964 | 0.908 | -0.034 [-0.105, +0.005] |
| undirected | 0.928 | 0.963 | 0.878 | -0.037 [-0.129, +0.013] |
| R0 TF-IDF LR (reference) | 0.982 | 0.990 | 0.935 | |

Learning curve (R7 / TF-IDF AUC): 25% of training families 0.945 / 0.970; 50% 0.956 / 0.978; 100% 0.964 / 0.982.

Reading:
1. **Coarse corruption helps, fine-grained masking does not.** Block masking, edge flipping and DropEdge raise
   AUC by about +0.02, with CIs above zero. Token masking at 15% or 50% does nothing. The edge-corrupting arms
   help as much as block masking, so this reads as regularisation of an overfitting model, not as evidence that
   real wiring matters.
2. **Block masking ties TF-IDF.** On AUC / PR-AUC / macro-F1 the block-masked GNN scores 0.986 / 0.990 / 0.935
   against TF-IDF's 0.982 / 0.990 / 0.935. Paired, dAUC is +0.004 [-0.004, +0.015]. This is the first GNN arm at
   TF-IDF level.
3. **Direction arms are inconsistent.** Backward-only is 0.980, forward-only 0.930 and undirected 0.928, and
   macro-F1 often moves opposite to AUC. This is noise from checkpoint selection, not a directional effect.
4. **Literature replications:** the predicted harm from DropEdge (AIA) did not appear here. Depth 8 did not
   oversmooth. Heavy token masking matched the opcode-augmentation result (no gain).
5. **Next.** Block masking on the no-edge (R2) and shuffled (R4) arms decides whether edges matter once the
   model is regularised. After that, seeds 1-2 for R2/R4/R7 and the block-masking trio, then the v3
   all-architecture runs with `--arch-balance`.

### Block-masking controls (2026-10-16, seed 0 only)

Every arm below has 15% block masking. Kfold x86, AUC-selected checkpoint at the validation threshold; paired
differences come from 2,000 cluster resamples.

| arm, all with 15% block masking | AUC | PR-AUC | macro-F1 |
|---|---|---|---|
| typed real edges (R7) | 0.986 | 0.990 | 0.935 |
| no edges (R2) | 0.977 | 0.983 | 0.941 |
| shuffled edges (R4) | 0.974 | 0.982 | 0.901 |

| comparison | dAUC [95% CI] | dPR-AUC | dmacro-F1 |
|---|---|---|---|
| typed - shuffled | +0.012 [+0.004, +0.021] | +0.009 [+0.002, +0.021] | +0.033 [-0.001, +0.081] |
| typed - no edges | +0.009 [+0.001, +0.019] | +0.007 [-0.001, +0.017] | -0.007 [-0.075, +0.055] |
| shuffled - no edges | -0.002 [-0.015, +0.008] | -0.001 | -0.040 [-0.100, -0.003] |
| typed - TF-IDF | +0.004 [-0.005, +0.017] | +0.000 | -0.001 |

Once the model is regularised, real wiring beats shuffled wiring on ranking: AUC and PR-AUC CIs are above zero,
and dAUC passes the +0.01 bar. Macro-F1 misses the rule by a hair (CI lower bound -0.001). Against no edges, the
AUC gain (+0.009) is just under the bar and macro-F1 does not follow. Shuffled wiring hurts macro-F1 against no
edges. This is the first structure signal in the ladder, but it is seed 0 only. The pre-registered test needs
seeds 1-2 of these three arms (30 runs, about 2.5 GPU h).

### Pre-registered test, block masking 15% (3 seeds x 5 folds, 2026-10-16)

| comparison (b - a) | dAUC [95% CI] | dPR-AUC | dmacro-F1 | rule |
|---|---|---|---|---|
| typed - no edges | +0.004 [-0.001, +0.010] | +0.004 [-0.000, +0.011] | +0.006 [-0.019, +0.026] | **fail** |
| typed - shuffled | +0.017 [+0.001, +0.047] | +0.010 [+0.001, +0.026] | +0.010 [+0.001, +0.020] | AUC passes; macro-F1 CI > 0 but below +0.02, so **fail** |
| shuffled - no edges | -0.012 [-0.042, +0.004] | -0.005 | -0.004 | |

Pooled 3-seed AUC: typed 0.979, no edges 0.974, shuffled 0.962.

Verdict under the pre-registered rule: **structure does not help.** Real wiring beats random wiring, mainly
because random wiring hurts. Real wiring does not beat no wiring. The CFG, as message passing uses it, adds
nothing measurable over a regularised bag of blocks.

## Reader inputs: operand-role vocabulary and unreferenced imports (step 14, 2026-10-16)

Status (step 25, 2026-10-20): the `--imports unreferenced` option described here was removed (its baseline reached AUC 0.526, the gate
failed and the GNN arm never ran), together with the `lcc` subgraph policy and the graph transformer (`--model transformer`).
`--vocab yanping|hybrid` and the side arrays are kept as experiments, not on the main path. See "Final results" at the end.

Two options use the team's reader output (`CT_GAT/IDA_FEATURE_READERS.md`) through cache-aligned side arrays (README, "Reader inputs").
Neither changes a default run: `RunConfig.vocab = 'ours'` and `imports = ''` keep every name, vocabulary and checkpoint.

- **Vocabulary** (`--vocab yanping|hybrid`, run-name suffix `-voc-<name>`). yanping: the reader's three ids per instruction plus our `api`
  tokens. hybrid: our `mn` and `beh`, the reader's two operand ids, `api`. Each kind except `beh` is cut on the training binaries
  (document frequency at least `min_df`). Question: does Yanping's operand categorisation carry more than our `sig`? Arms (kfold x86,
  seed 0): the no-edge arm with 15% block masking (`gnn-noedges-mblk15`, ours) against `-voc-yanping` and `-voc-hybrid`, plus the TF-IDF
  baselines over the same token kinds.
- **Unreferenced imports** (`--imports unreferenced`, suffix `-imp`). A bag of the imports in a binary's header that no instruction
  references (static fact of the export, not run-time evidence) and two log counts enter the head. Vocabulary and idf: training binaries
  only, `min_df` 2. Arms: three CPU baselines (`baselines.py --imports`: `imports_unref_lr`, `imports_header_lr`, `tfidf_sig_api_imp_lr`),
  then R7 + `--mask-blocks 0.15 --imports unreferenced` **only if** the pooled AUC of `imports_unref_lr` is at least the majority floor's
  AUC + 0.05 (`gates.py --stage imports --verdict`).

Side builds from the finished reader job (6517 binaries, `readers done` 20:51): `side/yanping` 2411 ok / 0 failed / 0 missing on the v2 cache and
5838 / 0 / 0 on the combined cache; `side/imports` 2411 / 0 / 0 and 5838 / 0 / 0. The reader also scans orphan instructions, which the
cache omits, so for 34 of the first 1343 v2 binaries it sees more referenced imports than the cache does; the converter requires only that
every import the cache references is referenced for the reader.

CPU baselines on the v2 cache, kfold x86, seed 0, pooled over the five folds (AUC checkpoint at the validation threshold):

| arm | AUC | PR-AUC | macro-F1 |
|---|---|---|---|
| TF-IDF, ours (sig + api) | 0.982 | 0.990 | 0.935 |
| TF-IDF, yanping | 0.981 | 0.990 | 0.938 |
| TF-IDF, hybrid | 0.981 | 0.989 | 0.917 |
| unreferenced imports, LR | 0.526 | 0.668 | 0.385 |
| header imports, LR | 0.985 | 0.991 | 0.954 |
| TF-IDF + unreferenced imports | 0.982 | 0.990 | 0.935 |

Paired differences, second minus first, dAUC [95% CI] with 1000 resamples: TF-IDF ours -> yanping -0.001 [-0.005, +0.003], ours -> hybrid
-0.001 [-0.003, +0.000], TF-IDF -> TF-IDF + unreferenced imports +0.000 [-0.000, +0.000]. The reader's operand categories give a bag of
tokens no better and no worse than ours, as the linear baseline sees them. Unreferenced imports are rare (x86: 16.6% of ransomware and 0.6%
of goodware have one or more; 81 of 1134 ransomware have no header imports at all) and under the family holdout a logistic regression on them barely ranks
the test binaries: `imports_unref_lr` pooled AUC 0.526 against a majority floor of 0.500 + 0.05. **The imports gate
fails, so the R7 + imports arm is not run.** (The header imports alone are a strong feature, AUC 0.985; that is a different feature from
the one the step asked for.) Smoke runs (2 epochs, fold 0, `--out smoke_step14_*`) confirm that both GNN options train end to end; the
GNN vocabulary arms (`queue_gnn.sh --stage vocab`, 10 runs) are not run yet.

## Header imports as a binary-level input (step 15, 2026-10-17)

`--imports header` (run-name suffix `-imph`) is the header-import feature the step 14 baselines singled out (`imports_header_lr`, AUC 0.985):
the same vocabulary, `EmbeddingBag` and two log counts as `--imports unreferenced`, over every header import identity instead of the
unreferenced ones (training binaries only, `min_df` 2, idf). `-imp` runs and all default names are unchanged (tested bit for bit). Queue:
`queue_gnn.sh --stage imports_header` runs R7 + `--mask-blocks 0.15 --imports header` (`gnn-mblk15-imph`, kfold x86, seed 0, folds 0-4)
with no gate; `gates.py --stage imports` adds R7 against R7 + header imports, `imports_header_lr` against R7 + header imports and TF-IDF
against R7 + header imports. The five-fold arm is not run yet. A 2-epoch smoke run on fold 0 (`--fixed-epochs 2 --batch-blocks 100000
--out smoke_step15_imports_header`) trains end to end: 4292 header identities in the vocabulary, test AUC 0.965 and macro-F1 0.966 after two
epochs (not a result).

## Vocabulary decision (2026-10-17): keep `ours`

`--vocab yanping|hybrid` is an experiment, not on the main path (step 25).

Pre-registered test. Arm: no-edge GNN + 15% block masking, kfold x86, 3 seeds x 5 folds, AUC-selected checkpoint at
the validation threshold, paired cluster bootstrap (2,000 resamples, 712 clusters). The rule for replacing `ours` is
dAUC >= +0.01 and dmacro-F1 >= +0.02, both 95% CIs above zero, and non-inferior on TF-IDF.

| candidate | dAUC [95% CI] | dPR-AUC | dmacro-F1 [95% CI] | TF-IDF dAUC | verdict |
|---|---|---|---|---|---|
| yanping (opcode + 2 operand roles + api) | +0.006 [-0.001, +0.014] | +0.003 [-0.002, +0.009] | +0.014 [-0.015, +0.041] | -0.001 [-0.005, +0.003] | tie, fails rule |
| hybrid (mn + beh + 2 operand roles + api) | -0.004 [-0.011, +0.003] | -0.007 [-0.016, -0.001] | +0.003 [-0.016, +0.021] | -0.001 [-0.003, +0.000] | tie/worse, fails rule |

Pooled 3-seed AUC: ours 0.974, yanping 0.980, hybrid 0.971.

**Decision.** `vocab='ours'` stays the default (mnemonic, operand-category signature, behaviour id, API name).
`--vocab yanping` remains available. The operand roles from the team's reader cannot separate stack memory from
other memory in the 2.1 exports, which is consistent with the near-tie.

## Final results (2026-10-20)

All runs are finished. Every number below is read from `RUNS/gates_{noapi,team,v3,paths_codebert,model1,imports_header}.md` or
reproduced with `python -m evaluation.stats` (the gates files and the command give the same values). Setup, unless a table says
otherwise: kfold x86 on the v2 cache (5 family-holdout folds, 1,810 test binaries in 712 clusters); kfold_v3 `arch all` on the
combined cache (5,652 test binaries in 2,586 clusters). Every GNN arm carries 15% block masking (`mblk15`): R7 is the typed
GNN, R2 the same network with `--control no_edges`, R4 with degree-preserving shuffled edges. AUC checkpoint at its validation
threshold. A table value is the mean over the listed seeds of the five-fold pooled metric; a difference is second minus first with
the 95% paired cluster bootstrap interval (2,000 resamples; clusters are ransomware families and single goodware binaries). The
pre-registered rule: a second arm beats the first when dAUC >= 0.01 and dmacro-F1 >= 0.02 and both intervals exclude zero.
TF-IDF and the header-imports LR exist for seed 0 only, and the kfold_v3 arms for seed 0 only, so those comparisons are a direction,
not the test.

### 1. Does CFG structure help?

kfold x86, 3 seeds x 5 folds:

| arm | AUC | PR-AUC | macro-F1 |
|---|---|---|---|
| R7 typed edges | 0.979 | 0.987 | 0.935 |
| R2 no edges | 0.974 | 0.982 | 0.929 |
| R4 shuffled edges | 0.962 | 0.977 | 0.925 |

| first → second | dAUC [95% CI] | dPR-AUC [95% CI] | dmacro-F1 [95% CI] | rule |
|---|---|---|---|---|
| no edges → typed | +0.004 [-0.001, +0.010] | +0.004 [-0.000, +0.011] | +0.006 [-0.019, +0.026] | fails |
| shuffled → typed | +0.017 [+0.001, +0.047] | +0.010 [+0.001, +0.026] | +0.010 [+0.001, +0.020] | fails (macro-F1 below +0.02) |
| no edges → shuffled | -0.012 [-0.042, +0.004] | -0.005 [-0.022, +0.004] | -0.004 [-0.028, +0.014] | |

kfold_v3 `arch all`, arch-balanced training (`-ab`), seed 0 (`gates_v3.md`):

| arm | AUC | PR-AUC | macro-F1 |
|---|---|---|---|
| R7 typed edges | 0.965 | 0.970 | 0.904 |
| R2 no edges | 0.961 | 0.966 | 0.890 |
| R4 shuffled edges | 0.956 | 0.964 | 0.907 |
| TF-IDF, arch-balanced | 0.969 | 0.976 | 0.868 |

| first → second | dAUC [95% CI] | dPR-AUC [95% CI] | dmacro-F1 [95% CI] |
|---|---|---|---|
| no edges → typed | +0.004 [-0.006, +0.015] | +0.004 [-0.006, +0.015] | +0.014 [-0.017, +0.051] |
| shuffled → typed | +0.009 [-0.012, +0.030] | +0.006 [-0.007, +0.021] | -0.003 [-0.032, +0.033] |

**Verdict: structure does not help.** On x86 the typed edges are 0.004 AUC above no edges, with an interval that includes zero,
and macro-F1 does not follow. They beat shuffled edges on AUC, but random wiring is what loses: shuffled edges are 0.012 below
no edges. On the all-architecture protocol, seed 0, typed edges are within noise of both controls. The unweighted R2, R4 and R7
arms of the v3 protocol (no `-ab`) are not on disk and are not reported.

### 2. Do imports explain the result?

Header imports are the PE import table of the binary (`imports_header_lr`: TF-IDF logistic regression over the import names;
`--imports header` for the GNN). kfold x86, seed 0 (`gates_imports_header.md`, `gates_team.md`):

| arm | AUC | PR-AUC | macro-F1 |
|---|---|---|---|
| TF-IDF (sig + api tokens) | 0.982 | 0.990 | 0.935 |
| header imports, LR | 0.985 | 0.991 | 0.954 |
| R7 typed (our GNN) | 0.986 | 0.990 | 0.935 |
| R2 no edges | 0.977 | 0.983 | 0.941 |
| R7 + header imports | 0.985 | 0.992 | 0.940 |

kfold_v3 `arch all`, seed 0:

| arm | AUC | PR-AUC | macro-F1 |
|---|---|---|---|
| header imports, LR | 0.970 | 0.977 | 0.914 |
| header imports, LR, arch-balanced | 0.966 | 0.974 | 0.908 |
| R7 typed, arch-balanced | 0.965 | 0.970 | 0.904 |
| R2 no edges, arch-balanced | 0.961 | 0.966 | 0.890 |

| first → second (seed 0) | dAUC [95% CI] | dPR-AUC [95% CI] | dmacro-F1 [95% CI] |
|---|---|---|---|
| x86: header LR → R7 typed | +0.001 [-0.006, +0.009] | -0.001 [-0.008, +0.005] | -0.020 [-0.082, +0.020] |
| x86: header LR → R2 no edges | -0.008 [-0.017, +0.001] | -0.008 [-0.019, -0.001] | -0.013 [-0.057, +0.019] |
| x86: R7 → R7 + header imports | -0.001 [-0.013, +0.007] | +0.002 [-0.004, +0.009] | +0.006 [-0.075, +0.084] |
| x86: header LR → R7 + header imports | +0.000 [-0.008, +0.008] | +0.001 [-0.004, +0.006] | -0.014 [-0.084, +0.031] |
| v3: header LR (arch-balanced) → R7 typed (arch-balanced) | -0.001 [-0.021, +0.018] | -0.003 [-0.019, +0.010] | -0.004 [-0.058, +0.045] |
| v3: header LR (arch-balanced) → R2 no edges (arch-balanced) | -0.005 [-0.025, +0.014] | -0.007 [-0.025, +0.008] | -0.018 [-0.077, +0.039] |

**Verdict: a logistic regression on the import table ties our GNN** on both protocols (AUC 0.985 against 0.986 on x86, 0.966
against 0.965 on v3), and feeding the same imports to the GNN adds nothing (-0.001). The variant that used only the imports no
instruction references reached AUC 0.526 as a baseline; it was dropped from the code in step 25 and its GNN arm never ran.

### 3. What do the API names carry?

kfold x86, 3 seeds (`noapi` removes the `api` token kind, so a block holds its `mn`, `sig` and `beh` tokens only; TF-IDF rows
seed 0). The seed-0 pooled R7 noapi row of `gates_noapi.md` reads 0.943; the table gives the 3-seed mean.

| arm | AUC | PR-AUC | macro-F1 |
|---|---|---|---|
| R7 typed | 0.979 | 0.987 | 0.935 |
| R7 typed, noapi | 0.961 | 0.979 | 0.910 |
| R2 no edges | 0.974 | 0.982 | 0.929 |
| R2 no edges, noapi | 0.981 | 0.988 | 0.932 |
| TF-IDF (sig + api) | 0.982 | 0.990 | 0.935 |
| TF-IDF noapi (mn + sig + beh) | 0.976 | 0.984 | 0.908 |

| first → second | dAUC [95% CI] | dPR-AUC [95% CI] | dmacro-F1 [95% CI] | note |
|---|---|---|---|---|
| R7 → R7 noapi (api removed) | -0.017 [-0.044, -0.001] | -0.008 [-0.023, +0.001] | -0.025 [-0.048, -0.007] | api loss shown |
| R2 → R2 noapi (api removed) | +0.007 [-0.001, +0.018] | +0.005 [-0.000, +0.013] | +0.002 [-0.025, +0.030] | api loss not shown |
| TF-IDF → TF-IDF noapi (seed 0) | -0.005 [-0.014, +0.007] | -0.007 [-0.015, -0.001] | -0.027 [-0.081, +0.012] | api loss not shown |
| R2 noapi → R7 noapi | -0.019 [-0.050, +0.001] | -0.009 [-0.026, +0.002] | -0.021 [-0.086, +0.023] | edges vs none, no api |
| TF-IDF → R2 noapi (seed 0) | +0.001 [-0.007, +0.012] | -0.002 [-0.010, +0.004] | +0.003 [-0.023, +0.028] | |
| header LR → R2 noapi (seed 0) | -0.001 [-0.009, +0.006] | -0.003 [-0.012, +0.003] | -0.016 [-0.046, +0.008] | |

**Verdict.** Removing the API names costs the typed GNN (0.017 AUC) but not the no-edge GNN, which is nominally better without
them (+0.007, interval includes zero). The 3-seed no-edge model without API names is level with TF-IDF and with the header-imports LR.

### 4. Depth and call graph

kfold x86, 3 seeds. L3 is the default (3 message-passing layers over each function's CFG); `nocall` turns off message passing over
the call graph.

| arm | AUC | PR-AUC | macro-F1 |
|---|---|---|---|
| R7 typed, L3 (default) | 0.979 | 0.987 | 0.935 |
| R7 typed, L2 | 0.968 | 0.981 | 0.933 |
| R7 typed, no call graph | 0.976 | 0.985 | 0.920 |

| first → second | dAUC [95% CI] | dPR-AUC [95% CI] | dmacro-F1 [95% CI] | rule |
|---|---|---|---|---|
| L2 → L3 | +0.011 [-0.002, +0.037] | +0.005 [-0.002, +0.018] | +0.003 [-0.007, +0.014] | fails |
| call graph off → on | +0.003 [-0.014, +0.020] | +0.001 [-0.007, +0.010] | +0.015 [-0.002, +0.040] | fails |

**Verdict.** Neither the third CFG layer nor the call-graph message passing is shown to matter: every interval includes zero.

### 5. The 500-block cap

`ours b500` is `--subgraph cap --budget 500` (whole functions ranked non-library, then API references, then size, until 500 blocks),
the budget of the teammates' pipelines. kfold x86, seed 0 (`gates_team.md`); the 3-seed mean is given where the arm has three
seeds (our b500 arms ran seed 0 only).

| arm | AUC | PR-AUC | macro-F1 | 3-seed AUC / macro-F1 |
|---|---|---|---|---|
| R7 typed, whole graph | 0.986 | 0.990 | 0.935 | 0.979 / 0.935 |
| R2 no edges, whole graph | 0.977 | 0.983 | 0.941 | 0.974 / 0.929 |
| ours b500 typed | 0.923 | 0.954 | 0.843 | |
| ours b500 no edges | 0.955 | 0.964 | 0.915 | |
| team gnn500 (their GAT, 57-d features) | 0.956 | 0.975 | 0.850 | 0.954 / 0.863 |
| team cb500 (their GAT, CodeBERT vectors) | 0.791 | 0.872 | 0.723 | 0.820 / 0.751 |
| team sig500 (their edge-typed GAT, 141-d subgraphs) | 0.953 | 0.974 | 0.894 | 0.949 / 0.886 |
| team sig500v2 (rebuild with the streaming preprocess) | 0.948 | 0.972 | 0.874 | 0.948 / 0.876 |

| first → second | dAUC [95% CI] | dmacro-F1 [95% CI] | seeds |
|---|---|---|---|
| R7 whole → ours b500 typed | -0.063 [-0.105, -0.029] | -0.092 [-0.182, -0.003] | 0 |
| R2 whole → ours b500 no edges | -0.021 [-0.041, -0.004] | -0.026 [-0.088, +0.031] | 0 |
| ours b500 typed → ours b500 no edges | +0.032 [+0.003, +0.068] | +0.073 [+0.023, +0.135] | 0 |
| ours b500 typed → team gnn500 | +0.032 [-0.002, +0.071] | +0.007 [-0.069, +0.081] | 0 |
| ours b500 no edges → team gnn500 | +0.001 [-0.018, +0.018] | -0.066 [-0.152, +0.015] | 0 |
| R7 whole → team gnn500 | -0.025 [-0.043, -0.009] | -0.072 [-0.136, -0.008] | 0 1 2 |
| R7 whole → team sig500 | -0.029 [-0.054, -0.010] | -0.050 [-0.118, +0.014] | 0 1 2 |
| R7 whole → team sig500v2 | -0.031 [-0.058, -0.009] | -0.059 [-0.105, -0.007] | 0 1 2 |
| R7 whole → team cb500 | -0.159 [-0.221, -0.104] | -0.184 [-0.242, -0.129] | 0 1 2 |
| team sig500 → team sig500v2 | -0.001 [-0.011, +0.009] | -0.009 [-0.039, +0.029] | 0 1 2 |

(The sig500 rows use the 1,808 binaries both arms have. Rows with seed 0 only are a direction, not the test.)

**Verdict.** The cap costs our own models between 0.021 AUC (no edges) and 0.063 AUC (typed; not shown non-inferior), which is of
the size of the 0.025 to 0.031 gap between the teammates' pipelines and our whole-graph R7. At the same 500 blocks, seed 0: gnn500
and sig500 are nominally above our typed arm (gnn500 +0.032 [-0.002, +0.071] dAUC; sig500 +0.030 [-0.003, +0.068] dAUC and +0.051
[+0.013, +0.104] dmacro-F1, the one interval that excludes zero), and level with our no-edge arm (+0.001 and -0.002). No team arm
passes the pre-registered rule against either of our b500 arms. The streaming rebuild sig500v2 is not better than sig500 (-0.001).

### 6. Model 3: our GNN on CodeBERT block vectors

Frozen `microsoft/codebert-base` on the text of each block (signature and API strings of the export, no operands), whole graph,
typed edges (path 2). kfold x86, seed 0 only (path 1, the no-edge twin, was not completed).

| arm | AUC | PR-AUC | macro-F1 |
|---|---|---|---|
| path 2: our GNN, typed, CodeBERT vectors | 0.961 | 0.979 | 0.937 |
| R7 typed, token counts | 0.986 | 0.990 | 0.935 |
| team cb500: their GAT on the same vectors, 500 blocks | 0.791 | 0.872 | 0.723 |

| first → second | dAUC [95% CI] | dPR-AUC [95% CI] | dmacro-F1 [95% CI] |
|---|---|---|---|
| R7 → path 2 | -0.025 [-0.059, -0.005] | -0.011 [-0.030, -0.001] | +0.003 [-0.042, +0.066] |
| path 2 → team cb500 | -0.169 [-0.255, -0.095] | -0.107 [-0.180, -0.062] | -0.214 [-0.313, -0.113] |

**Verdict.** The vectors do not beat token counts: path 2 is 0.025 AUC below R7 and level on macro-F1. Our GNN on the vectors is far
above the team GAT on the same vectors, but cb500 also runs under the 500-block cap, which by itself costs our own models 0.021 to 0.063 AUC
in section 5, so the 0.169 is not all model.

### 7. Model 1: the token-sequence transformer

`token_sequence/`: one position per instruction or API reference, windows of 512, a two-layer transformer, window vectors averaged
per binary, no graph. kfold x86; Model 1 has 3 seeds.

| arm | seeds | AUC | PR-AUC | macro-F1 |
|---|---|---|---|---|
| Model 1 tokenseq | 0 1 2 | 0.970 | 0.982 | 0.902 |
| Model 1 tokenseq | 0 | 0.964 | 0.980 | 0.882 |
| R7 typed | 0 1 2 | 0.979 | 0.987 | 0.935 |
| R2 no edges | 0 1 2 | 0.974 | 0.982 | 0.929 |
| TF-IDF | 0 | 0.982 | 0.990 | 0.935 |
| header imports, LR | 0 | 0.985 | 0.991 | 0.954 |

| first → second | seeds | dAUC [95% CI] | dPR-AUC [95% CI] | dmacro-F1 [95% CI] |
|---|---|---|---|---|
| TF-IDF → Model 1 | 0 | -0.018 [-0.034, -0.005] | -0.011 [-0.024, -0.003] | -0.053 [-0.127, +0.010] |
| header LR → Model 1 | 0 | -0.021 [-0.039, -0.006] | -0.012 [-0.026, -0.003] | -0.072 [-0.143, -0.011] |
| R7 → Model 1 | 0 1 2 | -0.009 [-0.021, +0.001] | -0.005 [-0.014, +0.003] | -0.033 [-0.101, +0.030] |
| R2 → Model 1 | 0 1 2 | -0.005 [-0.016, +0.005] | -0.001 [-0.009, +0.007] | -0.027 [-0.092, +0.028] |

**Verdict.** Model 1 does not beat any reference. It is behind the two baselines on seed 0 (AUC intervals below zero) and within the
noise of the GNN arms over 3 seeds, with lower macro-F1 on every pair.

### 8. Team pipelines after the fixes (2026-10-21)

Pipeline B rebuilt with main PR #10 (the `endp` fix; library functions no longer sampled), at 500 blocks and with no block cap;
`feat` is Model 2 (77 block features + CFG), `emb` Model 3 (64-d pooled five-channel embedding + CFG). Janice's five-channel Model 1
(main PR #12) reads the first 128 or 512 instructions of a pipeline-B build. kfold x86, 3-seed mean of the five-fold pooled metric
(`RUNS/gates_team.md`, `RUNS/gates_model1_variants.md`; details in `TEAM_PIPELINES_REVIEW.md`, section 13). The AUC of `1 - match
rate` as a ransomware score is 0.502 on the 500-block build and 0.447 uncapped (0.747 on the build before the fix).

| arm | AUC | PR-AUC | macro-F1 |
|---|---|---|---|
| R7 typed, whole graph | 0.979 | 0.987 | 0.935 |
| team sig500v3 (500 blocks, all inputs) | 0.948 | 0.973 | 0.893 |
| team sig500v3 feat (Model 2, 500 blocks) | 0.954 | 0.976 | 0.879 |
| team sig500v3 emb (Model 3, 500 blocks) | 0.966 | 0.982 | 0.909 |
| team sigwhole (uncapped, all inputs) | 0.970 | 0.982 | 0.899 |
| team sigwhole feat (Model 2, uncapped) | 0.970 | 0.983 | 0.929 |
| team sigwhole emb (Model 3, uncapped) | 0.981 | 0.989 | 0.901 |
| five-channel Model 1, L128, 500-block build | 0.919 | 0.954 | 0.863 |
| five-channel Model 1, L512, 500-block build | 0.926 | 0.958 | 0.878 |
| five-channel Model 1, L512, uncapped build | 0.919 | 0.955 | 0.863 |
| Model 1 tokenseq, validation threshold | 0.970 | 0.982 | 0.902 |
| Model 1 tokenseq, out-of-fold threshold | 0.968 | 0.981 | 0.918 |

Two pairs pass the pre-registered rule: Model 2 to Model 3 at 500 blocks, dAUC +0.012 [+0.000, +0.025] (above zero before rounding),
dmacro-F1 +0.031 [+0.002, +0.069]; Model 2 at 500 blocks to Model 2 uncapped, +0.015 [+0.003, +0.029], +0.050 [+0.006, +0.108].
Against R7 no arm passes; uncapped Model 3 is level on AUC, +0.002 [-0.005, +0.009], with macro-F1 -0.035 [-0.091, +0.019]. The
out-of-fold threshold moves Model 1's macro-F1 by +0.016 [-0.014, +0.054] at unchanged AUC (-0.002 [-0.007, +0.002]).

### Removed in step 25: last readings

Three options were removed from the code after the runs above (README, "Removed in step 25"); the run directories stay on disk. Their
last results, kfold x86 seed 0, R7 typed with 15% block masking as the reference (0.986 AUC), from `RUNS/gates_subgraph.md` and
`RUNS/gates_imports_header.md`: subgraph policy `lcc` at 20k blocks AUC 0.969 (dAUC -0.017 [-0.050, +0.001], 297 of 1,810 test binaries
truncated) and at 5k blocks 0.975 (-0.011 [-0.023, -0.001], 537 truncated), both not non-inferior; the unreferenced-imports baseline
(`imports_unref_lr`) AUC 0.526, which failed its gate, so `--imports unreferenced` never ran as a GNN arm. The graph transformer
(`--model transformer`, 3 seeds with and without the graph) is last reported in the Day-1 results section above.

### Headline

A bag-of-blocks model over instruction tokens alone (R2 noapi: block token counts, no edges and no API names; AUC 0.981 over
3 seeds) matches the table of PE imports (0.985) and TF-IDF (0.982): the paired differences on seed 0 are -0.001 [-0.009, +0.006]
against the import table and +0.001 [-0.007, +0.012] against TF-IDF. CFG structure adds nothing. With the API names it is worth
+0.004 AUC [-0.001, +0.010] and fails the rule; without them the typed-edge model is 0.019 AUC below the no-edge one (interval
-0.050 to +0.001). None of the other arms moves the number: a third CFG layer, call-graph message passing, header imports
added to the GNN, CodeBERT block vectors, the token-sequence transformer and the teammates' GATs are all level with or below the
no-edge bag of blocks and the import table. The signal is in the tokens and the imports, not in the graph.
