## Reading

Numbers are test macro-F1 on the shared cohort and the family-disjoint split unless stated. Floors: majority-class
accuracy is 0.74 (macro-F1 0.42); the architecture-only rule reaches macro-F1 0.43 on Mendeley goodware (it misfires,
because test goodware is 91% x86 while train goodware is 57%) and 0.68-0.71 on Goodware_Balanced (which is 77% x64).

1. **The simplest method is currently the best.** Mnemonic TF-IDF 1-3-grams with logistic regression, fit on train only
   and audited for leakage (`results/rules/summary.md` sections 5-6), scores 0.968 as reported and **0.955 after removing the
   62 test goodware files that duplicate training streams**, with 13 of 14 unseen families at recall 1.00 and 0.93 macro-F1
   inside x64 alone. On Goodware_Balanced it drops to 0.80. Its top n-grams are compiler idiom (wide-integer arithmetic vs
   MSVC CRT prologues), which is why the hard negatives cost 15 points and BlackCat is missed entirely there.
2. **The revised extractor helps the tokenization pipeline a lot.** Mendeley/Mendeley goes from 0.81 (expA, traditional
   features) to **0.93** (expC, mnemonic-only skip-data features, cohort); the cohort filter alone changes little
   (expA_cohort 0.79). The gain is not leakage: ransomware test duplication is 0/362 and goodware duplication is lower than
   expA's. Hive recall goes from 0.16 to 1.00. On Goodware_Balanced, eight of nine tokenization rows sit **below** the
   architecture-only floor; only expD's SVM (0.84) clears it.
3. **graph2vec on approximate CFGs reaches 0.84-0.90 on Mendeley** (AUC 0.98 for the WL histogram), but a size-only
   baseline of 10 scalars already gets AUC 0.92, and on Goodware_Balanced size-only is the best untuned graph row (0.60).
   The PV-DBOW embeddings never beat the histogram they compress. 51% of files hit the 5,000-block cap.
4. **The CNN-ViT encoder is the weakest and the least stable.** Five seeds: 0.63 +/- 0.02 on Mendeley (AUC 0.71-0.88,
   so it ranks but the operating point is poor), **0.50 +/- 0.11 on Goodware_Balanced**, at chance within each architecture.
   Below the majority floor on accuracy in both. It was trained on the unified disassembly, so these numbers are comparable
   with the other cohort rows but not with any earlier `asm_parse.py`-sourced CNN-ViT figure.
5. **Architecture is the confound in every pipeline.** Every Goodware_Balanced row that beats chance does so partly by
   reading bitness; the per-architecture columns show ransomware recall inside x64 collapsing to 0.00-0.25 in most rows.
   Per-architecture and family-holdout evaluation over all 40 families, not one fixed split, is the prerequisite for any
   further model comparison (`docs/RESEARCH_DIRECTIONS.md`, recommended order).

## Tuning (pre-registered protocol: group cross-validation on the training families, one test evaluation per chosen configuration)

Rows marked "chosen by CV (pre-registered)" are the results. Rows marked "post-hoc" were scored only to measure the
CV-to-test gap and must not be quoted as results. Details per pipeline: `results/summary.md` (tokenization, 787
configurations), `results/graph2vec/summary.md` section 7 (103), `results/cnn_vit/summary.md` section 9 (112).

6. **Tuning by cross-validation inside the 24 training families does not transfer to the 14 test families.** The
   tokenization search's CV winner (TF-IDF over tokenizer output, 50,000-token budget, LogReg) scores **0.858** on the
   Mendeley test set against 0.926 for the original configuration; its CV-rank-2/3/5 siblings score 0.954-0.968 post hoc, so
   the axis that won the search (sequence budget) lost the test set. On Goodware_Balanced the chosen configuration
   (0.683) falls below the architecture-only floor and turns into a bitness detector (x64 ransomware recall 0.17).
   Nothing that helped in CV (sequence budget, 1-3-grams, mean+max pooling, linear models) survived; subword tokenizers,
   architecture reweighting and fitted thresholds did not help even in CV.
7. **graph2vec is the one pipeline where tuning paid off on Mendeley, but not through the row the search chose.** The
   pre-registered pick (`wl_svd`/RF, CV rank 1) scores 0.887 against the untuned 0.895. The sparse WL histogram with a
   node labelling of length buckets at depth 4 (CV rank about 11) scores **0.947 post hoc**, AUC 0.982, and beats the
   size-only control (0.81) by a wide margin. The CV-to-test gap of the top five is +0.22 to +0.32. On Goodware_Balanced
   the best tuned row (0.675) still sits below the 0.705 architecture floor; the misclassified goodware is not explained
   by graph size (rank AUC by size about 0.32) and concentrates on a few projects, Steam above all.
8. **CNN-ViT tuning bought nothing.** Chosen configurations: Mendeley 0.655 +/- 0.049, Goodware_Balanced 0.518 +/- 0.106
   (five seeds), within noise of the untuned recipe re-run through the same harness (0.660 / 0.440). Only ROC-AUC moves
   consistently (0.80 to 0.87 on Mendeley). The search leader on Mendeley collapses when the cross-validation fold seed
   changes, so nothing there survives a re-split; on Goodware_Balanced the encoding change (strided sampling across the
   whole file) is the only axis where every level helped, and the gain vanishes inside each architecture. The sampler
   that removes the architecture shortcut was the worst configuration in the whole Balanced search.
9. **Consequence.** Ranking configurations by family-grouped CV on this training set is not reliable enough to pick
   between close alternatives: the CV-to-test gap is 0.2-0.5 and not rank-preserving in two of three pipelines. The
   defensible results remain the untuned ones plus the audited TF-IDF baseline. The next step that would actually
   change the picture is evaluation by family-holdout over all 40 families with per-architecture reporting, not more
   search (`docs/RESEARCH_DIRECTIONS.md`).

## Family holdout over all 38 cohort families (`results/family_holdout/summary.md`)

Every cohort ransomware family is held out exactly once (5 folds, families whole, goodware by duplicate group or
project), plus leave-one-family-out for per-family recall. Untuned, pre-registered configurations only. Fold mean of
macro-F1 (+/- sd over folds), and the x86-rule floor in macro-F1, which is 0.63 on Mendeley and **0.84 on Balanced**:

| model | Mendeley | Balanced | fixed split (Mendeley / Balanced) |
|---|---|---|---|
| TF-IDF 1-3gram + LogReg | **0.954 +/- 0.021** | **0.942 +/- 0.032** | 0.968 / 0.796 |
| graph2vec, untuned WL baseline, OOF threshold | 0.942 +/- 0.037 | 0.867 +/- 0.059 | 0.885 / 0.568 |
| graph2vec, tuning study's post-hoc winner | 0.885 +/- 0.057 | 0.751 +/- 0.056 | 0.947 / n/a |
| tokenization expC/expD config, MLP/WP | 0.869 +/- 0.035 | 0.820 +/- 0.043 | 0.926 / 0.597 |
| CNN-ViT, untuned recipe, 3 seeds per fold | 0.679 +/- 0.038 | 0.736 +/- 0.032 | 0.628 / 0.502 |

10. **The single split was misleading in both directions.** The fixed split understated the CNN-ViT encoder by 0.05 on
    Mendeley and 0.23 on Balanced, overstated the graph2vec post-hoc winner by 0.06, and understated TF-IDF on Balanced
    by 0.15. Differences smaller than about one fold sd (0.02-0.09 depending on the model) should not be read at all.
11. **The graph2vec post-hoc winner does not confirm.** Under family holdout it is worse than the untuned WL baseline on
    both datasets and worse on leave-one-family-out recall by 0.13-0.17. The untuned baseline is the configuration that
    holds up, at 0.94 on Mendeley.
12. **TF-IDF is the only pipeline that is strong on both goodware sources and the one that moves least** from its
    fixed-split number. On Balanced it is one of two models above the 0.84 architecture floor (with the graph2vec
    baseline); every tokenization row and the CNN-ViT encoder sit below it, so their Balanced numbers still do not show
    anything the PE machine field does not.
13. **x64 ransomware recall is the weak spot of every model** (0.13-0.79 pooled, against x64 goodware recall near 1.00),
    and 43 of the 114 x64 ransomware files are one family, Hive. Two families fail everywhere: Phobos (median LOFO
    recall 0.06) and Makop (0.23). Between models the failing set differs, which is the first evidence that an ensemble
    across representations could pay off.
14. **Consequence.** Report family-holdout fold means with their sd as the headline numbers from here on; the fixed
    split remains only for comparability with the paper. The next investment is not another model but an
    architecture-matched evaluation (or arch-balanced sampling) so that the Balanced numbers can be interpreted, and a
    look at why Phobos and Makop are invisible to every representation.

## Sequence transformer over mnemonics (replaces the image encoder; `results/family_holdout/summary_seq_transformer.md`)

Same six ViT blocks as the yanping model, but fed an embedded mnemonic sequence (full vocabulary, attention masks,
1-D stem) instead of token ids as pixels; 16 windows of 4,096 tokens strided over the whole file, mean of window
logits; masked-token pretraining on both corpora (unlabelled, transductive); architecture-balanced batches; 3 seeds
per fold. Configuration frozen before any test fold was scored. Family holdout, fold mean +/- sd of macro-F1:

| model | Mendeley | Balanced | LOFO mean recall (M / B) |
|---|---|---|---|
| sequence transformer (this work) | **0.923 +/- 0.057** | **0.889 +/- 0.057** | 0.92 / 0.84 |
| CNN-ViT image encoder (replaced) | 0.679 +/- 0.038 | 0.736 +/- 0.032 | 0.68 / 0.65 |
| TF-IDF 1-3gram + LogReg | 0.954 +/- 0.021 | 0.942 +/- 0.032 | 0.90 / 0.89 |
| graph2vec WL baseline | 0.942 +/- 0.037 | 0.867 +/- 0.059 | 0.91 / 0.79 |

15. **Fixing the representation recovers the transformer.** +0.24 on Mendeley and +0.15 on Balanced over the image
    encoder, AUC 0.98 / 0.95, now the second-best pipeline on Balanced and above its 0.84 architecture floor. It is
    still 0.03 to 0.05 behind TF-IDF on both datasets and needs the transductive pretraining to get there.
16. **Which of the six changes mattered (Mendeley K-fold, one seed).** Removing pretraining costs 0.13 (0.797 +/-
    0.117), so the masked-token pass on unlabelled streams is the single largest contributor; without it the model is
    below the graph and TF-IDF baselines. Reading only the first window scored 0.947 +/- 0.014, no worse than the
    whole-file windows, so multiple-instance coverage did not help here. Dropping the architecture-balanced sampler
    changed nothing (0.931 +/- 0.044). The gains therefore come from the embedding, the masking and the pretraining;
    the strictly inductive number to quote is 0.80.
17. **It fixes Phobos and misses Makop.** Phobos, invisible to every other representation (median LOFO recall 0.06),
    is recalled at 0.98 / 0.94 (K-fold / LOFO) on Mendeley. Makop stays at 0.53 / 0.07. Hive, the x64 family, is
    1.00 / 1.00 on Mendeley but 0.64 / 0.96 on Balanced, and x64 ransomware recall inside Balanced is 0.75 against
    goodware recall 0.94, so the architecture lean is reduced but not gone.
18. **What was not done.** The import-name side input exists and is unit-tested, but the ransomware import tables
    need the VM (`vm_package/README.md` step 4b), so no run used real import features. Seeds were cut from five to
    three per fold to fit the GPU budget; leave-one-family-out is one seed.

## Ensemble of TF-IDF and the sequence transformer (`results/family_holdout/summary_ensemble.md`)

Two parameter-free rules over the two members' held-out P(ransomware), pre-registered before scoring: `mean`
(primary) and `max` (the OR rule). No weight, threshold or stacker is fitted; argmax at 0.5. Both K-fold and
LOFO rows reuse the members' own held-out predictions, so nothing is retrained. Fold mean +/- sd of macro-F1:

| model | Mendeley | Balanced | LOFO mean recall (M / B) | FPR (M / B) |
|---|---|---|---|---|
| TF-IDF 1-3gram + LogReg (member) | 0.954 +/- 0.018 | 0.942 +/- 0.028 | 0.90 / 0.89 | 0.019 / 0.028 |
| sequence transformer (member) | 0.923 +/- 0.057 | 0.889 +/- 0.057 | 0.92 / 0.84 | 0.045 / 0.122 |
| ensemble, mean (primary) | 0.955 +/- 0.026 | 0.930 +/- 0.036 | 0.92 / 0.88 | 0.024 / 0.049 |
| ensemble, max (OR rule) | 0.951 +/- 0.009 | 0.911 +/- 0.050 | **0.95 / 0.94** | 0.053 / 0.126 |

19. **The mean ensemble does not beat TF-IDF.** It ties on Mendeley (0.955 against 0.954, inside one fold sd)
    and loses on Balanced (0.930 against 0.942), because the transformer's higher false-positive rate on the
    hard negatives is averaged in. The complementarity is real but a parameter-free rule does not harvest it: a
    per-file oracle that picks whichever member is right would reach ransomware recall 0.956 / 0.950 against
    0.928 / 0.912 for TF-IDF alone, on the 6.7% / 9.3% of files where the two members disagree.
20. **The OR rule is the one that changes a number.** Leave-one-family-out recall rises from 0.90 / 0.89 to
    0.95 / 0.94, the best LOFO figure in the study on both datasets, and it is the most stable row on Mendeley
    (fold sd 0.009). The price is the false-positive rate, which doubles on Mendeley (0.019 to 0.053) and
    quadruples on Balanced (0.028 to 0.126). Which of the two rows is preferable depends on the cost of a miss
    against the cost of an alert, not on macro-F1.
21. **Per family, the mean rule sits between its members, never above both.** DoppelPaymer and WastedLocker
    (TF-IDF right, transformer wrong) are pulled down; Nefilim and RansomEXX (transformer right) are pulled up;
    Makop and Maze on Balanced are dragged toward the weaker member. The OR rule recovers each family to its
    better member's recall by construction. A learned combiner (stacking with nested family-grouped CV) is the
    obvious next step, and the negative tuning result above is the reason it was not attempted here.

EMBER is not in this table; that rerun is being done by a teammate (`ember_pipeline/`, `vm_package/README.md`).
