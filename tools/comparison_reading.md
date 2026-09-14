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

EMBER is not in this table; that rerun is being done by a teammate (`ember_pipeline/`, `vm_package/README.md`).
