# Cross-pipeline comparison

All rows below are scored on the shared cohort (tag plain / upx_unpacked, Thanos excluded) and the
Mendeley family-disjoint split (24 train / 14 test ransomware families), except the two `expA`/`expB`
rows, which are the pre-cohort baselines. `mendeley` = Mendeley goodware; `balanced` = Goodware_Balanced
on expB's project-grouped goodware split. Ransomware is identical in every row.

Floors: **majority** = predict the majority class (ransomware) everywhere; **machine** = predict ransomware
iff the file is x86, from the cohort architecture. A score is only interesting relative to these.
`x86 r/g` and `x64 r/g` = ransomware recall / goodware recall inside that architecture.

| pipeline | dataset | model | acc | bal-acc | macro-F1 | AUC | majority | machine | x86 r/g | x64 r/g | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tokenization expA | mendeley (traditional) | RF/WPC/w2v | 0.780 | 0.802 | 0.750 | 0.883 | 0.745 | 0.593 | 0.93/0.84 | 0.20/0.92 |  |
| tokenization expA | mendeley (traditional) | MLP/WP/w2v | 0.834 | 0.866 | 0.810 | 0.815 | 0.745 | 0.593 | 0.99/0.92 | 0.22/1.00 |  |
| tokenization expA | mendeley (traditional) | SVM-RBF/SW/w2v | 0.766 | 0.803 | 0.740 | 0.915 | 0.745 | 0.593 | 0.90/0.87 | 0.19/0.92 |  |
| tokenization expA_cohort | mendeley, cohort (traditional) | RF/WPC/w2v | 0.739 | 0.781 | 0.717 | 0.842 | 0.737 | 0.615 | 0.82/0.86 | 0.19/0.92 |  |
| tokenization expA_cohort | mendeley, cohort (traditional) | MLP/WP/w2v | 0.809 | 0.840 | 0.785 | 0.821 | 0.737 | 0.615 | 0.91/0.90 | 0.21/1.00 |  |
| tokenization expA_cohort | mendeley, cohort (traditional) | SVM-RBF/SW/w2v | 0.784 | 0.816 | 0.760 | 0.915 | 0.737 | 0.615 | 0.88/0.88 | 0.21/0.92 |  |
| tokenization expC | mendeley, cohort (revised) | RF/WPC/w2v | 0.790 | 0.823 | 0.766 | 0.948 | 0.737 | 0.615 | 0.89/0.88 | 0.22/1.00 |  |
| tokenization expC | mendeley, cohort (revised) | MLP/WP/w2v | 0.943 | 0.924 | 0.926 | 0.952 | 0.737 | 0.615 | 0.97/0.88 | 0.94/0.92 |  |
| tokenization expC | mendeley, cohort (revised) | SVM-RBF/SW/w2v | 0.900 | 0.900 | 0.878 | 0.959 | 0.737 | 0.615 | 0.88/0.91 | 0.99/0.83 |  |
| tokenization expB | balanced (traditional) | RF/WPC/w2v | 0.684 | 0.738 | 0.663 | 0.778 | 0.745 | 0.731 | 0.80/0.59 | 0.09/0.99 |  |
| tokenization expB | balanced (traditional) | MLP/WP/w2v | 0.704 | 0.736 | 0.676 | 0.747 | 0.745 | 0.731 | 0.83/0.43 | 0.18/1.00 |  |
| tokenization expB | balanced (traditional) | SVM-RBF/SW/w2v | 0.595 | 0.690 | 0.586 | 0.821 | 0.745 | 0.731 | 0.64/0.70 | 0.02/0.99 |  |
| tokenization expB_cohort | balanced, cohort (traditional) | RF/WPC/w2v | 0.667 | 0.724 | 0.648 | 0.792 | 0.740 | 0.759 | 0.74/0.57 | 0.06/1.00 |  |
| tokenization expB_cohort | balanced, cohort (traditional) | MLP/WP/w2v | 0.609 | 0.657 | 0.590 | 0.732 | 0.740 | 0.759 | 0.65/0.35 | 0.19/0.99 |  |
| tokenization expB_cohort | balanced, cohort (traditional) | SVM-RBF/SW/w2v | 0.542 | 0.660 | 0.540 | 0.820 | 0.740 | 0.759 | 0.52/0.74 | 0.00/1.00 |  |
| tokenization expD | balanced, cohort (revised) | RF/WPC/w2v | 0.559 | 0.664 | 0.553 | 0.875 | 0.746 | 0.765 | 0.54/0.67 | 0.07/0.99 |  |
| tokenization expD | balanced, cohort (revised) | MLP/WP/w2v | 0.621 | 0.663 | 0.597 | 0.753 | 0.746 | 0.765 | 0.72/0.38 | 0.00/0.94 |  |
| tokenization expD | balanced, cohort (revised) | SVM-RBF/SW/w2v | 0.876 | 0.861 | 0.844 | 0.904 | 0.746 | 0.765 | 0.90/0.62 | 0.86/0.94 |  |
| cnn_vit (seed 1337) | mendeley, cohort (unified asm) | HierarchicalMalwareNet | 0.576 | 0.705 | 0.575 | 0.855 | 0.737 | 0.615 | 0.51/0.97 | 0.12/1.00 |  |
| cnn_vit (mean of 5 seeds) | mendeley, cohort (unified asm) | HierarchicalMalwareNet | 0.679 | 0.686 | 0.628 | 0.802 | 0.737 | 0.615 |  |  | sd macro-F1 0.018 |
| cnn_vit (seed 1337) | balanced, cohort (unified asm) | HierarchicalMalwareNet | 0.573 | 0.584 | 0.542 | 0.622 | 0.740 | 0.759 | 0.69/0.20 | 0.03/0.84 |  |
| cnn_vit (mean of 5 seeds) | balanced, cohort (unified asm) | HierarchicalMalwareNet | 0.516 | 0.593 | 0.502 | 0.645 | 0.740 | 0.759 |  |  | sd macro-F1 0.113 |
| graph2vec | balanced, cohort | LR/wl_tfidf/argmax (default 0.5) | 0.489 | 0.627 | 0.489 | 0.840 | 0.740 | 0.759 | 0.42/0.78 | 0.00/0.99 |  |
| graph2vec | balanced, cohort | LR/wl_tfidf/threshold from train out-of-fold scores/thr 0.41 | 0.503 | 0.626 | 0.502 | 0.840 | 0.740 | 0.759 | 0.44/0.70 | 0.07/0.99 |  |
| graph2vec | balanced, cohort | RF/wl_svd/argmax (default 0.5) | 0.532 | 0.666 | 0.531 | 0.837 | 0.740 | 0.759 | 0.48/0.85 | 0.00/1.00 |  |
| graph2vec | balanced, cohort | RF/wl_svd/threshold from train out-of-fold scores/thr 0.48 | 0.532 | 0.663 | 0.531 | 0.837 | 0.740 | 0.759 | 0.49/0.83 | 0.00/1.00 |  |
| graph2vec | balanced, cohort | SVM-RBF/wl_svd/argmax (default 0.5) | 0.470 | 0.622 | 0.470 | 0.831 | 0.740 | 0.759 | 0.37/0.83 | 0.07/1.00 |  |
| graph2vec | balanced, cohort | SVM-RBF/wl_svd/threshold from train out-of-fold scores/thr -0.36 | 0.513 | 0.641 | 0.512 | 0.831 | 0.740 | 0.759 | 0.45/0.76 | 0.07/0.99 |  |
| graph2vec | balanced, cohort | MLP/wl_svd/argmax (default 0.5) | 0.493 | 0.624 | 0.492 | 0.842 | 0.740 | 0.759 | 0.42/0.76 | 0.07/0.98 |  |
| graph2vec | balanced, cohort | MLP/wl_svd/threshold from train out-of-fold scores/thr 0.45 | 0.558 | 0.663 | 0.554 | 0.842 | 0.740 | 0.759 | 0.54/0.72 | 0.07/0.98 |  |
| graph2vec | balanced, cohort | RF/graph2vec/argmax (default 0.5) | 0.421 | 0.601 | 0.417 | 0.831 | 0.740 | 0.759 | 0.28/0.93 | 0.00/1.00 |  |
| graph2vec | balanced, cohort | RF/graph2vec/threshold from train out-of-fold scores/thr 0.38 | 0.530 | 0.662 | 0.529 | 0.831 | 0.740 | 0.759 | 0.48/0.83 | 0.00/1.00 |  |
| graph2vec | balanced, cohort | SVM-RBF/graph2vec/argmax (default 0.5) | 0.513 | 0.641 | 0.512 | 0.852 | 0.740 | 0.759 | 0.45/0.78 | 0.07/0.98 |  |
| graph2vec | balanced, cohort | SVM-RBF/graph2vec/threshold from train out-of-fold scores/thr 0.02 | 0.513 | 0.643 | 0.513 | 0.852 | 0.740 | 0.759 | 0.45/0.80 | 0.07/0.98 |  |
| graph2vec | balanced, cohort | MLP/graph2vec/argmax (default 0.5) | 0.546 | 0.663 | 0.543 | 0.851 | 0.740 | 0.759 | 0.51/0.78 | 0.07/0.98 |  |
| graph2vec | balanced, cohort | MLP/graph2vec/threshold from train out-of-fold scores/thr 0.53 | 0.542 | 0.660 | 0.540 | 0.851 | 0.740 | 0.759 | 0.50/0.78 | 0.07/0.98 |  |
| graph2vec | balanced, cohort | RF/size_only/argmax (default 0.5) | 0.413 | 0.586 | 0.410 | 0.702 | 0.740 | 0.759 | 0.28/0.91 | 0.00/0.96 |  |
| graph2vec | balanced, cohort | RF/size_only/threshold from train out-of-fold scores/thr 0.52 | 0.413 | 0.588 | 0.409 | 0.702 | 0.740 | 0.759 | 0.28/0.91 | 0.00/0.98 |  |
| graph2vec | balanced, cohort | LR/size_only/argmax (default 0.5) | 0.613 | 0.680 | 0.599 | 0.625 | 0.740 | 0.759 | 0.68/0.91 | 0.00/0.77 |  |
| graph2vec | balanced, cohort | LR/size_only/threshold from train out-of-fold scores/thr 0.42 | 0.624 | 0.664 | 0.602 | 0.625 | 0.740 | 0.759 | 0.72/0.85 | 0.01/0.69 |  |
| graph2vec | mendeley, cohort | LR/wl_tfidf/argmax (default 0.5) | 0.756 | 0.824 | 0.740 | 0.977 | 0.737 | 0.615 | 0.82/0.97 | 0.11/1.00 |  |
| graph2vec | mendeley, cohort | LR/wl_tfidf/threshold from train out-of-fold scores/thr 0.43 | 0.855 | 0.889 | 0.835 | 0.977 | 0.737 | 0.615 | 0.82/0.96 | 0.81/1.00 |  |
| graph2vec | mendeley, cohort | RF/wl_svd/argmax (default 0.5) | 0.817 | 0.871 | 0.799 | 0.968 | 0.737 | 0.615 | 0.74/0.98 | 0.81/1.00 |  |
| graph2vec | mendeley, cohort | RF/wl_svd/threshold from train out-of-fold scores/thr 0.48 | 0.831 | 0.880 | 0.813 | 0.968 | 0.737 | 0.615 | 0.77/0.98 | 0.81/1.00 |  |
| graph2vec | mendeley, cohort | SVM-RBF/wl_svd/argmax (default 0.5) | 0.568 | 0.692 | 0.566 | 0.932 | 0.737 | 0.615 | 0.52/0.95 | 0.07/1.00 |  |
| graph2vec | mendeley, cohort | SVM-RBF/wl_svd/threshold from train out-of-fold scores/thr 0.14 | 0.536 | 0.675 | 0.535 | 0.932 | 0.737 | 0.615 | 0.46/0.97 | 0.07/1.00 |  |
| graph2vec | mendeley, cohort | MLP/wl_svd/argmax (default 0.5) | 0.833 | 0.864 | 0.811 | 0.949 | 0.737 | 0.615 | 0.80/0.92 | 0.81/1.00 |  |
| graph2vec | mendeley, cohort | MLP/wl_svd/threshold from train out-of-fold scores/thr 0.42 | 0.862 | 0.881 | 0.839 | 0.949 | 0.737 | 0.615 | 0.85/0.91 | 0.81/1.00 |  |
| graph2vec | mendeley, cohort | RF/graph2vec/argmax (default 0.5) | 0.527 | 0.675 | 0.527 | 0.955 | 0.737 | 0.615 | 0.45/0.98 | 0.03/1.00 |  |
| graph2vec | mendeley, cohort | RF/graph2vec/threshold from train out-of-fold scores/thr 0.42 | 0.644 | 0.753 | 0.638 | 0.955 | 0.737 | 0.615 | 0.59/0.98 | 0.25/1.00 |  |
| graph2vec | mendeley, cohort | SVM-RBF/graph2vec/argmax (default 0.5) | 0.713 | 0.790 | 0.699 | 0.955 | 0.737 | 0.615 | 0.77/0.95 | 0.07/1.00 |  |
| graph2vec | mendeley, cohort | SVM-RBF/graph2vec/threshold from train out-of-fold scores/thr -0.25 | 0.914 | 0.917 | 0.895 | 0.955 | 0.737 | 0.615 | 0.95/0.91 | 0.76/1.00 |  |
| graph2vec | mendeley, cohort | MLP/graph2vec/argmax (default 0.5) | 0.817 | 0.861 | 0.796 | 0.936 | 0.737 | 0.615 | 0.94/0.95 | 0.07/1.00 |  |
| graph2vec | mendeley, cohort | MLP/graph2vec/threshold from train out-of-fold scores/thr 0.49 | 0.817 | 0.861 | 0.796 | 0.936 | 0.737 | 0.615 | 0.94/0.95 | 0.07/1.00 |  |
| graph2vec | mendeley, cohort | RF/size_only/argmax (default 0.5) | 0.550 | 0.687 | 0.549 | 0.919 | 0.737 | 0.615 | 0.50/0.98 | 0.00/0.92 |  |
| graph2vec | mendeley, cohort | RF/size_only/threshold from train out-of-fold scores/thr 0.54 | 0.436 | 0.612 | 0.432 | 0.919 | 0.737 | 0.615 | 0.30/0.98 | 0.00/1.00 |  |
| graph2vec | mendeley, cohort | LR/size_only/argmax (default 0.5) | 0.656 | 0.722 | 0.641 | 0.758 | 0.737 | 0.615 | 0.67/0.89 | 0.22/0.58 |  |
| graph2vec | mendeley, cohort | LR/size_only/threshold from train out-of-fold scores/thr 0.41 | 0.640 | 0.671 | 0.615 | 0.758 | 0.737 | 0.615 | 0.70/0.76 | 0.22/0.50 |  |
| rules | balanced, cohort | any_hit/ngram_rules/12 rules | 0.744 | 0.636 | 0.644 | 0.636 | 0.740 | 0.759 | 1.00/0.00 | 0.31/0.64 |  |
| rules | balanced, cohort | rule_votes/ngram_rules/41 rules/thr 0.64 | 0.526 | 0.634 | 0.523 | 0.757 | 0.740 | 0.759 | 0.51/0.61 | 0.00/1.00 |  |
| rules | balanced, cohort | weighted_logodds/ngram_rules/41 rules/thr 22.51 | 0.536 | 0.610 | 0.527 | 0.739 | 0.740 | 0.759 | 0.57/0.35 | 0.00/1.00 |  |
| rules | balanced, cohort | crypto_signature_any_hit/behaviour_rules/5 rules | 0.505 | 0.607 | 0.502 | 0.607 | 0.740 | 0.759 | 0.49/0.65 | 0.00/0.91 |  |
| rules | balanced, cohort | weighted_logodds/behaviour_rules/23 rules/thr 11.13 | 0.581 | 0.658 | 0.570 | 0.781 | 0.740 | 0.759 | 0.62/0.63 | 0.00/0.93 |  |
| rules | balanced, cohort | behaviour+ngram_rules_logreg/combined | 0.695 | 0.730 | 0.670 | 0.830 | 0.740 | 0.759 | 0.80/0.48 | 0.07/0.99 |  |
| rules | balanced, cohort | mnemonic_tfidf_1_3+LinearSVC/calibration_baseline | 0.828 | 0.851 | 0.802 | 0.929 | 0.740 | 0.759 | 0.80/0.72 | 0.81/1.00 |  |
| rules | balanced, cohort | mnemonic_tfidf_1_3+LogReg/calibration_baseline | 0.822 | 0.844 | 0.796 | 0.923 | 0.740 | 0.759 | 0.80/0.70 | 0.78/1.00 |  |
| rules | mendeley, cohort | any_hit/ngram_rules/14 rules | 0.798 | 0.636 | 0.656 | 0.636 | 0.737 | 0.615 | 0.99/0.30 | 0.92/0.25 |  |
| rules | mendeley, cohort | rule_votes/ngram_rules/36 rules/thr 0.06 | 0.788 | 0.819 | 0.763 | 0.789 | 0.737 | 0.615 | 0.94/0.87 | 0.00/1.00 |  |
| rules | mendeley, cohort | weighted_logodds/ngram_rules/36 rules/thr -0.23 | 0.794 | 0.826 | 0.770 | 0.789 | 0.737 | 0.615 | 0.95/0.88 | 0.00/1.00 |  |
| rules | mendeley, cohort | crypto_signature_any_hit/behaviour_rules/5 rules | 0.342 | 0.294 | 0.302 | 0.294 | 0.737 | 0.615 | 0.49/0.11 | 0.00/1.00 |  |
| rules | mendeley, cohort | weighted_logodds/behaviour_rules/23 rules/thr -1.48 | 0.774 | 0.729 | 0.720 | 0.695 | 0.737 | 0.615 | 0.79/0.61 | 0.96/0.92 |  |
| rules | mendeley, cohort | behaviour+ngram_rules_logreg/combined | 0.821 | 0.849 | 0.797 | 0.893 | 0.737 | 0.615 | 0.97/0.91 | 0.07/0.83 |  |
| rules | mendeley, cohort | mnemonic_tfidf_1_3+LinearSVC/calibration_baseline | 0.969 | 0.957 | 0.960 | 0.978 | 0.737 | 0.615 | 1.00/0.92 | 0.92/1.00 |  |
| rules | mendeley, cohort | mnemonic_tfidf_1_3+LogReg/calibration_baseline | 0.976 | 0.961 | 0.968 | 0.987 | 0.737 | 0.615 | 1.00/0.92 | 0.96/1.00 |  |

## Best macro-F1 per pipeline and dataset

| pipeline | mendeley-type dataset | balanced-type dataset |
|---|---|---|
| tokenization | 0.926 (tokenization expC · MLP/WP/w2v) | 0.844 (tokenization expD · SVM-RBF/SW/w2v) |
| cnn_vit | 0.628 (cnn_vit (mean of 5 seeds) · HierarchicalMalwareNet) | 0.542 (cnn_vit (seed 1337) · HierarchicalMalwareNet) |
| graph2vec | 0.895 (graph2vec · SVM-RBF/graph2vec/threshold from train out-of-fold scores/thr -0.25) | 0.602 (graph2vec · LR/size_only/threshold from train out-of-fold scores/thr 0.42) |
| rules | 0.968 (rules · mnemonic_tfidf_1_3+LogReg/calibration_baseline) | 0.802 (rules · mnemonic_tfidf_1_3+LinearSVC/calibration_baseline) |

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
   baseline of 10 scalars already gets AUC 0.92, and on Goodware_Balanced size-only is the best graph row (0.60). The
   embeddings do not improve on the histogram they compress. 51% of files hit the 5,000-block cap.
4. **The CNN-ViT encoder is the weakest and the least stable.** Five seeds: 0.63 +/- 0.02 on Mendeley (AUC 0.71-0.88,
   so it ranks but the operating point is poor), **0.50 +/- 0.11 on Goodware_Balanced**, at chance within each architecture.
   Below the majority floor on accuracy in both. It was trained on the unified disassembly, so these numbers are comparable
   with the other cohort rows but not with any earlier `asm_parse.py`-sourced CNN-ViT figure.
5. **Architecture is the confound in every pipeline.** Every Goodware_Balanced row that beats chance does so partly by
   reading bitness; the per-architecture columns show ransomware recall inside x64 collapsing to 0.00-0.25 in most rows.
   Per-architecture and family-holdout evaluation over all 40 families, not one fixed split, is the prerequisite for any
   further model comparison (`docs/RESEARCH_DIRECTIONS.md`, recommended order).

EMBER is not in this table; that rerun is being done by a teammate (`ember_pipeline/`, `vm_package/README.md`).
