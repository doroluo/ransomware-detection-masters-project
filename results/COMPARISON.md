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
| tokenization expC_tuned | mendeley, cohort (revised, tuned) | LogReg/SW/tfidf/thr 0.50 | 0.902 | 0.821 | 0.858 | 0.976 | 0.737 | 0.615 | 1.00/0.62 | 0.96/1.00 | chosen by CV (pre-registered) |
| tokenization expC_tuned | mendeley, cohort (revised, tuned) | LogReg/SW/tfidf/thr 0.50 | 0.965 | 0.942 | 0.954 | 0.987 | 0.737 | 0.615 | 1.00/0.88 | 0.96/1.00 | post-hoc, CV rank 2 |
| tokenization expC_tuned | mendeley, cohort (revised, tuned) | LinearSVC/SW/tfidf/thr 0.00 | 0.965 | 0.942 | 0.954 | 0.989 | 0.737 | 0.615 | 1.00/0.88 | 0.96/1.00 | post-hoc, CV rank 3 |
| tokenization expC_tuned | mendeley, cohort (revised, tuned) | LinearSVC/SW/tfidf/thr 0.00 | 0.902 | 0.821 | 0.858 | 0.975 | 0.737 | 0.615 | 1.00/0.62 | 0.96/1.00 | post-hoc, CV rank 4 |
| tokenization expC_tuned | mendeley, cohort (revised, tuned) | LogReg/SW/tfidf/thr 0.50 | 0.976 | 0.961 | 0.968 | 0.987 | 0.737 | 0.615 | 1.00/0.92 | 0.96/1.00 | post-hoc, CV rank 5 |
| tokenization expB | balanced (traditional) | RF/WPC/w2v | 0.684 | 0.738 | 0.663 | 0.778 | 0.745 | 0.731 | 0.80/0.59 | 0.09/0.99 |  |
| tokenization expB | balanced (traditional) | MLP/WP/w2v | 0.704 | 0.736 | 0.676 | 0.747 | 0.745 | 0.731 | 0.83/0.43 | 0.18/1.00 |  |
| tokenization expB | balanced (traditional) | SVM-RBF/SW/w2v | 0.595 | 0.690 | 0.586 | 0.821 | 0.745 | 0.731 | 0.64/0.70 | 0.02/0.99 |  |
| tokenization expB_cohort | balanced, cohort (traditional) | RF/WPC/w2v | 0.667 | 0.724 | 0.648 | 0.792 | 0.740 | 0.759 | 0.74/0.57 | 0.06/1.00 |  |
| tokenization expB_cohort | balanced, cohort (traditional) | MLP/WP/w2v | 0.609 | 0.657 | 0.590 | 0.732 | 0.740 | 0.759 | 0.65/0.35 | 0.19/0.99 |  |
| tokenization expB_cohort | balanced, cohort (traditional) | SVM-RBF/SW/w2v | 0.542 | 0.660 | 0.540 | 0.820 | 0.740 | 0.759 | 0.52/0.74 | 0.00/1.00 |  |
| tokenization expD | balanced, cohort (revised) | RF/WPC/w2v | 0.559 | 0.664 | 0.553 | 0.875 | 0.746 | 0.765 | 0.54/0.67 | 0.07/0.99 |  |
| tokenization expD | balanced, cohort (revised) | MLP/WP/w2v | 0.621 | 0.663 | 0.597 | 0.753 | 0.746 | 0.765 | 0.72/0.38 | 0.00/0.94 |  |
| tokenization expD | balanced, cohort (revised) | SVM-RBF/SW/w2v | 0.876 | 0.861 | 0.844 | 0.904 | 0.746 | 0.765 | 0.90/0.62 | 0.86/0.94 |  |
| tokenization expD_tuned | balanced, cohort (revised, tuned) | LinearSVC/SW/tfidf/thr 0.00 | 0.711 | 0.745 | 0.683 | 0.878 | 0.746 | 0.765 | 0.80/0.45 | 0.17/1.00 | chosen by CV (pre-registered) |
| tokenization expD_tuned | balanced, cohort (revised, tuned) | SVM-RBF/SW/w2v/thr 0.00 | 0.682 | 0.734 | 0.660 | 0.837 | 0.746 | 0.765 | 0.76/0.52 | 0.11/1.00 | post-hoc, CV rank 2 |
| tokenization expD_tuned | balanced, cohort (revised, tuned) | MLP/SW/w2v/thr 0.50 | 0.676 | 0.713 | 0.650 | 0.817 | 0.746 | 0.765 | 0.79/0.45 | 0.01/0.96 | post-hoc, CV rank 3 |
| tokenization expD_tuned | balanced, cohort (revised, tuned) | LinearSVC/SW/tfidf/thr 0.00 | 0.707 | 0.756 | 0.683 | 0.896 | 0.746 | 0.765 | 0.80/0.57 | 0.07/1.00 | post-hoc, CV rank 4 |
| tokenization expD_tuned | balanced, cohort (revised, tuned) | LogReg/SW/tfidf/thr 0.50 | 0.697 | 0.749 | 0.674 | 0.884 | 0.746 | 0.765 | 0.80/0.57 | 0.00/1.00 | post-hoc, CV rank 5 |
| cnn_vit (tuned) | mendeley, cohort (unified asm) | TunableMalwareNet (CNN-ViT, tuned)/thr 0.50 mean of 5 seeds | 0.670 | 0.744 | 0.655 | 0.865 | 0.737 | 0.615 | 0.68/0.89 | 0.21/0.97 | chosen by CV (pre-registered); sd macro-F1 0.049 |
| cnn_vit (seed 1337) | mendeley, cohort (unified asm) | HierarchicalMalwareNet | 0.576 | 0.705 | 0.575 | 0.855 | 0.737 | 0.615 | 0.51/0.97 | 0.12/1.00 |  |
| cnn_vit (mean of 5 seeds) | mendeley, cohort (unified asm) | HierarchicalMalwareNet | 0.679 | 0.686 | 0.628 | 0.802 | 0.737 | 0.615 |  |  | sd macro-F1 0.018 |
| cnn_vit (tuned) | balanced, cohort (unified asm) | TunableMalwareNet (CNN-ViT, tuned)/thr 0.50 mean of 5 seeds | 0.530 | 0.620 | 0.518 | 0.665 | 0.740 | 0.759 | 0.52/0.53 | 0.07/0.97 | chosen by CV (pre-registered); sd macro-F1 0.106 |
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
| graph2vec (tuned) | balanced, cohort | LR/wl_tfidf/argmax (default 0.5) | 0.472 | 0.608 | 0.472 | 0.823 | 0.740 | 0.759 | 0.41/0.70 | 0.00/1.00 | post-hoc / control: baseline_h2_cap5000 |
| graph2vec (tuned) | balanced, cohort | LR/wl_tfidf/threshold from train out-of-fold scores/thr 0.36 | 0.581 | 0.648 | 0.568 | 0.823 | 0.740 | 0.759 | 0.63/0.41 | 0.00/1.00 | post-hoc / control: baseline_h2_cap5000 |
| graph2vec (tuned) | balanced, cohort | LR/wl_tfidf/argmax (default 0.5) | 0.569 | 0.680 | 0.565 | 0.826 | 0.740 | 0.759 | 0.54/0.78 | 0.07/0.99 | chosen by CV (pre-registered) |
| graph2vec (tuned) | balanced, cohort | LR/wl_tfidf/threshold from train out-of-fold scores/thr 0.27 | 0.587 | 0.670 | 0.577 | 0.826 | 0.740 | 0.759 | 0.60/0.61 | 0.07/0.98 | chosen by CV (pre-registered) |
| graph2vec (tuned) | balanced, cohort | MLP/wl_svd/argmax (default 0.5) | 0.601 | 0.687 | 0.592 | 0.855 | 0.740 | 0.759 | 0.52/0.63 | 0.47/1.00 | post-hoc / control: tuned_best_wl_svd |
| graph2vec (tuned) | balanced, cohort | MLP/wl_svd/threshold from train out-of-fold scores/thr 0.33 | 0.697 | 0.744 | 0.675 | 0.855 | 0.740 | 0.759 | 0.61/0.57 | 0.81/1.00 | post-hoc / control: tuned_best_wl_svd |
| graph2vec (tuned) | balanced, cohort | MLP/graph2vec/argmax (default 0.5) | 0.587 | 0.690 | 0.581 | 0.882 | 0.740 | 0.759 | 0.58/0.74 | 0.07/1.00 | post-hoc / control: tuned_best_graph2vec |
| graph2vec (tuned) | balanced, cohort | MLP/graph2vec/threshold from train out-of-fold scores/thr 0.31 | 0.603 | 0.691 | 0.594 | 0.882 | 0.740 | 0.759 | 0.62/0.67 | 0.07/0.99 | post-hoc / control: tuned_best_graph2vec |
| graph2vec (tuned) | balanced, cohort | LR/size_only/argmax (default 0.5) | 0.605 | 0.672 | 0.591 | 0.675 | 0.740 | 0.759 | 0.67/0.80 | 0.00/0.81 | post-hoc / control: size_only_control |
| graph2vec (tuned) | balanced, cohort | LR/size_only/threshold from train out-of-fold scores/thr 0.48 | 0.611 | 0.674 | 0.596 | 0.675 | 0.740 | 0.759 | 0.68/0.78 | 0.01/0.81 | post-hoc / control: size_only_control |
| graph2vec (tuned) | mendeley, cohort | LR/wl_tfidf/argmax (default 0.5) | 0.739 | 0.811 | 0.724 | 0.966 | 0.737 | 0.615 | 0.82/0.96 | 0.01/1.00 | post-hoc / control: baseline_h2_cap5000 |
| graph2vec (tuned) | mendeley, cohort | LR/wl_tfidf/threshold from train out-of-fold scores/thr 0.38 | 0.914 | 0.872 | 0.885 | 0.966 | 0.737 | 0.615 | 1.00/0.76 | 0.81/1.00 | post-hoc / control: baseline_h2_cap5000 |
| graph2vec (tuned) | mendeley, cohort | RF/wl_svd/argmax (default 0.5) | 0.723 | 0.772 | 0.702 | 0.939 | 0.737 | 0.615 | 0.64/0.86 | 0.76/1.00 | chosen by CV (pre-registered) |
| graph2vec (tuned) | mendeley, cohort | RF/wl_svd/threshold from train out-of-fold scores/thr 0.35 | 0.912 | 0.886 | 0.887 | 0.939 | 0.737 | 0.615 | 0.94/0.82 | 0.93/0.92 | chosen by CV (pre-registered) |
| graph2vec (tuned) | mendeley, cohort | LR/wl_tfidf/argmax (default 0.5) | 0.790 | 0.848 | 0.772 | 0.982 | 0.737 | 0.615 | 0.74/0.97 | 0.65/1.00 | post-hoc / control: tuned_best_sparse_histogram |
| graph2vec (tuned) | mendeley, cohort | LR/wl_tfidf/threshold from train out-of-fold scores/thr 0.31 | 0.959 | 0.947 | 0.947 | 0.982 | 0.737 | 0.615 | 0.98/0.92 | 0.93/0.92 | post-hoc / control: tuned_best_sparse_histogram |
| graph2vec (tuned) | mendeley, cohort | RF/graph2vec/argmax (default 0.5) | 0.752 | 0.530 | 0.486 | 0.845 | 0.737 | 0.615 | 1.00/0.05 | 0.99/0.17 | post-hoc / control: tuned_best_graph2vec |
| graph2vec (tuned) | mendeley, cohort | RF/graph2vec/threshold from train out-of-fold scores/thr 0.39 | 0.737 | 0.500 | 0.424 | 0.845 | 0.737 | 0.615 | 1.00/0.00 | 1.00/0.00 | post-hoc / control: tuned_best_graph2vec |
| graph2vec (tuned) | mendeley, cohort | LR/size_only/argmax (default 0.5) | 0.825 | 0.846 | 0.799 | 0.939 | 0.737 | 0.615 | 0.92/0.89 | 0.31/0.92 | post-hoc / control: size_only_control |
| graph2vec (tuned) | mendeley, cohort | LR/size_only/threshold from train out-of-fold scores/thr 0.53 | 0.835 | 0.868 | 0.813 | 0.939 | 0.737 | 0.615 | 0.92/0.94 | 0.31/0.92 | post-hoc / control: size_only_control |
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

## Best macro-F1 per pipeline and dataset (tuned pipelines: pre-registered choice only)

| pipeline | mendeley-type dataset | balanced-type dataset |
|---|---|---|
| tokenization | 0.926 (tokenization expC · MLP/WP/w2v) | 0.844 (tokenization expD · SVM-RBF/SW/w2v) |
| tokenization (tuned) | 0.858 (tokenization expC_tuned · LogReg/SW/tfidf/thr 0.50) | 0.683 (tokenization expD_tuned · LinearSVC/SW/tfidf/thr 0.00) |
| cnn_vit (tuned) | 0.655 (cnn_vit (tuned) · TunableMalwareNet (CNN-ViT, tuned)/thr 0.50 mean of 5 seeds) | 0.518 (cnn_vit (tuned) · TunableMalwareNet (CNN-ViT, tuned)/thr 0.50 mean of 5 seeds) |
| cnn_vit | 0.628 (cnn_vit (mean of 5 seeds) · HierarchicalMalwareNet) | 0.542 (cnn_vit (seed 1337) · HierarchicalMalwareNet) |
| graph2vec | 0.895 (graph2vec · SVM-RBF/graph2vec/threshold from train out-of-fold scores/thr -0.25) | 0.602 (graph2vec · LR/size_only/threshold from train out-of-fold scores/thr 0.42) |
| graph2vec (tuned) | 0.887 (graph2vec (tuned) · RF/wl_svd/threshold from train out-of-fold scores/thr 0.35) | 0.577 (graph2vec (tuned) · LR/wl_tfidf/threshold from train out-of-fold scores/thr 0.27) |
| rules | 0.968 (rules · mnemonic_tfidf_1_3+LogReg/calibration_baseline) | 0.802 (rules · mnemonic_tfidf_1_3+LinearSVC/calibration_baseline) |

## Post-hoc maxima among tuned rows (NOT results: selected after seeing the test set)

| pipeline | mendeley-type dataset | balanced-type dataset |
|---|---|---|
| tokenization (tuned) | 0.968 (tokenization expC_tuned · LogReg/SW/tfidf/thr 0.50) | 0.683 (tokenization expD_tuned · LinearSVC/SW/tfidf/thr 0.00) |
| graph2vec (tuned) | 0.947 (graph2vec (tuned) · LR/wl_tfidf/threshold from train out-of-fold scores/thr 0.31) | 0.675 (graph2vec (tuned) · MLP/wl_svd/threshold from train out-of-fold scores/thr 0.33) |

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

EMBER is not in this table; that rerun is being done by a teammate (`ember_pipeline/`, `vm_package/README.md`).
