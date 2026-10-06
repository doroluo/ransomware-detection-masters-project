# GIN accuracy after opcode obfuscation

Generated 2026-10-05. Same training setup as `runs/gin`: 5 seeds (42–46), 60 epochs, early stop on validation ROC-AUC, threshold chosen on the validation split. Each trained model was scored twice: once on the original test graphs, once on test graphs rebuilt after NOP insertion and assembler synonym swaps (`je`/`jz` and the other equivalent pairs). Train and validation graphs were not edited. 287 of 294 test graphs changed.

Tokenization and the CNN-ViT images do not enter the GIN. Retraining on the original graphs reproduced `runs/gin/metrics.json` on every seed, including the same best epochs and the same decision thresholds.

## Test set, mean of 5 seeds

294 held-out samples.

| Metric | Clean test | Obfuscated test |
| --- | --- | --- |
| Accuracy | 96.19% ± 0.82 | 96.39% ± 0.90 |
| Precision | 96.51% ± 0.51 | 96.81% ± 0.90 |
| Recall | 95.40% ± 1.74 | 95.54% ± 2.10 |
| F1 | 95.94% ± 0.90 | 96.15% ± 1.01 |
| ROC-AUC | 99.31% ± 0.23 | 99.41% ± 0.11 |

## Best run (seed 46, epoch 33, threshold 0.0988)

Chosen by validation F1, the same rule as `train_gin.py`.

| Metric | Clean test | Obfuscated test |
| --- | --- | --- |
| Accuracy | 96.94% | 97.28% |
| Precision | 97.10% | 97.12% |
| Recall | 96.40% | 97.12% |
| F1 | 96.75% | 97.12% |
| ROC-AUC | 99.58% | 99.60% |

Clean confusion matrix: 151 true goodware, 4 false alarms, 5 missed ransomware, 134 detected.

Obfuscated confusion matrix: 151 true goodware, 4 false alarms, 4 missed ransomware, 135 detected.

## Per seed

| Seed | Clean accuracy | Obfuscated accuracy | Clean F1 | Obfuscated F1 |
| --- | --- | --- | --- | --- |
| 42 | 94.90% | 94.90% | 94.51% | 94.46% |
| 43 | 96.60% | 97.28% | 96.38% | 97.10% |
| 44 | 95.58% | 95.92% | 95.27% | 95.62% |
| 45 | 96.94% | 96.60% | 96.80% | 96.45% |
| 46 | 96.94% | 97.28% | 96.75% | 97.12% |

On the best run, per-family recall on the obfuscated test matches the clean test except Zeppelin, which went from 6/7 to 7/7. Makop and Nefilim stay at 4/5, Babuk at 5/6, and Conti at 6/7.

The obfuscation counts on the full test split were 911,565 inserted NOPs and 156,523 synonym swaps. Operand substitutions were not applied; these opcode files are bare mnemonics.
