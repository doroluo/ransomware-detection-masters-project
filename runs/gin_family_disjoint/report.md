# GIN ransomware detection - results

Generated 2026-09-16 07:29:39 from `data/graphs_fd/graphs.pt`.

## Test set

Best of 5 seed(s), evaluated once on 302 held-out samples.

| Metric | Best run | Mean | Std dev |
| --- | --- | --- | --- |
| Accuracy | 88.74% | 86.62% | 1.38% |
| Precision | 85.53% | 83.05% | 1.64% |
| Recall | 92.52% | 91.16% | 2.06% |
| F1 | 88.89% | 86.90% | 1.37% |
| ROC-AUC | 95.42% | 91.58% | 3.34% |

Decision threshold 0.0000, chosen on the validation split (best epoch 17).

## Confusion matrix

| | Called goodware | Called ransomware |
| --- | --- | --- |
| **Really goodware** | 132 | 23 (false alarms) |
| **Really ransomware** | 11 (missed) | 136 |

## Per-family recall (worst first)

| Family | Detected | Recall |
| --- | --- | --- |
| ryuk | 37/48 | 77.08% |
| nefilim | 37/37 | 100.00% |
| phobos | 49/49 | 100.00% |
| ransomexx | 13/13 | 100.00% |

## Configuration

3 GIN layers, hidden dim 64, embedding dim 64, dropout 0.5, sum pooling, row adjacency normalisation, both edge direction.

Trained with AdamW at lr 0.001, batch size 32, up to 60 epochs, early stopping after 15 epochs without a validation gain.
