# GIN ransomware detection - results

Generated 2026-09-16 07:27:07 from `data/graphs/graphs.pt`.

## Test set

Best of 5 seed(s), evaluated once on 294 held-out samples.

| Metric | Best run | Mean | Std dev |
| --- | --- | --- | --- |
| Accuracy | 96.94% | 96.19% | 0.82% |
| Precision | 97.10% | 96.51% | 0.51% |
| Recall | 96.40% | 95.40% | 1.74% |
| F1 | 96.75% | 95.94% | 0.90% |
| ROC-AUC | 99.58% | 99.31% | 0.23% |

Decision threshold 0.0988, chosen on the validation split (best epoch 33).

## Confusion matrix

| | Called goodware | Called ransomware |
| --- | --- | --- |
| **Really goodware** | 151 | 4 (false alarms) |
| **Really ransomware** | 5 (missed) | 134 |

## Per-family recall (worst first)

| Family | Detected | Recall |
| --- | --- | --- |
| makop | 4/5 | 80.00% |
| nefilim | 4/5 | 80.00% |
| babuk | 5/6 | 83.33% |
| conti | 6/7 | 85.71% |
| zeppelin | 6/7 | 85.71% |
| avaddon | 7/7 | 100.00% |
| blackmatter | 6/6 | 100.00% |
| darkside | 7/7 | 100.00% |
| dharma | 7/7 | 100.00% |
| doppelpaymer | 3/3 | 100.00% |
| exorcist | 3/3 | 100.00% |
| gandcrab | 7/7 | 100.00% |
| lockbit | 7/7 | 100.00% |
| maze | 7/7 | 100.00% |
| mountlocker | 2/2 | 100.00% |
| netwalker | 7/7 | 100.00% |
| phobos | 7/7 | 100.00% |
| pysa | 6/6 | 100.00% |
| ragnarok | 5/5 | 100.00% |
| ransomexx | 2/2 | 100.00% |
| revil | 7/7 | 100.00% |
| ryuk | 7/7 | 100.00% |
| stop | 7/7 | 100.00% |
| wastedlocker | 5/5 | 100.00% |

## Configuration

3 GIN layers, hidden dim 64, embedding dim 64, dropout 0.5, sum pooling, row adjacency normalisation, both edge direction.

Trained with AdamW at lr 0.001, batch size 32, up to 60 epochs, early stopping after 15 epochs without a validation gain.
