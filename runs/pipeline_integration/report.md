# Pipeline integration test

Run on 2026-10-05 against `data/opcodes_dataset/index.csv`.

This checks the three scripts added on top of the opcode index: tokenization, CNN-ViT image rendering, and test-split obfuscation, plus rebuilding GIN graphs from the obfuscated index. It is not a full-corpus training run.

## Sample

48 opcode files, 8 goodware and 8 ransomware from each of train, val, and test. Each sequence was truncated to 1,000 instructions. The temporary outputs were removed after the checks.

## Tokenization

Every method encoded all 48 samples. `sw` and `wp` vocabularies were fit on the train split only. No test-only token appeared in either vocabulary.

| Method | Vocabulary | Tokens, avg / max | Train | Val | Test |
| --- | ---: | --- | ---: | ---: | ---: |
| sw | 31 | 891 / 1000 | 16 | 16 | 16 |
| wp | 30 | 712 / 999 | 16 | 16 | 16 |
| bpe | 139 | 936 / 1461 | 16 | 16 | 16 |
| wpc | 171 | 942 / 1612 | 16 | 16 | 16 |
| uni | 59 | 1898 / 2420 | 16 | 16 | 16 |
| spc | 146 | 950 / 1697 | 16 | 16 | 16 |

The subword vocabularies are smaller than the requested 500 or 1,000 because 16 training files do not contain that many distinct pieces. On the full train split those vocabularies will be larger.

Retraining `sw` on the obfuscated index reproduced the clean train tokens exactly. All 16 obfuscated test sequences changed. That run reported vocab 31 and tokens avg/max 890/1000.

## CNN-ViT images

`sw` and `bpe` each wrote 48 images.

| Check | Result |
| --- | --- |
| PNG size | 256×256 grayscale |
| ViT mask | 16×16, on only where a real token falls |
| Padding | pixel value 0 after the real token stream |
| Classes | `goodware` and `ransomware` under train, val, and test |
| `sw` image ids | 31 train tokens mapped to ids 2–255; 0 rarer tokens collapsed |
| `bpe` image ids | 68 train tokens mapped to ids 2–255; 0 rarer tokens collapsed |

Id 0 is padding and id 1 is unknown.

## Obfuscation

Only the test split was rewritten (16 files). Train and val still point at the original opcode files.

| Edit | Count |
| --- | ---: |
| NOPs inserted | 1642 |
| Synonym swaps | 183 |
| Operand substitutions | 0 |
| Dead-code lines | 0 |

Edited sequences stayed inside the 15% insertion budget. Every original mnemonic was still present, aside from assembler synonym swaps such as `je`/`jz`. Operand substitutions did not run, because these files are bare mnemonics.

A separate check with operand text confirmed `xor reg, reg`, `inc`, `mov`, and `push` still rewrite, and that `lock xadd` is kept as a mnemonic rather than treated as operands.

## Graphs from the obfuscated index

`build_graphs.py` kept 40 opcodes (including `<unk>`) from 68 seen in train. 29 were dropped as rare or over the cap. No sample was skipped.

| Split | Graphs | Goodware | Ransomware |
| --- | ---: | ---: | ---: |
| train | 16 | 8 | 8 |
| val | 16 | 8 | 8 |
| test | 16 | 8 | 8 |

Nodes averaged 16.8 (max 37). Edges averaged 77.8 (max 225).

## Error exits

A missing index, an unknown tokenization method, and a missing token file each exited non-zero.

## Result

0 failures.
