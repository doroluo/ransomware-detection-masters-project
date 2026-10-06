# masters-project

## Datasets

The dataset archives are not stored in this repo — they exceed GitHub's 100 MB
per-file limit. Download them and place them in `data/` (which is gitignored):

| File | Size |
| --- | --- |
| `data/Ransomware PE Header Feature Dataset.zip` | 877 MB |
| `data/goodware.7z` | 573 MB |
| `data/Features_Extraction.7z` | 75 MB |

<!-- TODO: add the download link for each dataset above. -->

## Opcode representations

`extract_opcodes.py` writes `index.csv`. The GIN, the tokenizers, and the
CNN-ViT images all read that index, so they share one train / val / test split.

```bash
python tokenize_opcodes.py --index data/opcodes_dataset/index.csv --out data/tokens
python render_vit_images.py --tokens data/tokens/sw/records.pkl --out data/vit_images/sw
python obfuscate_opcodes.py --index data/opcodes_dataset/index.csv --out data/opcodes_obfuscated
```

`tokenize_opcodes.py` fits each vocabulary on the train split only, then
encodes train, val, and test. Methods are `sw` (top opcodes), `wp` (adjacent
pairs), and the subword tokenizers `bpe`, `wpc`, `uni`, and `spc`. The subword
methods need the `tokenizers` package from `requirements.txt`.

`render_vit_images.py` writes `256×256` grayscale PNGs plus a `16×16` ViT
mask, laid out as `<out>/<split>/<goodware|ransomware>/<sha256>.png`. That is
the directory layout the CNN-ViT loader expects. Ids `2..255` are the most
common training tokens; rarer tokens become unknown (`1`), and padding is `0`.

`obfuscate_opcodes.py` writes a new index and edits only the test split: NOP
insertion and assembler synonym swaps (`je`/`jz`, and the other equivalent
pairs). Train and val keep the original opcode files. Point `build_graphs.py`
and `tokenize_opcodes.py` at `data/opcodes_obfuscated/index.csv` to rebuild
both representations from the same attack. Operand substitutions (`xor reg, reg`
→ `sub reg, reg`, and the rest) run only for files extracted with
`--include-operands`.
