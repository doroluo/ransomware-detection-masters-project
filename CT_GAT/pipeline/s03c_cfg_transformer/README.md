# Stage 3c: Model 3, block embeddings + CFG

A per-block transformer encoder turns every basic block into a vector; the vectors replace the token bag of the GNN
(`s03b_cfg_feature`), which then runs over the block CFG and the call graph as before. The encoder is frozen: it is trained (or,
for CodeBERT, not trained at all) outside the GNN, once per fold, and its vectors are computed once per distinct block.

```
CFG cache --block_export--> block export --(encoder training, yours)--> <runs>/encoders/<name>/fold{k}/best.pt
                                                                                  |
          <cache>/side/blockvec/<export>_<name>/fold{k}/  <--precompute_blockvec--+
                                                                |
           python -m experiments.train --block-input vec --block-vec <export>_<name>   (GNN on the vectors)
```

Every command runs from `CT_GAT/pipeline` (`common/paths.py` resolves the cache, `GT_RUNS` and `GT_EXPORT`).

## Steps

1. **Export** the blocks of a cache, with the folds of the GNN experiments and a vocabulary per fold:
   `python -m s03c_cfg_transformer.block_export --protocol kfold --arch x86 --out RUNS/export/ours_cap0`
   (`RUNS` is the run directory). Layout: `EXPORT_SCHEMA.md`. It is read back through `common/reader.py` (`BlockStore`,
   `FoldView`), which the token-sequence model of `s03a_transformer` reads as well.
2. **Encoder checkpoints**, one per fold, in the format of `encoder_contract.py` (`TRANSFORMER_CONTRACT.md` has the rules and
   `python -m s03c_cfg_transformer.encoder_contract --ckpt ... --export ... --fold k` checks one):
   * `encoders/codebert.py`: the frozen `microsoft/codebert-base` on the text of each block, built from the signature and API
     strings of the vocabulary and never from operands. `HF_HOME` (an environment variable, never a path in the code) must
     point at the Hugging Face home that holds the weights; they are loaded with `local_files_only`, and the checkpoint records
     the sha256 of `model.safetensors`.
     `python -m s03c_cfg_transformer.encoders.codebert --write-checkpoints --export ours_cap0 --name codebert`
   * `encoders/reference.py`: a tiny seeded encoder for tests and smoke runs (`--name ref`).
   * Checkpoints written before this layout record the module `transformer_paths.codebert_encoder` or
     `transformer_paths.reference_encoder`; `encoder_contract.LEGACY_MODULES` maps both to `encoders/`, so they load unchanged.
3. **Precompute** the vectors, one per distinct block and fold:
   `python -m s03c_cfg_transformer.precompute_blockvec --encoder-name codebert --export ours_cap0 --folds 0 1 2 3 4`
   They go to `<cache>/side/blockvec/ours_cap0_codebert/fold{k}/`.
4. **Train the GNN on them**, protocol `kfold` only:
   `python -m experiments.train --protocol kfold --fold 0 --block-input vec --block-vec ours_cap0_codebert --mask-blocks 0.15`
   is path 2 (typed edges); add `--control no_edges` for path 1 (no edges).

## Adding your own encoder

Write a `torch.nn.Module` with `d_out`, `max_len` and `forward(tokens [B, L] long, mask [B, L] bool) -> [B, d_out]` that
depends only on the block it is given (`encoder_contract.BlockEncoder`), in a module that can be imported by name (a class
defined in a script run as `__main__` is refused). Train it on `FoldView(store, k, 'train')`, select on `'val'`, never read
`'test'`, and save it with `encoder_contract.save_checkpoint` to `<runs>/encoders/<name>/fold{k}/best.pt`.
`encoders/reference.py` is a working example of the whole chain, `TRANSFORMER_CONTRACT.md` the full rules. Steps 3 and 4 then
need no change.

## Results

CodeBERT, path 2 (typed edges), kfold x86, seed 0, five folds pooled: AUC 0.961, against 0.986 for the same GNN on the token bag
(R7) and 0.982 for TF-IDF; the paired difference to R7 is -0.025 [-0.059, -0.005]. The frozen general-purpose encoder does not
beat the counted tokens. The tables are in the project history (`docs/history/`, which arrives with a later pull request).

## Tests

`python -m pytest tests/test_transformer_paths.py tests/test_codebert_encoder.py tests/test_s03_package.py`. The export and
contract tests run on the synthetic cache of `tests/tiny_cache.py`; the tests that need the CodeBERT weights are marked
`real_data` and skip without `HF_HOME`.
