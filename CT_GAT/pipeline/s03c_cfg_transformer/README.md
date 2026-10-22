# Stage 3c: Model 3, the end-to-end block transformer and GAT

Model 3 trains a transformer over the tokens of each block together with the GAT over the block graph, end to end. It arrives in
this folder with the team-pipeline step (`block_transformer.py`, trained by `experiments/team_pipelines/train_b.py` with an `-e2e`
arm); until then the folder holds this README only.

The frozen-encoder path that stood here (block export, the encoder contract, the CodeBERT and reference encoders, the precomputed
block vectors the GNN reads with `--block-input vec`) is a reference model: its code is in
[`../reference/cfg_transformer/`](../reference/cfg_transformer/README.md) and the commands to run it are in
[`../reference/README.md`](../reference/README.md). Every command of this stage will run from `CT_GAT/pipeline`, as
`python -m experiments.team_pipelines.train_b ...` does today for the pipeline-B arms
([`../experiments/team_pipelines/README.md`](../experiments/team_pipelines/README.md)).
