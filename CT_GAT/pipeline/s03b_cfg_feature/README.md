# Stage 3b: Model 2, the team GAT on the 77 block features

Model 2 is Dorothy's GAT over the graph of a binary with the 77 features of each block, trained on our folds. It arrives in this
folder with the team-pipeline step (`gat/`, trained by `experiments/team_pipelines/train_b.py` with an `-feat` arm); until then
the folder holds this README only.

The hierarchical edge-typed GNN that stood here, with its R-ladder of controls, is a reference model: its code is in
[`../reference/gnn/`](../reference/gnn/README.md) and the commands to run it are in [`../reference/README.md`](../reference/README.md).
Every command of this stage will run from `CT_GAT/pipeline`, as `python -m experiments.team_pipelines.train_b ...` does today for
the pipeline-B arms ([`../experiments/team_pipelines/README.md`](../experiments/team_pipelines/README.md)).
