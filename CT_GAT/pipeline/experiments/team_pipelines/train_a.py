"""Pipeline A trainer: the teammates' BlockGAT (train_cfg) on our fold manifests.

python -m experiments.team_pipelines.train_a --fold 0 --seed 0 --arm gnn500            (RUNS/team/a_gnn500/<id>.pt)
       [--smoke N] [--device cuda|cpu]                               (run from CT_GAT/pipeline)

gnn500  the graphs of build_a.py (their gnn_graph_from_export, 500 blocks, 57 opcode / API-group / structure features).
BlockGAT(hidden 128, heads 4, dropout 0.2), AdamW lr 1e-3 / weight decay 1e-4, class-weighted cross-entropy, batch 8,
20 epochs, patience 5 on the validation ransomware F1 at 0.5 (see fit.py), our `fold_manifest(fold, seed)` for train,
val and test. Output RUNS/kfold{f}_x86_gnn_s{seed}_team-<arm>/ (common.write_run, history.csv, best.pt, best_auc.pt).
--smoke N trains on N available training ids for 2 epochs into a temp directory and writes nothing under RUNS.
"""
import argparse
import sys
import time

import torch

from common.config import RUNS
from common.manifests import run_command
from experiments.team_pipelines.common import fold_manifest, sample_rows
from experiments.team_pipelines.fit import run_arm, smoke_ids

ARMS = ('gnn500',)
BUDGET = 500
GRAPH_DIR = RUNS / 'team' / 'a_gnn500'
HP = {'lr': 1e-3, 'weight_decay': 1e-4, 'batch_size': 8, 'epochs': 20, 'patience': 5}


def load_gnn_graphs(ids, graph_dir=GRAPH_DIR):
    """({id: Data}, {id: (kept functions, kept blocks, total blocks)}) from the build_a .pt files."""
    graphs, sizes = {}, {}
    for i in ids:
        g = torch.load(graph_dir / f'{i}.pt', weights_only=False)
        if g.sample_id != i or int(g.y.item()) != sample_rows()[i]['label']:
            raise ValueError(f'{graph_dir / (i + ".pt")} holds sample {g.sample_id} / label {int(g.y.item())}')
        graphs[i] = g
        sizes[i] = (int(g.func_batch.max()) + 1, int(g.x.size(0)), int(g.num_blocks_total))
    return graphs, sizes


def run(arm, fold, seed, smoke=None, device=None, graph_dir=GRAPH_DIR, log=print):
    if arm not in ARMS:
        raise ValueError(f'arm {arm}: one of {ARMS}')
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    from s03b_cfg_feature.gat import train_cfg as tc      # imported here: this module imports without torch_geometric
    manifest = fold_manifest(fold, seed)
    want = [r['sample_id'] for r in manifest.train + manifest.val + manifest.test]
    extra = {'budget': BUDGET}
    t0 = time.time()
    split = None
    if smoke:  # N available training ids and the validation ids are all a smoke run reads
        available = {i for i in want if (graph_dir / f'{i}.pt').is_file()}
        split = smoke_ids(manifest, available, smoke)
        want = sorted(set(split[0]) | set(split[1]))
    graphs, sizes = load_gnn_graphs(want, graph_dir)
    in_dim = int(next(iter(graphs.values())).x.size(1))
    extra['in_dim'] = in_dim
    log(f'{arm}: {len(graphs)} graphs loaded in {time.time() - t0:.0f} s, in_dim {in_dim}')

    def build_model():
        return tc.BlockGAT(in_dim=in_dim, hidden=128, dropout=0.2)

    return run_arm(arm=arm, fold=fold, seed=seed, graphs=graphs, sizes=sizes, build_model=build_model, hp=HP,
                   device=device, smoke=smoke, smoke_split=split, extra=extra, log=log)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--arm', choices=ARMS, required=True)
    ap.add_argument('--smoke', type=int, help='train on N available training ids for 2 epochs into a temp directory')
    ap.add_argument('--device', choices=['cuda', 'cpu'])
    args = ap.parse_args(argv)
    out = run(args.arm, args.fold, args.seed, args.smoke, args.device)
    if args.smoke:
        print(f"smoke: val AUC {out['val_auc']:.4f}, peak GPU {out['peak_gpu_mb']} MB, loss "
              f"{out['history'][0]['loss']} -> {out['history'][-1]['loss']}, temp dir {out['dir']} (not under {RUNS})")
    return out


if __name__ == '__main__':
    run_command(main)
    sys.exit(0)
