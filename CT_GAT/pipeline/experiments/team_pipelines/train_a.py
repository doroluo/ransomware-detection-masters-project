"""Pipeline A trainer: the teammates' BlockGAT (train_cfg) on our fold manifests.

python -m experiments.team_pipelines.train_a --fold 0 --seed 0 --arm gnn500            (RUNS/team/a_gnn500/<id>.pt)
python -m experiments.team_pipelines.train_a --fold 0 --seed 0 --arm cb500 [--block-vec ours_cap0_codebert]
       [--smoke N] [--device cuda|cpu]                               (run from CT_GAT/pipeline)

gnn500  the graphs of build_a.py (their gnn_graph_from_export, 500 blocks, 57 opcode / API-group / structure features).
cb500   graphs from OUR cache: the blocks of `select_subgraph(..., 'cap', 500, rng=None)` (what the matched
        `--subgraph cap --budget 500` arm evaluates), the cache's directed intra-function CFG edges among them, and the
        block vectors of reference.cfg_transformer.precompute_blockvec (`BlockVecSource`, keyed by the export's block_key). No raw
        export is parsed. The graphs are built in memory per fold; x is held as float16 and cast per batch.
Both: BlockGAT(hidden 128, heads 4, dropout 0.2), AdamW lr 1e-3 / weight decay 1e-4, class-weighted cross-entropy, batch 8,
20 epochs, patience 5 on the validation ransomware F1 at 0.5 (see fit.py), our `fold_manifest(fold, seed)` for train,
val and test. Output RUNS/kfold{f}_x86_gnn_s{seed}_team-<arm>/ (common.write_run, history.csv, best.pt, best_auc.pt).
--smoke N trains on N available training ids for 2 epochs into a temp directory and writes nothing under RUNS.
"""
import argparse
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from common.config import RUNS, RunConfig
from common.manifests import load_sample, run_command
from experiments.team_pipelines.common import fold_manifest, import_team, sample_rows
from experiments.team_pipelines.fit import run_arm, smoke_ids
from s02_preprocess.subgraph import select_subgraph
from reference.gnn.spec import open_block_vecs

ARMS = ('gnn500', 'cb500')
BUDGET = 500
GRAPH_DIR = RUNS / 'team' / 'a_gnn500'
DEFAULT_BLOCK_VEC = 'ours_cap0_codebert'
HP = {'lr': 1e-3, 'weight_decay': 1e-4, 'batch_size': 8, 'epochs': 20, 'patience': 5}


class FloatInput(nn.Module):
    """BlockGAT over graphs whose x is stored as float16: x is cast to float32 on the way in."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, data):
        data.x = data.x.float()
        return self.net(data)


def cache_graph(sample_id, vecs, budget=BUDGET):
    """The cb500 graph of one binary and its (kept functions, kept blocks, total blocks): x = block vectors of the blocks
    `select_subgraph(z, 'cap', budget, None)` keeps, edge_index = the cache's CFG edges with both ends kept and in the same
    function (directed, local ids)."""
    from torch_geometric.data import Data              # imported here: this module imports without torch_geometric
    z = load_sample(sample_id)
    fn, blk = select_subgraph(z, 'cap', budget, None, sample_id)
    fn_ptr = np.asarray(z['fn_blk_ptr'], dtype=np.int64)
    local = np.full(len(z['blk_feat']), -1, dtype=np.int64)
    local[blk] = np.arange(len(blk))
    e = np.asarray(z['cfg_edges'], dtype=np.int64).reshape(-1, 2)
    src, dst = local[e[:, 0]], local[e[:, 1]]
    same_fn = np.searchsorted(fn_ptr, e[:, 0], 'right') == np.searchsorted(fn_ptr, e[:, 1], 'right')
    keep = (src >= 0) & (dst >= 0) & same_fn
    graph = Data(x=torch.from_numpy(vecs.block_vectors(sample_id, blk)),
                 edge_index=torch.from_numpy(np.stack([src[keep], dst[keep]])).long(),
                 y=torch.tensor([sample_rows()[sample_id]['label']], dtype=torch.long))
    graph.sample_id = sample_id
    return graph, (len(fn), len(blk), int(np.diff(fn_ptr).sum()))


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


def run(arm, fold, seed, smoke=None, block_vec=DEFAULT_BLOCK_VEC, device=None, graph_dir=GRAPH_DIR, log=print):
    if arm not in ARMS:
        raise ValueError(f'arm {arm}: one of {ARMS}')
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    tc = import_team('train_cfg')
    manifest = fold_manifest(fold, seed)
    want = [r['sample_id'] for r in manifest.train + manifest.val + manifest.test]
    extra = {'budget': BUDGET}
    t0 = time.time()
    split = None
    if smoke:  # N available training ids and the validation ids are all a smoke run reads
        available = {i for i in want if arm == 'cb500' or (graph_dir / f'{i}.pt').is_file()}
        split = smoke_ids(manifest, available, smoke)
        want = sorted(set(split[0]) | set(split[1]))
    if arm == 'gnn500':
        graphs, sizes = load_gnn_graphs(want, graph_dir)
    else:
        # this config only opens the vectors and checks their export; the 500-block selection is cache_graph's, so its budget stays 0
        cfg = RunConfig(protocol='kfold', fold=fold, arch='x86', seed=seed, model='gnn', block_input='vec',
                        block_vec=block_vec)
        vecs = open_block_vecs(cfg)
        graphs, sizes = {}, {}
        for i in want:
            graphs[i], sizes[i] = cache_graph(i, vecs)
        extra['block_vec'] = block_vec
        extra['block_vec_d'] = vecs.d
    in_dim = int(next(iter(graphs.values())).x.size(1))
    extra['in_dim'] = in_dim
    log(f'{arm}: {len(graphs)} graphs loaded in {time.time() - t0:.0f} s, in_dim {in_dim}')

    def build_model():
        net = tc.BlockGAT(in_dim=in_dim, hidden=128, dropout=0.2)
        return FloatInput(net) if arm == 'cb500' else net

    return run_arm(arm=arm, fold=fold, seed=seed, graphs=graphs, sizes=sizes, build_model=build_model, hp=HP,
                   device=device, smoke=smoke, smoke_split=split, extra=extra, log=log)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--arm', choices=ARMS, required=True)
    ap.add_argument('--smoke', type=int, help='train on N available training ids for 2 epochs into a temp directory')
    ap.add_argument('--block-vec', default=DEFAULT_BLOCK_VEC, help='cb500: <export>_<name> under <cache>/side/blockvec')
    ap.add_argument('--device', choices=['cuda', 'cpu'])
    args = ap.parse_args(argv)
    out = run(args.arm, args.fold, args.seed, args.smoke, args.block_vec, args.device)
    if args.smoke:
        print(f"smoke: val AUC {out['val_auc']:.4f}, peak GPU {out['peak_gpu_mb']} MB, loss "
              f"{out['history'][0]['loss']} -> {out['history'][-1]['loss']}, temp dir {out['dir']} (not under {RUNS})")
    return out


if __name__ == '__main__':
    run_command(main)
    sys.exit(0)
