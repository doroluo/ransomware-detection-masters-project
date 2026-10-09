"""python test_structure.py: shortest-path buckets against BFS, and the ablation's blindness to graph structure."""
from collections import deque

import numpy as np
import torch

from config import RunConfig
from data import TokenVocab, load_index, loader, make_manifest
from model import GraphTransformer, shortest_path_buckets
from train import to_device


def bfs_buckets(adj):
    n = len(adj)
    out = np.full((n, n), 5)
    for s in range(n):
        dist = {s: 0}
        queue = deque([s])
        while queue:
            u = queue.popleft()
            for v in np.nonzero(adj[u])[0]:
                if v not in dist:
                    dist[int(v)] = dist[u] + 1
                    queue.append(int(v))
        for v, d in dist.items():
            out[s, v] = min(d, 4)
    return out


def test_shortest_paths():
    rng = np.random.default_rng(0)
    for length, p in ((8, 0.2), (32, 0.05), (128, 0.012), (128, 0.0)):
        adj = rng.random((6, length, length)) < p
        chain = np.zeros((length, length), dtype=bool)
        chain[np.arange(length - 1), np.arange(1, length)] = True
        adj[0] = chain
        got = shortest_path_buckets(torch.from_numpy(adj).cuda()).cpu().numpy()
        for i in range(len(adj)):
            assert np.array_equal(got[i], bfs_buckets(adj[i])), f'buckets differ (L={length}, graph {i})'
    print('shortest-path buckets match BFS')


def test_no_graph_ignores_structure():
    base = RunConfig(num_workers=0, dropout=0.0)
    manifest = make_manifest(base, load_index())
    vocab = TokenVocab(manifest.train[:100], 2)
    rows = manifest.train[:4]
    for no_graph in (True, False):
        cfg = RunConfig(num_workers=0, dropout=0.0, no_graph=no_graph)
        torch.manual_seed(0)
        model = GraphTransformer(cfg, vocab.total).cuda().eval()
        batch = to_device(next(iter(loader(rows, vocab, cfg, train=False))), 'cuda')
        scrambled = dict(batch)
        scrambled['edge_src'], scrambled['edge_dst'] = batch['edge_dst'], batch['edge_src']
        scrambled['deg_in'], scrambled['deg_out'] = batch['deg_out'], batch['deg_in']
        scrambled['blk_feat'] = batch['blk_feat'][:, [0, 1, 2, 3, 5, 4]]
        scrambled['call_src'], scrambled['call_dst'] = batch['call_dst'], batch['call_src']
        with torch.no_grad():
            same = torch.allclose(model(batch), model(scrambled), atol=1e-5)
        assert same == no_graph, f'no_graph={no_graph}: output {"ignores" if same else "depends on"} structure'
    print('no-graph output is invariant to reversing every edge, degree and call; graph output is not')


if __name__ == '__main__':
    test_shortest_paths()
    test_no_graph_ignores_structure()
