"""Chain-merge measurement: how many blocks of the kfold x86 pool could merge into a neighbour.

python -m experiments.subgraph_chain

Reads the cache's npz arrays only. Counts the edges u -> v with edge_kind 0 (fallthrough), out-degree(u) == 1,
in-degree(v) == 1, u != v and the same function (`s02_preprocess.subgraph.chain_reduction`), over every x86 binary that has
a kfold fold, and writes <runs>/subgraph_chain.json, which the gates stage `subgraph` reads. A chain-merge policy is worth
building only when the reduction is at least 5%.
"""
import json

from common.config import CACHE, RUNS
from common.manifests import load_index, run_command
from s02_preprocess.subgraph import chain_reduction

WORTH_BUILDING = 0.05


def main():
    rows = [r for r in load_index(protocol='kfold') if r['arch'] == 'x86' and r['fold'] is not None]
    out = {'cache': str(CACHE), 'pool': 'kfold x86', **chain_reduction(rows, CACHE)}
    out['worth_building'] = out['reduction'] >= WORTH_BUILDING
    (RUNS / 'subgraph_chain.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    run_command(main)
