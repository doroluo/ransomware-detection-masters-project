"""Pipeline B trainer: the teammates' FullBlockGAT (train_full_gat recipe) on the build_b.py subgraphs, our fold manifests.

python -m experiments.team_pipelines.train_b --fold 0 --seed 0 --arm sig500|sig500v2|sig500v3|sig500v3-feat|sig500v3-emb
       [--features all|feat|emb] [--source SOURCE] [--build DIR] [--smoke N] [--device cuda|cpu]
       (run from the block_level_counting root; reads the build directory and never writes there)

Arms. `sig500` reads the first build `RUNS/team/b_sig500` (teammate source gnn-331ce0b); `sig500v2` reads `RUNS/team/b_sig500_v2`
(source main-c7899f0, Yanping's streaming preprocess). --build DIR overrides the directory of either arm; the build's
`settings.json` names its source (none: gnn-331ce0b), the arm must match it, and the build's own `preprocess_full` is the one
imported for the embedder and the pooling. The graph loader and the model are always the gnn source's (main has neither).
Run names: `kfold{f}_x86_gnn_s{seed}_team-sig500` and `..._team-sig500v2`. Everything below is the same for both arms.

Arms on the fixed preprocessing (step 35). `sig500v3`, `sig500v3-feat` and `sig500v3-emb` read `RUNS/team/b_sig500_v3`, the
rebuild with source `main-fix` (build_b --source main-fix: the `endp` and library-function fixes of the s02 preprocess). They differ
in the node input only (--features): `all` the 141-d concatenation of today (64 pooled embedding + 77 handcrafted block features),
`feat` only the 77 scaled block features (the team diagram's Model 2: block features + CFG), `emb` only the 64-d pooled channel
embedding (Model 3: block embeddings + CFG). Edges and edge attributes (edge_dim 8) are the same in the three. The arm carries the
mode: `--features` defaults to the arm's (`all` for every arm but the two suffixed ones) and a value that contradicts the arm is
refused, so no run is written under a name that does not describe its input. The per-fold API remap and the one-sample re-pool
check run in every mode: the three api-kind count columns of the features come from the remapped ids, and the check validates the
build; `feat` alone skips the re-pool of the embeddings, which it never reads. --source (default: the arm's) must equal the source
in the build's settings.json, else the run is refused.

The build fixed one API vocabulary over all 1,810 binaries and pooled the block embeddings with the embedder it saved
(`channel_embedder.pt`). Per fold this script makes the vocabulary a training-set one again: every API id (>= FIRST_API_ID)
that no training binary of the fold uses becomes UNK_API in the saved `instruction_ids[:, 3]`, the instruction vectors are
re-embedded with `FiveChannelEmbedder` (the saved state) and re-pooled with `pool_block_embeddings`, and the three api-kind
columns of `block_features` (api_none / api_unresolved / api_known counts) are recomputed the way `build_model_inputs`
counts them. The other channels (opcode, operands, behaviour tag) are the build's: an API name outside the training set
keeps the behaviour tag it was given. Before any swap, one sample is re-pooled with the unchanged ids and must reproduce
its stored `block_embeddings` (1e-5) and api-kind columns (exactly). The graphs are `graph_full_loader`'s: its
`_node_features` (unit-length embedding, log / rate-scaled features, both concatenated: 141) and `_edge_attr` (one-hot
relation + back-edge bit: edge_dim 8). Model `FullBlockGAT` with its defaults; recipe of train_full_gat.py (AdamW lr 1e-3,
weight decay 1e-4, class-weighted cross-entropy, batch 16, 40 epochs, patience 8 on the validation ransomware F1 at 0.5),
fit.py for the loop and both selection rules. Only ids listed in the build's checkpoint.json are used (their embedder state
is the one saved with it); a full run needs all 1,810, less the ids the build excluded (`build_b --exclude`, read from the
build's settings.json): those are left out of training, validation and test and recorded in metrics.json (`excluded`,
`excluded_reason`). --smoke N: N available training ids, 2 epochs, temp dir only.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

from common.config import RUNS
from experiments.team_pipelines.build_b import load_vocab_embedder, resolve_dir
from experiments.team_pipelines.common import DEFAULT_SOURCE, SOURCES, fold_manifest, import_team, sample_rows, source_pins
from experiments.team_pipelines.fit import run_arm, smoke_ids

ARMS = ('sig500', 'sig500v2', 'sig500v3', 'sig500v3-feat', 'sig500v3-emb')
B_DIR = RUNS / 'team' / 'b_sig500'
B_V2_DIR = RUNS / 'team' / 'b_sig500_v2'
B_V3_DIR = RUNS / 'team' / 'b_sig500_v3'
ARM_DIR = {'sig500': B_DIR, 'sig500v2': B_V2_DIR, 'sig500v3': B_V3_DIR, 'sig500v3-feat': B_V3_DIR, 'sig500v3-emb': B_V3_DIR}
ARM_SOURCE = {'sig500': 'gnn-331ce0b', 'sig500v2': 'main-c7899f0', 'sig500v3': 'main-fix', 'sig500v3-feat': 'main-fix',
              'sig500v3-emb': 'main-fix'}
FEATURES = ('all', 'feat', 'emb')
ARM_FEATURES = {'sig500': 'all', 'sig500v2': 'all', 'sig500v3': 'all', 'sig500v3-feat': 'feat', 'sig500v3-emb': 'emb'}
FEATURE_MODE = {'all': 'both', 'feat': 'features', 'emb': 'embeddings'}      # graph_full_loader._node_features modes
HP = {'lr': 1e-3, 'weight_decay': 1e-4, 'batch_size': 16, 'epochs': 40, 'patience': 8}
KIND_COLUMNS = ('api_none_count', 'api_unresolved_count', 'api_known_count')


def read_settings(b_dir=B_DIR):
    path = Path(b_dir) / 'settings.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


def read_source(b_dir=B_DIR):
    """The teammate source a build was made with (its settings.json `source`; the first build has none: gnn-331ce0b)."""
    source = read_settings(b_dir).get('source', DEFAULT_SOURCE)
    if source not in SOURCES:
        raise SystemExit(f'{b_dir}/settings.json names the unknown source {source}')
    return source


def build_state(b_dir=B_DIR):
    """(pp, embedder, done ids): the build's `preprocess_full` (from its own source), its FiveChannelEmbedder state and the
    ids its checkpoint covers. The checkpoint is read before the embedder: the embedder only ever gains rows, so a newer
    file is a superset of the one the checkpoint was written with."""
    pp = import_team('preprocess_full', source=read_source(b_dir))
    state = json.loads((Path(b_dir) / 'checkpoint.json').read_text(encoding='utf-8'))
    _, embedder = load_vocab_embedder(b_dir, pp)
    embedder.eval()
    return pp, embedder, set(state['done'])


def read_excluded(b_dir=B_DIR):
    """{sample_id: reason} of the ids the build left out on purpose (its settings.json); {} when none or no file."""
    path = b_dir / 'settings.json'
    return dict(json.loads(path.read_text(encoding='utf-8')).get('excluded', {})) if path.is_file() else {}


def block_counts(tok, instruction_ids, matched, insn_to_block, n_blocks):
    """[n_blocks, 3] float counts (api_none, api_unresolved, api_known) of a payload's instructions, from the api channel:
    known <=> id >= FIRST_API_ID, unresolved or unseen <=> UNK_API, none <=> NO_API on a matched instruction (an unmatched
    instruction also carries NO_API and is counted in none of the three)."""
    api = instruction_ids[:, 3]
    kinds = torch.stack([(api == tok.NO_API) & matched.bool(), api == tok.UNK_API, api >= tok.FIRST_API_ID], 1).float()
    out = torch.zeros(n_blocks, 3)
    out.index_add_(0, insn_to_block.long(), kinds)
    return out


def kind_slice(pp):
    cols = [pp.HANDCRAFTED_FEATURE_NAMES.index(n) for n in KIND_COLUMNS]
    if cols != list(range(cols[0], cols[0] + 3)):
        raise AssertionError(f'api-kind columns are not consecutive: {cols}')
    return slice(cols[0], cols[0] + 3)


def remap_api(tok, instruction_ids, seen):
    """instruction_ids with every API id >= FIRST_API_ID that is not in `seen` replaced by UNK_API."""
    out = instruction_ids.clone()
    api = out[:, 3]
    unseen = (api >= tok.FIRST_API_ID) & ~torch.isin(api, seen)
    api[unseen] = tok.UNK_API
    return out


def repool(pp, embedder, payload, instruction_ids):
    n_blocks = int(payload['block_features'].shape[0])
    return pp.pool_block_embeddings(instruction_ids.long(), payload['insn_to_block'].long(), n_blocks, embedder)


def check_repool(pp, embedder, payload, atol=1e-5):
    """Assert that re-pooling the stored instruction ids with the saved embedder reproduces the stored
    block_embeddings (atol) and that block_counts reproduces the stored api-kind columns. Returns the largest
    embedding difference."""
    pooled = repool(pp, embedder, payload, payload['instruction_ids'])
    diff = float((pooled - payload['block_embeddings'].float()).abs().max()) if pooled.numel() else 0.0
    if diff > atol:
        raise AssertionError(f"{payload.get('file_id')}: re-pooled block embeddings differ from the stored ones by "
                             f'{diff:.3g} (> {atol})')
    counts = block_counts(pp.tok, payload['instruction_ids'], payload['instruction_matched'], payload['insn_to_block'],
                          int(payload['block_features'].shape[0]))
    stored = payload['block_features'][:, kind_slice(pp)].float()
    if not torch.equal(counts, stored):
        raise AssertionError(f"{payload.get('file_id')}: api-kind columns recomputed from the ids differ from the "
                             f'stored ones in {int((counts != stored).any(1).sum())} blocks')
    return diff


def slim(payload):
    """The tensors of a build payload that the graph needs (the lists of block keys and rvas are dropped)."""
    keep = ('instruction_ids', 'insn_to_block', 'instruction_matched', 'block_features', 'block_embeddings', 'edge_index',
            'edge_attr', 'edge_back', 'edge_type', 'label', 'file_id')
    return {k: payload[k] for k in keep}


def load_slim(b_dir, sample_id):
    payload = torch.load(b_dir / f'{sample_id}.pt', map_location='cpu', weights_only=False)
    if payload['file_id'] != sample_id or int(payload['label']) != sample_rows()[sample_id]['label']:
        raise ValueError(f'{sample_id}.pt holds sample {payload["file_id"]} / label {payload["label"]}')
    return slim(payload)


def train_api_ids(tok, payloads, train_ids):
    """Sorted tensor of the API ids (>= FIRST_API_ID) that occur in the training binaries."""
    seen = set()
    for i in train_ids:
        api = payloads[i]['instruction_ids'][:, 3]
        seen.update(torch.unique(api[api >= tok.FIRST_API_ID]).tolist())
    return torch.tensor(sorted(seen), dtype=torch.long)


def make_graph(pp, gl, embedder, payload, seen, features='all'):
    """The PyG Data of one payload with API ids outside `seen` mapped to UNK_API (seen=None: the stored ids), built as
    graph_full_loader.load_full_graph builds it from a payload. `features`: 'all' (64 embedding + 77 block features), 'feat'
    (the 77 scaled block features) or 'emb' (the 64 pooled embedding); the edges are the same in the three."""
    from torch_geometric.data import Data
    if features not in FEATURE_MODE:
        raise ValueError(f'features {features}: one of {FEATURES}')
    p = dict(payload)
    if seen is not None:
        ids = remap_api(pp.tok, p['instruction_ids'], seen)
        if features != 'feat':
            p['block_embeddings'] = repool(pp, embedder, p, ids)
        block_features = p['block_features'].float().clone()
        block_features[:, kind_slice(pp)] = block_counts(pp.tok, ids, p['instruction_matched'], p['insn_to_block'],
                                                         block_features.shape[0])
        p['block_features'] = block_features
    x = gl._node_features(p, FEATURE_MODE[features], True)
    data = Data(x=x, edge_index=p['edge_index'].long(), edge_attr=gl._edge_attr(p),
                y=torch.tensor(int(p['label']), dtype=torch.long), num_nodes=int(x.shape[0]))
    data.edge_type = p['edge_type'].long()
    data.sample_id = str(p['file_id'])
    return data


def read_build_log(b_dir):
    """{sample_id: (n_blocks_total, n_blocks_kept)} of the build log (the last row of an id wins)."""
    out = {}
    with (b_dir / 'build_log.csv').open(encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            if r['status'] == 'ok':
                out[r['sample_id']] = (int(r['n_blocks_total']), int(r['n_blocks_kept']))
    return out


def run(arm, fold, seed, smoke=None, device=None, b_dir=None, log=print, features=None, source=None):
    if arm not in ARMS:
        raise ValueError(f'arm {arm}: one of {ARMS}')
    features = ARM_FEATURES[arm] if features is None else features
    if features not in FEATURES:
        raise ValueError(f'features {features}: one of {FEATURES}')
    if features != ARM_FEATURES[arm]:
        raise SystemExit(f'arm {arm} trains on --features {ARM_FEATURES[arm]}, not {features} (the arm names the input)')
    if source is not None and source != ARM_SOURCE[arm]:
        raise SystemExit(f'arm {arm} reads builds of {ARM_SOURCE[arm]}, not --source {source}')
    b_dir = Path(ARM_DIR[arm] if b_dir is None else b_dir)
    source = read_source(b_dir)
    if source != ARM_SOURCE[arm]:
        raise SystemExit(f'arm {arm} reads builds of {ARM_SOURCE[arm]}; {b_dir} was built with {source}')
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    pp, embedder, done = build_state(b_dir)
    gl = import_team('graph_full_loader')
    gm = import_team('gat_full_model')
    manifest = fold_manifest(fold, seed)
    excluded = read_excluded(b_dir)
    want = [r['sample_id'] for r in manifest.train + manifest.val + manifest.test if r['sample_id'] not in excluded]
    t0 = time.time()
    split = None
    if smoke:
        available = {i for i in want if i in done and (b_dir / f'{i}.pt').is_file()}
        split = smoke_ids(manifest, available, smoke)
        want, train_ids = sorted(set(split[0]) | set(split[1])), list(split[0])
    else:
        missing = [i for i in want if i not in done]
        if missing:
            raise SystemExit(f'the build covers {len(done)} binaries; {len(missing)} of this fold\'s {len(want)} are '
                             f'missing from checkpoint.json, e.g. {missing[:3]}')
        train_ids = [r['sample_id'] for r in manifest.train if r['sample_id'] not in excluded]
    payloads = {i: load_slim(b_dir, i) for i in want}
    first = train_ids[0]
    err = check_repool(pp, embedder, payloads[first])
    log(f're-pool check on {first[:12]}: max |difference| {err:.2e} (<= 1e-5), api-kind columns reproduced')
    seen = train_api_ids(pp.tok, payloads, train_ids)
    every = set()
    for i in want:
        api = payloads[i]['instruction_ids'][:, 3]
        every.update(torch.unique(api[api >= pp.tok.FIRST_API_ID]).tolist())
    log(f'api ids in the fold\'s training binaries: {len(seen)}; unseen in val/test binaries mapped to UNK_API: '
        f'{len(every - set(seen.tolist()))} of {len(every)} ids used')
    graphs = {i: make_graph(pp, gl, embedder, payloads[i], seen, features) for i in want}
    del payloads
    log_rows = read_build_log(b_dir)
    sizes = {i: (None, int(graphs[i].x.size(0)), log_rows[i][0]) for i in want}
    in_dim, edge_dim = int(graphs[want[0]].x.size(1)), int(graphs[want[0]].edge_attr.size(1))
    log(f'{arm}: {len(graphs)} graphs built in {time.time() - t0:.0f} s, features {features}, in_dim {in_dim}, '
        f'edge_dim {edge_dim}')
    extra = {'in_dim': in_dim, 'edge_dim': edge_dim, 'features': features, 'n_api_ids_train': int(len(seen)),
             'n_api_ids_unseen_mapped': len(every - set(seen.tolist())), 'build_dir': str(b_dir),
             'build_binaries': len(done), 'build_source': source, 'build_pins': source_pins(source)}
    reasons = sorted(set(excluded.values()))
    return run_arm(arm=arm, fold=fold, seed=seed, graphs=graphs, sizes=sizes,
                   build_model=lambda: gm.FullBlockGAT(in_dim=in_dim, edge_dim=edge_dim), hp=HP, device=device,
                   smoke=smoke, smoke_split=split, extra=extra, log=log, excluded=sorted(excluded),
                   excluded_reason='; '.join(reasons))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--arm', choices=ARMS, default='sig500')
    ap.add_argument('--features', choices=FEATURES, help="node input: all (141-d), feat (77 block features), emb (64-d "
                    "embedding); default and only value allowed: the arm's")
    ap.add_argument('--source', choices=sorted(SOURCES), help="teammate source the build must have been made with (default: the arm's)")
    ap.add_argument('--build', type=Path, help='build directory (default: b_sig500, b_sig500_v2 or b_sig500_v3 by arm)')
    ap.add_argument('--smoke', type=int, help='train on N available training ids for 2 epochs into a temp directory')
    ap.add_argument('--device', choices=['cuda', 'cpu'])
    args = ap.parse_args(argv)
    out = run(args.arm, args.fold, args.seed, args.smoke, args.device,
              resolve_dir(args.build) if args.build else None, features=args.features, source=args.source)
    if args.smoke:
        print(f"smoke: val AUC {out['val_auc']:.4f}, peak GPU {out['peak_gpu_mb']} MB, loss "
              f"{out['history'][0]['loss']} -> {out['history'][-1]['loss']}, temp dir {out['dir']} (not under {RUNS})")
    return out


if __name__ == '__main__':
    main()
    sys.exit(0)
