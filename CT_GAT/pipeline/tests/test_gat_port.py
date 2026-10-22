"""python -m pytest tests/test_gat_port.py: Dorothy's graph model, loader and training helpers, copied into
`s03b_cfg_feature/gat` (branch origin/gnn, commit 331ce0b).

What runs everywhere with torch_geometric: that the copies carry their attribution, that `FullBlockGAT` gives two logits per graph
on a batch with the loader's 141-d nodes and 8-d edge attributes, and that the training loop of `experiments/team_pipelines/fit.py`
gives the same history with and without a node budget that no batch reaches. What needs `GT_TEAM_GNN_ROOT` (the worktree at
331ce0b; it skips with a reason without): that the source text of `FullBlockGAT`, of every function of `graph_full_loader.py` and of
every function kept from `train_cfg.py` equals the original's `inspect.getsource`, so a copy cannot drift from what the pins name.
"""
import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.team_pipelines import common as tp, fit
from test_team_pipelines import (fresh_team_caches, needs_pyg, pinned_team_modules, synthetic_graphs,  # noqa: F401
                                 tiny_model)
from tiny_cache import tiny  # noqa: F401  (a fixture)

NO_GNN_ROOT = None
try:
    tp.root_of('gnn-331ce0b')
except tp.TeamSourceNotConfigured as e:
    NO_GNN_ROOT = str(e)
needs_gnn_worktree = pytest.mark.skipif(NO_GNN_ROOT is not None, reason=f'needs the pinned gnn worktree ({NO_GNN_ROOT})')

KEPT_FROM_TRAIN_CFG = ('BlockGAT', '_batches', 'ransomware_scores', 'predict_proba', 'metrics_at', 'best_threshold')


def own_modules():
    from s03b_cfg_feature.gat import gat_full_model, graph_full_loader, train_cfg
    return gat_full_model, graph_full_loader, train_cfg


def definitions(module):
    """{name: source text} of the functions and classes defined in `module`."""
    return {name: inspect.getsource(obj) for name, obj in vars(module).items()
            if (inspect.isfunction(obj) or inspect.isclass(obj)) and obj.__module__ == module.__name__}


@needs_pyg
@needs_gnn_worktree
def test_every_copied_definition_equals_the_pinned_original():
    gat_full_model, graph_full_loader, train_cfg = own_modules()
    gm, gl, tc = tp.import_team('gat_full_model'), tp.import_team('graph_full_loader'), tp.import_team('train_cfg')
    theirs = definitions(gm)
    assert definitions(gat_full_model) == {'FullBlockGAT': theirs['FullBlockGAT']}
    assert definitions(graph_full_loader) == definitions(gl) and len(definitions(gl)) == 10      # every function, none left out
    for name in ('FEATURE_MODES', '_LOG_COLUMNS', '_COUNT_START', 'BATCH_SIZE'):
        assert getattr(graph_full_loader, name) == getattr(gl, name)
    kept, original = definitions(train_cfg), definitions(tc)
    assert tuple(sorted(kept)) == tuple(sorted(KEPT_FROM_TRAIN_CFG))
    assert all(kept[name] == original[name] for name in kept)
    left_out = sorted(set(original) - set(kept))
    assert left_out == ['_is_sample_graph', 'evaluate', 'load_split', 'main', 'stratified_holdout', 'train']
    for name in left_out:
        assert not hasattr(train_cfg, name)


@pytest.mark.parametrize('module,original', [('gat_full_model', 'CT_GAT/preprocessing/gat_full_model.py'),
                                             ('graph_full_loader', 'CT_GAT/preprocessing/graph_full_loader.py'),
                                             ('train_cfg', 'train_cfg.py')])
def test_each_file_names_dorothy_the_commit_the_original_path_and_the_pin(module, original):
    path = Path(importlib.util.find_spec(f's03b_cfg_feature.gat.{module}').origin)
    text = path.read_text(encoding='utf-8')
    docstring = text.split('"""')[1]
    assert 'Dorothy' in docstring and '331ce0b' in docstring and 'origin/gnn' in docstring
    assert f'`{original}`' in docstring
    assert tp.PINNED_SHA256[original] in docstring                       # the pin of the CRLF checkout the worktree is compared with
    assert 'Left out' in docstring and '---- the original docstring ----' in docstring
    assert (path.parent / '__init__.py').read_text(encoding='utf-8') == ''                 # importing the package imports no PyG


@needs_pyg
def test_the_only_edit_of_the_loader_is_the_graph_directory():
    from common.catalog import PROCESSED
    _, graph_full_loader, _ = own_modules()
    assert graph_full_loader.GRAPH_DIR == PROCESSED / 'cfg_subgraphs'
    assert not hasattr(graph_full_loader, 'PROJECT_ROOT')


@needs_pyg
def test_full_block_gat_gives_two_logits_per_graph_on_a_two_graph_batch():
    from torch_geometric.data import Batch, Data
    from s03b_cfg_feature.gat import gat_full_model
    rng = np.random.default_rng(0)
    graphs = []
    for label, n in ((0, 6), (1, 9)):
        edges = torch.tensor(rng.integers(0, n, (2, 2 * n)))
        graphs.append(Data(x=torch.tensor(rng.normal(size=(n, 141)), dtype=torch.float32), edge_index=edges,
                           edge_attr=torch.tensor(rng.random((2 * n, 8)), dtype=torch.float32), y=torch.tensor(label)))
    model = gat_full_model.FullBlockGAT(in_dim=141, edge_dim=8).eval()
    logits = model(Batch.from_data_list(graphs))
    assert tuple(logits.shape) == (2, 2) and logits.device.type == 'cpu' and torch.isfinite(logits).all()
    alone = model(Batch.from_data_list(graphs[:1]))
    assert tuple(alone.shape) == (1, 2)
    empty = Batch.from_data_list([Data(x=torch.randn(3, 141), edge_index=torch.zeros(2, 0, dtype=torch.long),
                                       edge_attr=torch.zeros(0, 8), y=torch.tensor(0))])
    assert tuple(model(empty).shape) == (1, 2)                            # no edges: the self-loop fallback


@needs_pyg
def test_block_gat_of_pipeline_a_runs_on_a_batch():
    from torch_geometric.data import Batch, Data
    from s03b_cfg_feature.gat import train_cfg
    graphs = [Data(x=torch.randn(n, 57), edge_index=torch.tensor([[0, 1], [1, 2]]), y=torch.tensor(k)) for k, n in enumerate((4, 5))]
    assert tuple(train_cfg.BlockGAT(in_dim=57).eval()(Batch.from_data_list(graphs)).shape) == (2, 2)


@needs_pyg
def test_fit_without_a_node_budget_equals_fit_with_one_no_batch_reaches(tiny, tmp_path):   # noqa: F811
    manifest = tp.fold_manifest(2, 1)
    train_ids = [r['sample_id'] for r in manifest.train]
    val_ids = [r['sample_id'] for r in manifest.val]
    graphs, _ = synthetic_graphs(train_ids + val_ids, np.random.default_rng(0))
    hp = {'lr': 1e-2, 'weight_decay': 0.0, 'batch_size': 4, 'epochs': 3, 'patience': 8}

    def go(extra):
        torch.manual_seed(0)
        out = fit.fit(tiny_model()(), train_ids, val_ids, graphs, hp={**hp, **extra}, seed=0, device=torch.device('cpu'),
                      log=lambda *_: None)
        return out['history'], out['auc_state']
    base, state = go({})
    same, state_same = go({'node_budget': 10 ** 9})
    assert all('split_batches' not in h for h in base) and all(h['split_batches'] == 0 for h in same)
    keys = ('loss', 'val_f1', 'val_macro_f1', 'val_auc', 'val_acc')
    assert [[h[k] for k in keys] for h in base] == [[h[k] for k in keys] for h in same]       # the same history, to the printed digits
    assert all(torch.equal(state[k], state_same[k]) for k in state)                           # a budget never hit changes nothing
    cut, _ = go({'node_budget': 6})
    assert all(h['split_batches'] > 0 for h in cut)                                           # a small budget cuts and says so
