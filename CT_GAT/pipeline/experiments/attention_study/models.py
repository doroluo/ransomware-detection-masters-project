"""Attention-only controls layered over the frozen paper model implementations."""
from contextlib import contextmanager
import hashlib
from pathlib import Path

from torch import nn
from torch_geometric.nn import GATConv

from experiments.final_paper import trial as paper_trial
from experiments.final_paper.controlled import controlled_spec
from experiments.grid_study import grids


def source_hashes():
    folder = Path(__file__).parent
    return {p.name: hashlib.sha256(p.read_text(encoding='utf-8').encode()).hexdigest()
            for p in sorted(folder.glob('*.py'))}


def candidates(model, base, heads=(2, 4, 8, 12), dropouts=(0.0, 0.1, 0.2), width=96):
    if model in (1, 3) and any(width % h for h in heads):
        raise ValueError('The fixed Transformer width must be divisible by every requested head count')
    spec = controlled_spec(base, model)
    spec.update(attention_width=width if model in (1, 3) else None,
                attention_sources=source_hashes(), gat_head_policy='all_layers',
                study='attention_followup')
    if model in (1, 3):
        spec['width'] = width
    axes = dict(attention_dropout=list(dropouts))
    if model in (1, 3):
        axes['transformer_heads'] = list(heads)
    if model in (2, 3):
        axes['gat_heads'] = list(heads)
    if model == 3:
        axes['pooling'] = ['mean', 'cls']
    return grids.expand(spec, axes)


def make_model(model, api_size, spec, seed):
    network = paper_make_model(model, api_size, spec, seed)
    if model in (2, 3) and spec['gat_head_policy'] == 'all_layers':
        graph = network if model == 2 else network.gat
        previous = graph.convs[-1]
        graph.convs[-1] = GATConv(
            previous.in_channels, previous.out_channels, heads=spec['gat_heads'],
            concat=False, dropout=spec['attention_dropout'], edge_dim=graph.edge_dim,
        )
    for module in network.modules():
        if isinstance(module, nn.MultiheadAttention):
            module.dropout = spec['attention_dropout']
        elif isinstance(module, GATConv):
            module.dropout = spec['attention_dropout']
    return network


paper_make_model = paper_trial.make_model


@contextmanager
def use_attention_factory():
    """Replace the factory only within this sequential study process."""
    previous = paper_trial.make_model
    paper_trial.make_model = make_model
    try:
        yield
    finally:
        paper_trial.make_model = previous
