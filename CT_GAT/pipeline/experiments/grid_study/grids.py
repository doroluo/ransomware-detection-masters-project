"""Coarse Cartesian stages; freeze earlier choices using validation only."""
import itertools
import statistics

from experiments.model_grid import runner
from experiments.model_grid.data import digest


def base_spec(model):
    args = runner.parser(model).parse_args(['--build', '.', '--index', 'index.csv', '--out', '.'])
    return dict(runner.cases(args, model)[0], lr=3e-4)


def expand(base, axes):
    return [dict(base, **dict(zip(axes, vals))) for vals in itertools.product(*axes.values())]


def optimizer(model, rates, drops, decays):
    # Baseline values first enable early matched comparisons and reuse previous work.
    return expand(base_spec(model), dict(lr=rates, dropout=drops, weight_decay=decays,
                   pooling=['mean', 'cls'] if model == 3 else [base_spec(model)['pooling']]))


def architecture(model, base):
    if model == 1:
        axes = dict(sequence_length=[128, 512, 1024], width=[64, 32, 128], transformer_layers=[2, 1, 4])
    elif model == 2:
        axes = dict(gat_hidden=[128, 64, 256], gat_layers=[2, 1, 3, 4], gat_heads=[4, 2, 8])
    else:
        axes = dict(width=[64, 32, 128], transformer_layers=[2, 1, 3],
                    gat_hidden=[128, 64, 256], gat_layers=[2, 1, 3], pooling=['mean', 'cls'])
    # Deterministic breadth-first ordering covers extremes of each axis before interactions.
    combos = expand(base, axes)
    rank = lambda s: sum(axes[k].index(s[k]) > 0 for k in axes if k != 'pooling')
    return sorted(combos, key=rank)


def instruction(model, base):
    return expand(base, dict(block_length=[64, 32, 128, 256], pooling=['mean', 'cls'])) if model == 3 else []


def trial_id(spec, epochs, seed, test):
    key = digest(dict(spec=spec, epochs=epochs, seed=seed, evaluate_test=test))[:16]
    return f"lr{spec['lr']:g}_d{spec['dropout']:g}_{spec['pooling']}_{key}"


def choose(model, results):
    if not results:
        return []
    score = lambda r: (r['validation_fixed']['auc'], r['validation_fixed']['macro_f1'])
    if model != 3:
        return [max(results, key=score)['spec']]
    pairs = {}
    for r in results:
        key = digest({k:v for k,v in r['spec'].items() if k != 'pooling'})
        pairs.setdefault(key, {})[r['spec']['pooling']] = r
    complete = [p for p in pairs.values() if set(p) == {'mean','cls'}]
    if not complete:
        return []
    best = max(complete, key=lambda p: tuple(statistics.mean(score(r)[i] for r in p.values()) for i in (0,1)))
    return [best[p]['spec'] for p in ('mean','cls')]
