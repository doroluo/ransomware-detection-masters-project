"""Team entry point: python grid_search.py --models 1 3 --build ... --index ... --dataset combined.

Models run sequentially on one GPU. Different machines use disjoint model selections.
The per-model runners remain available as python -m experiments.model_grid.model1/2/3.
"""
import argparse
from pathlib import Path
import sys

from experiments.model_grid.runner import main as run_model, parser as model_parser


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if '--help' in argv or '-h' in argv:
        print(__doc__)
        print('\nAdditional options: --models 1 2 3 (default: all); --out runs/model_grid (output root).')
        print(model_parser(1).format_help())
        return
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--models', nargs='+', type=int, choices=[1, 2, 3], default=[1, 2, 3])
    p.add_argument('--out', type=Path, default=Path('runs/model_grid'))
    args, rest = p.parse_known_args(argv)
    if len(set(args.models)) != len(args.models):
        p.error('--models must not repeat a model')
    for model in args.models:
        print(f'=== Model {model}: {args.out / f"model{model}"} ===', flush=True)
        run_model(model, [*rest, '--out', str(args.out / f'model{model}')])


if __name__ == '__main__':
    main()
