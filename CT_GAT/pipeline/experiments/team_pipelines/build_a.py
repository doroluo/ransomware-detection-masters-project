"""Pipeline A, opcode mode: the teammates' `construct_blocks.gnn_graph_from_export` over the 1,810 kfold x86 binaries.

python -m experiments.team_pipelines.build_a --mode gnn --budget 500 --out RUNS/team/a_gnn500
       [--limit N] [--ids <sample_id> ...] [--workers W]       (run from CT_GAT/pipeline;
a leading RUNS/ in --out is the run directory (GT_RUNS))

For every id (sorted) the row `construct_blocks` expects (the Mendeley bundle's samples.csv row, an int `label` and the
`cfg_file`; `common.team_row`) goes through `gnn_graph_from_export(row, max_blocks=budget)`, which keeps whole
functions in its own ranking. Per sample <out>/<sample_id>.pt holds the PyG `Data` (x [n, 57] opcode and API-group rates
and structure, edge_index, func_batch, y) with `sample_id` and `label` attached; `num_blocks_total` is their count of all
blocks. Written through a temp file, so a killed build never leaves a half file that a rerun would skip. A rerun resumes:
ids with a .pt are skipped. A sample that raises stops the build with its id (nothing is skipped; with --workers the
samples already running are finished or dropped, none is logged as done unless its .pt exists). `build_log.csv`:
sample_id, status, n_blocks_total, n_blocks_kept, seconds. Only `--mode gnn` exists.
"""
import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

from common.config import RUNS
from common.manifests import run_command
from experiments.team_pipelines.common import all_ids, import_team, sample_rows, team_row

LOG_FIELDS = ('sample_id', 'status', 'n_blocks_total', 'n_blocks_kept', 'seconds')
_CB = None


def _module():
    global _CB
    if _CB is None:
        _CB = import_team('construct_blocks')
    return _CB


def build_one(args):
    """(sample_id, n_blocks_total, n_blocks_kept, seconds); writes <out>/<sample_id>.pt. Raises on any failure."""
    sample_id, out, budget = args
    t0 = time.time()
    try:
        graph = _module().gnn_graph_from_export(team_row(sample_id), max_blocks=budget)
    except Exception as e:
        raise RuntimeError(f'sample_id={sample_id}: {e!r}; ' + traceback.format_exc().replace('\n', ' | ')) from None
    graph.sample_id = sample_id
    graph.label = int(graph.y.item())
    path = Path(out) / f'{sample_id}.pt'
    tmp = path.with_name(path.name + '.tmp')
    torch.save(graph, tmp)
    os.replace(tmp, path)
    return sample_id, int(graph.num_blocks_total), int(graph.x.size(0)), time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['gnn'], default='gnn')
    ap.add_argument('--budget', type=int, default=500)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--limit', type=int, help='the first N ids in sorted order')
    ap.add_argument('--ids', nargs='+', help='only these sample ids')
    ap.add_argument('--workers', type=int, default=1, help='processes (each parses one export at a time)')
    args = ap.parse_args(argv)

    ids = all_ids() if args.ids is None else list(args.ids)
    unknown = sorted(set(ids) - set(sample_rows()))
    if unknown:
        raise SystemExit(f'not kfold x86 ids: {unknown}')
    ids = sorted(ids)
    if args.limit is not None:
        ids = ids[:args.limit]
    out = RUNS.joinpath(*args.out.parts[1:]) if args.out.parts and args.out.parts[0] == 'RUNS' else args.out
    out.mkdir(parents=True, exist_ok=True)
    settings = {'mode': args.mode, 'budget': args.budget}
    settings_path = out / 'settings.json'
    if settings_path.is_file():
        have = json.loads(settings_path.read_text(encoding='utf-8'))
        if have != settings:
            raise SystemExit(f'{out} was built with {have}, this run asks for {settings}')
    else:
        settings_path.write_text(json.dumps(settings), encoding='utf-8')
    for stale in out.glob('*.tmp'):
        stale.unlink()
    todo = [i for i in ids if not (out / f'{i}.pt').is_file()]
    print(f'{out}: {len(ids)} ids, {len(ids) - len(todo)} already built, {len(todo)} to build, '
          f'{args.workers} worker(s)', flush=True)

    log_path = out / 'build_log.csv'
    new_log = not log_path.exists()
    t_start, built = time.time(), 0
    jobs = [(i, str(out), args.budget) for i in todo]
    pool = None
    if args.workers > 1:
        import multiprocessing
        pool = multiprocessing.get_context('spawn').Pool(args.workers)
        results = pool.imap_unordered(build_one, jobs)
    else:
        results = map(build_one, jobs)
    with log_path.open('a', newline='', encoding='utf-8') as log_file:
        log = csv.writer(log_file)
        if new_log:
            log.writerow(LOG_FIELDS)
        try:
            for sample_id, total, kept, seconds in results:
                log.writerow([sample_id, 'ok', total, kept, f'{seconds:.1f}'])
                log_file.flush()
                built += 1
                if built % 10 == 0 or built == len(todo):
                    wall = time.time() - t_start
                    print(f'{built}/{len(todo)} built, {len(ids) - len(todo) + built}/{len(ids)} done, {wall:.0f} s, '
                          f'{wall / built:.1f} s/sample, last {sample_id[:12]} {seconds:.1f} s ({kept}/{total} blocks)',
                          flush=True)
        except Exception:
            traceback.print_exc()
            print(f'FAILED (see sample_id above) after {built} built in this run; nothing skipped, build stopped', flush=True)
            if pool is not None:
                pool.terminate()
            raise SystemExit(1)
    if pool is not None:
        pool.close()
        pool.join()
    wall = time.time() - t_start
    summary = {'processed': sum((out / f'{i}.pt').is_file() for i in ids), 'total': len(ids), 'built_this_run': built,
               'wall_seconds_this_run': round(wall, 1), 'settings': settings}
    (out / 'build_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"processed {summary['processed']}/{summary['total']} in {wall:.0f} s this run ({built} built)", flush=True)
    return summary


if __name__ == '__main__':
    sys.exit(0 if run_command(main) else 1)
