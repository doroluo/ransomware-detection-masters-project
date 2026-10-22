"""Pipeline B preprocessing: the teammates' `preprocess_full.process_sample` over the 1,810 kfold x86 binaries.

python -m experiments.team_pipelines.build_b --sample signal --num-blocks 500 --seed 0 [--out RUNS/team/b_sig500]
       [--source gnn-331ce0b|main-c7899f0|repo] [--limit N] [--ids <sample_id> ...] [--exclude <sample_id> ... --exclude-reason TEXT]
       [--frozen-from <build dir>]
       (run from CT_GAT/pipeline; a leading RUNS/ in --out is the run directory (GT_RUNS); --out defaults to the `team_b_build`
       source, `CT_GAT/processed/cfg_subgraphs`, where `--model fivechannel` reads)

--source picks the pinned teammate code (common.SOURCES): `gnn-331ce0b` (default; the v1 build `b_sig500`), `main-c7899f0`
(Yanping's streaming `preprocess_full`; the v2 build `b_sig500_v2`) or `repo`, the same two files in this repository
(`CT_GAT/pipeline/s02_preprocess`), with no fixed pin: their sha256 are computed from the files and recorded in the
checkpoint's settings, so a resume after either file changed stops. The source and its sha256 pins go to `settings.json`, the
source also to the checkpoint's settings, so a resume with another source stops.

For every id (sorted) the CFG and the IDA listing come from `paths_for` (the Mendeley bundle and GT_TEAM_ASM_DIR, else the
Mendeley collections of `CT_GAT/data`) and go through
`process_sample` with the explicit cfg path, no size cap (the caps live in the teammates' `main`, which is not called),
one shared `ApiVocab` and one `FiveChannelEmbedder` created after `torch.manual_seed(seed)`. Per sample
<out>/<sample_id>.pt and <out>/<sample_id>.manifest.json (their `save_sample`). The vocabulary (`api_vocab.json`), the
embedder (`channel_embedder.pt`) and `checkpoint.json` (the ids they cover) are written every 50 samples and at the end,
under <out>, never under the teammates' tree. <out> must be empty or an earlier build of this command (an unzipped preprocessed
zip is refused). The fold ids come from the CFG cache index (`CACHE`: GT_CACHE, else the first dataset of the registry, with
nothing configured `CT_GAT/cache/mendeley_x86`, see `s02_preprocess.cache.build_cache`). A rerun resumes: it loads that state, keeps the ids in the checkpoint and
rebuilds every other id (a sample finished after the last checkpoint used vocabulary ids the saved vocabulary may lack).
A sample that raises stops the build with its id; nothing is skipped. `build_log.csv`: sample_id, status,
n_blocks_total, n_blocks_kept, seconds.

--frozen-from DIR (equivalence checks): start from DIR's `api_vocab.json` and `channel_embedder.pt` (read only) and call
`process_sample(freeze_vocab=True)`, so the token and embedding ids of two sources are comparable sample by sample.

--exclude: ids that are left out on purpose (a listing `process_sample` cannot hold in memory), with --exclude-reason. They are
not attempted, count as neither built nor failed, and are written with the reason to `settings.json` (next to the checkpoint's own
settings, which stay as they were so an earlier build resumes) and to `build_summary.json` as `excluded`; there `processed` +
the number excluded = `total` is a finished build. Pass the same --exclude on every resume: an id that is not excluded is
attempted again. An id the checkpoint already holds cannot be excluded.
"""
import argparse
import csv
import gc
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import torch

from common import manifests
from common.config import RUNS
from common.paths import source_for
from experiments.team_pipelines.common import (DEFAULT_SOURCE, SOURCES, all_ids, import_team, paths_for, sample_rows,
                                               source_pins)

SAVE_EVERY = 50
LOG_FIELDS = ('sample_id', 'status', 'n_blocks_total', 'n_blocks_kept', 'seconds')


def atomic(path, write):
    tmp = path.with_name(path.name + '.tmp')
    write(tmp)
    os.replace(tmp, path)


def save_state(out, pp, vocab, embedder, done, settings):
    atomic(out / 'api_vocab.json', lambda p: pp.tok.save_api_vocab(vocab, p))
    atomic(out / 'channel_embedder.pt', lambda p: torch.save(embedder.state_dict(), p))
    atomic(out / 'checkpoint.json', lambda p: p.write_text(
        json.dumps({'settings': settings, 'done': sorted(done), 'api_names': len(vocab.names)}), encoding='utf-8'))


def load_vocab_embedder(directory, pp, vocab_path=None):
    """(vocab, embedder) from a build directory's api_vocab.json (`vocab_path` names another file, as the unzipped Drive zips
    keep it one level up) and channel_embedder.pt (the embedder is grown to the saved table first). A missing vocabulary file is an
    empty vocabulary (`load_api_vocab`), so the embedder is then sized from the rows of `api_embedding.weight` alone: a trainer
    needs the embedder, not the names."""
    directory = Path(directory)
    vocab = pp.tok.load_api_vocab(Path(vocab_path) if vocab_path else directory / 'api_vocab.json')
    embedder = pp.FiveChannelEmbedder(pp.api_table_size(vocab))
    weights = torch.load(directory / 'channel_embedder.pt', map_location='cpu')
    embedder.expand_api(int(weights['api_embedding.weight'].shape[0]))
    embedder.load_state_dict(weights)
    return vocab, embedder


def load_state(out, pp, settings):
    """(vocab, embedder, done ids) from `out`, or a fresh pair seeded with settings['seed'] (or the pair of
    settings['frozen_from']) when it has none."""
    ckpt = out / 'checkpoint.json'
    if not ckpt.is_file():
        if settings.get('frozen_from'):
            vocab, embedder = load_vocab_embedder(settings['frozen_from'], pp)
            return vocab, embedder, set()
        torch.manual_seed(settings['seed'])
        vocab = pp.tok.ApiVocab()
        return vocab, pp.FiveChannelEmbedder(pp.api_table_size(vocab)), set()
    state = json.loads(ckpt.read_text(encoding='utf-8'))
    saved = {'source': DEFAULT_SOURCE, **state['settings']}      # a checkpoint of the first build has no source
    if saved != settings:
        pins, now = saved.get('pins', {}), settings.get('pins', {})
        changed = sorted(rel for rel in pins.keys() | now.keys() if pins.get(rel) != now.get(rel))
        raise SystemExit(f"{out} was built with {saved}, this run asks for {settings}"
                         + (f'; the files {changed} changed since the checkpoint' if changed else ''))
    vocab, embedder = load_vocab_embedder(out, pp)
    if len(vocab.names) != state['api_names']:
        raise SystemExit(f"{out}/api_vocab.json has {len(vocab.names)} names, the checkpoint says {state['api_names']}")
    return vocab, embedder, set(state['done'])


def drop_unfinished(out, done):
    """Delete the .pt / manifest files of ids the checkpoint does not cover."""
    gone = 0
    for p in out.glob('*.pt'):
        if p.name != 'channel_embedder.pt' and p.stem not in done:
            p.unlink()
            (out / f'{p.stem}.manifest.json').unlink(missing_ok=True)
            gone += 1
    return gone


def refuse_foreign_folder(out):
    """Stop when `out` holds sample `.pt` files but is no build of this command (neither checkpoint.json nor settings.json:
    an unzipped preprocessed zip, say): `drop_unfinished` would delete them."""
    if (out / 'checkpoint.json').is_file() or (out / 'settings.json').is_file():
        return
    if any(p.name != 'channel_embedder.pt' for p in out.glob('*.pt')):
        raise SystemExit(f'{out} holds .pt files but no checkpoint.json or settings.json, so it is not a build of this command '
                         f'(an unzipped preprocessed zip?): give --out an empty folder')


def require_cache():
    """The fold ids come from the cache index: stop with the build command when it has none."""
    if not (manifests.CACHE / 'index.csv').is_file():
        raise SystemExit(f'{manifests.CACHE / "index.csv"} is missing: build the CFG cache first '
                         f'(python -m s02_preprocess.cache.build_cache --cache mendeley_x86), or set GT_CACHE to a built one')


def resolve_dir(path):
    """`path`, with a leading RUNS/ meaning the run directory (common.config.RUNS)."""
    return RUNS.joinpath(*path.parts[1:]) if path.parts and path.parts[0] == 'RUNS' else path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sample', choices=['signal', 'walk', 'callwalk'], default='signal')
    ap.add_argument('--num-blocks', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--source', choices=sorted(SOURCES), default=DEFAULT_SOURCE,
                    help='pinned teammate code (default: the one the first build used; repo: the files of this repository)')
    ap.add_argument('--frozen-from', type=Path, help='start from the vocabulary and embedder of this build directory and freeze the vocabulary')
    ap.add_argument('--out', type=Path, help='the build folder (default: the team_b_build source, CT_GAT/processed/cfg_subgraphs)')
    ap.add_argument('--limit', type=int, help='the first N ids in sorted order')
    ap.add_argument('--ids', nargs='+', help='only these sample ids')
    ap.add_argument('--exclude', nargs='+', default=[], metavar='ID', help='sample ids left out of the build on purpose')
    ap.add_argument('--exclude-reason', help='why they are left out (required with --exclude)')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')

    require_cache()
    ids = all_ids() if args.ids is None else list(args.ids)
    unknown = sorted(set(ids) - set(sample_rows()))
    if unknown:
        raise SystemExit(f'not kfold x86 ids: {unknown}')
    if args.limit is not None:
        ids = ids[:args.limit]
    ids = sorted(ids)
    if bool(args.exclude) != bool(args.exclude_reason):
        raise SystemExit('--exclude and --exclude-reason go together')
    not_in_build = sorted(set(args.exclude) - set(ids))
    if not_in_build:
        raise SystemExit(f'--exclude ids that are not among the ids of this build: {not_in_build}')
    excluded = {i: args.exclude_reason for i in sorted(set(args.exclude))}
    pp = import_team('preprocess_full', source=args.source)
    settings = {'sample': args.sample, 'num_blocks': args.num_blocks, 'seed': args.seed, 'source': args.source}
    if SOURCES[args.source]['env'] is None:
        settings['pins'] = dict(source_pins(args.source))     # the files of this repository are the pin: a resume sees a change
    if args.frozen_from:
        settings['frozen_from'] = str(resolve_dir(args.frozen_from))
    out = resolve_dir(args.out or source_for('team_b_build'))
    out.mkdir(parents=True, exist_ok=True)
    refuse_foreign_folder(out)
    vocab, embedder, done = load_state(out, pp, settings)
    built_excluded = sorted(set(excluded) & done)
    if built_excluded:
        raise SystemExit(f'the checkpoint already holds {built_excluded}: they cannot be excluded')
    dropped = drop_unfinished(out, done)
    todo = [i for i in ids if i not in done and i not in excluded]
    atomic(out / 'settings.json', lambda p: p.write_text(
        json.dumps({**settings, 'pins': source_pins(args.source), 'excluded': excluded}, indent=2), encoding='utf-8'))
    print(f'{out}: {len(ids)} ids, {len(ids) - len(todo) - len(excluded)} done in the checkpoint, {len(excluded)} excluded, '
          f'{dropped} unfinished files removed, {len(todo)} to build', flush=True)

    log_path = out / 'build_log.csv'
    new_log = not log_path.exists()
    t_start, built, since_save = time.time(), 0, 0
    with log_path.open('a', newline='', encoding='utf-8') as log_file:
        log = csv.writer(log_file)
        if new_log:
            log.writerow(LOG_FIELDS)
        for n, sample_id in enumerate(todo, 1):
            row = sample_rows()[sample_id]
            cfg_path, asm_path = paths_for(sample_id)
            t0 = time.time()
            try:
                result = pp.process_sample(
                    asm_path, cfg_path, label=row['label'], seed=args.seed, api_vocab=vocab,
                    num_blocks=args.num_blocks, sample_mode=args.sample, file_id=sample_id,
                    channel_embedder=embedder, freeze_vocab=bool(args.frozen_from),
                    dataset_metadata={'sample_id': sample_id, 'label': row['label'], 'arch': row['arch'],
                                      'family': row['family'], 'fold': row['fold']})
                pp.save_sample(result, out)
            except Exception:
                traceback.print_exc()
                print(f'FAILED sample_id={sample_id} after {built} built in this run; nothing skipped, build stopped',
                      flush=True)
                raise SystemExit(1)
            seconds = time.time() - t0
            sample = result['sample']
            log.writerow([sample_id, 'ok', sample.original_blocks, sample.sampled_blocks, f'{seconds:.1f}'])
            log_file.flush()
            del result
            gc.collect()
            done.add(sample_id)
            built += 1
            since_save += 1
            if since_save >= SAVE_EVERY:
                save_state(out, pp, vocab, embedder, done, settings)
                since_save = 0
            if n % 10 == 0 or n == len(todo):
                wall = time.time() - t_start
                print(f'{n}/{len(todo)} built, {len(done)}/{len(ids)} done, {wall:.0f} s, '
                      f'{wall / n:.1f} s/sample, last {sample_id[:12]} {seconds:.1f} s '
                      f'({sample.sampled_blocks}/{sample.original_blocks} blocks), vocab {len(vocab.names)}',
                      flush=True)
    save_state(out, pp, vocab, embedder, done, settings)
    wall = time.time() - t_start
    summary = {'processed': len(done & set(ids)), 'total': len(ids), 'excluded': excluded, 'built_this_run': built,
               'wall_seconds_this_run': round(wall, 1), 'api_names': len(vocab.names), 'settings': settings}
    (out / 'build_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"processed {summary['processed']}/{summary['total']} in {wall:.0f} s this run "
          f"({built} built, {len(excluded)} excluded); vocabulary {len(vocab.names)} names", flush=True)
    return summary


if __name__ == '__main__':
    sys.exit(0 if manifests.run_command(main) else 1)
