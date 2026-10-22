"""Pipeline B preprocessing: the teammates' `preprocess_full.process_sample` over the 1,810 kfold x86 binaries, or over another dataset.

python -m experiments.team_pipelines.build_b --sample signal --num-blocks 500 --seed 0 [--out RUNS/team/b_sig500]
       [--source gnn-331ce0b|main-c7899f0|repo] [--limit N] [--ids <sample_id> ...] [--exclude <sample_id> ... --exclude-reason TEXT]
       [--frozen-from <build dir>] [--seeded-rows]
       [--protocol kfold|kfold_v3 --arch x86|all] [--max-instructions N] [--extend]
       (run from CT_GAT/pipeline; a leading RUNS/ in --out is the run directory (GT_RUNS); --out defaults to the `team_b_build`
       source, `CT_GAT/processed/cfg_subgraphs`, where `--model fivechannel` reads)

The dataset is a flag: `--protocol kfold` (default; the Mendeley cache, x86 by default) or `--protocol kfold_v3` (the combined
cache `combined_v3`, `--arch all`), and `--arch` x86 | all (`matched` is a training-time filter of the all build, never built).
The flag is the literal `--protocol`, so that the cache is resolved from it as in every other command (`common/paths.py`), and
a build for another dataset than kfold x86 needs `--out`. The ids are the five folds' test sets of the dataset
(`common.all_ids(protocol, arch)`: kfold x86 1,810, kfold all 2,411, kfold_v3 all 5,652). A binary of the mendeley corpus goes
through `paths_for` as always (bundle and GT_TEAM_ASM_DIR, else the Mendeley collections); a binary of another corpus has its
CFG and listing in its own collection. `settings.json` and the checkpoint's settings of a kfold x86 build carry no `protocol` or
`arch` key (those builds, and their resumes, are unchanged); any other build records both, `load_state` reads a missing key as
kfold x86, and a resume for another dataset stops. `dataset_metadata.fold` is the fold of the build's protocol (`fold_v3` for
kfold_v3). Vocabulary clash: `sig500v3` is the s02-fix build version, `kfold_v3` the fold protocol, `_v3.zip` the preprocessing
version of the zips.

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

--seeded-rows: every API row the embedder adds as the vocabulary grows is drawn from a generator seeded with (--seed, API id)
(`FiveChannelEmbedder(seed=)`), so the table depends only on the vocabulary and the seed and a build stopped and resumed equals one
run straight through. Without it the added rows come from the global torch generator, which a resume does not reproduce (the
rows added after the resume differ and so do the block embeddings pooled from them). The build records `embedder_rows:
seeded-per-id` in `settings.json` and in the checkpoint's settings; a build whose settings lack the key keeps the global-generator
rows, a resume must pass the flag exactly when the build has the key, and the files of the repository (`--source repo`) are the
only code with the seeded embedder.

--frozen-from DIR (equivalence checks): start from DIR's `api_vocab.json` and `channel_embedder.pt` (read only) and call
`process_sample(freeze_vocab=True)`, so the token and embedding ids of two sources are comparable sample by sample.

--exclude: ids that are left out on purpose (a listing `process_sample` cannot hold in memory), with --exclude-reason. They are
not attempted, count as neither built nor failed, and are written with the reason to `settings.json` (next to the checkpoint's own
settings, which stay as they were so an earlier build resumes) and to `build_summary.json` as `excluded`; there `processed` +
the number excluded = `total` is a finished build. Pass the same --exclude on every resume: an id that is not excluded is
attempted again. An id the checkpoint already holds cannot be excluded.

--max-instructions N: the ids of the build whose index `num_instructions` exceeds N are excluded with the automatic reason
"num_instructions > N: load_cfg needs about 3 KB per instruction (31 GB RAM)" (`load_cfg` holds the whole CFG; a binary above
about 4,000,000 instructions does not fit this machine). They are recorded like --exclude ids, in `settings.json` and
`build_summary.json` `excluded`, and the trainers leave them out of every split; pass the same N on every resume (a lower N on a
resume is how a MemoryError is answered). An id the checkpoint already holds stays built and is not excluded. N is recorded as
`max_instructions` in `settings.json` (only by a build that passes it, so the settings of the kfold x86 builds keep their keys, and
not in the checkpoint's settings); a resume of a build that records a cap prints a warning naming it when the flag is omitted or
differs, and goes on (the cap of the run is what counts, and replaces the record).

--extend (the full-Mendeley builds): grow a finished kfold x86 build, copied to --out, to all 2,411 Mendeley ids with
`--protocol kfold --arch all --source repo`, with the copy's own --sample, --num-blocks and --seed. The copy's vocabulary, embedder
and finished ids are loaded and only the x64 ids are built, in sorted order, continuing the vocabulary (ids only grow), so every
x86 `.pt` stays byte-identical. The comparison of the settings tolerates exactly the dataset (`arch`) and the code source: the
build may be a `repo` or a `main-fix` build (both run the files of this repository), the pins of the files now are recorded in
`settings.json` `extensions` ([{arch, ids_added, done_before, source, pins, embedder_rows, date}]) while the build's own `source` and pins stay
as they were, so the trainers keep reading it as what it was. `ids_added` is the number of ids the checkpoint holds beyond the
`done_before` of the entry (the size of the finished set when the extension began), rewritten with every checkpoint and at the end
of each run, so a resumed extension counts the ids of all its runs. The API rows the embedder adds during the extension are drawn per
(--seed, API id), as --seeded-rows does (do not pass it): the extension is reproducible from the x86 build and a stopped and
resumed extension equals a straight one; a resume passes --extend again and must run the files the extension began with. The
checkpoint's settings gain `protocol: kfold, arch: all`, so a plain kfold x86 run on the extended folder stops. A build of the
`gnn-331ce0b` or `main-c7899f0` source (their code is the pinned worktrees), of another dataset, or an extension to anything but
kfold all is refused.
"""
import argparse
import csv
import gc
import inspect
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
from experiments.team_pipelines.common import (BUILD_ARCHES, DATASET_OF, DEFAULT_ARCH, DEFAULT_SOURCE, SOURCE_CODE, SOURCES,
                                               all_ids, import_team, paths_for, sample_rows, source_pins)

SAVE_EVERY = 50
EXTEND_FREE = ('protocol', 'arch', 'source', 'pins', 'embedder_rows')       # what --extend may differ in, besides the ids it adds
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


SEEDED_ROWS = 'seeded-per-id'      # settings['embedder_rows'] of a build whose added API rows depend only on (seed, API id)


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


def dataset_of(settings):
    """(protocol, arch) that a build's settings name; settings without the keys are those of a kfold x86 build."""
    return settings.get('protocol', 'kfold'), settings.get('arch', 'x86')


def read_extensions(out):
    """The `extensions` list of out/settings.json: [] for a build that was never extended (or has no settings.json)."""
    path = out / 'settings.json'
    return json.loads(path.read_text(encoding='utf-8')).get('extensions', []) if path.is_file() else []


def read_max_instructions(out):
    """The `max_instructions` out/settings.json records, None for a build that passed no cap (or has no settings.json)."""
    path = out / 'settings.json'
    return json.loads(path.read_text(encoding='utf-8')).get('max_instructions') if path.is_file() else None


def changed_files(saved, now):
    """The pinned files whose sha256 differ between two pin tables."""
    return sorted(rel for rel in saved.keys() | now.keys() if saved.get(rel) != now.get(rel))


def check_extension(out, saved, settings, extensions):
    """SystemExit unless `out`, whose checkpoint holds the settings `saved`, may be grown to the dataset of `settings` by
    `--extend`: a kfold x86 build to kfold all, or an extension begun earlier (`extensions` is not empty) and resumed with the
    files it began with. Both runs must use the files of this repository (SOURCE_CODE), and apart from the keys in
    EXTEND_FREE the settings must agree."""
    protocol, arch = dataset_of(saved)
    if dataset_of(settings) != ('kfold', 'all'):
        raise SystemExit('--extend adds the x64 ids of the Mendeley dataset: pass --protocol kfold --arch all')
    if protocol != 'kfold' or arch not in ('x86', 'all') or (arch == 'all' and not extensions):
        raise SystemExit(f'{out} was built for {protocol} {arch}: --extend grows a kfold x86 build (or resumes an extension of one)')
    for what, source in (('the build', saved['source']), ('this run', settings['source'])):
        if SOURCE_CODE.get(source) != 'repo':
            raise SystemExit(f'--extend continues builds made with the files of this repository (source repo or main-fix): '
                             f'{what} has the source {source}')
    if extensions:
        changed = changed_files(extensions[-1]['pins'], settings['pins'])
        if changed:
            raise SystemExit(f'{out}: the extension began with other files; {changed} changed since')
    rest = lambda s: {k: v for k, v in s.items() if k not in EXTEND_FREE}      # noqa: E731
    if rest(saved) != rest(settings):
        raise SystemExit(f'{out} was built with {saved}, this run asks for {settings}')


def load_state(out, pp, settings, extend=False):
    """(vocab, embedder, done ids) from `out`, or a fresh pair seeded with settings['seed'] (or the pair of
    settings['frozen_from']) when it has none. With settings['embedder_rows'] the rows the embedder adds are seeded per API id.
    `extend` (--extend): `out` must hold the checkpoint of a build that `check_extension` accepts, and the rows the embedder adds
    are seeded per API id from here on, whether the build was seeded or not."""
    ckpt = out / 'checkpoint.json'
    seeded = settings.get('embedder_rows') == SEEDED_ROWS or extend
    row_seed = settings['seed'] if seeded else None
    if not ckpt.is_file():
        if extend:
            raise SystemExit(f'--extend grows a finished build: {out} has no checkpoint.json (copy the build to extend into --out)')
        if settings.get('frozen_from'):
            vocab, embedder = load_vocab_embedder(settings['frozen_from'], pp)
            return vocab, embedder, set()
        torch.manual_seed(settings['seed'])
        vocab = pp.tok.ApiVocab()
        extra = {'seed': row_seed} if seeded else {}
        return vocab, pp.FiveChannelEmbedder(pp.api_table_size(vocab), **extra), set()
    state = json.loads(ckpt.read_text(encoding='utf-8'))
    saved = {'source': DEFAULT_SOURCE, **state['settings']}      # a checkpoint of the first build has no source
    if extend:
        check_extension(out, saved, settings, read_extensions(out))
    elif dataset_of(saved) != dataset_of(settings):
        raise SystemExit(f'{out} was built for {" ".join(dataset_of(saved))}, this run asks for {" ".join(dataset_of(settings))}')
    elif read_extensions(out):
        raise SystemExit(f'{out} was extended (settings.json lists its extensions): pass --extend to resume it')
    elif saved != settings:
        changed = changed_files(saved.get('pins', {}), settings.get('pins', {}))
        hint = ''
        if saved.get('embedder_rows') != settings.get('embedder_rows'):
            hint = '; pass --seeded-rows exactly when the build has embedder_rows'
        raise SystemExit(f"{out} was built with {saved}, this run asks for {settings}"
                         + (f'; the files {changed} changed since the checkpoint' if changed else '') + hint)
    vocab, embedder = load_vocab_embedder(out, pp)
    if seeded:
        embedder.seed = row_seed             # the rows it adds from here on are the ones an uninterrupted build adds
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


def require_cache(protocol='kfold'):
    """The fold ids come from the cache index: stop with the build command of the protocol's dataset when it has none."""
    if not (manifests.CACHE / 'index.csv').is_file():
        raise SystemExit(f'{manifests.CACHE / "index.csv"} is missing: build the CFG cache first '
                         f'(python -m s02_preprocess.cache.build_cache --cache {DATASET_OF[protocol]}), or set GT_CACHE to a built one')


def resolve_dir(path):
    """`path`, with a leading RUNS/ meaning the run directory (common.config.RUNS)."""
    return RUNS.joinpath(*path.parts[1:]) if path.parts and path.parts[0] == 'RUNS' else path


def extension_documents(out, args, settings, excluded, done):
    """(the settings.json of an extended build, the settings its checkpoint carries). settings.json is the one of the build being
    extended with the dataset (`protocol`, `arch`), the exclusions of this run and an `extensions` entry appended on the first run
    of the extension (a resumed extension keeps its entry, with `ids_added` recounted from `done`, the ids the checkpoint holds); its `source` and `pins` stay the build's own. The checkpoint's settings
    are the build's own with the dataset added."""
    path = out / 'settings.json'
    if not path.is_file():
        raise SystemExit(f'--extend needs the settings.json of the build to extend, {path} is missing')
    document = json.loads(path.read_text(encoding='utf-8'))
    extensions = document.get('extensions') or [{
        'arch': settings['arch'], 'ids_added': 0, 'done_before': len(done), 'source': args.source, 'pins': dict(source_pins(args.source)),
        'embedder_rows': SEEDED_ROWS, 'date': time.strftime('%Y-%m-%d')}]
    first = extensions[0]
    if 'done_before' in first:                  # an entry of an earlier version keeps the count it was written with
        first['ids_added'] = len(done) - first['done_before']
    checkpoint = {**json.loads((out / 'checkpoint.json').read_text(encoding='utf-8'))['settings'],
                  'protocol': settings['protocol'], 'arch': settings['arch']}
    return {**document, 'protocol': settings['protocol'], 'arch': settings['arch'], 'excluded': excluded,
            'extensions': extensions}, checkpoint


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sample', choices=['signal', 'walk', 'callwalk'], default='signal')
    ap.add_argument('--num-blocks', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--source', choices=sorted(SOURCES), default=DEFAULT_SOURCE,
                    help='pinned teammate code (default: the one the first build used; repo: the files of this repository)')
    ap.add_argument('--frozen-from', type=Path, help='start from the vocabulary and embedder of this build directory and freeze the vocabulary')
    ap.add_argument('--seeded-rows', action='store_true', help='draw every API row the embedder adds from a generator seeded with '
                    '(--seed, API id), so a resumed build equals a straight one; recorded as embedder_rows in settings.json')
    ap.add_argument('--protocol', choices=sorted(DATASET_OF), default='kfold',
                    help='the fold protocol, which picks the dataset: kfold (mendeley_x86 cache) or kfold_v3 (combined_v3 cache)')
    ap.add_argument('--arch', choices=['x86', 'all', 'matched'],
                    help='the architectures built (default: x86 for kfold, all for kfold_v3); matched is refused, it is a filter of the all build')
    ap.add_argument('--max-instructions', type=int, metavar='N',
                    help='leave out the ids whose index num_instructions exceeds N, with an automatic reason (load_cfg needs about 3 KB per instruction)')
    ap.add_argument('--extend', action='store_true', help='grow the finished kfold x86 build in --out to kfold all (the x64 ids): see the module docstring')
    ap.add_argument('--out', type=Path, help='the build folder (default: the team_b_build source, CT_GAT/processed/cfg_subgraphs; '
                    'required for any dataset but kfold x86)')
    ap.add_argument('--limit', type=int, help='the first N ids in sorted order')
    ap.add_argument('--ids', nargs='+', help='only these sample ids')
    ap.add_argument('--exclude', nargs='+', default=[], metavar='ID', help='sample ids left out of the build on purpose')
    ap.add_argument('--exclude-reason', help='why they are left out (required with --exclude)')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')

    protocol = args.protocol
    arch = args.arch or DEFAULT_ARCH[protocol]
    if arch not in BUILD_ARCHES[protocol]:
        raise SystemExit(f'--arch {arch}: ' + ('matched is a training-time filter of the all build, which is the one built'
                                              if arch == 'matched' else f'a {protocol} build is made for arch {" or ".join(BUILD_ARCHES[protocol])}'))
    if (protocol, arch) != ('kfold', 'x86') and args.out is None:
        raise SystemExit(f'--out is required for a {protocol} {arch} build (the default folder is the one the x86 arms read)')
    if args.extend and args.seeded_rows:
        raise SystemExit('--extend draws the rows it adds per API id by itself: do not pass --seeded-rows')
    require_cache() if protocol == 'kfold' else require_cache(protocol)
    ids = all_ids(protocol, arch) if args.ids is None else list(args.ids)
    rows = sample_rows(protocol, arch)
    unknown = sorted(set(ids) - set(rows))
    if unknown:
        raise SystemExit(f'not {protocol} {arch} ids: {unknown}')
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
    if (protocol, arch) != ('kfold', 'x86'):
        settings.update(protocol=protocol, arch=arch)         # a kfold x86 build records neither: its settings stay as they were
    if SOURCES[args.source]['env'] is None:
        settings['pins'] = dict(source_pins(args.source))     # the files of this repository are the pin: a resume sees a change
    if args.seeded_rows or args.extend:
        if 'seed' not in inspect.signature(pp.FiveChannelEmbedder).parameters:
            raise SystemExit(f"{'--seeded-rows' if args.seeded_rows else '--extend'} needs an embedder that takes a seed; "
                             f'the files of --source {args.source} do not')
    if args.seeded_rows:
        settings['embedder_rows'] = SEEDED_ROWS
    if args.frozen_from:
        settings['frozen_from'] = str(resolve_dir(args.frozen_from))
    out = resolve_dir(args.out or source_for('team_b_build'))
    out.mkdir(parents=True, exist_ok=True)
    refuse_foreign_folder(out)
    recorded_cap = read_max_instructions(out)
    vocab, embedder, done = load_state(out, pp, settings, **({'extend': True} if args.extend else {}))
    if recorded_cap is not None and args.max_instructions != recorded_cap:
        print(f'WARNING {out}: the build records --max-instructions {recorded_cap}, this run '
              + ('omits it' if args.max_instructions is None else f'passes {args.max_instructions}'), file=sys.stderr, flush=True)
    built_excluded = sorted(set(excluded) & done)
    if built_excluded:
        raise SystemExit(f'the checkpoint already holds {built_excluded}: they cannot be excluded')
    if args.max_instructions is not None:
        reason = f'num_instructions > {args.max_instructions}: load_cfg needs about 3 KB per instruction (31 GB RAM)'
        too_big = {i: reason for i in ids if i not in done and i not in excluded
                   and int(rows[i]['num_instructions']) > args.max_instructions}      # an id the checkpoint holds stays built
        excluded = dict(sorted({**too_big, **excluded}.items()))
    dropped = drop_unfinished(out, done)
    todo = [i for i in ids if i not in done and i not in excluded]
    if args.extend:
        document, settings = extension_documents(out, args, settings, excluded, done)
    else:
        document = {**settings, 'pins': source_pins(args.source), 'excluded': excluded}
    document.pop('max_instructions', None)           # the record is the cap of this run
    if args.max_instructions is not None:
        document['max_instructions'] = args.max_instructions
    write_settings = lambda: atomic(out / 'settings.json', lambda p: p.write_text(json.dumps(document, indent=2), encoding='utf-8'))      # noqa: E731

    def save():
        save_state(out, pp, vocab, embedder, done, settings)
        if args.extend:
            first = document['extensions'][0]
            if 'done_before' in first:
                first['ids_added'] = len(done) - first['done_before']
            write_settings()

    write_settings()
    if args.extend:
        save_state(out, pp, vocab, embedder, done, settings)       # the checkpoint names the dataset from the first moment on
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
            row = rows[sample_id]
            cfg_path, asm_path = (paths_for(sample_id) if row['corpus'] == 'mendeley' else paths_for(sample_id, row['corpus']))
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
                save()
                since_save = 0
            if n % 10 == 0 or n == len(todo):
                wall = time.time() - t_start
                print(f'{n}/{len(todo)} built, {len(done)}/{len(ids)} done, {wall:.0f} s, '
                      f'{wall / n:.1f} s/sample, last {sample_id[:12]} {seconds:.1f} s '
                      f'({sample.sampled_blocks}/{sample.original_blocks} blocks), vocab {len(vocab.names)}',
                      flush=True)
    save()
    wall = time.time() - t_start
    summary = {'processed': len(done & set(ids)), 'total': len(ids), 'excluded': excluded, 'built_this_run': built,
               'wall_seconds_this_run': round(wall, 1), 'api_names': len(vocab.names), 'settings': settings}
    (out / 'build_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"processed {summary['processed']}/{summary['total']} in {wall:.0f} s this run "
          f"({built} built, {len(excluded)} excluded); vocabulary {len(vocab.names)} names", flush=True)
    return summary


if __name__ == '__main__':
    sys.exit(0 if manifests.run_command(main) else 1)
