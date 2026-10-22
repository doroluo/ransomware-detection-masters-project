"""Package a pipeline-B build as the zip the team unzips at `CT_GAT/`.

python -m experiments.team_pipelines.package_build <build dir> [--out DIR] [--no-manifests] [--deflate-pt]
       (run from CT_GAT/pipeline; a leading RUNS/ in <build dir> is the run directory (GT_RUNS); --out defaults to the current directory)

<build dir> is a finished `build_b` folder (settings.json, build_summary.json, build_log.csv, api_vocab.json, channel_embedder.pt,
the `<sha256>.pt` samples and their manifests). It is read only. The zip unzips at `CT_GAT/` to the layout `preprocess_full` writes
by default, which `train_b` (`build_layout`: an unzipped Drive zip), `--model fivechannel` and the teammates' loaders read:

    processed/api_vocab.json
    processed/cfg_subgraphs/<sha256>.pt, <sha256>.manifest.json, channel_embedder.pt
    processed/cfg_subgraphs/build_info/   index.csv, build_settings.json, build_summary.json, build_log.csv

The name is `Mendeley_x86_cfg_subgraphs_<signal|callwalk>_<N>blocks_v3.zip` from the build's settings (sampler and block budget), or
`Mendeley_x86_cfg_subgraphs_uncapped_v3.zip` when the budget is 0 or at least `train_b.UNCAPPED_BLOCKS`: with no budget the callwalk
restarts until every allowed block is chosen and the payloads are saved in canonical block order, so one uncapped build serves both
samplers. Beside the zip go `<zip>.sha256` and `<zip stem>.README.md`, the text that is also the zip's first file; it is generated
from settings.json and build_summary.json, so it names no machine path.

--no-manifests leaves the per-sample manifest json files out (alignment diagnostics that no model reads: 3-6 MB each in the
500-block builds, far more in the uncapped one, about 102 GB in all; the README says so).
--deflate-pt compresses the `.pt` files too (level 1); by default they are stored, since tensors barely compress.
A build that is not finished (`processed` plus the excluded ids below `total`, or a missing sample) is refused.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import torch

from experiments.team_pipelines.build_b import resolve_dir
from experiments.team_pipelines.train_b import UNCAPPED_BLOCKS

VERSION = 'v3'
SAMPLERS = ('signal', 'callwalk')
SAMPLER_TEXT = {
    'signal': 'the blocks of whole functions that call imports, highest signal first, up to the budget (several functions, so the '
              'result is not one connected subgraph)',
    'callwalk': 'a walk that grows from a seeded random block over the CFG edges and the call edges in both directions, with seeded '
                'restarts when the reachable region is smaller than the budget; the saved edges are the CFG edges among the chosen '
                'blocks',
}
INDEX_FIELDS = ('sample_id', 'label', 'family', 'arch', 'fold', 'n_instructions', 'match_rate', 'n_blocks_total', 'n_blocks_kept')
BASE = 'processed/cfg_subgraphs/'


def zip_name(settings):
    """`Mendeley_x86_cfg_subgraphs_<signal|callwalk>_<N>blocks_v3.zip`, or the uncapped name when the budget is 0 or >= UNCAPPED_BLOCKS."""
    sampler, blocks = settings.get('sample'), settings.get('num_blocks')
    if sampler not in SAMPLERS:
        raise SystemExit(f'the build\'s sampler is {sampler}: only {" and ".join(SAMPLERS)} builds are packaged')
    if not isinstance(blocks, int) or blocks < 0:
        raise SystemExit(f'the build\'s num_blocks is {blocks}')
    if blocks == 0 or blocks >= UNCAPPED_BLOCKS:
        return f'Mendeley_x86_cfg_subgraphs_uncapped_{VERSION}.zip'
    return f'Mendeley_x86_cfg_subgraphs_{sampler}_{blocks}blocks_{VERSION}.zip'


def read_build(build):
    """(settings, summary, sample paths) of a finished build folder; SystemExit when a file is missing or the build is not finished."""
    build = Path(build)
    for name in ('settings.json', 'build_summary.json', 'build_log.csv', 'api_vocab.json', 'channel_embedder.pt'):
        if not (build / name).is_file():
            raise SystemExit(f'{build} is no finished build_b folder: {name} is missing')
    settings = json.loads((build / 'settings.json').read_text(encoding='utf-8'))
    summary = json.loads((build / 'build_summary.json').read_text(encoding='utf-8'))
    excluded = summary.get('excluded') or {}
    if summary['processed'] + len(excluded) != summary['total']:
        raise SystemExit(f"{build} is not finished: {summary['processed']} of {summary['total']} processed, {len(excluded)} excluded")
    paths = sorted(p for p in build.glob('*.pt') if p.name != 'channel_embedder.pt')
    if len(paths) != summary['processed']:
        raise SystemExit(f"{build} holds {len(paths)} sample files, its summary says {summary['processed']} processed")
    return settings, summary, paths


def index_rows(build, paths):
    """One row per sample (`INDEX_FIELDS`): label, family, arch, fold, instructions, match rate (the share of the selected instructions
    paired with a listing line), blocks in the binary and blocks kept (from the build log)."""
    with (Path(build) / 'build_log.csv').open(encoding='utf-8', newline='') as f:
        log = {r['sample_id']: r for r in csv.DictReader(f)}
    rows = []
    for p in paths:
        payload = torch.load(p, weights_only=False, map_location='cpu')
        meta = payload.get('dataset_metadata', {})
        matched = payload['instruction_matched'].float()
        row = log.get(p.stem, {})
        rows.append({'sample_id': p.stem, 'label': int(payload['label']), 'family': meta.get('family', ''), 'arch': meta.get('arch', ''),
                     'fold': meta.get('fold', ''), 'n_instructions': int(matched.numel()),
                     'match_rate': round(float(matched.mean()) if matched.numel() else 0.0, 6),
                     'n_blocks_total': row.get('n_blocks_total', ''), 'n_blocks_kept': row.get('n_blocks_kept', '')})
    return rows


def readme_text(settings, summary, rows, manifests=True):
    """The README of the zip, from the build's settings.json and build_summary.json and the index rows."""
    sampler, blocks, uncapped = settings['sample'], settings['num_blocks'], zip_name(settings).endswith(f'uncapped_{VERSION}.zip')
    ransomware = sum(r['label'] == 1 for r in rows)
    budget = ('no block budget: every non-library block of every binary, in canonical block order. With no budget the callwalk '
              'restarts until every allowed block is chosen, so this one archive is the uncapped build of both `--sample signal` and '
              '`--sample callwalk`' if uncapped else f'`--sample {sampler}`, budget {blocks} blocks')
    sampling = '' if uncapped else f'\n`{sampler}` is {SAMPLER_TEXT[sampler]}.\n'
    pins = settings.get('pins') or {}
    pinned = ''.join(f'\n- `{name}`: sha256 `{digest}`' for name, digest in sorted(pins.items()))
    rows_note = (' The API rows the embedder added were drawn per (seed, API id) (`--seeded-rows`), so the table does not depend on '
                 'a pause and resume of the build.' if settings.get('embedder_rows') == 'seeded-per-id' else '')
    excluded = summary.get('excluded') or {}
    manifest_line = 'CT_GAT/processed/cfg_subgraphs/<sha256>.manifest.json   per-sample alignment report\n' if manifests else ''
    left_out = ('' if manifests else '\n(the per-sample manifest json files, alignment diagnostics that the models do not read, are '
                                      'left out of this archive; available on request)\n')
    return f"""# Mendeley x86 CFG subgraphs ({'uncapped' if uncapped else f'{sampler}, {blocks} blocks'}), pipeline B of the team pipelines

Unzip this archive at `CT_GAT/` of the repository. It creates exactly what `python -m s02_preprocess.preprocess_full` writes by
default, so the loaders and Janice's model (`s03a_transformer/five_channel`,
`python -m experiments.train --model fivechannel --build CT_GAT/processed/cfg_subgraphs`) read it unchanged:

```
CT_GAT/processed/api_vocab.json              API vocabulary (ids >= FIRST_API_ID)
CT_GAT/processed/cfg_subgraphs/<sha256>.pt   one sample per binary ({summary['processed']})
{manifest_line}CT_GAT/processed/cfg_subgraphs/channel_embedder.pt      the seeded five-channel embedder
CT_GAT/processed/cfg_subgraphs/build_info/   index.csv, build_settings.json, build_summary.json, build_log.csv (not read by the code)
```
{left_out}
What is in a sample: `instruction_ids [N,5]` (opcode, operand 1, operand 2, API id, behaviour), `insn_to_block`,
`instruction_matched`, `block_features [n,77]`, `block_embeddings [n,64]`, `edge_index`, `edge_type`, `edge_attr`, `edge_back`,
`block_keys`, `label`, `dataset_metadata` (sample_id, label, arch, family, fold), and the rest of the keys `save_sample` writes.

How it was built: the {summary['total']} Mendeley x86 binaries of our 5-fold family-holdout set ({ransomware} ransomware,
{len(rows) - ransomware} goodware in this archive{f', {len(excluded)} excluded on purpose' if excluded else ', no exclusions'}), {budget}, seed {settings.get('seed', 0)}.
{sampling}Library functions are used only when a binary has no other code (they are collapsed in the IDA listing and cannot be matched).
CFGs come from the Mendeley bundle, listings from the Mendeley IDA export. Code: source `{settings.get('source', 'unknown')}`{', files of the repository when built:' if pins else ''}{pinned}

Before training: `api_vocab.json` was grown over all binaries in one pass.{rows_note} Map API ids that do not occur in your training
binaries to `UNK_API` per fold (`--model fivechannel` and our team trainers do this), or rebuild the vocabulary from your training
split (`--split train`, then the other splits frozen).

`build_info/index.csv` lists, per binary: label, family, arch, our kfold fold, number of instructions, match rate (share of selected
instructions paired with a listing line), blocks in the binary and blocks kept.
"""


def index_csv(rows):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=INDEX_FIELDS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            digest.update(chunk)
    return digest.hexdigest()


def package(build, out, manifests=True, deflate_pt=False):
    """Write the zip, `<zip>.sha256` and `<zip stem>.README.md` into `out`; returns (zip path, sha256)."""
    build, out = Path(build), Path(out)
    settings, summary, paths = read_build(build)
    name = zip_name(settings)
    rows = index_rows(build, paths)
    readme = readme_text(settings, summary, rows, manifests)
    out.mkdir(parents=True, exist_ok=True)
    zpath = out / name
    part = zpath.with_name(zpath.name + '.part')
    stored = zipfile.ZIP_DEFLATED if deflate_pt else zipfile.ZIP_STORED
    deflated = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(part, 'w', allowZip64=True) as z:
        z.writestr(f'README_{zpath.stem}.md', readme, compress_type=deflated)
        z.write(build / 'api_vocab.json', 'processed/api_vocab.json', compress_type=deflated)
        z.write(build / 'channel_embedder.pt', BASE + 'channel_embedder.pt', compress_type=deflated)
        z.writestr(BASE + 'build_info/index.csv', index_csv(rows), compress_type=deflated)
        for source, target in (('settings.json', 'build_settings.json'), ('build_summary.json', 'build_summary.json'),
                               ('build_log.csv', 'build_log.csv')):
            z.write(build / source, BASE + 'build_info/' + target, compress_type=deflated)
        for p in paths:
            z.write(p, BASE + p.name, compress_type=stored, compresslevel=1 if deflate_pt else None)
            if manifests:
                z.write(p.with_suffix('.manifest.json'), BASE + p.stem + '.manifest.json', compress_type=deflated)
    os.replace(part, zpath)
    digest = sha256_of(zpath)
    (out / f'{name}.sha256').write_bytes(f'{digest}  {name}\n'.encode('utf-8'))   # bytes, not write_text: on Windows that writes CRLF
    (out / f'{zpath.stem}.README.md').write_bytes(readme.encode('utf-8'))         # and `sha256sum -c` on macOS and Linux fails on it
    return zpath, digest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('build', type=Path, help='a finished build_b folder (a leading RUNS/ is the run directory)')
    ap.add_argument('--out', type=Path, default=Path('.'), help='where the zip, its .sha256 and the README go (default: the current directory)')
    ap.add_argument('--no-manifests', action='store_true', help='leave the per-sample manifest json files out of the zip')
    ap.add_argument('--deflate-pt', action='store_true', help='compress the .pt files too (default: stored)')
    args = ap.parse_args(argv)
    zpath, digest = package(resolve_dir(args.build), args.out, not args.no_manifests, args.deflate_pt)
    print(f'{zpath} {zpath.stat().st_size} bytes sha256 {digest}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
