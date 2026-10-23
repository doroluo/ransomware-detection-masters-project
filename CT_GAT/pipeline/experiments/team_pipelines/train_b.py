"""Pipeline B trainer: the teammates' FullBlockGAT (train_full_gat recipe) on the build_b.py subgraphs, our fold manifests.

python -m experiments.team_pipelines.train_b --fold 0 --seed 0 --arm <arm> [--features all|feat|emb|e2e] [--block-len 64] [--source LABEL]
       [--protocol kfold|kfold_v3 --arch x86|all|matched] [--arch-balance] [--build DIR] [--smoke N] [--device cuda|cpu]
       (run from CT_GAT/pipeline; reads the build directory and never writes there)

Arms (`ARM_TABLE`, one row per arm: sampler, block budget, node input, the source labels its build may record, the default build
directory, the hyperparameters; a new arm is a new row).

  sig500, sig500v2   the first two builds `RUNS/team/b_sig500` (source gnn-331ce0b, the teammates' preprocess) and `b_sig500_v2`
                     (source main-c7899f0, Yanping's streaming preprocess). They need the pinned teammate worktrees
                     (GT_TEAM_GNN_ROOT, GT_TEAM_MAIN_ROOT): the embedder and the pooling come from the files that built them.
  sig500v3[-feat|-emb]   `RUNS/team/b_sig500_v3` (500 blocks, source main-fix: the `endp` and library-function fixes of the s02
                     preprocess, merged into main; a build of this repository's own files records `repo`, and both labels are
                     accepted). They differ in the node input only: `all` the 141-d concatenation (64 pooled embedding + 77
                     handcrafted block features), `feat` the 77 scaled block features (Model 2: block features + CFG), `emb` the
                     64-d pooled channel embedding (Model 3: block embeddings + CFG). Edges and edge attributes (edge_dim 8) are
                     the same in the three.
  sig500v3-e2e       Model 3 trained end to end on `b_sig500_v3` (input `e2e`): Janice's `Model1TokenSequence` encodes the first
                     `--block-len` (64, the only value) instructions of every block from the build's five-channel ids, and
                     Dorothy's `FullBlockGAT` (in_dim 64, edge_dim 8) combines the block vectors over the CFG; both are trained
                     together (`s03c_cfg_transformer/block_transformer.py`). The stored block embeddings and the 77 block features
                     are not read. Recipe `HP_E2E`: the others' (AdamW weight decay 1e-4, batch 16, 40 epochs, patience 8) with lr 3e-4
                     instead of 1e-3, plus gradient clipping at 1.0 and the 12 GB GPU cap.
  <build>-m1frozen   Model 3 on a frozen trained Model 1 (sig500v3, sig250, callwalk500, callwalk250; input `m1frozen`): the
                     best_auc.pt of the per-block Model 1 of the same fold, seed, dataset and weighting (`fivechannel --per-block
                     --max-instructions 64`, or --model1-run DIR) encodes the first 64 instructions of every block once, its
                     weights never change, and only `FullBlockGAT` (in_dim 64) is trained on the block vectors (recipe `HP`). The
                     run refuses a Model 1 whose manifest.json is not this run's train / val / test (`check_same_splits`), and one
                     that trained on a build with another api_vocab.json than the arm's (`check_same_api_vocab`: API ids are
                     numbered per build, so a Model 1 of another build gets the wrong embedding rows). The run name does not
                     say which build the Model 1 used: train it with `--build` the arm's build. The guard finds that build in the `build_dir` of the
                     Model 1's metrics.json, else in RUNS/team/<its folder name> (a path of the training machine is not on yours),
                     else takes `--model1-build DIR`, which overrides both.
  sig250[-feat|-emb|-e2e], callwalk500[-feat|-emb|-e2e], callwalk250[-feat|-emb|-e2e]
                     `RUNS/team/b_sig250_main`, `b_callwalk500_main` and `b_callwalk250_main`: builds of this repository's own files
                     (`build_b --source repo`) with `--sample signal --num-blocks 250`, `--sample callwalk --num-blocks 500` and
                     `--sample callwalk --num-blocks 250` (callwalk: `s02_preprocess/README.md`, a walk across the calls). The three
                     inputs of `sig500v3` (`-feat` Model 2, `-emb` Model 3) and the end-to-end Model 3 (`-e2e`, recipe `HP_E2E`);
                     there is no `all` arm for them. A callwalk arm refuses a build of the other sampler, a 250 arm one of 500 blocks.
  sigwhole[-feat|-emb]   `RUNS/team/b_sigwhole_v3` (`build_b --sample signal --num-blocks 1000000`: every non-library block of every
                     binary, no cap), the same three inputs. An uncapped callwalk build is the same set of blocks (with no budget
                     the walk restarts until every allowed block is chosen), so this build serves both samplers. They train with a node budget (`HP_WHOLE`, fit.py): a binary has up to
                     190k blocks and a batch of 16 can hold over a million nodes, which does not fit the GPU, so a batch above the
                     budget is cut in order into parts whose gradients are summed. Because the build is 14 GB of tensors, `run`
                     reads each payload twice instead of holding them all: once for the API ids it uses (and the one-sample re-pool
                     check), once to build its graph, which is all it keeps.

The dataset is a flag, not an arm. `--protocol kfold --arch x86` (the defaults) is the 1,810 Mendeley x86 binaries and everything above;
`--protocol kfold --arch all` the 2,411 Mendeley binaries (x86 and x64) and `--protocol kfold_v3 --arch all|matched` the combined
dataset (5,652 binaries with a v3 fold; `matched`: the 4,850 of them in `arch_matched.csv`, trained on the `all` build). The flag is the
literal `--protocol`, so that the cache is resolved from it as in every command (`common/paths.py`). The folds, the validation carve, the
run name and the build all follow: `kfold{f}_all_gnn_s{seed}_team-<arm>`, `v3kfold{f}_matched_gnn_s{seed}_team-<arm>`, ... Vocabulary
clash: `sig500v3` is the s02-fix *build* version, `kfold_v3` the fold *protocol*, `_v3.zip` the preprocessing version of the zips.
The legacy arms (`sig500`, `sig500v2`) read the x86 builds of the pinned worktrees and run on kfold x86 only; the uncapped arms
(`sigwhole*`) on kfold only. A run refuses a build made for another dataset, in both directions (`check_build`): the build's
`settings.json` names `protocol` and `arch` (a kfold x86 build records neither), `all` serves an `all` or `matched` run, and an
`x86` run needs an x86 build, because a build that grew over more binaries has another API vocabulary and embedder.
--arch-balance (needs --arch all or matched) weights every training binary by its (class, architecture) cell, N / (C n_cell) over the
non-empty cells of the fold's training manifest, instead of the class weight (fit.py); the run is named `...gnn-ab_s...`, its
metrics.json `recipe` has `arch_balance` and the four cell weights. Selection stays the teammates' (ransomware F1 at 0.5).

--build DIR overrides the directory. Without it the arm's default directory is used when it exists, else the `team_b_build` source
(`CT_GAT/processed/cfg_subgraphs`, where the preprocessed zips unpack). The default directory is `RUNS/team/b_<family>` for kfold
x86 (`ARM_TABLE`), `RUNS/team/b_<family>_mendeley_all` for kfold all and `RUNS/team/b_<family>_combined_v3` for kfold_v3
(`default_build_dir`; <family> is the arm without `-feat|-emb|-e2e` and without the `v3` of `sig500v3`). The build is read in either of its two layouts
(`build_layout`): a `build_b` folder, or an unzipped Drive zip (`build_info/build_settings.json`, `build_info/build_log.csv`,
`checkpoint.json` when the zip has one, else the ids are the `<sha256>.pt` stems, `api_vocab.json` in the folder or one level up,
else the embedder is sized from `channel_embedder.pt`).

The build's settings.json names its sampler, block budget and source. A run refuses a build that contradicts the arm in any of the
three (the message names the three values); the two first arms skip the keys their builds never wrote (the first has no `source`:
it is gnn-331ce0b). `--features` and `--source` are refused when they contradict the arm. The code that embeds and pools is chosen by
the build's label (`common.SOURCE_CODE`): `main-fix` and `repo` run this repository's `s02_preprocess` files, the older labels
their pinned worktrees. `main-fix` is an alias: the embedder and pooling functions this trainer calls are the same in the files
of 145ed35 and in the repository's, and `check_repool` below (a stored sample re-pooled with the stored ids must reproduce its
block embeddings) runs at the start of every run. metrics.json records `build_source` (the label), `build_pins` (the pins in
the build's settings.json, never those of the files now in the repository) and `code_pins` (the sha256 of the files this run
imported).

The graph loader, the model and the training recipe are Dorothy's, copied into `s03b_cfg_feature/gat` (no GT_TEAM_* variable).
Run names: `kfold{f}_x86_gnn_s{seed}_team-<arm>` (kfold x86; `common.team_config` names the others).

The build fixed one API vocabulary over all 1,810 binaries and pooled the block embeddings with the embedder it saved
(`channel_embedder.pt`). Per fold this script makes the vocabulary a training-set one again: every API id (>= FIRST_API_ID)
that no training binary of the fold uses becomes UNK_API in the saved `instruction_ids[:, 3]`, the instruction vectors are
re-embedded with `FiveChannelEmbedder` (the saved state) and re-pooled with `pool_block_embeddings`, and the three api-kind
columns of `block_features` (api_none / api_unresolved / api_known counts) are recomputed the way `build_model_inputs`
counts them. The other channels (opcode, operands, behaviour tag) are the build's: an API name outside the training set
keeps the behaviour tag it was given. Before any swap, one sample is re-pooled with the unchanged ids and must reproduce
its stored `block_embeddings` (1e-5) and api-kind columns (exactly); `feat` alone skips the re-pool of the embeddings, which
it never reads. The graphs are `graph_full_loader`'s: its `_node_features` (unit-length embedding, log / rate-scaled features)
and `_edge_attr` (one-hot relation + back-edge bit: edge_dim 8). Model `FullBlockGAT` with its defaults; recipe of
train_full_gat.py (AdamW lr 1e-3, weight decay 1e-4, class-weighted cross-entropy, batch 16, 40 epochs, patience 8 on the
validation ransomware F1 at 0.5), fit.py for the loop and both selection rules. Only ids listed in the build's checkpoint
are used (their embedder state is the one saved with it); a full run needs all 1,810, less the ids the build excluded
(`build_b --exclude`, read from the build's settings): those are left out of training, validation and test and recorded in
metrics.json (`excluded`, `excluded_reason`). --smoke N: N available training ids, 2 epochs, temp dir only.
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path, PureWindowsPath
from typing import NamedTuple, Optional

import torch

from common.config import RUNS
from common.manifests import run_command
from common.paths import MissingSource, source_for
from experiments.team_pipelines.build_b import dataset_of, load_vocab_embedder, resolve_dir
from experiments.team_pipelines.common import (DATASET_OF, DEFAULT_ARCH, DEFAULT_SOURCE, SOURCE_CODE, code_source, fold_manifest,
                                               import_team, sample_rows, source_label_ok, source_pins)
from experiments.team_pipelines.fit import run_arm, smoke_ids

UNCAPPED_BLOCKS = 1_000_000     # build_b --num-blocks of the uncapped build (every non-library block)
HP = {'lr': 1e-3, 'weight_decay': 1e-4, 'batch_size': 16, 'epochs': 40, 'patience': 8}
NODE_BUDGET = 300_000     # nodes per part of a batch: training ~28 KB per node (8.4 GB), evaluation ~15 KB (4.5 GB), under the 12 GB limit
GPU_CAP_GB = 12          # the allocator may reserve at most this much (fit.py: beyond the card the driver pages into system RAM)
HP_WHOLE = {**HP, 'node_budget': NODE_BUDGET, 'gpu_cap_gb': GPU_CAP_GB}
HP_E2E = {**HP, 'lr': 3e-4, 'grad_clip': 1.0, 'gpu_cap_gb': GPU_CAP_GB}      # post-LN transformer without warmup: clip, and lr 3e-4 (1e-3 swung the validation F1 by up to 0.24 per epoch on fold 0)
BLOCK_LEN = 64            # instructions per block the end-to-end arm reads (block_transformer.BLOCK_LEN)
TEAM_RUNS = RUNS / 'team'
B_DIR = RUNS / 'team' / 'b_sig500'
B_V2_DIR = RUNS / 'team' / 'b_sig500_v2'
B_V3_DIR = RUNS / 'team' / 'b_sig500_v3'
B_SIG250_DIR = RUNS / 'team' / 'b_sig250_main'
B_CALLWALK500_DIR = RUNS / 'team' / 'b_callwalk500_main'
B_CALLWALK250_DIR = RUNS / 'team' / 'b_callwalk250_main'
B_WHOLE_DIR = RUNS / 'team' / 'b_sigwhole_v3'
FEATURES = ('all', 'feat', 'emb')
E2E = 'e2e'               # the node input of the end-to-end arm: no graph features, the block's instructions (block_transformer.py)
M1FROZEN = 'm1frozen'     # the node input of the frozen arm: block vectors of the fold's trained per-block Model 1, not updated
INPUTS = FEATURES + (E2E, M1FROZEN)
FEATURE_MODE = {'all': 'both', 'feat': 'features', 'emb': 'embeddings'}      # graph_full_loader._node_features modes
KIND_COLUMNS = ('api_none_count', 'api_unresolved_count', 'api_known_count')
MAIN_FIX_LABELS = ('main-fix', 'repo')       # the builds of the fixed s02 preprocess: the pinned fix commit, or this repository's files
REPO_LABELS = ('repo',)                      # the builds made from this repository's own files (sig250, callwalk500, callwalk250)
RUN_ARCHES = {'kfold': ('x86', 'all'), 'kfold_v3': ('all', 'matched')}      # what a run can ask for: a build's arch (BUILD_ARCHES) and `matched`, a filter of the all build
BUILD_SUFFIX = {'kfold': 'mendeley_all', 'kfold_v3': 'combined_v3'}      # RUNS/team/b_<family>_<suffix>: the default build directory of a dataset other than kfold x86


class Arm(NamedTuple):
    sampler: str                  # build_b --sample
    num_blocks: Optional[int]     # the block budget; None: uncapped (the build's num_blocks is at least UNCAPPED_BLOCKS)
    features: str                 # the node input: one of INPUTS
    sources: tuple                # the labels a build's settings.json `source` may carry
    build: Path                   # the default build directory
    hp: dict
    legacy: bool = False          # the two first builds, whose settings.json lacks keys (the first has no `source`)


ARM_TABLE = {
    'sig500': Arm('signal', 500, 'all', ('gnn-331ce0b',), B_DIR, HP, True),
    'sig500v2': Arm('signal', 500, 'all', ('main-c7899f0',), B_V2_DIR, HP, True),
    'sig500v3': Arm('signal', 500, 'all', MAIN_FIX_LABELS, B_V3_DIR, HP),
    'sig500v3-feat': Arm('signal', 500, 'feat', MAIN_FIX_LABELS, B_V3_DIR, HP),
    'sig500v3-emb': Arm('signal', 500, 'emb', MAIN_FIX_LABELS, B_V3_DIR, HP),
    'sig500v3-e2e': Arm('signal', 500, E2E, MAIN_FIX_LABELS, B_V3_DIR, HP_E2E),
    'sig500v3-m1frozen': Arm('signal', 500, M1FROZEN, MAIN_FIX_LABELS, B_V3_DIR, HP),
    'sig250-feat': Arm('signal', 250, 'feat', REPO_LABELS, B_SIG250_DIR, HP),
    'sig250-emb': Arm('signal', 250, 'emb', REPO_LABELS, B_SIG250_DIR, HP),
    'sig250-e2e': Arm('signal', 250, E2E, REPO_LABELS, B_SIG250_DIR, HP_E2E),
    'sig250-m1frozen': Arm('signal', 250, M1FROZEN, REPO_LABELS, B_SIG250_DIR, HP),
    'callwalk500-feat': Arm('callwalk', 500, 'feat', REPO_LABELS, B_CALLWALK500_DIR, HP),
    'callwalk500-emb': Arm('callwalk', 500, 'emb', REPO_LABELS, B_CALLWALK500_DIR, HP),
    'callwalk500-e2e': Arm('callwalk', 500, E2E, REPO_LABELS, B_CALLWALK500_DIR, HP_E2E),
    'callwalk500-m1frozen': Arm('callwalk', 500, M1FROZEN, REPO_LABELS, B_CALLWALK500_DIR, HP),
    'callwalk250-feat': Arm('callwalk', 250, 'feat', REPO_LABELS, B_CALLWALK250_DIR, HP),
    'callwalk250-emb': Arm('callwalk', 250, 'emb', REPO_LABELS, B_CALLWALK250_DIR, HP),
    'callwalk250-e2e': Arm('callwalk', 250, E2E, REPO_LABELS, B_CALLWALK250_DIR, HP_E2E),
    'callwalk250-m1frozen': Arm('callwalk', 250, M1FROZEN, REPO_LABELS, B_CALLWALK250_DIR, HP),
    'sigwhole': Arm('signal', None, 'all', MAIN_FIX_LABELS, B_WHOLE_DIR, HP_WHOLE),
    'sigwhole-feat': Arm('signal', None, 'feat', MAIN_FIX_LABELS, B_WHOLE_DIR, HP_WHOLE),
    'sigwhole-emb': Arm('signal', None, 'emb', MAIN_FIX_LABELS, B_WHOLE_DIR, HP_WHOLE),
}
ARMS = tuple(ARM_TABLE)
ARM_DIR = {arm: row.build for arm, row in ARM_TABLE.items()}
ARM_FEATURES = {arm: row.features for arm, row in ARM_TABLE.items()}
ARM_HP = {arm: row.hp for arm, row in ARM_TABLE.items()}


class Layout(NamedTuple):
    """Where the parts of a build are: `settings` (settings.json or build_info/build_settings.json), `build_log`, `checkpoint`
    (checkpoint.json, None when the build has none: its ids are then the sample files) and `vocab` (api_vocab.json, None when
    there is none)."""
    kind: str
    settings: Optional[Path]
    build_log: Path
    checkpoint: Optional[Path]
    vocab: Optional[Path]


def build_layout(b_dir):
    """The `Layout` of a build directory: a `build_b` folder (settings.json, build_log.csv, checkpoint.json, api_vocab.json in
    it), or an unzipped Drive zip, which has `build_info/build_settings.json` and no settings.json (`build_info/build_log.csv`,
    `checkpoint.json` when the zip has one, `api_vocab.json` in the folder or one level up)."""
    b_dir = Path(b_dir)
    info = b_dir / 'build_info'
    if not (b_dir / 'settings.json').is_file() and (info / 'build_settings.json').is_file():
        vocab = next((p for p in (b_dir / 'api_vocab.json', b_dir.parent / 'api_vocab.json') if p.is_file()), None)
        ckpt = b_dir / 'checkpoint.json'
        return Layout('zip', info / 'build_settings.json', info / 'build_log.csv', ckpt if ckpt.is_file() else None, vocab)
    settings = b_dir / 'settings.json'
    vocab = b_dir / 'api_vocab.json'
    return Layout('build', settings if settings.is_file() else None, b_dir / 'build_log.csv', b_dir / 'checkpoint.json',
                  vocab if vocab.is_file() else None)


def dataset_error(protocol, arch):
    """None when (protocol, arch) is a dataset the team trainers run on (`RUN_ARCHES`), else the reason."""
    if protocol not in RUN_ARCHES:
        return f'protocol {protocol!r}: one of {", ".join(RUN_ARCHES)}'
    if arch not in RUN_ARCHES[protocol]:
        return f'--protocol {protocol} runs on --arch {" or ".join(RUN_ARCHES[protocol])}, not {arch!r}'
    return None


def build_arch(arch):
    """The `arch` of the build_b folder that a run of `arch` reads: `matched` is a training-time filter of the all build."""
    return 'all' if arch == 'matched' else arch


def check_arm_dataset(arm, protocol='kfold', arch='x86', arch_balance=False):
    """SystemExit when `arm` cannot run on the dataset (protocol, arch): a legacy arm reads the x86 build of the pinned worktrees,
    an uncapped arm has no build on the combined dataset, --arch-balance needs both architectures."""
    error = dataset_error(protocol, arch)
    if error:
        raise SystemExit(error)
    row = ARM_TABLE[arm]
    if row.legacy and (protocol, arch) != ('kfold', 'x86'):
        raise SystemExit(f'arm {arm} reads the x86 build {row.build.name}; use sig500v3-* on another dataset')
    if row.num_blocks is None and protocol != 'kfold':
        raise SystemExit(f'arm {arm} reads an uncapped build, which is made for kfold only (a binary of the combined dataset has up to '
                         f'8M blocks: no GPU holds it)')
    if arch_balance and arch == 'x86':
        raise SystemExit('--arch-balance weights the (class, architecture) cells: it needs --arch all or matched')


def build_family(arm):
    """The family of an arm in the name of its build directory: the arm without `-feat|-emb|-e2e` and without the `v3` of
    `sig500v3` (the s02-fix version is implied by the `repo` source): sig500v3-emb -> sig500, callwalk250-e2e -> callwalk250."""
    return re.sub(r'-(feat|emb|e2e|m1frozen)$', '', arm).replace('sig500v3', 'sig500')


def default_build_dir(arm, protocol='kfold', arch='x86'):
    """The arm's default directory when it exists, else the `team_b_build` source (`CT_GAT/processed/cfg_subgraphs`, where the
    preprocessed zips unpack); a message naming both when neither is a directory. The arm's directory is `ARM_DIR[arm]` for kfold
    x86 and `RUNS/team/b_<family>_mendeley_all` / `b_<family>_combined_v3` for the other datasets (an `all` build also serves
    `matched`)."""
    default = ARM_DIR[arm] if (protocol, arch) == ('kfold', 'x86') else TEAM_RUNS / f'b_{build_family(arm)}_{BUILD_SUFFIX[protocol]}'
    if default.is_dir():
        return default
    try:
        unzipped = source_for('team_b_build')
    except MissingSource:
        unzipped = None
    if unzipped is not None and Path(unzipped).is_dir():
        return Path(unzipped)
    raise SystemExit(f'arm {arm}: {default} does not exist and neither does the team_b_build source ({unzipped}): '
                     f'pass --build DIR (a build_b folder or an unzipped preprocessed zip)')


def read_settings(b_dir=B_DIR):
    path = build_layout(b_dir).settings
    return json.loads(path.read_text(encoding='utf-8')) if path is not None else {}


def read_source(b_dir=B_DIR):
    """The source label a build recorded (its settings `source`; the first build has none: gnn-331ce0b)."""
    source = read_settings(b_dir).get('source', DEFAULT_SOURCE)
    if not source_label_ok(source):
        raise SystemExit(f'{b_dir} (settings) names the unknown source {source}')
    return source


def done_ids(b_dir):
    """The ids a build covers: its checkpoint.json `done`, else (an unzipped Drive zip) the stems of its sample files."""
    layout = build_layout(b_dir)
    if layout.checkpoint is not None:
        return set(json.loads(layout.checkpoint.read_text(encoding='utf-8'))['done'])
    return {p.stem for p in Path(b_dir).glob('*.pt') if len(p.stem) == 64}


def build_state(b_dir=B_DIR):
    """(pp, embedder, done ids): the `preprocess_full` that embeds and pools for the build (`common.SOURCE_CODE` of its label),
    its FiveChannelEmbedder state and the ids it covers. The ids are read before the embedder: the embedder only ever gains rows,
    so a newer file is a superset of the one the checkpoint was written with."""
    layout = build_layout(b_dir)
    pp = import_team('preprocess_full', source=code_source(read_source(b_dir)))
    done = done_ids(b_dir)
    _, embedder = load_vocab_embedder(b_dir, pp, layout.vocab)
    embedder.eval()
    return pp, embedder, done


def read_excluded(b_dir=B_DIR):
    """{sample_id: reason} of the ids the build left out on purpose (its settings); {} when none or no file."""
    return dict(read_settings(b_dir).get('excluded', {}))


def check_source_flag(arm, source):
    """SystemExit when the --source flag names a label the arm does not accept."""
    if source is not None and source not in ARM_TABLE[arm].sources:
        raise SystemExit(f'arm {arm} reads builds of {" or ".join(ARM_TABLE[arm].sources)}, not --source {source}')


def check_build_dataset(b_dir, protocol='kfold', arch='x86'):
    """SystemExit unless `b_dir` was built for the dataset of the run: its settings name `protocol` and `arch` (a build without the
    keys is a kfold x86 build, `build_b.dataset_of`), the protocol must be the run's and the arch the one the run reads (`x86` for
    x86, `all` for all and matched). Refused in both directions: an x86 run on an all build trains on another vocabulary."""
    have = dataset_of(read_settings(b_dir))
    if have != (protocol, build_arch(arch)):
        hint = ' (matched is a training-time filter of the all build: build_b does not make it)' if have[1] == 'matched' else \
            f' (a {protocol} {arch} run reads the {protocol} {build_arch(arch)} build)' if arch == 'matched' else ''
        raise SystemExit(f'{b_dir} was built for {have[0]} {have[1]}, this run asks for {protocol} {arch}{hint}')


def check_build(arm, b_dir, source=None, protocol='kfold', arch='x86'):
    """The source label of `b_dir` when its settings agree with `arm` in sampler, block budget and source label; SystemExit
    with the three values of both sides otherwise. `source` (the --source flag) must be a label the arm accepts and the one the
    build recorded. The build must also be one of the run's dataset (`check_build_dataset`)."""
    row = ARM_TABLE[arm]
    check_arm_dataset(arm, protocol, arch)
    check_source_flag(arm, source)
    check_build_dataset(b_dir, protocol, arch)
    settings = read_settings(b_dir)
    label = read_source(b_dir)
    sampler, blocks = settings.get('sample'), settings.get('num_blocks')
    want = f'sampler {row.sampler}, num_blocks {row.num_blocks if row.num_blocks is not None else f">= {UNCAPPED_BLOCKS} (uncapped)"}, ' \
           f'source {" or ".join(row.sources)}'
    have = f'{label}, sampler {sampler}, num_blocks {blocks}'
    if row.legacy:
        blocks_ok = blocks is None or blocks == row.num_blocks
        sampler_ok = sampler is None or sampler == row.sampler
    else:
        blocks_ok = blocks is not None and (blocks >= UNCAPPED_BLOCKS if row.num_blocks is None else blocks == row.num_blocks)
        sampler_ok = sampler == row.sampler
    if label not in row.sources or not blocks_ok or not sampler_ok:
        raise SystemExit(f'arm {arm} reads builds with {want}; {b_dir} was built with {have}')
    if source is not None and source != label:
        raise SystemExit(f'{b_dir} was built with {label}, not --source {source}')
    return label


def build_pins(b_dir, label):
    """The pins the build recorded in its settings; for a build that recorded none (the first) the pins of its source."""
    return dict(read_settings(b_dir).get('pins') or source_pins(code_source(label)))


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


def load_slim(b_dir, sample_id, protocol='kfold', arch='x86'):
    """The slim payload of one sample of a build; its label is checked against the cache index of the build's dataset (a matched
    run reads the all build)."""
    payload = torch.load(Path(b_dir) / f'{sample_id}.pt', map_location='cpu', weights_only=False)
    if payload['file_id'] != sample_id or int(payload['label']) != sample_rows(protocol, build_arch(arch))[sample_id]['label']:
        raise ValueError(f'{sample_id}.pt holds sample {payload["file_id"]} / label {payload["label"]}')
    return slim(payload)


def payload_api_ids(tok, payload):
    """Sorted tensor of the API ids (>= FIRST_API_ID) that occur in one payload."""
    api = payload['instruction_ids'][:, 3]
    return torch.unique(api[api >= tok.FIRST_API_ID])


def train_api_ids(tok, payloads, train_ids):
    """Sorted tensor of the API ids (>= FIRST_API_ID) that occur in the training binaries."""
    seen = set()
    for i in train_ids:
        seen.update(payload_api_ids(tok, payloads[i]).tolist())
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
    with build_layout(b_dir).build_log.open(encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            if r['status'] == 'ok':
                out[r['sample_id']] = (int(r['n_blocks_total']), int(r['n_blocks_kept']))
    return out


def check_block_len(arm, block_len):
    """SystemExit when --block-len names a value the arm does not train with (the end-to-end arm reads BLOCK_LEN instructions per
    block; the other arms read no instructions)."""
    if block_len is None:
        return
    if ARM_FEATURES[arm] != E2E:
        raise SystemExit(f'arm {arm} reads no instructions, --block-len {block_len} does not apply (the arm names the input)')
    if block_len != BLOCK_LEN:
        raise SystemExit(f'arm {arm} trains with --block-len {BLOCK_LEN}, not {block_len} (the arm names the input; another length '
                         f'is another arm)')


def e2e_extras(graphs, test_ids, block_len, api_rows):
    """The metrics.json entries of the end-to-end arm: parameters, block length, the blocks cut at it and the instructions per
    binary in the test split, and the names of the encoder and the GAT."""
    from s03c_cfg_transformer import block_transformer as bt
    test = [graphs[i] for i in test_ids]
    n_blocks = sum(int(g.num_nodes) for g in test)
    cut = sum(g.n_truncated for g in test)
    return {'n_parameters': bt.Model3EndToEnd(api_rows, block_len).n_parameters(), 'block_len': block_len,
            'truncated_blocks': {'test_count': cut, 'test_blocks': n_blocks, 'test_share': cut / max(n_blocks, 1)},
            'tokens_per_binary': sum(g.n_tokens for g in test) / max(len(test), 1),
            'instructions_per_binary': sum(g.n_instructions for g in test) / max(len(test), 1),
            'encoder': bt.ENCODER, 'gat': bt.GAT}


def model1_run_dir(fold, seed, protocol='kfold', arch='x86', arch_balance=False):
    """RUNS/<name> of the per-block Model 1 (`fivechannel --per-block --max-instructions BLOCK_LEN`) of the same fold, seed,
    dataset and loss weighting."""
    from common.config import RunConfig
    cfg = RunConfig(protocol=protocol, fold=fold, arch=arch, seed=seed, model='fivechannel', max_instructions=BLOCK_LEN,
                    per_block=True, arch_balance=arch_balance)
    return RUNS / cfg.name


def check_same_splits(model1_dir, manifest):
    """SystemExit unless the Model 1 run's manifest.json holds exactly the train, val and test ids of `manifest`."""
    path = Path(model1_dir) / 'manifest.json'
    if not path.is_file():
        raise SystemExit(f'{model1_dir} has no manifest.json (an unfinished Model 1 run)')
    theirs = json.loads(path.read_text(encoding='utf-8'))
    for split in ('train', 'val', 'test'):
        ours = sorted(r['sample_id'] for r in getattr(manifest, split))
        if sorted(theirs[split]) != ours:
            raise SystemExit(f'{model1_dir}: its {split} split ({len(theirs[split])} ids) is not this run\'s ({len(ours)} ids); '
                             f'the frozen encoder must come from the same fold and seed')


def locate_model1_build(model1_dir, explicit=None):
    """The build folder to compare with the arm's: `explicit` (`--model1-build`, a leading RUNS/ is the run directory) when given, else the
    `build_dir` of the Model 1 run's metrics.json (an absolute path on the machine that trained it, a leading RUNS/ the run directory)
    when that folder exists here, else `TEAM_RUNS/<its last path component>` (where the root README has a teammate place the unzipped
    build; a Windows or POSIX path both give the name). SystemExit when the run records no build_dir and none is given, or when
    neither folder exists, naming the recorded path, the looked-up one and `--model1-build DIR`."""
    if explicit:
        return resolve_dir(Path(explicit))
    path = Path(model1_dir) / 'metrics.json'
    recorded = json.loads(path.read_text(encoding='utf-8')).get('build_dir') if path.is_file() else None
    if not recorded:
        raise SystemExit(f"{model1_dir} does not record the build it trained on (no build_dir in its metrics.json): the frozen "
                         f"encoder's API embedding cannot be matched to this arm's build; name the Model 1's build with "
                         f"--model1-build DIR, or train the Model 1 again (python -m experiments.train --model fivechannel "
                         f"--per-block --max-instructions {BLOCK_LEN} --build <the arm's build> ...)")
    found = resolve_dir(Path(recorded))
    if found.is_dir():
        return found
    here = TEAM_RUNS / PureWindowsPath(recorded).name
    if here.is_dir():
        return here
    raise SystemExit(f"{model1_dir}: the Model 1 build {recorded} has no api_vocab.json (or is not a directory on this machine), and "
                     f"{here} is not a directory either; place the unzipped build at {here} or name its folder with "
                     f"--model1-build DIR")


def api_names(build, role, model1_dir):
    """The API names of the build's api_vocab.json (the position of a name is its id minus `tok.FIRST_API_ID`). SystemExit when the
    build has no vocabulary file: nothing can be compared."""
    from s03a_transformer.five_channel.model import tok
    path = build_layout(build).vocab if Path(build).is_dir() else None
    if path is None:
        raise SystemExit(f"{model1_dir}: the {role} build {build} has no api_vocab.json (or is not a directory on this machine), so "
                         f"its API ids cannot be compared with the other build's")
    return list(tok.load_api_vocab(path).names)


def check_same_api_vocab(model1_dir, b_dir, model1_build=None):
    """SystemExit unless the Model 1 run trained on a build whose api_vocab.json is that of `b_dir`, the arm's build: the same names
    in the same order. API ids are numbered per build (the position in api_vocab.json), so a Model 1 of another build maps an id to
    another API name and the frozen encoder embeds the wrong rows without an error (b_sig250_main has 4.6% of the names it shares with
    b_sig500_v3 at the same position, b_callwalk500_main 19.6%). A vocabulary that only extends the Model 1's (`build_b --extend`
    appends names) is refused too: the old ids agree, but the arm's new ids have no row in the Model 1's embedding, and an extended
    build is another dataset, whose folds `check_same_splits` already refuses. Returns (the Model 1's build, the API rows that the
    arm's build can hold: FIRST_API_ID + its names)."""
    from s03a_transformer.five_channel.model import tok
    theirs_dir = locate_model1_build(model1_dir, model1_build)
    ours = api_names(b_dir, 'arm', model1_dir)
    theirs = api_names(theirs_dir, 'Model 1', model1_dir)
    if theirs != ours:
        shared = min(len(theirs), len(ours))
        same = sum(a == b for a, b in zip(theirs, ours))
        if same == shared:
            why = (f"one build's vocabulary extends the other's ({len(theirs)} vs {len(ours)} names), so the Model 1 was trained "
                   f"on another dataset's build (an extended build is another dataset)")
        else:
            why = (f"API ids are numbered per build and {same} of the {shared} positions they share hold the same name, so the "
                   f"frozen encoder would read the wrong API embeddings")
        raise SystemExit(f"{model1_dir} trained on the build {theirs_dir} ({len(theirs)} API names), not on this arm's build "
                         f"{b_dir} ({len(ours)} API names): {why}. Train the Model 1 on the arm's build (python -m experiments.train --model "
                         f"fivechannel --per-block --max-instructions {BLOCK_LEN} --build {b_dir} ...) or name a run of that build "
                         f"with --model1-run DIR")
    return theirs_dir, tok.FIRST_API_ID + len(ours)


def load_frozen_model1(model1_dir, manifest, smoke, log=print, *, b_dir, model1_build=None):
    """Janice's `Model1TokenSequence` with the weights of the Model 1 run's best_auc.pt, in eval mode and with no gradient, after
    `check_same_splits` and `check_same_api_vocab` against `b_dir`, the arm's build, and a refusal when the checkpoint's API embedding
    has more rows than that build can hold. A smoke run without that run uses random weights (plumbing only); a full run refuses."""
    from s03a_transformer.five_channel.model import Model1TokenSequence
    model1_dir = Path(model1_dir)
    ckpt = model1_dir / 'best_auc.pt'
    if not ckpt.is_file():
        if not smoke:
            raise SystemExit(f'{ckpt} does not exist: train the per-block Model 1 of this fold and seed first (python -m '
                             f'experiments.train --model fivechannel --per-block --max-instructions {BLOCK_LEN} ...)')
        log(f'smoke: {ckpt} does not exist, the frozen encoder has random weights')
        net, source = Model1TokenSequence(api_vocab_size=1, max_seq_len=BLOCK_LEN), None
    else:
        config = json.loads((model1_dir / 'config.json').read_text(encoding='utf-8'))
        if not config.get('per_block') or config.get('max_instructions') != BLOCK_LEN:
            raise SystemExit(f'{model1_dir} is not a per-block Model 1 at {BLOCK_LEN} instructions per block')
        check_same_splits(model1_dir, manifest)
        _, capacity = check_same_api_vocab(model1_dir, b_dir, model1_build)
        state = torch.load(ckpt, map_location='cpu')
        rows = int(state['embedder.api_embedding.weight'].shape[0])
        if rows > capacity:
            raise SystemExit(f'{ckpt} has {rows} API embedding rows, more than the {capacity} ids that {b_dir} can hold '
                             f'(FIRST_API_ID + the names of its api_vocab.json): its Model 1 read ids that build does not have')
        net = Model1TokenSequence(api_vocab_size=rows, max_seq_len=BLOCK_LEN)
        net.load_state_dict(state)
        source = str(ckpt)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, source


@torch.no_grad()
def frozen_graph(net, seq, device):
    """The PyG Data of one `BlockSeqData`: `x` the frozen encoder's vector of every block (zero for an empty block), the edges kept."""
    from torch_geometric.data import Data
    from s03c_cfg_transformer import block_transformer as bt
    n_blocks = int(seq.num_nodes)
    ids, pad_mask, lengths = bt.pad_blocks(seq.insn_ids.to(device), seq.insn_block.to(device), n_blocks, BLOCK_LEN)
    x = bt.encode_blocks(net, ids, pad_mask, lengths).float().cpu()
    data = Data(x=x, edge_index=seq.edge_index, edge_attr=seq.edge_attr, y=seq.y, num_nodes=n_blocks)
    data.edge_type = seq.edge_type
    data.sample_id = seq.sample_id
    return data


def run(arm, fold, seed, smoke=None, device=None, b_dir=None, log=print, features=None, source=None, block_len=None,
        protocol='kfold', arch='x86', arch_balance=False, model1_run=None, model1_build=None):
    if arm not in ARM_TABLE:
        raise ValueError(f'arm {arm}: one of {ARMS}')
    check_arm_dataset(arm, protocol, arch, arch_balance)
    features = ARM_FEATURES[arm] if features is None else features
    if features not in INPUTS:
        raise ValueError(f'features {features}: one of {INPUTS}')
    if features != ARM_FEATURES[arm]:
        raise SystemExit(f'arm {arm} trains on --features {ARM_FEATURES[arm]}, not {features} (the arm names the input)')
    check_block_len(arm, block_len)
    check_source_flag(arm, source)
    b_dir = default_build_dir(arm, protocol, arch) if b_dir is None else Path(b_dir)
    label = check_build(arm, b_dir, source, protocol, arch)
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    from s03b_cfg_feature.gat import gat_full_model as gm, graph_full_loader as gl      # imported here: no torch_geometric at import
    pp, embedder, done = build_state(b_dir)
    manifest = fold_manifest(fold, seed, protocol, arch)
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
                             f'missing from its checkpoint, e.g. {missing[:3]}')
        train_ids = [r['sample_id'] for r in manifest.train if r['sample_id'] not in excluded]
    first = train_ids[0]
    api_ids, err = {}, None                                      # first pass: the API ids of every payload, one payload in memory
    for i in want:
        payload = load_slim(b_dir, i, protocol, arch)
        api_ids[i] = payload_api_ids(pp.tok, payload).tolist()
        if i == first:
            err = check_repool(pp, embedder, payload)
        del payload
    log(f're-pool check on {first[:12]}: max |difference| {err:.2e} (<= 1e-5), api-kind columns reproduced')
    seen = torch.tensor(sorted({a for i in train_ids for a in api_ids[i]}), dtype=torch.long)
    every = {a for i in want for a in api_ids[i]}
    log(f'api ids in the fold\'s training binaries: {len(seen)}; unseen in val/test binaries mapped to UNK_API: '
        f'{len(every - set(seen.tolist()))} of {len(every)} ids used')
    e2e, frozen = features == E2E, features == M1FROZEN
    if e2e or frozen:
        from s03c_cfg_transformer import block_transformer as bt      # imported here: it needs torch_geometric
    if frozen:
        model1_dir = Path(model1_run) if model1_run else model1_run_dir(fold, seed, protocol, arch, arch_balance)
        frozen_net, frozen_source = load_frozen_model1(model1_dir, manifest, smoke, log, b_dir=b_dir, model1_build=model1_build)
        frozen_net.to(device)
        log(f'frozen Model 1 encoder: {frozen_source or "random weights (smoke)"}; same train/val/test as {model1_dir.name}')
    graphs = {}                                                  # second pass: each graph from its payload, which is then dropped
    for i in want:
        payload = load_slim(b_dir, i, protocol, arch)
        if frozen:
            graphs[i] = frozen_graph(frozen_net, bt.make_e2e_graph(pp, gl, payload, seen, BLOCK_LEN), device)
        elif e2e:
            graphs[i] = bt.make_e2e_graph(pp, gl, payload, seen, BLOCK_LEN)
        else:
            graphs[i] = make_graph(pp, gl, embedder, payload, seen, features)
    log_rows = read_build_log(b_dir)
    sizes = {i: (None, int(graphs[i].num_nodes), log_rows[i][0]) for i in want}
    in_dim, edge_dim = (bt.D_MODEL if e2e else int(graphs[want[0]].x.size(1))), int(graphs[want[0]].edge_attr.size(1))
    log(f'{arm}: {len(graphs)} graphs built in {time.time() - t0:.0f} s, features {features}, in_dim {in_dim}, '
        f'edge_dim {edge_dim}')
    api_rows = int(embedder.api_embedding.num_embeddings)
    extra = {'in_dim': in_dim, 'edge_dim': edge_dim, 'features': features, 'n_api_ids_train': int(len(seen)),
             'n_api_ids_unseen_mapped': len(every - set(seen.tolist())), 'build_dir': str(b_dir),
             'build_binaries': len(done), 'build_source': label, 'build_pins': build_pins(b_dir, label),
             'code_pins': dict(source_pins(code_source(label)))}
    if e2e:
        build_model = lambda: bt.Model3EndToEnd(api_rows, BLOCK_LEN, edge_dim)                  # noqa: E731
        if not smoke:
            extra.update(e2e_extras(graphs, [r['sample_id'] for r in manifest.test if r['sample_id'] not in excluded], BLOCK_LEN,
                                    api_rows))
    else:
        build_model = lambda: gm.FullBlockGAT(in_dim=in_dim, edge_dim=edge_dim)                 # noqa: E731
    if frozen:
        extra.update({'model1_checkpoint': frozen_source, 'model1_run': str(model1_dir), 'block_len': BLOCK_LEN})
        if frozen_source:                       # the random-weights smoke has no Model 1 run to name
            extra['model1_build'] = str(locate_model1_build(model1_dir, model1_build))
        del frozen_net
    reasons = sorted(set(excluded.values()))
    return run_arm(arm=arm, fold=fold, seed=seed, graphs=graphs, sizes=sizes,
                   build_model=build_model, hp=ARM_HP[arm], device=device,
                   smoke=smoke, smoke_split=split, extra=extra, log=log, excluded=sorted(excluded),
                   excluded_reason='; '.join(reasons), protocol=protocol, arch=arch, arch_balance=arch_balance)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--arm', choices=ARMS, default='sig500')
    ap.add_argument('--features', choices=INPUTS, help="node input: all (141-d), feat (77 block features), emb (64-d "
                    "embedding), e2e (the block's instructions, encoder trained with the GAT); default and only value allowed: the arm's")
    ap.add_argument('--block-len', type=int, help=f"instructions per block the encoder reads (the -e2e arms: {BLOCK_LEN}, the only "
                    "value allowed; the other arms read no instructions)")
    ap.add_argument('--protocol', choices=sorted(DATASET_OF), default='kfold', help="the fold protocol, which picks the dataset: kfold "
                    "(Mendeley, mendeley_x86 cache) or kfold_v3 (combined_v3 cache); the literal flag also picks the cache")
    ap.add_argument('--arch', choices=['x86', 'all', 'matched'], help="the binaries to train on (default: x86 under kfold, all under "
                    "kfold_v3); matched (kfold_v3 only) trains on the all build")
    ap.add_argument('--arch-balance', action='store_true', help="weight the training binaries by (class, architecture) cell instead "
                    "of by class (needs --arch all or matched; run name gnn-ab)")
    ap.add_argument('--source', choices=sorted(SOURCE_CODE), help="the source label the build must have recorded (default: any "
                    "the arm accepts)")
    ap.add_argument('--build', type=Path, help="build directory: a build_b folder or an unzipped preprocessed zip (default: the "
                    "arm's directory under the run directory when it exists, else the team_b_build source)")
    ap.add_argument('--smoke', type=int, help='train on N available training ids for 2 epochs into a temp directory')
    ap.add_argument('--device', choices=['cuda', 'cpu'])
    ap.add_argument('--model1-run', type=Path, help="the -m1frozen arms: the Model 1 run directory whose best_auc.pt is frozen "
                    f"(default: RUNS/<the fivechannel --per-block --max-instructions {BLOCK_LEN} run of this fold, seed and dataset>)")
    ap.add_argument('--model1-build', type=Path, help="the -m1frozen arms: the build folder the Model 1 trained on, compared with the "
                    "arm's (default: the build_dir its metrics.json records, else RUNS/team/<that folder's name> when the recorded "
                    "path is not on this machine)")
    args = ap.parse_args(argv)
    arch = args.arch or DEFAULT_ARCH[args.protocol]
    out = run(args.arm, args.fold, args.seed, args.smoke, args.device,
              resolve_dir(args.build) if args.build else None, features=args.features, source=args.source,
              block_len=args.block_len, protocol=args.protocol, arch=arch, arch_balance=args.arch_balance,
              model1_run=args.model1_run, model1_build=args.model1_build)
    if args.smoke:
        print(f"smoke: val AUC {out['val_auc']:.4f}, peak GPU {out['peak_gpu_mb']} MB, loss "
              f"{out['history'][0]['loss']} -> {out['history'][-1]['loss']}, temp dir {out['dir']} (not under {RUNS})")
    return out


if __name__ == '__main__':
    run_command(main)
    sys.exit(0)
