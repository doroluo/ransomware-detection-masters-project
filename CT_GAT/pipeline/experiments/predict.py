"""Score binaries with the weights of a saved run, without retraining.

python -m experiments.predict --run RUNS/<run folder> [--checkpoint auc|f1] [--build DIR] [--split val|test|all] [--ids ID ...]
       [--out FILE] [--device cpu|cuda] [--check [--tolerance X]] [--autocast auto|bf16|off] [--threshold saved|val|X]
       [--model1-run DIR] [--model1-build DIR]
       (run from CT_GAT/pipeline; reads the run folder and the build, never writes there)

A run folder is what `experiments.train --model fivechannel` and `experiments.team_pipelines.train_b` write (and what the
`*_model_runs_v1.zip` of the Drive unpacks to under `CT_GAT/pipeline/runs/`): config.json, manifest.json, metrics.json, best.pt (the
checkpoint of the best validation F1), best_auc.pt (the best validation AUC) and predictions*.csv. This command rebuilds the network
of the run from the trainers' own functions (`train_b.new_model`, `five_channel.train.new_net`), loads the checkpoint (`--checkpoint
auc`, the default, or `f1`), builds the model's input from the build exactly as the trainer does and writes one row per binary in the
columns of the saved predictions.

Run kinds: Model 1 (`fivechannel`: L128, L512, `-ab`, `-pb`, a run on `b_sigwhole_v3` is the same kind), and every arm of
`train_b.ARM_TABLE` (inputs `all`, `feat`, `emb`, `e2e`, `m1frozen`, the `-ab` twins, the sigwhole arms), on kfold x86, kfold all and
kfold_v3 all|matched. A run folder of another model (`experiments.train` gnn / tokenseq, a baseline) is refused, as is one without
metrics.json (a run still being written) or without the checkpoint.

The model's input is the build's: `--build DIR` (a build_b folder or an unzipped preprocessed zip), else the `build_dir` that
metrics.json records when it exists on this machine, else the arm's default build (`train_b.default_build_dir`, the `team_b_build`
source). The API ids that no training binary of the run used become UNK_API, as in training: the set comes from the train ids of the
run's manifest.json (read from the build; the other splits never enter it), and its size must equal metrics.json `n_api_ids_train`.

Which binaries: `--ids` names binaries of the build; else `--split` (default `all`: the run's validation and test binaries; `train`
is never a split of this command, a training binary is scored by `--ids`). An id the build does not hold is refused, not skipped.
The label of every row is the one the build's .pt holds (for the run's own ids it must equal the saved prediction's); arch, family and
fold of a binary the run did not score come from the CFG cache index when this machine has one, else they are empty.

What a build must be (all refused with a message naming both sides): of the run's dataset, and for a team arm of the arm's sampler,
block budget and source (`train_b.check_build`); the same sampler and budget as the build the run was made from when that build is on
this machine; with an API vocabulary (`api_vocab.json`) that equals the run's build's, or that the run's build's extends (the ids
keep their meaning, a new id is not in the run's training set and becomes UNK_API), and that has at least as many ids as the
checkpoint's embedding has rows. When the run's build is not on this machine the vocabulary is verified by the run itself: up to 8
of its own validation binaries are rescored and must reproduce their saved probabilities (within the tolerance below); `--check` is
the whole of that test. A `-m1frozen` run also loads the Model 1 run it was made with (`--model1-run`, else `model1_run` of
metrics.json when that folder exists, else `RUNS/<name of the per-block Model 1 of this fold and seed>`) and passes it to the
trainer's own guard (`train_b.load_frozen_model1`, the arm's build being the scored one), which refuses a Model 1 whose build
vocabulary differs. The guard finds the Model 1's build in the `build_dir` of its metrics.json, else in `RUNS/team/<that folder's
name>`; `--model1-build DIR` names it, and when it is nowhere on this machine and the scored build is the run's own, the scored
build stands in for it (the run passed the guard when it trained). The guard then cannot tell which Model 1 weights were loaded, so
whenever the scored build stands in, the 8 validation binaries are rescored before anything else, with or without the run's build,
and a Model 1 that does not reproduce them is refused (`--check` is still the whole test).

The `pred` column: `--threshold saved` (default) is the threshold the trainer used for that checkpoint's predictions file:
`selection.auc.threshold_val` of metrics.json for `--checkpoint auc` (predictions_auc.csv), and the RunConfig threshold, 0.5, for
`--checkpoint f1` (predictions.csv; the trainers threshold that file at 0.5, and the validation threshold of the F1 state,
`selection.f1.threshold_val`, is in metrics.json only). `--threshold val` is `selection.<rule>.threshold_val` for either checkpoint,
a number is any threshold. The summary says which one was used and why.

--check rescoring the run's own validation and test binaries must reproduce its saved predictions file: the same ids, labels and
splits, max |dprob| within the tolerance, the test AUC within 1e-3 of the one in metrics.json (the printed dAUC is now minus
saved) and `pred` agreeing in at least 99.5% of the rows; the exit status is 1 otherwise.

Numerics. A network without a bf16 autocast part (the GAT arms `all`, `feat`, `emb`, `m1frozen`) is fp32 in training and here: the
tolerance is 1e-4 (the probabilities are saved with 6 decimals) and the measured difference is about 1e-6. Model 1 and the `e2e` arms
train under bf16 autocast on cuda, and an fp32 rescoring of such a run differs from its saved probabilities by up to 5.2e-2
(measured on the published runs; the same difference is 1.5e-6 between fp32 and fp64, so it is the bf16 rounding of the training
run). With `--autocast auto` (default) the cpu runs those parts under bf16 autocast too (the trainers' `AUTOCAST_ON_CPU` switch;
torch's fused encoder-layer fast path, which does not run under autocast, is off meanwhile): the measured difference is then at
most 2.5e-3, the tolerance 5e-3. `--autocast off` rescores in fp32 and allows 6e-2: a diagnostic of how far bf16 rounding moves a run, not the way to
reproduce one. `--tolerance` overrides either. On cuda the trainers' own autocast runs.
"""
import argparse
import contextlib
import csv
import json
import sys
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from common.config import RUNS, RemovedOption, RunConfig
from common.manifests import Manifest, load_index, run_command
from common.paths import cache_for
from common.run_io import write_predictions
from experiments.team_pipelines import fit, train_b
from experiments.team_pipelines.build_b import resolve_dir

RULES = {'auc': ('best_auc.pt', 'predictions_auc.csv'), 'f1': ('best.pt', 'predictions.csv')}
PROBE_IDS = 8                  # own validation binaries rescored when the run's build is not on this machine
AGREEMENT = 0.995              # --check: share of rows whose `pred` equals the saved one
TOLERANCE_FP32, TOLERANCE_BF16, TOLERANCE_FP32_OF_BF16 = 1e-4, 5e-3, 6e-2     # see default_tolerance
AUC_TOLERANCE = 1e-3           # --check: largest |test AUC now - saved|
AUTOCAST_MODES = ('auto', 'bf16', 'off')
CHUNK = 64                     # training binaries read at a time to collect their API ids
SPLITS = ('val', 'test', 'all')


class Run(NamedTuple):
    dir: Path
    cfg: RunConfig
    metrics: dict
    manifest: dict                # manifest.json: train / val / test id lists
    kind: str                     # 'model1' or 'team'
    arm: Optional[str]            # the train_b arm of a team run
    rule: str                     # 'auc' or 'f1'
    checkpoint: Path

    @property
    def excluded(self):
        return set(self.metrics.get('excluded', []))

    @property
    def autocast(self):
        """True when the network has a bf16 autocast part on cuda (Model 1, the end-to-end arms)."""
        return self.kind == 'model1' or train_b.ARM_FEATURES[self.arm] == train_b.E2E

    def ids(self, split):
        return [i for i in self.manifest[split] if i not in self.excluded]


def say(*args):
    print(*args, flush=True)


# ---- the run -------------------------------------------------------------------------------------------------------------------

def load_run(path, rule='auc'):
    """The `Run` of a run folder; SystemExit naming what is missing or unsupported."""
    path = resolve_dir(Path(path))
    if not path.is_dir():
        raise SystemExit(f'--run {path} is not a directory (a run folder of the model_runs zip, RUNS/<name>)')
    for name in ('config.json', 'manifest.json', 'metrics.json'):
        if not (path / name).is_file():
            raise SystemExit(f'{path} has no {name}: it is not a finished run folder (a run still being written has no metrics.json)')
    checkpoint = path / RULES[rule][0]
    if not checkpoint.is_file():
        raise SystemExit(f'{path} has no {checkpoint.name}, the checkpoint of --checkpoint {rule}: a run folder without weights '
                         f'cannot score anything (the weights are in the *_model_runs_v1.zip of the Drive)')
    try:
        cfg = RunConfig.load(path / 'config.json')
    except RemovedOption as e:
        raise SystemExit(f'{path}: {e}') from None
    metrics = json.loads((path / 'metrics.json').read_text(encoding='utf-8'))
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    arm = None
    if cfg.model == 'fivechannel':
        kind = 'model1'
    elif cfg.model == 'gnn' and cfg.tag.startswith('team-'):
        kind, arm = 'team', cfg.tag[len('team-'):]
        if arm not in train_b.ARM_TABLE:
            raise SystemExit(f'{path}: arm {arm!r} is not one of train_b.ARMS ({", ".join(train_b.ARMS)}); this command scores '
                             f'the arms of the current trainer')
        if metrics.get('features') not in (None, train_b.ARM_FEATURES[arm]):
            raise SystemExit(f'{path}: metrics.json says features {metrics["features"]}, the arm {arm} has {train_b.ARM_FEATURES[arm]}')
    else:
        raise SystemExit(f'{path}: model {cfg.model!r} (tag {cfg.tag!r}) is not scored by this command: it scores the five-channel '
                         f'Model 1 (fivechannel) and the team arms of train_b (tag team-<arm>)')
    return Run(path, cfg, metrics, manifest, kind, arm, rule, checkpoint)


def saved_rows(run):
    """[row dict] of the run's saved predictions file for its checkpoint (the rows `write_predictions` wrote)."""
    path = run.dir / RULES[run.rule][1]
    if not path.is_file():
        raise SystemExit(f'{run.dir} has no {path.name}')
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def threshold_of(run, mode='saved'):
    """(threshold, text saying where it comes from) for the `pred` column. See the module docstring."""
    val = float(run.metrics['selection'][run.rule]['threshold_val'])
    name = f'{val} = selection.{run.rule}.threshold_val of metrics.json'
    if mode == 'val' or (mode == 'saved' and run.rule == 'auc'):
        return val, name
    if mode == 'saved':
        return float(run.cfg.threshold), (f'{run.cfg.threshold} = the RunConfig threshold, with which the trainer thresholded '
                                           f'predictions.csv (the validation threshold of this checkpoint is {val}, --threshold val)')
    try:
        return float(mode), f'{float(mode)} = the --threshold given'
    except ValueError:
        raise SystemExit(f'--threshold {mode}: saved, val or a number') from None


# ---- the build -----------------------------------------------------------------------------------------------------------------

def vocab_names(b_dir):
    """The API names of a build (its api_vocab.json, in the folder or one level up for an unzipped Drive zip); None when it has none."""
    from s03a_transformer.five_channel.model import tok
    path = train_b.build_layout(b_dir).vocab
    return None if path is None else list(tok.load_api_vocab(path).names)


def first_api_id():
    from s03a_transformer.five_channel.model import tok
    return tok.FIRST_API_ID


def budget_class(n):
    return 'uncapped' if n is not None and n >= train_b.UNCAPPED_BLOCKS else n


def reference_build(run):
    """The build the run was made from, when metrics.json names it and it exists on this machine, else None."""
    path = run.metrics.get('build_dir')
    return Path(path) if path and Path(path).is_dir() else None


def default_build(run):
    ref = reference_build(run)
    if ref is not None:
        return ref
    if run.kind == 'team':
        return train_b.default_build_dir(run.arm, run.cfg.protocol, run.cfg.arch)
    from s03a_transformer.five_channel.train import resolve_build
    return resolve_build(None, run.cfg.protocol, run.cfg.arch)


def check_build(run, b_dir, rows):
    """SystemExit unless the build `b_dir` is one the run can score: its dataset (and for a team arm its sampler, block budget and
    source), the sampler and budget of the run's own build, and its API vocabulary (module docstring). `rows` is the number of rows
    of the checkpoint's API embedding. Returns the reference build (or None) for the caller's messages."""
    cfg = run.cfg
    if run.kind == 'team':
        train_b.check_build(run.arm, b_dir, None, cfg.protocol, cfg.arch)
    else:
        train_b.check_build_dataset(b_dir, cfg.protocol, cfg.arch)
    names = vocab_names(b_dir)
    ref = reference_build(run)
    if ref is not None:
        have, want = train_b.read_settings(b_dir), train_b.read_settings(ref)
        for key in ('sample', 'num_blocks'):
            x, y = have.get(key), want.get(key)
            if key == 'num_blocks':
                x, y = budget_class(x), budget_class(y)
            if x != y:
                raise SystemExit(f'{b_dir} was built with {key} {have.get(key)}; {run.dir.name} was made from {ref} with {want.get(key)}')
    if names is not None and rows is not None and first_api_id() + len(names) < rows:
        raise SystemExit(f'{b_dir} has an api_vocab.json of {len(names)} names (ids up to {first_api_id() + len(names) - 1}); the '
                         f'checkpoint of {run.dir.name} embeds {rows} ids: it was trained on a build with a larger vocabulary')
    if ref is None:
        return None
    ref_names = vocab_names(ref)
    if names is not None and ref_names is not None and names[:len(ref_names)] != ref_names:
        same = sum(x == y for x, y in zip(names, ref_names))
        raise SystemExit(f'{b_dir} numbers the API names differently from {ref} (the build of {run.dir.name}): {same} of its '
                         f'{len(ref_names)} names are at the same position; the checkpoint embeds ids, so the vocabularies must be '
                         f'equal (or the build must extend the one of the run)')
    if (names is None) != (ref_names is None):
        raise SystemExit(f'{b_dir if names is None else ref} has no api_vocab.json, so its vocabulary cannot be compared with '
                         f'{ref if names is None else b_dir}')
    return ref


def check_ids_in_build(b_dir, ids, done):
    """SystemExit naming the ids that the build does not hold (not in its checkpoint / sample files, or no file)."""
    missing = [i for i in ids if i not in done or not (Path(b_dir) / f'{i}.pt').is_file()]
    if missing:
        raise SystemExit(f'{len(missing)} of the {len(ids)} binaries asked for are not in the build {b_dir}, e.g. {missing[:3]}')


# ---- scorers (one per run kind): ids -> (probs, sizes, labels) -----------------------------------------------------------------

class Model1Scorer:
    """The five-channel Model 1 of a run, on the instructions of a build (`five_channel/train.py`: load_binaries, make_items, predict)."""

    def __init__(self, run, b_dir, device, log=say):
        from s03a_transformer.five_channel import train as fc
        self.fc, self.run, self.device = fc, run, device
        self.build = fc.resolve_build(b_dir, run.cfg.protocol, run.cfg.arch)
        state = torch.load(run.checkpoint, map_location='cpu')
        self.rows = int(state['embedder.api_embedding.weight'].shape[0])
        self.ref = check_build(run, self.build, self.rows)
        self.done = fc.build_done(self.build)
        train_ids = run.ids('train')
        if self.done is not None:
            check_ids_in_build(self.build, train_ids, self.done)
        parts = []
        for start in range(0, len(train_ids), CHUNK):
            binaries = fc.load_binaries(self.build, dict.fromkeys(train_ids[start:start + CHUNK]), run.cfg.max_instructions, run.cfg.per_block)
            parts += [b.api for b in binaries.values()]
        self.seen = torch.unique(torch.cat(parts)) if parts else torch.zeros(0, dtype=torch.long)
        self.net = fc.new_net(run.cfg, self.rows, device)
        self.net.load_state_dict(state)
        self.net.eval()
        log(f'Model 1 {run.dir.name}: {sum(p.numel() for p in self.net.parameters())} parameters, {len(self.seen)} API ids in the '
            f'{len(train_ids)} training binaries')

    def has(self, ids):
        check_ids_in_build(self.build, ids, self.done if self.done is not None else {p.stem for p in self.build.glob('*.pt')})

    @torch.no_grad()
    def score(self, ids):
        fc, cfg = self.fc, self.run.cfg
        probs, sizes, labels = [], [], []
        for start in range(0, len(ids), fc.BATCH_SIZE):
            chunk = list(ids[start:start + fc.BATCH_SIZE])
            binaries = fc.load_binaries(self.build, dict.fromkeys(chunk), cfg.max_instructions, cfg.per_block)
            items = fc.make_items(binaries, chunk, self.seen)
            _, p, s = fc.predict(self.net, items, chunk, binaries, cfg, self.device)
            probs.append(p)
            sizes += s
            labels += [binaries[i].label for i in chunk]
        return (np.concatenate(probs) if probs else np.zeros(0)), sizes, labels

    @property
    def n_api_train(self):
        return int(len(self.seen))


class TeamScorer:
    """The network of a `train_b` arm on the graphs of a build (`train_b`: build_state, check_repool, make_graph, make_e2e_graph,
    frozen_graph; `fit.predict` for the batches)."""

    def __init__(self, run, b_dir, device, model1_run=None, log=say, model1_build=None):
        from s03b_cfg_feature.gat import graph_full_loader as gl, train_cfg
        self.run, self.device, self.gl, self.tc = run, device, gl, train_cfg
        self.build = Path(b_dir)
        arm, cfg, metrics = run.arm, run.cfg, run.metrics
        self.features = train_b.ARM_FEATURES[arm]
        state = torch.load(run.checkpoint, map_location='cpu')
        self.rows = int(state['encoder.embedder.api_embedding.weight'].shape[0]) if self.features == train_b.E2E else None
        self.ref = check_build(run, self.build, self.rows)
        self.pp, self.embedder, self.done = train_b.build_state(self.build)
        train_ids = run.ids('train')
        check_ids_in_build(self.build, train_ids, self.done)
        api, first = set(), True
        for i in train_ids:
            payload = train_b.read_slim(self.build, i)
            api.update(train_b.payload_api_ids(self.pp.tok, payload).tolist())
            if first:
                first = False
                try:
                    train_b.check_repool(self.pp, self.embedder, payload)
                except AssertionError as e:
                    raise SystemExit(f'{self.build} does not reproduce its own block embeddings ({e}); it is not the build the '
                                     f'run was made from') from None
        self.seen = torch.tensor(sorted(api), dtype=torch.long)
        self.frozen, self.standin = None, False
        if self.features == train_b.M1FROZEN:
            self.frozen = self.load_frozen(model1_run, log, model1_build)
        self.model = train_b.new_model(self.features, int(metrics['in_dim']), int(metrics['edge_dim']), self.rows)
        self.model.load_state_dict(state)
        self.model.to(device).eval()
        recipe = metrics.get('recipe', {})
        hp = train_b.ARM_HP[arm]
        self.batch_size = int(recipe.get('batch_size', hp['batch_size']))
        self.budget = recipe.get('node_budget', hp.get('node_budget'))
        self.log_rows = train_b.read_build_log(self.build)
        log(f'{arm} {run.dir.name}: features {self.features}, {sum(p.numel() for p in self.model.parameters())} parameters, '
            f'{len(self.seen)} API ids in the {len(train_ids)} training binaries')

    def model1_build(self, model1_dir, explicit):
        """The `model1_build` that `train_b.load_frozen_model1` is given: `--model1-build` when named, else None (the guard reads the
        `build_dir` of the Model 1's metrics.json, else RUNS/team/<its folder name>), except when that build is not on this machine
        and the scored build is the one the run was made from: then the scored build stands in for it. The run passed the same guard
        when it trained (`model1_build` in its metrics.json), which required the Model 1's vocabulary to equal that of the run's
        build, so the scored build is the Model 1's when it has the run's vocabulary. `check_build` compares them when the run's build
        is here (a scored build whose vocabulary only extends the run's is not the run's: the guard gets no stand-in and refuses the
        missing build). When it is not here, the rescoring of the run's own validation binaries (the probe, or `--check`) shows a
        build that gives other inputs; an extension of the run's build gives the same inputs (its new ids are UNK_API), so it is
        scored. A run without `model1_build` was not guarded and gets the guard's refusal. `self.standin` records that the stand-in
        was used: nothing then verifies the Model 1 checkpoint itself, and `predict` runs the probe."""
        if explicit:
            return explicit
        try:
            train_b.locate_model1_build(model1_dir)
            return None
        except SystemExit:
            pass
        ref = reference_build(self.run)
        if not self.run.metrics.get('model1_build') or (ref is not None and vocab_names(self.build) != vocab_names(ref)):
            return None
        return self.build

    def load_frozen(self, model1_run, log, model1_build=None):
        run = self.run
        c = run.cfg
        recorded = run.metrics.get('model1_run')
        default = train_b.model1_run_dir(c.fold, c.seed, c.protocol, c.arch, c.arch_balance)
        tries = [Path(model1_run)] if model1_run else [Path(recorded)] * bool(recorded) + [default]
        found = next((p for p in tries if p.is_dir()), None)
        if found is None:
            raise SystemExit(f'{run.dir.name} is a -m1frozen run: its frozen Model 1 is the per-block Model 1 run of the same fold '
                             f'and seed, and that folder is missing ({" / ".join(str(p) for p in tries)}); unpack it next to this '
                             f'run or pass --model1-run DIR')
        model1_dir = found
        manifest = Manifest(*[[{'sample_id': i} for i in run.manifest[s]] for s in ('train', 'val', 'test')], run.manifest.get('val_families', []))
        used = self.model1_build(model1_dir, model1_build)
        self.standin = used is not None and not model1_build
        net, source = train_b.load_frozen_model1(model1_dir, manifest, False, log, b_dir=self.build, model1_build=used)
        net.to(self.device)
        return net

    def has(self, ids):
        check_ids_in_build(self.build, ids, self.done)

    def graph(self, sample_id):
        payload = train_b.read_slim(self.build, sample_id)
        if self.features == train_b.E2E or self.frozen is not None:
            from s03c_cfg_transformer import block_transformer as bt
            seq = bt.make_e2e_graph(self.pp, self.gl, payload, self.seen, train_b.BLOCK_LEN)
            return seq if self.frozen is None else train_b.frozen_graph(self.frozen, seq, self.device)
        return train_b.make_graph(self.pp, self.gl, self.embedder, payload, self.seen, self.features)

    @torch.no_grad()
    def score(self, ids):
        probs, sizes, labels = [], [], []
        for start in range(0, len(ids), self.batch_size):
            graphs = [self.graph(i) for i in ids[start:start + self.batch_size]]
            p, y = fit.predict(self.tc, self.model, graphs, self.device, self.batch_size, self.budget)
            probs.append(p.numpy().astype(np.float64))
            labels += y.tolist()
            sizes += [(None, int(g.num_nodes), self.log_rows[g.sample_id][0] if g.sample_id in self.log_rows else None) for g in graphs]
            del graphs
        return (np.concatenate(probs) if probs else np.zeros(0)), sizes, labels

    @property
    def n_api_train(self):
        return int(len(self.seen))


def make_scorer(run, b_dir, device, model1_run=None, log=say, model1_build=None):
    fit.configure_threads()
    scorer = (Model1Scorer(run, b_dir, device, log) if run.kind == 'model1'
              else TeamScorer(run, b_dir, device, model1_run, log, model1_build))
    want = run.metrics.get('n_api_ids_train')
    if want is not None and scorer.n_api_train != int(want):
        raise SystemExit(f'the training binaries of {run.dir.name} use {want} API ids in the build it was made from and {scorer.n_api_train} '
                         f'in {scorer.build}: it is not the same build (a different vocabulary or other binaries)')
    return scorer


# ---- the check, the probe, the table ---------------------------------------------------------------------------------------------

def auc_of(labels, probs):
    return float(roc_auc_score(np.asarray(labels), np.asarray(probs)))


def trained_in_bf16(run):
    """True for a run whose network has a bf16 autocast part (Model 1, the end-to-end arms) and that was trained on cuda."""
    return run.autocast and str(run.metrics.get('device', '')).startswith('cuda')


def bf16_emulated(run, device, mode):
    """True when this command runs the bf16 autocast part of the network in bf16 on the cpu: `--autocast auto` does for a run
    trained under bf16 autocast on cuda (its probabilities carry bf16 rounding that an fp32 rescoring does not reproduce:
    measured up to 5.2e-2), `bf16` always (for a network that has such a part), `off` never.
    On cuda the trainers' own autocast runs and nothing is emulated."""
    if mode not in AUTOCAST_MODES:
        raise SystemExit(f'--autocast {mode}: one of {", ".join(AUTOCAST_MODES)}')
    return device.type == 'cpu' and run.autocast and (mode == 'bf16' or (mode == 'auto' and trained_in_bf16(run)))


@contextlib.contextmanager
def numerics(emulated):
    """With `emulated`, the trainers' autocast switches (`AUTOCAST_ON_CPU` of five_channel/train.py and block_transformer.py) are
    on: bf16 autocast on the cpu as on cuda. The fused encoder-layer fast path of torch does not run under autocast, so it is
    switched off meanwhile (it computes the same function). Everything is restored on exit."""
    if not emulated:
        yield
        return
    from s03a_transformer.five_channel import train as fc
    from s03c_cfg_transformer import block_transformer as bt
    fast = torch.backends.mha.get_fastpath_enabled()
    before = fc.AUTOCAST_ON_CPU, bt.AUTOCAST_ON_CPU
    fc.AUTOCAST_ON_CPU = bt.AUTOCAST_ON_CPU = True
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        yield
    finally:
        fc.AUTOCAST_ON_CPU, bt.AUTOCAST_ON_CPU = before
        torch.backends.mha.set_fastpath_enabled(fast)


def default_tolerance(run, emulated):
    """Largest |dprob| that --check allows. 1e-4 where the network is fp32 in training and here (the probabilities are saved with
    6 decimals; measured about 1e-6). For a run trained in bf16 on cuda: 5e-3 when this command runs bf16 too (`emulated`;
    measured up to 2.5e-3 on cpu), 6e-2 for an fp32 rescoring of it (`--autocast off`, a diagnostic; measured up to 5.2e-2)."""
    if not trained_in_bf16(run):
        return TOLERANCE_FP32
    return TOLERANCE_BF16 if emulated else TOLERANCE_FP32_OF_BF16


def saved_auc(run):
    """The test AUC that the trainer wrote to metrics.json for this checkpoint (from the unrounded probabilities)."""
    return float(run.metrics['selection']['auc']['test']['auc'] if run.rule == 'auc' else run.metrics['test']['auc'])


def compare(saved, ids, probs, labels, threshold, tolerance, auc_metrics=None):
    """The numbers of --check for the saved rows `saved` ({id: row}) of `ids`, the probabilities and labels just scored."""
    old = np.array([float(saved[i]['prob']) for i in ids])
    old_labels = [int(saved[i]['label']) for i in ids]
    test = [k for k, i in enumerate(ids) if saved[i]['split'] == 'test']
    pred_now, pred_old = (probs >= threshold).astype(int), np.array([int(saved[i]['pred']) for i in ids])
    out = {'n': len(ids), 'max_dprob': float(np.abs(probs - old).max()) if len(ids) else 0.0,
           'labels_equal': labels == old_labels, 'pred_agreement': float((pred_now == pred_old).mean()) if len(ids) else 1.0,
           'tolerance': tolerance}
    out['worst'] = ids[int(np.abs(probs - old).argmax())] if len(ids) else None
    if test and len(set(old_labels[k] for k in test)) == 2:
        out['auc_csv'] = auc_of([old_labels[k] for k in test], old[test])
        out['auc_saved'] = out['auc_csv'] if auc_metrics is None else auc_metrics
        out['auc_now'] = auc_of([old_labels[k] for k in test], probs[test])
        out['dauc'] = out['auc_now'] - out['auc_saved']
    return out


def probe(run, scorer, saved, threshold, tolerance, log):
    """The identity test when nothing else verifies the inputs (the run's build is not here, or the scored build stands in for the
    Model 1's build): rescore a few validation binaries (both classes) of the run itself."""
    val = [r for r in saved if r['split'] == 'val']
    pick = [r['sample_id'] for lab in ('1', '0') for r in [x for x in val if x['label'] == lab][:PROBE_IDS // 2]]
    if not pick:
        return
    probs, _, _ = scorer.score(pick)
    old = np.array([float(next(r['prob'] for r in val if r['sample_id'] == i)) for i in pick])
    worst = float(np.abs(probs - old).max())
    if worst > tolerance:
        if getattr(scorer, 'standin', False):
            why = (f"the scored build stands in for the build of the frozen Model 1, which is not on this machine, so the Model 1 "
                   f"checkpoint was tested by rescoring {len(pick)} of the run's own binaries")
            what = ("This build does not give the run's inputs (another API vocabulary, sampler or budget), or the Model 1 weights "
                    "loaded are not the run's")
        else:
            why = (f"the build of {run.dir.name} is not on this machine, so its vocabulary was tested by rescoring {len(pick)} of "
                   f"the run's own binaries")
            what = "This build does not give the run's inputs (another API vocabulary, sampler or budget)"
        raise SystemExit(f'{scorer.build}: {why}: max |dprob| {worst:.3g} > {tolerance:g}. {what}')
    log(f'build identity: {len(pick)} of the run\'s own validation binaries rescored, max |dprob| {worst:.3g} <= {tolerance:g}')


def index_meta(cfg, ids):
    """{sample_id: row of the CFG cache index} for `ids`; {} when this machine has no cache for the protocol."""
    try:
        index = load_index(cache=cache_for(cfg.protocol), protocol=cfg.protocol)
    except FileNotFoundError:
        return {}
    wanted = set(ids)
    out = {}
    for r in index:
        if r['sample_id'] in wanted:
            fold = r.get('fold_v3') if cfg.protocol == 'kfold_v3' else r.get('fold')
            out[r['sample_id']] = {**r, 'fold': fold}
    return out


def table(run, ids, probs, sizes, labels, saved, meta):
    """(rows, by_split) for `write_predictions`: one row per id, the split the run's manifest says (`new` for another binary)."""
    where = {i: s for s in ('train', 'val', 'test') for i in run.manifest[s]}
    rows, by_split = [], {}
    for k, i in enumerate(ids):
        if i in saved and int(saved[i]['label']) != labels[k]:
            raise SystemExit(f'{i}: the build says label {labels[k]}, {run.dir.name} scored it with label {saved[i]["label"]}')
        src = saved.get(i) or meta.get(i) or {}
        rows.append({'sample_id': i, 'label': labels[k], 'arch': src.get('arch', ''), 'family': src.get('family', ''),
                     'fold': '' if src.get('fold') is None else src.get('fold', '')})
        group = by_split.setdefault(where.get(i, 'new'), ([], [], []))
        group[0].append(i)
        group[1].append(probs[k])
        group[2].append(sizes[k])
    return rows, {s: (a, np.array(b), c) for s, (a, b, c) in by_split.items()}


def default_out(run):
    return Path.cwd() / f'{run.dir.name}_{run.rule}_scores.csv'


def check_out(run, out):
    """SystemExit when `out` would be inside the run folder."""
    out = Path(out).resolve()
    if run.dir.resolve() == out.parent or run.dir.resolve() in out.parents:
        raise SystemExit(f'--out {out} is inside the run folder {run.dir}: a run folder is never written to')
    return out


# ---- the command ---------------------------------------------------------------------------------------------------------------

def predict(run_dir, checkpoint='auc', build=None, split='all', ids=None, out=None, device='cpu', check=False, tolerance=None,
            threshold='saved', model1_run=None, autocast='auto', log=say, model1_build=None):
    """Score binaries with the weights of a run (module docstring); returns a dict with the probabilities, the rows written and,
    for `check`, its numbers. SystemExit for every refusal and for a failed check."""
    if checkpoint not in RULES:
        raise SystemExit(f'--checkpoint {checkpoint}: auc or f1')
    if split not in SPLITS:
        raise SystemExit(f'--split {split}: one of {", ".join(SPLITS)}')
    if check and ids:
        raise SystemExit('--check rescoring the run\'s own validation and test binaries takes no --ids')
    run = load_run(run_dir, checkpoint)
    out = check_out(run, out or default_out(run))
    device = torch.device(device)
    thr, thr_text = threshold_of(run, threshold)
    b_dir = resolve_dir(Path(build)) if build else default_build(run)
    if not b_dir.is_dir():
        raise SystemExit(f'the build {b_dir} is not a directory: pass --build DIR (a build_b folder or an unzipped preprocessed zip)')
    saved_list = saved_rows(run)
    saved = {r['sample_id']: r for r in saved_list}
    scorer = make_scorer(run, b_dir, device, model1_run, log, model1_build)
    emulated = bf16_emulated(run, device, autocast)
    tol = default_tolerance(run, emulated) if tolerance is None else float(tolerance)
    saved_thr = threshold_of(run, 'saved')[0]
    if ids:
        want = list(dict.fromkeys(ids))
    elif check:
        want = [r['sample_id'] for r in saved_list]
    else:
        want = [i for s in (('val', 'test') if split == 'all' else (split,)) for i in run.ids(s)]
    scorer.has(want)
    with numerics(emulated):
        if (scorer.ref is None and not check) or getattr(scorer, 'standin', False):
            probe(run, scorer, saved_list, saved_thr, tol, log)
        probs, sizes, labels = scorer.score(want)
    rows, by_split = table(run, want, probs, sizes, labels, saved, index_meta(run.cfg, [i for i in want if i not in saved]))
    result = {'run': run, 'ids': want, 'probs': probs, 'threshold': thr, 'out': out, 'build': scorer.build}
    problems = []
    if check:
        expected = {s: run.ids(s) for s in ('val', 'test')}
        if sorted(want) != sorted(expected['val'] + expected['test']) or len(want) != len(saved):
            problems.append(f'the saved {RULES[run.rule][1]} does not hold the manifest\'s validation and test ids')
        if any(saved[i]['split'] != ('val' if i in set(expected['val']) else 'test') for i in want):
            problems.append('a saved row has another split than the manifest gives it')
        numbers = compare(saved, want, probs, labels, saved_thr, tol, saved_auc(run))
        if not numbers['labels_equal']:
            problems.append('the labels of the build differ from the saved ones')
        if numbers['max_dprob'] > tol:
            problems.append(f'max |dprob| {numbers["max_dprob"]:.3g} > tolerance {tol:g} (worst {numbers["worst"]})')
        if 'auc_now' in numbers and abs(numbers['auc_now'] - numbers['auc_saved']) > AUC_TOLERANCE:
            problems.append(f'test AUC {numbers["auc_now"]:.6f} differs from the saved {numbers["auc_saved"]:.6f} by '
                            f'{numbers["auc_now"] - numbers["auc_saved"]:+.2e} (> {AUC_TOLERANCE:g})')
        if numbers['pred_agreement'] < AGREEMENT:
            problems.append(f'pred agreement {numbers["pred_agreement"]:.4f} < {AGREEMENT}')
        result['check'] = {**numbers, 'problems': problems}
    write_predictions(out, rows, by_split, thr)
    auc = auc_of(labels, probs) if len(set(labels)) == 2 else None
    log(f'run {run.dir.name} ({run.kind}{" " + run.arm if run.arm else ""}), checkpoint {checkpoint} ({run.checkpoint.name}), build {scorer.build}')
    log(f'threshold for pred: {thr_text}')
    log('numerics: ' + ('bf16 autocast on the cpu, as the run was trained' if emulated else 'fp32' if device.type == 'cpu'
                        else "cuda, the trainers' own autocast"))
    log(f'scored {len(want)} binaries' + ''.join(f', {s} {len(g[0])}' for s, g in by_split.items()) + ', AUC '
        + (f'{auc:.4f}' if auc is not None else 'n/a (one class)') + f'; wrote {out}')
    if check:
        c = result['check']
        log(f'check against {RULES[run.rule][1]}: n {c["n"]}, max |dprob| {c["max_dprob"]:.3g} (tolerance {c["tolerance"]:g}, worst '
            f'{c["worst"]}), test AUC saved {c.get("auc_saved", float("nan")):.6f} (metrics.json; '
            f'{c.get("auc_csv", float("nan")):.6f} from the csv) / now {c.get("auc_now", float("nan")):.6f}, '
            f'dAUC (now - saved) {c.get("dauc", float("nan")):+.2e} (tolerance {AUC_TOLERANCE:g}), pred agreement {c["pred_agreement"]:.4f} (at {saved_thr})')
        if problems:
            raise SystemExit('CHECK FAILED: ' + '; '.join(problems))
        log('CHECK PASSED')
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True, type=Path, help='the run folder (RUNS/<name>, as unpacked from the model_runs zip)')
    ap.add_argument('--checkpoint', choices=sorted(RULES), default='auc', help='auc: best_auc.pt (threshold: validation, '
                    'predictions_auc.csv); f1: best.pt (predictions.csv, thresholded at 0.5 by the trainer)')
    ap.add_argument('--build', type=Path, help="the build to read (default: the run's build_dir when it exists here, else the "
                    "arm's default / team_b_build source)")
    ap.add_argument('--split', choices=SPLITS, default='all', help="the run's validation and/or test binaries (default all)")
    ap.add_argument('--ids', nargs='+', help='sample ids of the build to score instead of a split')
    ap.add_argument('--out', type=Path, help='CSV to write (default: ./<run>_<checkpoint>_scores.csv; never inside the run folder)')
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--check', action='store_true', help="rescore the run's own validation and test binaries and compare with its "
                    'saved predictions; exit 1 when they differ')
    ap.add_argument('--autocast', choices=AUTOCAST_MODES, default='auto', help="auto: for a run trained under bf16 autocast on cuda "
                    "(Model 1, the e2e arms) run those parts in bf16 on the cpu too (the probabilities then agree to 2.5e-3 at most, "
                    "against up to 5.2e-2 in fp32); bf16: always; off: fp32, a diagnostic")
    ap.add_argument('--tolerance', type=float, help='--check: largest |dprob| allowed (default 1e-4; for a bf16-autocast run '
                    'trained on cuda 5e-3, or 6e-2 with --autocast off)')
    ap.add_argument('--threshold', default='saved', help="the pred threshold: saved (the trainer's for this checkpoint), val "
                    "(metrics.json selection.<rule>.threshold_val) or a number")
    ap.add_argument('--model1-run', type=Path, help='-m1frozen runs: the folder of the per-block Model 1 run that was frozen')
    ap.add_argument('--model1-build', type=Path, help="-m1frozen runs: the build the Model 1 trained on (default: the build_dir of its "
                    "metrics.json, else RUNS/team/<that folder's name>, else the build being scored)")
    args = ap.parse_args(argv)
    try:
        return predict(args.run, args.checkpoint, args.build, args.split, args.ids, args.out, args.device, args.check, args.tolerance,
                       args.threshold, args.model1_run, args.autocast, model1_build=args.model1_build)
    except FileNotFoundError as e:                  # a missing build folder or sample file, a teammate source: the message says which
        raise SystemExit(str(e)) from None


if __name__ == '__main__':
    run_command(main)
    sys.exit(0)
