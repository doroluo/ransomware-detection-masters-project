"""Resumable experiment queue, from CT_GAT/pipeline:

    python -m experiments.queue --plan experiments/plans/v3.txt [--budget-min N] [--dry] [--stop-file PATH]

A plan is a text file with one command per line; blank lines and lines that start with `#` are ignored.

    train <experiments.train arguments>      python -m experiments.train <arguments>
    baselines <s04_evaluation.baselines arguments>
    report                                   python -m s04_evaluation.report
    val_threshold                            python -m s04_evaluation.val_threshold      > <runs>/val_threshold.md
    gates <stage> [--out NAME]               python -m s04_evaluation.gates --stage <stage>  > <runs>/gates_<NAME or stage>.md

The whole plan is parsed first (a bad line, an unknown gates stage, an unknown training flag stops the queue before it starts).
`<runs>` is `GT_RUNS` (or `--runs`).

Semantics, from the shell queues the experiments were run with:
  * a `train` line is skipped (`skip <name>`) when `<runs>/<name>/metrics.json` exists, <name> being `RunConfig.name` of its
    arguments (or `--out`); a `baselines` line is skipped when every directory it would write already holds a metrics.json with
    a validation threshold (`stack_logits.csv` with --stack-logits);
  * when the STOP file (`<runs>/STOP`, or `--stop-file`) exists before a `train` or `baselines` line, the line is logged as
    `stopped before <name>` and skipped; `report`, `val_threshold` and `gates` still run, so the tables of what finished are
    written. A STOP file left from an earlier session is removed at the start and logged. `--stop` creates it;
  * `--budget-min N` caps the wall time spent in `train` lines (retries included): once N minutes are used, the remaining
    `train` lines are logged as `skipped (budget) <name>`; the other lines still run;
  * a line that exits non-zero is retried once (`retry <name>`; the first log is kept as `<name>.log.try1`), not when the STOP
    file exists, so stopping the queue and killing the run does not restart it;
  * every line logs to `<runs>/logs/<name>.log`; `start <name> HH:MM:SS` and `finished <name> rc=<rc> HH:MM:SS` mark each one;
  * the exit code is 1 when a line failed after its retry (the queue goes on to the next line), 0 otherwise, 2 for a bad plan.

`--dry` prints `run <name> <arguments>` for every `train` line and `cpu <command>` for the others, writes nothing and skips
nothing that exists; `--dry-min N` charges N minutes per `train` line so that the budget guard can be tried. `--python CMD`
(or `GT_PY`) names the interpreter, default the one running the queue. Every child runs with the pipeline root as its working
directory, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the GPU is shared) and `PYTHONUNBUFFERED=1`.
Python only: the same plan runs on Windows and Linux.
"""
import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
PLANS = PIPELINE_ROOT / 'experiments' / 'plans'
VERBS = ('train', 'baselines', 'report', 'val_threshold', 'gates')
WORK = ('train', 'baselines')                     # the lines that the STOP file and the budget hold back


class PlanError(ValueError):
    pass


@dataclass(frozen=True)
class Step:
    """One parsed line: what to run, what it is called in the logs, and where its output goes."""
    verb: str
    args: tuple
    label: str                                     # the log name and the name printed by `start` / `finished`
    module: str
    module_args: tuple
    stdout: str = ''                               # a file name under <runs> that receives stdout, '' = the log
    done_files: tuple = ()                         # a work line is finished when all of these exist (relative to <runs>)
    done_needs: str = ''                           # ... and, when set, the first of them contains this text

    @property
    def text(self):
        return ' '.join(self.args)


def _lazy_train():
    from experiments import train
    return train


def _train_step(args, where):
    try:
        cfg, out, _ = _lazy_train().parse_args(list(args))
    except SystemExit as e:
        raise PlanError(f'{where}: `train {" ".join(args)}` is not accepted by experiments.train') from e
    name = out or cfg.name
    return Step('train', tuple(args), name, 'experiments.train', tuple(args), done_files=(f'{name}/metrics.json',))


def _baselines_step(args, where):
    from s04_evaluation import baselines
    try:
        parsed = baselines.parse_args(list(args))
    except SystemExit as e:
        raise PlanError(f'{where}: `baselines {" ".join(args)}` is not accepted by s04_evaluation.baselines') from e
    names = [baselines.baseline_name(cfg) if parsed.stack_logits else baselines.output_name(cfg, parsed.vocab, parsed.imports)
             for cfg in baselines.configs(parsed)]
    if parsed.out_root:
        base = Path(os.path.abspath(PIPELINE_ROOT / parsed.out_root))
    else:
        base = Path('baselines')
    marker = 'stack_logits.csv' if parsed.stack_logits else 'metrics.json'
    files = tuple((base / n / marker).as_posix() for n in names)
    label = 'baselines_' + re.sub(r'(kfold)\d+', lambda m: m.group(1), names[0], count=1)
    return Step('baselines', tuple(args), label, 's04_evaluation.baselines', tuple(args), done_files=files,
                done_needs='' if parsed.stack_logits else 'threshold_val')


def parse_line(line, where='plan'):
    tokens = shlex.split(line, posix=os.name != 'nt')
    verb, args = tokens[0], tokens[1:]
    if verb not in VERBS:
        raise PlanError(f'{where}: unknown command {verb!r}; the plan commands are {", ".join(VERBS)}')
    if verb == 'train':
        return _train_step(args, where)
    if verb == 'baselines':
        return _baselines_step(args, where)
    if verb == 'report':
        if args:
            raise PlanError(f'{where}: `report` takes no arguments')
        return Step('report', (), 'report', 's04_evaluation.report', ())
    if verb == 'val_threshold':
        if args:
            raise PlanError(f'{where}: `val_threshold` takes no arguments')
        return Step('val_threshold', (), 'val_threshold', 's04_evaluation.val_threshold', (), stdout='val_threshold.md')
    from s04_evaluation.gates import stage_names
    out = None
    if len(args) == 3 and args[1] == '--out':
        args, out = args[:1], args[2]
    if len(args) != 1:
        raise PlanError(f'{where}: usage `gates <stage> [--out NAME]`')
    if args[0] not in stage_names():
        raise PlanError(f'{where}: no gates stage {args[0]!r}; the stages are {", ".join(stage_names())}')
    name = out or args[0]
    shown = tuple(args) if out is None else (args[0], '--out', out)
    return Step('gates', shown, f'gates_{name}', 's04_evaluation.gates', ('--stage', args[0]), stdout=f'gates_{name}.md')


def parse_plan(text, source='plan'):
    steps = []
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if line and not line.startswith('#'):
            steps.append(parse_line(line, f'{source}:{number}'))
    if not steps:
        raise PlanError(f'{source}: the plan has no commands')
    return steps


def load_plan(path):
    path = Path(path)
    if not path.is_file() and (PLANS / path).is_file():
        path = PLANS / path
    if not path.is_file() and (PLANS / f'{path}.txt').is_file():
        path = PLANS / f'{path}.txt'
    if not path.is_file():
        raise PlanError(f'no plan file {path}')
    return parse_plan(path.read_text(encoding='utf-8'), path.name)


def python_command(text=None):
    text = text or os.environ.get('GT_PY') or ''
    if not text:
        return [sys.executable]
    parts = shlex.split(text, posix=os.name != 'nt')
    return [p[1:-1] if len(p) > 1 and p[0] == p[-1] == '"' else p for p in parts]


def clock():
    return time.strftime('%H:%M:%S')


class Queue:
    """Runs the steps of a plan in order; `launch` is the one place a process starts (tests replace it)."""

    def __init__(self, steps, runs, stop_file=None, budget_min=None, dry=False, dry_min=0, python=None, out=None):
        self.steps = steps
        self.runs = Path(runs)
        self.stop = Path(stop_file) if stop_file else self.runs / 'STOP'
        self.budget_s = None if budget_min is None else budget_min * 60
        self.dry, self.dry_s = dry, dry_min * 60
        self.python = python or python_command()
        self.out = out or sys.stdout
        self.used_s = 0.0
        self.failed = []

    def say(self, text):
        print(text, file=self.out, flush=True)

    def finished(self, step):
        """True when a work line already has its results on disk."""
        if not step.done_files:
            return False
        paths = [self.runs / f for f in step.done_files]
        if not all(p.is_file() for p in paths):
            return False
        return not step.done_needs or all(step.done_needs in p.read_text(encoding='utf-8', errors='replace') for p in paths)

    def env(self):
        return {**os.environ, 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'PYTHONUNBUFFERED': '1',
                'GT_RUNS': str(self.runs)}

    def launch(self, command, stdout_path, stderr_path):
        """Run `command` in the pipeline root with its output in the given files; the exit code."""
        with open(stdout_path, 'wb') as out:
            if stderr_path is None:
                return subprocess.run(command, cwd=PIPELINE_ROOT, env=self.env(), stdout=out, stderr=subprocess.STDOUT).returncode
            with open(stderr_path, 'wb') as err:
                return subprocess.run(command, cwd=PIPELINE_ROOT, env=self.env(), stdout=out, stderr=err).returncode

    def command(self, step):
        return [*self.python, '-m', step.module, *step.module_args]

    def attempt(self, step):
        logs = self.runs / 'logs'
        log = logs / f'{step.label}.log'
        command = self.command(step)
        if step.stdout:
            return self.launch(command, self.runs / step.stdout, log)
        return self.launch(command, log, None)

    def run_step(self, step):
        """Run one step with its single retry; the exit code and the seconds it took."""
        logs = self.runs / 'logs'
        start = time.monotonic()
        self.say(f'start {step.label} {clock()}')
        rc = self.attempt(step)
        if rc != 0 and not self.stop.exists():
            self.say(f'retry {step.label}')
            log = logs / f'{step.label}.log'
            if log.exists():
                shutil.move(log, logs / f'{step.label}.log.try1')
            rc = self.attempt(step)
        return rc, time.monotonic() - start

    def dry_line(self, step):
        if step.verb == 'train':
            return f'run {step.label} {step.text}'
        target = f' > <runs>/{step.stdout}' if step.stdout else ''
        return f'cpu {step.verb}{" " + step.text if step.text else ""}{target}'

    def over_budget(self):
        return self.budget_s is not None and self.used_s >= self.budget_s

    def run(self):
        """Run (or print) the plan; the process exit code."""
        if not self.dry:
            (self.runs / 'logs').mkdir(parents=True, exist_ok=True)
            if self.stop.exists():
                self.stop.unlink()
                self.say(f'cleared stale {self.stop}')
        for step in self.steps:
            work = step.verb in WORK
            if not self.dry and work and self.finished(step):
                self.say(f'skip {step.label}')
            elif work and self.stop.exists():
                self.say(f'stopped before {step.label}')
            elif step.verb == 'train' and self.over_budget():
                self.say(f'skipped (budget) {step.label}')
            elif self.dry:
                self.say(self.dry_line(step))
                if step.verb == 'train':
                    self.used_s += self.dry_s
            else:
                rc, seconds = self.run_step(step)
                if step.verb == 'train':
                    self.used_s += seconds
                self.say(f'finished {step.label} rc={rc} {clock()}' + (f' gpu_min={int(self.used_s // 60)}'
                                                                      if step.verb == 'train' else ''))
                if rc != 0:
                    self.failed.append(step.label)
        if not self.dry:
            self.say(f'queue done {clock()} gpu_min={int(self.used_s // 60)}'
                     + (f' failed: {", ".join(self.failed)}' if self.failed else ''))
        return 1 if self.failed else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--plan', help='a plan file, or the name of one in experiments/plans (v3, noapi, ...)')
    ap.add_argument('--budget-min', type=float, help='minutes of wall time the train lines may use')
    ap.add_argument('--dry', action='store_true', help='print what would run; write nothing, skip nothing')
    ap.add_argument('--dry-min', type=float, default=0, help='with --dry: minutes charged per train line (tries the budget)')
    ap.add_argument('--stop-file', help='default: <runs>/STOP')
    ap.add_argument('--stop', action='store_true', help='create the STOP file and exit (the queue stops before its next line)')
    ap.add_argument('--runs', help='run directory (default GT_RUNS, see common/paths.py)')
    ap.add_argument('--python', help='interpreter command (default GT_PY, else the one running this queue)')
    args = ap.parse_args(argv)
    if args.runs:
        os.environ['GT_RUNS'] = args.runs                # before common.paths is imported, as train / gates read it
    from common.paths import runs_for
    runs = Path(args.runs) if args.runs else runs_for()
    if args.stop:
        stop = Path(args.stop_file) if args.stop_file else runs / 'STOP'
        stop.parent.mkdir(parents=True, exist_ok=True)
        stop.touch()
        print(f'created {stop}: the queue finishes the run in progress and skips the rest')
        return 0
    if not args.plan:
        ap.error('--plan is required')
    try:
        steps = load_plan(args.plan)
    except PlanError as e:
        print(e, file=sys.stderr)
        return 2
    queue = Queue(steps, runs, args.stop_file, args.budget_min, args.dry, args.dry_min,
                  python_command(args.python))
    return queue.run()


if __name__ == '__main__':
    sys.exit(main())
