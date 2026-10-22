"""What a gate stage is, and the rules and names its definition file uses.

A stage is data: `experiments/stages/<name>.py` defines `STAGE = Stage(...)` with the arms (label, group) it reads, the pairs
of arms it compares (labels, second minus first), the seeds the pairs use, the rule column and the intro text. The engine
(`gates.py`, `python -m s04_evaluation.gates --stage <name>`) loads the file and lays the stage out: a pooled table, then the paired
differences.

This module imports nothing but `common.config`, so a stage file can import it while the engine is `__main__`.
"""
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Union

from common.config import RunConfig

SEED = 0
LABEL = 'seed 0 only, not the pre-registered test'
MIN_AUC, MIN_F1 = 0.01, 0.02            # the pre-registered minimum effect (PLAN_GNN.md section 5)
NI_AUC, NI_F1 = 0.01, 0.02              # non-inferiority margins of stage subgraph


@dataclass(frozen=True)
class Stage:
    """name      the stage name: the file name in experiments/stages and the value of --stage
    intro     the paragraph under the title; a function of the number of resamples when the text quotes it
    arms      ((label, group), ...): the groups of RunConfig.group (or a baseline group such as `tfidf`) the stage reads
    pairs     ((first label, second label), ...): paired differences, second minus first
    seeds     the seeds the pairs use (those complete in both arms); the pooled tables are always seed 0
    rule      None, or rule(first, second, result, info) -> the text of the extra `rule` column of the paired table
    extra     None, or the name of a table printed after the pooled one (`gate_tables.EXTRAS`)
    options   `match_rate_builds`, ((name, folder under the run directory), ...): a match-rate line
              (`gate_tables.match_rate_lines`) per pipeline-B build, after `extra`"""
    name: str
    intro: Union[str, Callable[[int], str]] = ''
    arms: tuple = ()
    pairs: tuple = ()
    seeds: tuple = (SEED,)
    rule: Optional[Callable] = None
    extra: Optional[str] = None
    options: Mapping = field(default_factory=dict)

    def intro_text(self, resamples):
        return self.intro(resamples) if callable(self.intro) else self.intro


def variant_of(**kw):
    """RunConfig.variant of a GNN arm, so the group names here cannot drift from the run names."""
    return RunConfig(**{'model': 'gnn', 'protocol': 'kfold_v3', 'fold': 0, 'arch': 'all', **kw}).variant


def x86_group(variant):
    return f'kfold_x86_{variant}'


def signed_ci(r):
    return f"{r['diff'] + 0.0:+.3f} [{r['lo'] + 0.0:+.3f}, {r['hi'] + 0.0:+.3f}]"


def beats(r):
    """The pre-registered minimum effect on one paired comparison (b - a): dAUC >= 0.01 and dmacro-F1 >= 0.02 with
    both 95% intervals above zero."""
    return all(r[k]['diff'] >= m and r[k]['lo'] > 0 for k, m in (('auc', MIN_AUC), ('macro_f1', MIN_F1)))


def non_inferior(result):
    """Lower bounds of the paired 95% intervals above -0.01 (dAUC) and -0.02 (dmacro-F1)."""
    return result['auc']['lo'] > -NI_AUC and result['macro_f1']['lo'] > -NI_F1
