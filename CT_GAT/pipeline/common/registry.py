"""The model registry: what `experiments/train.py` needs to know about a model, and the table of models.

A model is one `ModelSpec` and one line in `MODELS`. The training loop (`experiments/train.py`) knows nothing else about
a model: it asks the registry for the spec of `cfg.model`, lets the spec build the network and the loaders, and calls
`forward`; class weights, AdamW, gradient clipping, patience, the F1 and AUC checkpoints, the validation thresholds and
the run directory (`common/run_io.py`) are the loop's.

    name            the value of `RunConfig.model` the spec serves
    requires        paths under the cache directory that every run of the model reads (checked before anything is written)
    check(cfg)      raise for options the model does not support (before anything is written)
    build(cfg, info)  fit what the model fits on the training manifest (vocabulary, ...), write its files to `info.out`, and
                    return the `torch.nn.Module`; also keeps what `loader` needs
    loader(cfg, rows, train)  an iterable of batches over `rows`, in training (augmenting, shuffling) or evaluation mode
    forward(model, batch)     the logits [batch size] of a batch that is already on the device
    variant_tokens(cfg)       the dash-separated tokens of the run-name variant (`RunConfig.variant`) that the model's options
                    produce; the name itself stays `RunConfig.name`
    run(cfg, out_name, **options)   optional: a model whose loop is not the registry's (Model 1: `fivechannel`) trains and writes its
                    run directory itself; `experiments.train.run` calls it after `check` and does not call `build`, `loader`
                    or `forward`

`MODELS` maps a name to `'module:Class'`; the class is imported on first use, so that importing this module costs nothing
(no torch). `spec_for(cfg)` returns a fresh instance per run, because a spec keeps the state it fitted in `build`.
"""
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable

from common.manifests import Manifest

MODELS = {
    'fivechannel': 's03a_transformer.five_channel.spec:FiveChannelSpec',
}
REMOVED_MODELS = ('gnn', 'tokenseq')
REFERENCE_BRANCH = 'jsonl-reference-models'


def unknown_model_message(name):
    """Why `name` has no spec: the earlier jsonl-only models are on a branch, anything else is a typo or a spec to add."""
    if name in REMOVED_MODELS:
        return (f'model {name!r} is not on this branch: the earlier jsonl-only models ({", ".join(REMOVED_MODELS)}) and their '
                f'commands are on branch `{REFERENCE_BRANCH}`; the models here are {", ".join(MODELS)}')
    return f'model {name!r} has no spec in common/registry.py MODELS ({", ".join(MODELS)}); add a ModelSpec and one line there'


@dataclass
class RunInfo:
    """What the loop hands to `ModelSpec.build`: the manifest of the run, its directory, and a place for the spec to leave
    a one-line description of what it fitted (`summary`, printed at the start of the run)."""
    manifest: Manifest
    out: Path
    summary: dict = field(default_factory=dict)


@runtime_checkable
class ModelSpec(Protocol):
    name: str
    requires: tuple

    def check(self, cfg) -> None: ...

    def build(self, cfg, info: RunInfo) -> Any: ...

    def loader(self, cfg, rows: list, train: bool) -> Iterable: ...

    def forward(self, model, batch) -> Any: ...

    def variant_tokens(self, cfg) -> tuple: ...


def spec_for(cfg) -> ModelSpec:
    """A new spec of `cfg.model`. A model `RunConfig` still names (for the run directories on disk) but no spec serves raises
    NotImplementedError."""
    if cfg.model not in MODELS:
        raise NotImplementedError(unknown_model_message(cfg.model))
    module, _, attr = MODELS[cfg.model].partition(':')
    spec = getattr(importlib.import_module(module), attr)()
    if spec.name != cfg.model:
        raise ValueError(f'{MODELS[cfg.model]} is the spec of {spec.name!r}, registered as {cfg.model!r}')
    return spec
