"""Janice Park's five-channel Model 1 as a `common.registry.ModelSpec`: the model `fivechannel`.

Like `tokenseq` it owns its loop (`five_channel/train.py`): two logits per binary, instruction tuples from the `.pt` files of a
pipeline-B build instead of the CFG cache. A spec that owns its loop says so with `run(cfg, out_name, **options)`;
`experiments.train.run` hands the run over to it, and `build`, `loader` and `forward` are not used and refuse. The loop module
is imported on `run`, not when the registry builds the spec.
"""
import shutil

from common.config import RUNS, RunConfig

OPTIONS = ('build', 'device', 'smoke')           # what `run` takes besides the config (experiments.train flags)


class FiveChannelSpec:
    name = 'fivechannel'
    requires = ('index.csv',)       # the folds come from the cache's index (`common.manifests`); the tuples come from the build

    def check(self, cfg: RunConfig):
        cfg.check_fivechannel()

    def run(self, cfg: RunConfig, out_name=None, **options):
        """Train the model on `cfg` and write <runs>/<out_name or cfg.name>; returns the metrics. `options` are `OPTIONS`."""
        from s03a_transformer.five_channel import train
        unknown = set(options) - set(OPTIONS)
        if unknown:
            raise TypeError(f'fivechannel takes {", ".join(OPTIONS)}, not {", ".join(sorted(unknown))}')
        if out_name in (None, cfg.name) or options.get('smoke') is not None:
            return train.run(cfg, **options)
        stage = RUNS / f'.stage-{out_name}'                  # write_run insists on a directory named like the config
        shutil.rmtree(stage, ignore_errors=True)
        try:
            metrics = train.run(cfg, out_root=stage, **options)
            (RUNS / out_name).mkdir(parents=True, exist_ok=True)
            for path in (stage / cfg.name).iterdir():
                path.replace(RUNS / out_name / path.name)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        return metrics

    def build(self, cfg, info):
        raise NotImplementedError('fivechannel runs its own loop (run); the registry loop does not build it')

    loader = forward = build

    def variant_tokens(self, cfg: RunConfig):
        return tuple(cfg.variant.split('-'))
