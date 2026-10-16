"""Where the CFG cache and the run directories live. GT_CACHE overrides the cache."""
import os
import sys
from pathlib import Path

DEFAULT_CACHE = Path('C:/Users/chaoa/Downloads/graph_transformer_cache_v2')
COMBINED_CACHE = Path('C:/Users/chaoa/Downloads/graph_transformer_cache_combined')


def cache_for(protocol, env=None):
    """GT_CACHE when set; otherwise the combined cache for kfold_v3 and the v2 cache for every other protocol."""
    env = os.environ if env is None else env
    if env.get('GT_CACHE'):
        return Path(env['GT_CACHE'])
    return COMBINED_CACHE if protocol == 'kfold_v3' else DEFAULT_CACHE


def protocol_in(argv):
    """The value of --protocol in a command line, or None."""
    for i, a in enumerate(argv):
        if a.startswith('--protocol='):
            return a.partition('=')[2]
        if a == '--protocol' and i + 1 < len(argv):
            return argv[i + 1]


CACHE = cache_for(protocol_in(sys.argv[1:]))
RUNS = Path('C:/Users/chaoa/Downloads/graph_transformer_runs')
