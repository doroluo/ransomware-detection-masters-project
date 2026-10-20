import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                       # tiny_cache
sys.path.insert(0, str(HERE.parent))                # the package root: common, s02_preprocess, ...


def pytest_collection_modifyitems(config, items):
    """`real_data` tests need the cache of the run (config.CACHE); `gpu` tests need CUDA. Both skip cleanly without."""
    from common.config import CACHE
    no_data = not (CACHE / 'index.csv').is_file()
    no_cuda = False
    if any('gpu' in item.keywords for item in items):
        import torch
        no_cuda = not torch.cuda.is_available()
    for item in items:
        if no_data and 'real_data' in item.keywords:
            item.add_marker(pytest.mark.skip(reason='needs the CFG cache (datasets.json or GT_CACHE)'))
        if no_cuda and 'gpu' in item.keywords:
            item.add_marker(pytest.mark.skip(reason='needs CUDA'))
