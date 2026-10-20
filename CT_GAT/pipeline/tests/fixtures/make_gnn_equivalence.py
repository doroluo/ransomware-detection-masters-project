"""Record tests/fixtures/gnn_equivalence.npz: the batches and one forward/backward step of the GNN on the tiny cache, as the
3-block-level-counting branch (3764cc7, CT_GAT/block_level_counting) computes them. Two steps, two processes (both packages are
called `common`):

    # 1. from CT_GAT/pipeline of this checkout: write the tiny cache (tests/tiny_cache.py) to a directory
    python tests/fixtures/make_gnn_equivalence.py cache <cache dir>
    # 2. from anywhere, with the interpreter of the branch's checkout: compute on it with the branch's code
    python tests/fixtures/make_gnn_equivalence.py source <CT_GAT/block_level_counting of the branch> <cache dir> <out.npz>

The computation is tests/gnn_equivalence_cases.py `compute`; `test_gnn_equivalence.py` runs it on this checkout.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

TESTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS))


def write_cache(cache):
    sys.path.insert(0, str(TESTS.parent))
    import tiny_cache
    tiny_cache.build(Path(cache))


def record(source_root, cache, out):
    sys.path.insert(0, str(source_root))
    import numpy as np
    import torch
    from common import manifests
    from common.config import RunConfig
    from common.manifests import load_index, make_manifest
    from gnn_equivalence_cases import compute
    from pipeline.batching import loader
    from pipeline.model import HierGNN
    from pipeline.s03_tokenizer import vocab as vocab_module
    from pipeline.s03_tokenizer.vocab import TokenVocab
    from pipeline.s10_unused_import_features import imports_feat
    from pipeline.s10_unused_import_features.imports_feat import ImportVocab
    cache = Path(cache)
    for module in (manifests, vocab_module, imports_feat):
        module.CACHE = cache
    api = SimpleNamespace(RunConfig=RunConfig, load_index=load_index, make_manifest=make_manifest, TokenVocab=TokenVocab,
                          ImportVocab=ImportVocab, loader=loader, HierGNN=HierGNN)
    arrays = compute(api, cache)
    arrays['meta/torch'] = np.array(torch.__version__)
    np.savez_compressed(out, **arrays)
    print(f'{len(arrays)} arrays to {out}')


if __name__ == '__main__':
    if sys.argv[1] == 'cache':
        write_cache(sys.argv[2])
    else:
        record(*sys.argv[2:5])
