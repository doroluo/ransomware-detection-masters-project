"""One-off builder of tests/fixtures/reader_side: a real binary's reader output (copied byte for byte from the team's
output directory) and the matching cache sample, with the cache vocabulary shrunk to the strings it uses.

python tests/fixtures/build_reader_fixture.py <sample_id> [cache] [reader root]
"""
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sid = sys.argv[1]
cache = Path(sys.argv[2] if len(sys.argv) > 2 else 'C:/Users/chaoa/Downloads/graph_transformer_cache_v2')
root = Path(sys.argv[3] if len(sys.argv) > 3 else 'C:/Users/chaoa/Downloads/ida_features/out')
out = Path(__file__).parent / 'reader_side'
shutil.rmtree(out, ignore_errors=True)
(out / 'reader' / sid).mkdir(parents=True)
for name in ('operand_roles.jsonl.gz', 'imports.json', 'manifest.json'):
    shutil.copyfile(root / sid / name, out / 'reader' / sid / name)
vocab = json.loads((cache / 'vocab.json').read_text(encoding='utf-8'))
with np.load(cache / 'samples' / f'{sid}.npz') as z:
    keep = {k: z[k] for k in ('fn_blk_ptr', 'fn_flags', 'blk_insn_ptr', 'blk_api_ptr', 'mn', 'api', 'num_functions',
                              'num_blocks', 'num_instructions')}
small = {}
for k in ('mn', 'api'):
    used = np.unique(keep[k])
    small[k] = [vocab[k][i] for i in used]
    keep[k] = np.searchsorted(used, keep[k]).astype(np.int32)
(out / 'cache' / 'samples').mkdir(parents=True)
np.savez_compressed(out / 'cache' / 'samples' / f'{sid}.npz', **keep)
(out / 'cache' / 'vocab.json').write_text(json.dumps(small), encoding='utf-8')
with (out / 'cache' / 'index.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['sample_id', 'status'])
    w.writerow([sid, 'ok'])
print(sid, {k: len(v) for k, v in small.items()})
