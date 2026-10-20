"""A stand-in for the interpreter of the experiment queue: `python queue_stub.py -m <module> <args>`, in the place of
`python -m <module> <args>`. It records every call, writes the files a real run would write and fails on demand, so that
tests/test_queue.py drives the real subprocess loop of experiments/queue.py without training anything.

Environment: GT_RUNS (the queue sets it), STUB_FAIL_ONCE / STUB_FAIL_ALWAYS (comma lists of run names), STUB_STOP_AT (a run name
during which the STOP file appears, as stop_queue does), STUB_KILL (a run name whose attempt fails without writing metrics.json
and creates the STOP file, as a kill with --now does).
"""
import json
import os
import sys
from pathlib import Path

runs = Path(os.environ['GT_RUNS'])
module, args = sys.argv[2], sys.argv[3:]
name = args[args.index('--out') + 1] if '--out' in args else module
(runs / 'calls.log').open('a').write(json.dumps({'module': module, 'name': name, 'args': args,
                                                 'alloc': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
                                                 'unbuffered': os.environ.get('PYTHONUNBUFFERED')}) + '\n')
listed = lambda key: [n for n in os.environ.get(key, '').split(',') if n]
print(f'stub {module} {name}')
print(f'stub stderr {name}', file=sys.stderr)
if module == 's04_evaluation.gates':
    print(f'# gates {args[args.index("--stage") + 1]}')
if name in listed('STUB_FAIL_ALWAYS'):
    sys.exit(1)
flag = runs / f'.failed_once_{name}'
if name in listed('STUB_FAIL_ONCE') and not flag.exists():
    flag.touch()
    sys.exit(1)
if name in listed('STUB_KILL'):
    (runs / 'STOP').touch()
    sys.exit(1)
if module == 'experiments.train':
    (runs / name).mkdir(parents=True, exist_ok=True)
    (runs / name / 'metrics.json').write_text('{}')
if name in listed('STUB_STOP_AT'):
    (runs / 'STOP').touch()
