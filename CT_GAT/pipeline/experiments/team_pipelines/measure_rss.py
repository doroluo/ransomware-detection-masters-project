"""Run a command and record its wall time and the peak resident memory of its process tree (a psutil poller).

python -m experiments.team_pipelines.measure_rss --out <file.json> [--interval 0.5] -- <command> [args ...]

The command runs with this script's environment and working directory; its output goes to this script's. The JSON holds
`command`, `returncode`, `wall_seconds`, `peak_rss_bytes` (the largest sum of the resident sets of the child and all its
descendants seen at a poll), `peak_rss_gb` and `polls`. The exit code is the command's.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def tree_rss(root):
    """Resident bytes of `root` and its descendants (processes that vanish between listing and reading count 0)."""
    import psutil                                      # imported here: the module imports without psutil
    total = 0
    try:
        procs = [root] + root.children(recursive=True)
    except psutil.NoSuchProcess:
        return 0
    for proc in procs:
        try:
            total += proc.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def run_measured(command, interval=0.5):
    import psutil
    t0 = time.time()
    child = subprocess.Popen(command)
    watch = psutil.Process(child.pid)
    peak, polls = 0, 0
    while child.poll() is None:
        peak, polls = max(peak, tree_rss(watch)), polls + 1
        time.sleep(interval)
    return {'command': command, 'returncode': child.returncode, 'wall_seconds': round(time.time() - t0, 1),
            'peak_rss_bytes': peak, 'peak_rss_gb': round(peak / 2 ** 30, 3), 'polls': polls, 'child_pid': child.pid}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--interval', type=float, default=0.5)
    ap.add_argument('command', nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        ap.error('no command')
    result = run_measured(command, args.interval)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'command'}), flush=True)
    return result['returncode']


if __name__ == '__main__':
    sys.exit(main())
