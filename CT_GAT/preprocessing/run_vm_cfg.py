#!/usr/bin/env python3
"""Bounded VM supervisor: wait for inventory, extract, then package safe outputs."""
import json
from pathlib import Path
import subprocess
import sys
import time

from vm_ida_cfg import check_out, write_json


def main():
    out = check_out(Path.home() / 'work/out/ida_cfg_20261006')
    script = out / 'vm_ida_cfg.py'
    stage = 'waiting_for_inventory'
    try:
        write_json(out / 'pipeline_state.json', dict(stage=stage))
        deadline = time.monotonic() + 3 * 3600
        while not (out / 'summary.json').exists():
            if time.monotonic() > deadline:
                raise RuntimeError('inventory_wait_limit')
            time.sleep(10)
        for stage, command in [('extracting', 'run'), ('packaging', 'package')]:
            write_json(out / 'pipeline_state.json', dict(stage=stage))
            subprocess.run([sys.executable, '-u', str(script), command, '--out', str(out)], check=True)
        receipt = json.loads((out / 'transfer_ready.json').read_text())
        write_json(out / 'pipeline_state.json', dict(stage='ready_for_transfer', graphs=receipt['graphs']))
    except Exception as error:
        write_json(out / 'pipeline_failure.json', dict(stage=stage, reason=type(error).__name__))
        raise


if __name__ == '__main__':
    main()
