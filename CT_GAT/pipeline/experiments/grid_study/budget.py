"""Persist active training/validation wall time, excluding CPU data preparation and idle waits."""
from contextlib import contextmanager
import json
from pathlib import Path
import threading
import time
import uuid

from experiments.model_grid.runner import atomic_json


class BudgetExhausted(InterruptedError):
    pass


class Budget:
    def __init__(self, path, hours, shares=None):
        self.path = Path(path)
        shares = shares or dict(optimizer=8/30, architecture=8/30, coverage=6/30, confirm=8/30)
        self.limits = {k: hours * 3600 * v for k, v in shares.items()}
        self.doc = dict(limit_seconds=hours*3600, phase_limits=self.limits,
                        used_seconds={k: 0. for k in shares}, active=None,
                        accounting='Active training, validation, checkpoint and test wall time; CPU preparation excluded. '
                                   'Stops at epoch boundaries. Crash accounting may miss at most one 2-second heartbeat.')
        if self.path.exists():
            self.doc = json.loads(self.path.read_text())
            if self.doc['phase_limits'] != self.limits:
                raise ValueError('budget differs from saved budget; use a new output directory')
            active=self.doc.get('active')
            if active and active.get('stop_path'):
                old_stop=Path(active['stop_path'])
                if old_stop.exists() and old_stop.read_text(encoding='utf-8')==active.get('stop_tag'):
                    old_stop.unlink()
            self.doc['active']=None
        atomic_json(self.path, self.doc)

    def remaining(self, phase):
        return min(self.limits[phase] - self.doc['used_seconds'][phase],
                   self.doc['limit_seconds'] - sum(self.doc['used_seconds'].values()))

    @contextmanager
    def charge(self, phase, stop_path, trial):
        remaining = self.remaining(phase)
        if remaining <= 0:
            raise BudgetExhausted(phase)
        stop_path = Path(stop_path)
        if stop_path.exists():
            raise InterruptedError(f'user STOP file: {stop_path}')
        stop_path.parent.mkdir(parents=True, exist_ok=True)
        tag = 'budget:' + uuid.uuid4().hex
        done, expired = threading.Event(), threading.Event()
        base, started = self.doc['used_seconds'][phase], time.perf_counter()
        def save():
            self.doc['used_seconds'][phase] = base + time.perf_counter() - started
            self.doc['active'] = dict(phase=phase, trial=str(trial),stop_path=str(stop_path.resolve()),stop_tag=tag)
            atomic_json(self.path, self.doc)
        def watch():
            while not done.wait(min(2., max(.001, remaining - (time.perf_counter()-started)))):
                save()
                if time.perf_counter() - started >= remaining:
                    expired.set()
                    # Never replace a user's STOP request.
                    try:
                        with stop_path.open('x', encoding='utf-8') as f:
                            f.write(tag)
                    except FileExistsError:
                        pass
                    return
        save()
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        try:
            yield
        except InterruptedError as exc:
            if expired.is_set():
                raise BudgetExhausted(phase) from exc
            raise
        finally:
            done.set(); watcher.join()
            save()
            self.doc['active'] = None
            atomic_json(self.path, self.doc)
            if stop_path.exists() and stop_path.read_text(encoding='utf-8') == tag:
                stop_path.unlink()
