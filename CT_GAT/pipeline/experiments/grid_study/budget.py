"""Persist active training/validation wall time, excluding CPU data preparation and idle waits."""
from contextlib import contextmanager
import json
import hashlib
import os
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
        saved = self._load()
        if saved is not None:
            self.doc = saved
            if self.doc['phase_limits'] != self.limits:
                raise ValueError('budget differs from saved budget; use a new output directory')
            active=self.doc.get('active')
            if active and active.get('stop_path'):
                old_stop=Path(active['stop_path'])
                if old_stop.exists() and old_stop.read_text(encoding='utf-8')==active.get('stop_tag'):
                    old_stop.unlink()
            self.doc['active']=None
        self._persist()

    def _load(self):
        # Legacy .tmp may be newer than a Windows-locked display file.
        candidates = []
        paths = [self.path, self.path.with_suffix(self.path.suffix + '.tmp')]
        paths += list(self.path.with_suffix('.journal').glob('*.json'))
        for path in paths:
            try:
                record = json.loads(path.read_text(encoding='utf-8'))
                if 'payload' in record:
                    payload = record['payload']
                    if hashlib.sha256(payload.encode()).hexdigest() != record['sha256']:
                        continue
                    record = json.loads(payload)
                values = record['used_seconds']
                if set(values) != set(self.limits) or any(not isinstance(v, (int, float))
                        or not (0 <= v < float('inf')) for v in values.values()):
                    continue
                candidates.append((sum(values.values()), path.stat().st_mtime_ns, record))
            except (OSError, ValueError, KeyError, TypeError):
                continue  # A crash can leave a partial new journal record.
        if candidates:
            return max(candidates, key=lambda item: item[:2])[2]
        if self.path.exists() or paths[1].exists() or len(paths) > 2:
            raise RuntimeError('No valid budget record; refusing to reset consumed time')
        return None

    def _persist(self):
        # New immutable files avoid replacing a file held open by a Windows viewer.
        journal = self.path.with_suffix('.journal')
        journal.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.doc, sort_keys=True, allow_nan=False)
        snapshot = journal / f'{time.time_ns():020d}_{uuid.uuid4().hex}.json'
        with snapshot.open('x', encoding='utf-8') as f:
            json.dump(dict(payload=payload, sha256=hashlib.sha256(payload.encode()).hexdigest()), f)
            f.flush()
            os.fsync(f.fileno())
        # The ledger is durable before attempting the human-readable display copy.
        try:
            atomic_json(self.path, self.doc)
        except PermissionError:
            if not getattr(self, '_reported_lock', False):
                print(f'Budget display locked: authoritative snapshots are in {journal}', flush=True)
                self._reported_lock = True
        for old in sorted(journal.glob('*.json'))[:-8]:
            try:
                old.unlink()
            except OSError:
                pass

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
        errors = []
        base, started = self.doc['used_seconds'][phase], time.perf_counter()
        def save():
            self.doc['used_seconds'][phase] = base + time.perf_counter() - started
            self.doc['active'] = dict(phase=phase, trial=str(trial),stop_path=str(stop_path.resolve()),stop_tag=tag)
            self._persist()
        def request_stop():
            # Never replace a user's STOP request.
            try:
                with stop_path.open('x', encoding='utf-8') as f:
                    f.write(tag)
            except FileExistsError:
                pass
        def watch():
            while not done.wait(min(2., max(.001, remaining - (time.perf_counter()-started)))):
                try:
                    save()
                except OSError as exc:
                    errors.append(exc)
                    request_stop()  # Fail closed if the durable ledger cannot be written.
                    return
                if time.perf_counter() - started >= remaining:
                    expired.set()
                    request_stop()
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
            self.doc['used_seconds'][phase] = base + time.perf_counter() - started
            self.doc['active'] = None
            self._persist()
            if stop_path.exists() and stop_path.read_text(encoding='utf-8') == tag:
                stop_path.unlink()
            if errors:
                raise OSError('Budget ledger write failed; training stopped at a checkpoint') from errors[0]
