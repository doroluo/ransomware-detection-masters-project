import hashlib
import json

import pytest

from experiments.remote_model3 import run


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_qualification_gate_rejects_smoke_stale_source_and_missing_epoch(tmp_path, monkeypatch):
    monkeypatch.setattr(run.runner, 'source_hashes', lambda: {'model.py': 'source'})
    monkeypatch.setattr(run, 'protocol_hashes', lambda: {'trial.py': 'adapter'})
    selection, vocab, build = tmp_path / 'selection.json', tmp_path / 'vocab.json', tmp_path / 'data'
    write(selection, {'frozen': True})
    write(vocab, {'fingerprint': 'vocab'})
    path = tmp_path / 'qualification'
    case = path / 'trials' / 'one'
    config = dict(evaluate_test=False)
    plan = dict(execution_profile='gpu-resident', qualification=True, epochs=1, seeds=[0],
                smoke=0, device='cuda', model=3, fold=3, ablation='none', evaluate_test=False,
                build=str(build.resolve()), scientific_sources={'model.py': 'source'},
                adapter_sources={'trial.py': 'adapter'},
                selection_sha256=hashlib.sha256(selection.read_bytes()).hexdigest(),
                common_vocabulary_fingerprint='vocab')
    write(path / 'plan.json', plan)
    write(path / 'status.json', dict(state='complete', completed=1, test_evaluated=False))
    write(case / 'config.json', config)
    write(case / 'metrics.json', dict(epochs_run=1, device='cuda', fingerprint=run.controlled.digest(config),
                                     spec={'pooling': 'mean'}, seed=0))
    write(case / 'history.json', [{'epoch': 1}])
    (case / 'last.pt').write_bytes(b'checkpoint existence marker; never deserialized')
    assert run.qualification_passed(path, build, selection, vocab, 'none')
    write(path / 'plan.json', dict(plan, smoke=8))
    assert not run.qualification_passed(path, build, selection, vocab, 'none')
    write(path / 'plan.json', plan)
    monkeypatch.setattr(run, 'protocol_hashes', lambda: {'trial.py': 'changed'})
    assert not run.qualification_passed(path, build, selection, vocab, 'none')
    monkeypatch.setattr(run, 'protocol_hashes', lambda: {'trial.py': 'adapter'})
    write(case / 'history.json', [])
    assert not run.qualification_passed(path, build, selection, vocab, 'none')


def test_unqualified_full_run_cannot_start_training(tmp_path, monkeypatch):
    for name, value in (('fold3/selection.json', {}), ('fold3/common_api.json', {'fingerprint': 'v'}),
                        ('index.csv', {}), ('data/signal250/settings.json', {}), ('data/signal250/api_vocab.json', [])):
        write(tmp_path / name, value)
    def forbidden(*args):
        raise AssertionError('Training started without qualification')
    monkeypatch.setattr(run.controlled, 'main', forbidden)
    with pytest.raises(SystemExit):
        run.main(['--handoff-root', str(tmp_path), '--variant', 'signal250'])
