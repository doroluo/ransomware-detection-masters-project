import subprocess
import sys

import psutil
import pytest

from experiments.remote_model3.benchmark import profile_spec, select_training_rows, terminate_worker_tree


def test_profiles_keep_logical_batch_and_record_execution_changes():
    base = dict(batch_size=8, micro_batch_size=1, attention_backend='math',
                activation_checkpoint=True, activation_offload=True, block_length=64)
    assert profile_spec(base, 'original') == base
    resident = profile_spec(base, 'resident')
    assert resident['micro_batch_size'] == 1 and resident['batch_size'] == 8
    assert resident['activation_checkpoint'] is False and resident['activation_offload'] is False
    batched = profile_spec(base, 'batched')
    assert batched['micro_batch_size'] == batched['batch_size'] == 8
    assert batched['block_length'] == 64 and batched['attention_backend'] == 'math'
    assert base['activation_checkpoint'] is True
    with pytest.raises(ValueError, match='Unknown profile'):
        profile_spec(base, 'unknown')


def test_sample_selection_uses_both_classes_and_larger_training_examples():
    rows = [dict(sample_id=str(label) + str(i), label=label, num_instructions=i)
            for label in (0, 1) for i in range(10)]
    selected = select_training_rows(rows, 8)
    assert len(selected) == len({r['sample_id'] for r in selected}) == 8
    assert all(sum(r['label'] == label for r in selected) == 4 for label in (0, 1))
    assert {r['sample_id'] for r in selected} >= {'09', '19'}
    with pytest.raises(ValueError, match='Too few'):
        select_training_rows(rows[:2], 8)


def test_timeout_terminates_only_the_diagnostic_process_tree():
    bystander = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    program = (
        'import subprocess,sys,time; '
        'child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
        'print(child.pid,flush=True); time.sleep(60)'
    )
    worker = subprocess.Popen([sys.executable, '-c', program], stdout=subprocess.PIPE, text=True)
    try:
        child = psutil.Process(int(worker.stdout.readline().strip()))
        terminate_worker_tree(worker)
        assert worker.poll() is not None
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
        assert bystander.poll() is None
    finally:
        if worker.poll() is None:
            terminate_worker_tree(worker)
        bystander.terminate()
        bystander.wait(timeout=5)
