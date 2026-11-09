import json
import copy
from types import SimpleNamespace

import pytest
import torch

from experiments.final_paper.selection import ValidationStudy, batch_cases, head_cases, reuse_metadata
from experiments.grid_study import grids
from experiments.model_grid.data import digest
from experiments.final_paper import canonical, trial
from experiments.final_paper.controlled import resource_failure
from experiments.final_paper.data import ControlledPrepared, ablate_sample
from experiments.model_grid import runner
from experiments.model_grid.models import BlockData, Model3


def test_extended_model3_grids_keep_pooling_pairs():
    for cases, size in ((head_cases(3, grids.base_spec(3)), 18), (batch_cases(3, grids.base_spec(3)), 6)):
        assert len(cases) == size
        groups = {}
        for spec in cases:
            groups.setdefault(digest({k: v for k, v in spec.items() if k != 'pooling'}), set()).add(spec['pooling'])
            assert spec['width'] % spec['transformer_heads'] == 0
        assert all(pools == {'mean', 'cls'} for pools in groups.values())


def test_selection_worker_rejects_outer_test_before_data_access():
    worker = object.__new__(ValidationStudy)
    with pytest.raises(ValueError, match='outer test'):
        worker.trial(1, {}, 'confirm', 'confirm', None, '500', 1, 0, test=True)


def test_reuse_requires_exact_contract_and_retains_original_weights(tmp_path):
    src, out = tmp_path / 'original', tmp_path / 'new'
    src.mkdir()
    contract = dict(model=1, seed=0, epochs=12, source_hashes={'models.py': 'abc'}, dataset_fingerprint='data')
    result = dict(fingerprint=digest(contract), seed=0)
    (src / 'config.json').write_text(json.dumps(contract))
    (src / 'metrics.json').write_text(json.dumps(result))
    (src / 'best.pt').write_bytes(b'original checkpoint')
    assert reuse_metadata([src / 'metrics.json'], {**contract, 'seed': 1}, out) is None
    assert reuse_metadata([src / 'metrics.json'], contract, out) == result
    assert (src / 'best.pt').read_bytes() == b'original checkpoint'
    assert not (out / 'best.pt').exists()
    assert json.loads((out / 'reused_from.json').read_text())['path'] == str(src.resolve())


def test_api_remapping_uses_training_names_not_source_numbers():
    base = canonical.tok.FIRST_API_ID
    a = canonical.vocabulary_lookup(['Open', 'Read', 'ValidationOnly'], ['Open', 'Read'])
    b = canonical.vocabulary_lookup(['Read', 'Open', 'ValidationOnly'], ['Open', 'Read'])
    assert a[base].item() == b[base + 1].item() == base
    assert a[base + 1].item() == b[base].item() == base + 1
    assert a[base + 2].item() == b[base + 2].item() == canonical.tok.UNK_API
    assert torch.equal(a[:base], torch.arange(base))


@pytest.fixture
def controlled_build(tmp_path, monkeypatch):
    from experiments.grid_study import coverage
    splits = {}
    names = ['API0', 'API1', 'ValidationOnly']
    torch.set_num_threads(1)
    for i, split in enumerate(('train', 'val', 'test')):
        splits[split] = []
        for label in (0, 1):
            sid = f'{i * 2 + label:064x}'
            row = dict(sample_id=sid, label=label, family=f'{split}{label}', identity_group=sid,
                       arch='x86', num_instructions=4, num_blocks=2)
            splits[split].append(row)
            ids = torch.ones(4, 5, dtype=torch.long)
            ids[:, 0] = torch.tensor([2, 3, 4, 5])
            ids[:, 3] = canonical.tok.FIRST_API_ID + (label if split == 'train' else 2)
            torch.save(dict(instruction_ids=ids, insn_to_block=torch.tensor([0, 0, 1, 1]),
                instruction_matched=torch.ones(4, dtype=torch.bool), block_features=torch.ones(2, 77),
                edge_index=torch.tensor([[0], [1]]), edge_attr=torch.eye(7)[:1],
                edge_back=torch.tensor([False]), label=label), tmp_path / f'{sid}.pt')
    (tmp_path / 'settings.json').write_text('{"excluded": {}}')
    (tmp_path / 'api_vocab.json').write_text(json.dumps(names))
    (tmp_path / 'index.csv').write_text('synthetic')
    monkeypatch.setattr(canonical, 'manifest_for', lambda *a: copy.deepcopy(splits))
    monkeypatch.setattr(coverage, 'manifest_for', lambda *a: copy.deepcopy(splits))
    from experiments.final_paper import data as paper_data
    monkeypatch.setattr(paper_data, 'manifest_for', lambda *a: copy.deepcopy(splits))
    return tmp_path, splits


def test_common_vocabulary_never_fits_validation_or_test(controlled_build, monkeypatch):
    root, splits = controlled_build
    reads = []
    load = torch.load
    def track(path, **kwargs):
        reads.append(path.stem)
        return load(path, **kwargs)
    monkeypatch.setattr(torch, 'load', track)
    v = canonical.fit_common_vocabulary({'source': root}, root / 'index.csv', 'combined', 0, 0,
                                        root / 'fitted.json')
    assert v['names'] == ['API0', 'API1']
    assert set(reads) == {r['sample_id'] for r in splits['train']}
    assert v['validation_or_test_used_for_fitting'] is False


@pytest.mark.parametrize('model', [1, 2, 3])
def test_controlled_preparation_keeps_test_closed_and_source_unchanged(controlled_build, monkeypatch, model):
    root, splits = controlled_build
    v = canonical.fit_common_vocabulary({'source': root}, root / 'index.csv', 'combined', 0, 0,
                                        root / 'fitted.json')
    before = {p: canonical.file_sha256(p) for p in root.glob('*.pt')}
    reads = []
    old = ControlledPrepared.read
    def track(self, sid):
        reads.append(sid)
        return old(self, sid)
    monkeypatch.setattr(ControlledPrepared, 'read', track)
    spec = dict(grids.base_spec(model), sequence_length=4, block_length=4)
    data = ControlledPrepared(root, root / 'index.csv', 'combined', 0, 0, model, spec, root / 'packed', v)
    assert not {r['sample_id'] for r in splits['test']} & set(reads)
    assert data.api_size == canonical.tok.FIRST_API_ID + 2
    val = data.samples[splits['val'][0]['sample_id']]
    if model == 1:
        assert val['ids'][:, 3].eq(canonical.tok.UNK_API).all()
    if model == 3:
        assert val.insn_ids[:, 3].eq(canonical.tok.UNK_API).all()
    data.load_test()
    assert {r['sample_id'] for r in splits['test']} <= set(reads)
    assert all(canonical.file_sha256(p) == fingerprint for p, fingerprint in before.items())


def test_shuffle_control_preserves_block_ownership_and_input_multiset():
    ids = torch.arange(30).reshape(6, 5)
    sample = BlockData(insn_ids=ids.clone(), insn_block=torch.tensor([0, 0, 0, 1, 1, 1]), num_nodes=2)
    other = sample.clone()
    ablate_sample(3, sample, 'shuffle_instructions', 'sample')
    ablate_sample(3, other, 'shuffle_instructions', 'sample')
    assert torch.equal(sample.insn_ids, other.insn_ids)
    assert sample.insn_block.tolist() == [0, 0, 0, 1, 1, 1]
    for start in (0, 3):
        assert sorted(map(tuple, sample.insn_ids[start:start + 3].tolist())) == sorted(map(tuple, ids[start:start + 3].tolist()))


def tiny_training_data():
    samples, rows, splits = {}, {}, {}
    for split, labels in [('train', [0, 1, 0, 1, 0]), ('val', [0, 1])]:
        splits[split] = []
        for i, label in enumerate(labels):
            sid = f'{split}{i}'
            row = dict(sample_id=sid, label=label, family=sid, arch='x86')
            rows[sid] = row; splits[split].append(row)
            ids = torch.ones(4, 5, dtype=torch.long); ids[:, 0] = torch.tensor([2, 3, 4, 5]) + label
            samples[sid] = dict(ids=ids, label=label)
    return SimpleNamespace(samples=samples, rows=rows, splits=splits, api_size=canonical.tok.FIRST_API_ID,
        fingerprint='synthetic-training', load_test=lambda: (_ for _ in ()).throw(AssertionError('test opened')))


def small_spec():
    return dict(grids.base_spec(1), width=8, transformer_heads=2, transformer_layers=1, dropout=0.,
                sequence_length=4, batch_size=3)


def test_microbatch_gradients_match_weighted_full_batches_with_short_tail(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    gradients = []
    step = torch.optim.AdamW.step
    def record(self, *args, **kwargs):
        gradients.append([p.grad.clone() if p.grad is not None else None
                          for group in self.param_groups for p in group['params']])
        return step(self, *args, **kwargs)
    monkeypatch.setattr(torch.optim.AdamW, 'step', record)
    spec = small_spec(); data = tiny_training_data(); sources = runner.source_hashes()
    runner.run_trial(1, spec, 7, 2, 4, data, tmp_path / 'full', torch.device('cpu'), sources)
    trial.run_trial(1, dict(spec, micro_batch_size=1), 7, 2, 4, data, tmp_path / 'micro', torch.device('cpu'), sources)
    assert len(gradients) == 8  # Two logical batches per epoch, including a short tail.
    for full, micro in zip(gradients[:4], gradients[4:]):
        for a, b in zip(full, micro):
            assert (a is None) == (b is None)
            if a is not None:
                assert torch.allclose(a, b, atol=1e-6, rtol=1e-5)
    # Attention key-bias gradients are analytically zero; tiny FP32 residuals can
    # produce different Adam updates there without changing predictions.
    assert (tmp_path / 'full' / 'validation_predictions.csv').read_bytes() == (tmp_path / 'micro' / 'validation_predictions.csv').read_bytes()


def test_microbatch_resume_preserves_weights_exactly(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    spec = dict(small_spec(), micro_batch_size=1, dropout=.2); data = tiny_training_data(); sources = runner.source_hashes()
    trial.run_trial(1, spec, 7, 2, 4, data, tmp_path / 'full', torch.device('cpu'), sources)
    save = trial.save_checkpoint
    def interrupt(path, value):
        save(path, value)
        if path == tmp_path / 'resume' / 'last.pt' and value['epoch'] == 1:
            raise InterruptedError('checkpoint stop')
    monkeypatch.setattr(trial, 'save_checkpoint', interrupt)
    with pytest.raises(InterruptedError):
        trial.run_trial(1, spec, 7, 2, 4, data, tmp_path / 'resume', torch.device('cpu'), sources)
    monkeypatch.setattr(trial, 'save_checkpoint', save)
    trial.run_trial(1, spec, 7, 2, 4, data, tmp_path / 'resume', torch.device('cpu'), sources)
    a = torch.load(tmp_path / 'full' / 'last.pt', weights_only=False)['model']
    b = torch.load(tmp_path / 'resume' / 'last.pt', weights_only=False)['model']
    assert all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize('pool', ['mean', 'cls'])
@pytest.mark.parametrize('dropout', [0., .2])
def test_activation_checkpoint_preserves_model3_outputs_and_gradients(pool, dropout):
    torch.set_num_threads(1)
    spec = dict(small_spec(), block_length=4, block_chunk=2, pooling=pool, gat_hidden=8, gat_heads=2, gat_layers=1, dropout=dropout)
    original = Model3(canonical.tok.FIRST_API_ID, spec, 3)
    checkpointed = trial.CheckpointModel3(canonical.tok.FIRST_API_ID, spec, 3)
    checkpointed.load_state_dict(original.state_dict())
    ids = torch.ones(5, 5, dtype=torch.long); ids[:, 0] = torch.tensor([2, 3, 4, 5, 6])
    sample = BlockData(insn_ids=ids, insn_block=torch.tensor([0, 0, 0, 1, 1]), num_nodes=3)
    torch.manual_seed(44); a = original.block_vectors(sample)
    torch.manual_seed(44); b = checkpointed.block_vectors(sample)
    assert torch.allclose(a, b, atol=1e-6)
    weight = torch.arange(8)
    (a * weight).sum().backward(); (b * weight).sum().backward()
    for x, y in zip(original.encoder.parameters(), checkpointed.encoder.parameters()):
        if x.grad is not None:
            assert y.grad is not None and torch.allclose(x.grad, y.grad, atol=1e-5, rtol=1e-5)


def test_resource_errors_do_not_hide_programming_errors():
    assert resource_failure(RuntimeError('CUDA error: out of memory'))
    assert resource_failure(MemoryError('allocation'))
    assert not resource_failure(RuntimeError('mat1 and mat2 shapes cannot be multiplied'))


def test_campaign_graph_is_bounded_and_has_per_fold_selections(tmp_path):
    from experiments.final_paper.plan import REPRESENTATIONS, counts, jobs_for, validate_plan
    config = dict(root=str(tmp_path), builds={k: str(tmp_path / k) for k in REPRESENTATIONS},
        index=str(tmp_path / 'index.csv'), build_workers=2, folds=list(range(5)), seeds=[0, 1, 2],
        reuse_from=[], maximum_hours=480, checkpoint_hours=168, device='cuda', threads=4,
        confirm_epochs=40, patience=8)
    jobs = jobs_for(config); validate_plan(config, jobs)
    assert counts(config)['representation_confirmations'] == 540
    assert counts(config)['ablation_confirmations'] == 210
    for job in jobs:
        if job['id'].startswith(('primary_', 'compare_', 'ablate_')):
            fold = int(next(x[1:] for x in job['id'].split('_') if x.startswith('f') and x[1:].isdigit()))
            assert f'select_f{fold}' in job['dependencies']


def test_cache_cleanup_guard_rejects_original_inputs_and_unmarked_dirs(tmp_path):
    from experiments.final_paper.campaign import owned_cache
    root = tmp_path / 'campaign'; cache = root / '_cache' / 'job'; cache.mkdir(parents=True)
    with pytest.raises(ValueError, match='outside'):
        owned_cache(tmp_path / 'original', root)
    with pytest.raises(ValueError, match='marker'):
        owned_cache(cache, root)
    (cache / '_paper_owned_cache.json').write_text(json.dumps(dict(job=str(root / 'comparisons' / 'job'))))
    assert owned_cache(cache, root) == cache.resolve()


def test_campaign_journal_survives_a_locked_display_file(tmp_path, monkeypatch):
    from experiments.final_paper.campaign import load_state, save_state
    state = dict(jobs={'job': {'state': 'running'}})
    save_state(tmp_path, state)
    original = runner.atomic_json
    def locked(path, value):
        raise PermissionError('viewer holds file')
    monkeypatch.setattr(runner, 'atomic_json', locked)
    state['jobs']['job']['state'] = 'complete'; save_state(tmp_path, state)
    assert load_state(tmp_path)['jobs']['job']['state'] == 'complete'
    monkeypatch.setattr(runner, 'atomic_json', original)


def test_oof_metrics_use_fold_specific_decisions_not_a_global_half_threshold():
    from experiments.final_paper.report import scores
    result = scores([0, 1], [.3, .6], [True, False])
    assert result['accuracy'] == 0 and result['detection_rate'] == 0 and result['fpr'] == 1
    assert result['auc'] == 1


def test_oof_requires_exact_nonduplicated_outer_test_cohort(tmp_path):
    import csv
    from experiments.final_paper.report import oof_for
    rows, expected = [], {}
    for fold in range(5):
        job = tmp_path / str(fold); case = job / 'trials' / 'case'; case.mkdir(parents=True)
        sid = f's{fold}'; expected[sid] = fold % 2
        (job / 'dataset.json').write_text(json.dumps({'splits': {'test': [sid]}}))
        with (case / 'test_predictions.csv').open('w', newline='') as handle:
            w = csv.DictWriter(handle, fieldnames=['sample_id', 'split', 'label', 'prob', 'family', 'arch'])
            w.writeheader(); w.writerow(dict(sample_id=sid, split='test', label=fold % 2, prob=.5, family=sid, arch='x86'))
        rows.append(dict(fold=fold, seed=0, case_path=str(case), job_path=str(job), threshold=.2 if fold % 2 else .8))
    assert [r['decision'] for r in oof_for(rows, expected)] == [False, True, False, True, False]
    assert oof_for(rows[:4], expected) is None
    with pytest.raises(ValueError, match='cohort'):
        oof_for(rows, {**expected, 'extra': 1})


def test_compact_sampling_retains_noncall_import_signal_and_call_targets(tmp_path):
    import gzip
    from experiments.final_paper.sampling import compact_cfg, pp
    sid = 'a' * 64
    header = dict(record='header', schema=pp.SCHEMA, sha256=sid, imagebase=0x400000,
                  pe={'imports': [{'name': 'Open', 'dll': 'test', 'rva': 9000}]})
    def function(fid, rva, mnemonic, imports, refs):
        return dict(record='function', id=fid, name=f'fn{fid}', rva=rva, library=False,
            blocks=[dict(id=0, start_rva=rva, end_rva=rva + 1, instructions=[dict(rva=rva, size=1,
                mnemonic=mnemonic, import_ids=imports, code_refs=refs)])], edges=[])
    path = tmp_path / 'cfg.jsonl.gz'
    with gzip.open(path, 'wt', encoding='utf-8') as handle:
        for record in (header, function(0, 4096, 'mov', [0], []), function(1, 8192, 'call', [], [4096]),
                       dict(record='footer', complete=True)):
            handle.write(json.dumps(record) + '\n')
    compact, full = compact_cfg(path, sid), pp.load_cfg(path)
    assert pp.sample_signal_subgraph(compact, seed=0, num_blocks=1).selected_keys == pp.sample_signal_subgraph(full, seed=0, num_blocks=1).selected_keys
    assert [(e.src, e.dst) for e in pp.call_edges_of(compact)] == [(e.src, e.dst) for e in pp.call_edges_of(full)]
