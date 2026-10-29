"""Scientific-contract tests for the deadline grid, including real training on tiny tensor builds."""
import copy
import json

import pytest
import torch

from experiments.model_grid import data as dm, runner
from experiments.model_grid.models import BlockTransformer, collate, forward, make_model


@pytest.fixture
def spec():
    args = runner.parser(3).parse_args(['--build', '.', '--index', 'index.csv', '--out', '.'])
    s = runner.cases(args, 3)[0]
    s.update(width=8, transformer_heads=2, transformer_layers=1, gat_hidden=8, gat_heads=2,
             gat_layers=2, dropout=0., batch_size=2, block_length=4, sequence_length=4, block_chunk=2)
    return s


@pytest.fixture
def tiny_build(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    splits = {}
    for i, split in enumerate(('train', 'val', 'test')):
        rows = []
        for label in (0, 1):
            sid = f'{2*i+label:064x}'
            rows.append(dict(sample_id=sid, label=label, family=f'f{i}_{label}',
                             identity_group=sid, arch='x86', num_instructions=5, num_blocks=3))
            ids = torch.ones(5, 5, dtype=torch.long)
            ids[:, 0] = torch.tensor([2, 3, 4, 5, 6])
            ids[:, 3] = dm.tok.FIRST_API_ID + (50 if split != 'train' else label)
            feat = torch.zeros(3, 77); feat[:, 0] = torch.tensor([4, 1, 0])
            torch.save(dict(instruction_ids=ids, insn_to_block=torch.tensor([0, 0, 0, 0, 1]),
                            instruction_matched=torch.ones(5, dtype=torch.bool), block_features=feat,
                            edge_index=torch.tensor([[0, 1], [1, 2]]), edge_attr=torch.eye(7)[:2],
                            edge_back=torch.tensor([False, False]), label=label), tmp_path/f'{sid}.pt')
        splits[split] = rows
    (tmp_path/'index.csv').write_text('synthetic index')
    monkeypatch.setattr(dm, 'manifest_for', lambda *a: copy.deepcopy(splits))
    return tmp_path, splits


@pytest.mark.parametrize('pool', ['mean', 'cls'])
def test_order_padding_empty_and_gradients(pool):
    torch.manual_seed(12)
    encoder = BlockTransformer(20, block_length=4, width=8, heads=2, layers=1, dropout=0, pooling=pool)
    ids = torch.ones(2, 4, 5, dtype=torch.long)
    ids[0, :, 0] = torch.tensor([2, 3, 4, 5])
    pad = torch.tensor([[False, False, False, True], [True, True, True, True]])
    out = encoder(ids, pad)
    altered = ids.clone(); altered[:, 3] = 19
    assert torch.allclose(out, encoder(altered, pad), atol=1e-6)
    assert torch.equal(out[1], torch.zeros(8))
    reverse = ids.clone(); reverse[0, :3] = ids[0, :3].flip(0)
    assert not torch.allclose(out[0], encoder(reverse, pad)[0], atol=1e-6)
    (out[0] * torch.arange(8)).sum().backward()
    assert encoder.base.pos_encoder.grad.abs().sum() > 0
    if pool == 'cls':
        assert encoder.cls.grad.abs().sum() > 0


@pytest.mark.parametrize('model', [1, 2, 3])
def test_input_contract_and_test_isolation(tiny_build, spec, monkeypatch, model):
    root, splits = tiny_build
    reads = []
    old_read = dm.Prepared.read
    def track(self, sid):
        reads.append(sid)
        return old_read(self, sid)
    monkeypatch.setattr(dm.Prepared, 'read', track)
    data = dm.Prepared(root, root/'index.csv', 'mendeley-x86', 0, 0, model, spec)
    test_ids = {r['sample_id'] for r in splits['test']}
    assert not test_ids & set(reads)
    assert data.api_size == dm.tok.FIRST_API_ID + 2
    val = data.samples[splits['val'][0]['sample_id']]
    if model == 1:
        assert torch.all(val['ids'][:, 3] == dm.tok.UNK_API)
    elif model == 3:
        assert torch.all(val.insn_ids[:, 3] == dm.tok.UNK_API)
    else:
        assert val.x.shape == (3, 77)
    samples = [data.samples[r['sample_id']] for r in splits['train']]
    batch, y = collate(model, samples, torch.device('cpu'))
    net = make_model(model, data.api_size, spec, 0)
    loss = torch.nn.functional.cross_entropy(forward(model, net, batch), y)
    loss.backward()
    assert torch.isfinite(loss)
    if model == 3:
        assert net.encoder.base.embedder.opcode_embedding.weight.grad.abs().sum() > 0
        assert samples[0].insn_block.tolist() == [0, 0, 0, 0, 1]
        assert batch.insn_block.tolist() == [0, 0, 0, 0, 1, 3, 3, 3, 3, 4]
    data.load_test()
    assert test_ids <= set(reads)


def test_pooling_arms_have_identical_real_input_and_shared_weights(tiny_build, spec):
    root, _ = tiny_build
    data = dm.Prepared(root, root/'index.csv', 'mendeley-x86', 0, 0, 3, spec)
    torch.manual_seed(9); mean = make_model(3, data.api_size, spec, 9)
    torch.manual_seed(9); cls = make_model(3, data.api_size, dict(spec, pooling='cls'), 9)
    assert mean.encoder.block_length == cls.encoder.block_length == 4
    for k,v in mean.encoder.base.state_dict().items():
        assert torch.equal(v, cls.encoder.base.state_dict()[k])
    for k,v in mean.gat.state_dict().items():
        assert torch.equal(v, cls.gat.state_dict()[k])


def test_resume_matches_uninterrupted_training(tiny_build, spec, tmp_path, monkeypatch):
    root, _ = tiny_build
    data = dm.Prepared(root, root/'index.csv', 'mendeley-x86', 0, 0, 1, spec)
    sources = runner.source_hashes()
    uninterrupted = tmp_path/'full'
    runner.run_trial(1, spec, 5, 2, 4, data, uninterrupted, torch.device('cpu'), sources)
    resumed = tmp_path/'resumed'
    save = runner.save_checkpoint
    def interrupt(path, value):
        save(path, value)
        if path == resumed/'last.pt' and value['epoch'] == 1:
            raise InterruptedError('simulated interruption after checkpoint')
    monkeypatch.setattr(runner, 'save_checkpoint', interrupt)
    with pytest.raises(InterruptedError):
        runner.run_trial(1, spec, 5, 2, 4, data, resumed, torch.device('cpu'), sources)
    monkeypatch.setattr(runner, 'save_checkpoint', save)
    runner.run_trial(1, spec, 5, 2, 4, data, resumed, torch.device('cpu'), sources)
    a = torch.load(uninterrupted/'last.pt', weights_only=False)['model']
    b = torch.load(resumed/'last.pt', weights_only=False)['model']
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert (uninterrupted/'validation_predictions.csv').read_bytes() == (resumed/'validation_predictions.csv').read_bytes()
    with pytest.raises(ValueError, match='settings/input/source changed'):
        runner.run_trial(1, dict(spec, lr=.01), 5, 2, 4, data, resumed, torch.device('cpu'), sources)


def test_grid_counts_and_pair_selection():
    for model, count in [(1, 6), (2, 6), (3, 12)]:
        args = runner.parser(model).parse_args(['--build', '.', '--index', 'index.csv', '--out', '.'])
        assert len(runner.cases(args, model)) == count
    rr = [dict(spec=dict(lr=lr, dropout=.1, pooling=pool), validation_fixed=dict(auc=auc, macro_f1=.5))
          for lr,pool,auc in [(.001,'mean',.9),(.001,'cls',.7),(.0001,'mean',.85),(.0001,'cls',.86)]]
    assert all(s['lr'] == .0001 for s in runner.select_cases(3, rr))


def test_groups_and_samples_cannot_cross_splits():
    rows = {s: [dict(sample_id=f'{s}{y}', label=y, family=f'f{s}', identity_group=f'g{s}') for y in (0,1)]
            for s in ('train','val','test')}
    dm.assert_disjoint(rows)
    rows['test'][1]['family'] = rows['train'][1]['family']
    with pytest.raises(ValueError, match='leakage'):
        dm.assert_disjoint(rows)


@pytest.mark.parametrize('model', [1, 2, 3])
def test_runner_screen_confirm_and_completed_resume(tiny_build, tmp_path, model):
    root, _ = tiny_build
    out = tmp_path/f'model{model}'
    argv = ['--build', str(root), '--index', str(root/'index.csv'), '--out', str(out),
            '--device', 'cpu', '--learning-rates', '.0003', '--dropouts', '0', '--epochs', '1',
            '--confirm-epochs', '1', '--seeds', '0', '--sequence-length', '4', '--block-length', '4',
            '--threads', '1', '--stage', 'screen']
    runner.main(model, argv)
    fold = out/'fold0'
    assert not list(fold.rglob('test_predictions.csv'))
    hashes = json.loads((fold/'payload_hashes.json').read_text())
    assert len(hashes) == 4
    before = {p: p.stat().st_mtime_ns for p in fold.glob('screen/*/metrics.json')}
    runner.main(model, [*argv[:-1], 'confirm'])
    assert all(p.stat().st_mtime_ns == t for p,t in before.items())
    assert len(list(fold.rglob('test_predictions.csv'))) == (2 if model == 3 else 1)
    assert len(json.loads((fold/'payload_hashes.json').read_text())) == 6
    assert 'Held-out confirmation' in (fold/'REPORT.md').read_text()


def test_entry_point_dispatches_models_sequentially(monkeypatch):
    import grid_search
    calls = []
    monkeypatch.setattr(grid_search, 'run_model', lambda m,a: calls.append((m,a)))
    grid_search.main(['--models', '1', '3', '--out', 'results', '--build', 'build', '--index', 'index.csv'])
    assert [m for m,_ in calls] == [1,3]
    assert calls[0][1][-1].replace('\\', '/') == 'results/model1'
    assert calls[1][1][-1].replace('\\', '/') == 'results/model3'
