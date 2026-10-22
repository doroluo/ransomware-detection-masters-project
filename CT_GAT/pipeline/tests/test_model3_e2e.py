"""python -m pytest tests/test_model3_e2e.py: Model 3 trained end to end (`s03c_cfg_transformer/block_transformer.py`, the arm
`sig500v3-e2e` of `train_b`): the padding of the flat instruction list, the offset of the block index in a batch, the block encoder
against Janice's `Model1TokenSequence`, empty and long blocks, the graph a payload becomes, the gradient clip of `fit`, a smoke run and
a full run on a synthetic build, the files `stats`, `report` and `gates --stage team` read, and the refusals.

Everything runs on CPU on the tiny cache of `tiny_cache.py` and the synthetic builds of `test_team_v3.synthetic_build` (40 payloads
with `insn_to_block`, `instruction_matched`, `edge_back` and `edge_type`). The tests need torch_geometric and skip with a reason
without it."""
import json
import random
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from experiments.stages.team import E2E, team_group
from experiments.team_pipelines import common as tp, fit, train_b
from s04_evaluation import gate_tables, gates, report, stats
from test_team_pipelines import fresh_team_caches, needs_pyg, pinned_team_modules, write_tfidf  # noqa: F401
from test_team_v3 import synthetic_build
from tiny_cache import tiny  # noqa: F401  (a fixture)

pytestmark = needs_pyg
ARM = 'sig500v3-e2e'
QUIET = lambda *_: None                                                                       # noqa: E731


@pytest.fixture
def bt():
    from s03c_cfg_transformer import block_transformer
    return block_transformer


def flat(lengths, seed=0, n_api=6):
    """(insn_ids [M, 5], insn_block [M]) of blocks with `lengths` instructions each, in block order; every row is distinct."""
    rng = torch.Generator().manual_seed(seed)
    m = sum(lengths)
    ids = torch.stack([torch.arange(m) + 1, torch.randint(1, 30, (m,), generator=rng), torch.randint(1, 30, (m,), generator=rng),
                       torch.randint(0, 76 + n_api, (m,), generator=rng), torch.randint(0, 8, (m,), generator=rng)], 1)
    block = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths, dtype=torch.long))
    return ids.long(), block


def pad_reference(ids, block, n_blocks, block_len):
    """pad_blocks as Python loops: the first `block_len` instructions of each block in order, an empty block one unmasked PAD row."""
    rows = [[ids[k] for k in range(len(block)) if int(block[k]) == b][:block_len] for b in range(n_blocks)]
    width = max([len(r) for r in rows] + [1])
    out = torch.zeros(n_blocks, width, 5, dtype=torch.long)
    mask = torch.ones(n_blocks, width, dtype=torch.bool)
    for b, r in enumerate(rows):
        for j, row in enumerate(r):
            out[b, j] = row
            mask[b, j] = False
        if not r:
            mask[b, 0] = False
    return out, mask, torch.tensor([len(r) for r in rows])


def test_pad_blocks_equals_a_python_loop_on_random_lengths_including_empty_and_longer_than_the_cap(bt):
    rnd = random.Random(0)
    for trial in range(30):
        lengths = [rnd.choice([0, 0, 1, 2, 5, 17, 63, 64, 65, 90]) for _ in range(rnd.randint(1, 12))]
        ids, block = flat(lengths, seed=trial)
        for cap in (64, 16):
            got = bt.pad_blocks(ids, block, len(lengths), cap)
            for a, b in zip(got, pad_reference(ids, block, len(lengths), cap)):
                assert a.shape == b.shape and torch.equal(a, b), (lengths, cap)
            assert got[1].all(1).sum() == 0                                                     # no all-masked row
    ids, block = flat([3, 0, 2])
    order = torch.tensor([4, 0, 3, 1, 2])                                                       # a list that is not sorted by block
    shuffled = bt.pad_blocks(ids[order], block[order], 3, 8)
    reference = pad_reference(ids[order], block[order], 3, 8)
    assert all(torch.equal(a, b) for a, b in zip(shuffled, reference))
    ids, mask, lengths = bt.pad_blocks(ids[:0], block[:0], 2, 8)                                # no instruction at all: one PAD column
    assert ids.shape == (2, 1, 5) and not mask.any() and lengths.tolist() == [0, 0]


def test_the_block_index_is_offset_by_the_blocks_of_the_earlier_graphs_in_a_batch(bt):
    from torch_geometric.data import Batch

    def graph(lengths, seed):
        ids, block = flat(lengths, seed)
        n = len(lengths)
        return bt.BlockSeqData(insn_ids=ids, insn_block=block, edge_index=torch.tensor([[0], [n - 1]]), edge_attr=torch.zeros(1, 8),
                               y=torch.tensor(0), num_nodes=n)
    graphs = [graph([2, 1, 3], 0), graph([1, 4], 1), graph([2, 2, 2, 2], 2)]
    batch = Batch.from_data_list(graphs)
    assert batch.num_nodes == 9 and batch.insn_ids.shape == (sum(g.insn_ids.shape[0] for g in graphs), 5)
    first, second = graphs[0].insn_block.numel(), graphs[1].insn_block.numel()
    assert torch.equal(batch.insn_block[:first], graphs[0].insn_block)
    assert torch.equal(batch.insn_block[first:first + second], graphs[1].insn_block + 3)
    assert torch.equal(batch.insn_block[first + second:], graphs[2].insn_block + 5)
    assert torch.equal(batch.edge_index, torch.tensor([[0, 3, 5], [2, 4, 8]]))                  # the graph's own offsets are unchanged
    ids, mask, lengths = bt.pad_blocks(batch.insn_ids, batch.insn_block, batch.num_nodes, 64)
    assert lengths.tolist() == [2, 1, 3, 1, 4, 2, 2, 2, 2]


def test_encode_blocks_is_the_forward_of_janices_model_up_to_its_classifier(bt):
    torch.manual_seed(0)
    model = bt.Model3EndToEnd(76 + 6, block_len=16)
    m1 = model.encoder.eval()
    ids, mask, lengths = bt.pad_blocks(*flat([5, 9, 1, 16], seed=3), 4, 16)
    mine = bt.encode_blocks(m1, ids, mask, lengths)
    classifier, m1.classifier = m1.classifier, nn.Identity()
    try:
        hers = m1(ids, padding_mask=mask)
    finally:
        m1.classifier = classifier
    assert mine.shape == (4, 64) and torch.allclose(mine, hers, atol=1e-6)
    one = bt.encode_blocks(m1, ids[1:2, :9], mask[1:2, :9])                                      # one block, no padding at all
    m1.classifier = nn.Identity()
    try:
        assert torch.allclose(one, m1(ids[1:2, :9], padding_mask=mask[1:2, :9]), atol=1e-6)
    finally:
        m1.classifier = classifier


def test_an_empty_block_gives_a_zero_vector_and_the_batch_has_no_nan(bt):
    from torch_geometric.data import Batch
    torch.manual_seed(0)
    ids, block = flat([4, 0, 3, 0])
    graph = bt.BlockSeqData(insn_ids=ids, insn_block=block, edge_index=torch.tensor([[0, 2], [2, 1]]), edge_attr=torch.zeros(2, 8),
                            y=torch.tensor(1), num_nodes=4)
    batch = Batch.from_data_list([graph, graph])
    model = bt.Model3EndToEnd(76 + 6, block_len=64)
    for training in (True, False):
        model.train(training)
        vectors = model.block_vectors(batch)
        assert torch.isfinite(vectors).all() and vectors.shape == (8, 64)
        assert (vectors[[1, 3, 5, 7]] == 0).all() and (vectors[[0, 2, 4, 6]].abs().sum(1) > 0).all()
        logits = model(batch)
        assert logits.shape == (2, 2) and torch.isfinite(logits).all()


def test_gradients_reach_the_encoder_and_the_gat_but_not_janices_unused_classifier(bt):
    from torch_geometric.data import Batch
    torch.manual_seed(0)
    ids, block = flat([4, 2, 3])
    graph = bt.BlockSeqData(insn_ids=ids, insn_block=block, edge_index=torch.tensor([[0, 1], [1, 2]]), edge_attr=torch.zeros(2, 8),
                            y=torch.tensor(1), num_nodes=3)
    model = bt.Model3EndToEnd(76 + 6)
    model.train()
    nn.functional.cross_entropy(model(Batch.from_data_list([graph, graph])), torch.tensor([1, 1])).backward()
    grads = {n: p.grad for n, p in model.named_parameters()}
    assert all(g is None for n, g in grads.items() if n.startswith('encoder.classifier'))
    assert grads['encoder.embedder.opcode_embedding.weight'].abs().sum() > 0
    assert grads['encoder.transformer_encoder.layers.0.linear1.weight'].abs().sum() > 0
    assert grads['norm.weight'].abs().sum() > 0 and grads['gat.project.weight'].abs().sum() > 0
    counts = model.n_parameters()
    assert counts['unused_classifier'] == 4290 and counts['total'] == counts['encoder'] + counts['gat']
    assert counts['total'] == sum(p.numel() for p in model.parameters()) - 4290


def test_make_e2e_graph_maps_unseen_apis_to_unk_and_caps_a_long_block(bt):
    from s03b_cfg_feature.gat import graph_full_loader as gl
    pp = tp.import_team('preprocess_full', source='repo')
    first, unk = pp.tok.FIRST_API_ID, pp.tok.UNK_API
    lengths = [3, 70, 0, 2]
    ids, block = flat(lengths)
    ids[:, 3] = torch.tensor([first] * 5 + [first + 1] * 70)[:ids.shape[0]]
    ids[0, 3] = pp.tok.NO_API
    perm = torch.randperm(ids.shape[0], generator=torch.Generator().manual_seed(1))             # the payload need not be in block order
    ids, block = ids[perm], block[perm]
    n_edges = 4
    payload = {'file_id': 'x' * 64, 'label': 1, 'instruction_ids': ids, 'insn_to_block': block, 'block_features': torch.zeros(4, 77),
               'edge_index': torch.stack([torch.arange(n_edges), (torch.arange(n_edges) + 1) % 4]),
               'edge_attr': torch.nn.functional.one_hot(torch.arange(n_edges) % 7, 7).float(),
               'edge_back': torch.tensor([True, False, False, True]), 'edge_type': torch.arange(n_edges)}
    g = bt.make_e2e_graph(pp, gl, payload, torch.tensor([first]), block_len=64)
    assert g.num_nodes == 4 and g.insn_ids.shape == (3 + 64 + 0 + 2, 5) and g.n_truncated == 1
    assert g.n_instructions == 75 and g.n_tokens == 69 and g.y == 1 and g.sample_id == 'x' * 64
    assert g.edge_attr.shape == (4, 8) and torch.equal(g.edge_attr[:, 7], torch.tensor([1., 0., 0., 1.]))
    assert torch.equal(g.insn_block, torch.repeat_interleave(torch.arange(4), torch.tensor([3, 64, 0, 2])))
    api = g.insn_ids[:, 3]                                                                       # first+1 is not in `seen`
    assert int((api == unk).sum()) == 64 and int((api == first).sum()) == 4 and int((api == pp.tok.NO_API).sum()) == 1
    kept = (block == 1).nonzero().flatten()[:64]                                                 # the long block keeps its first 64, in order
    assert torch.equal(g.insn_ids[3:67][:, 0], ids[kept][:, 0])
    stored = bt.make_e2e_graph(pp, gl, payload, None, block_len=64)                             # seen=None: the stored ids
    assert int((stored.insn_ids[:, 3] == first + 1).sum()) == 64 and int((stored.insn_ids[:, 3] == unk).sum()) == 0
    short = bt.make_e2e_graph(pp, gl, payload, None, block_len=8)
    assert short.n_truncated == 1 and short.n_tokens == 3 + 8 + 2 and short.insn_ids.shape[0] == 13


def test_fit_clips_the_gradient_norm_only_when_asked(bt):
    from torch_geometric.data import Batch
    ids, block = flat([4, 2, 3])
    graph = bt.BlockSeqData(insn_ids=ids, insn_block=block, edge_index=torch.tensor([[0, 1], [1, 2]]), edge_attr=torch.zeros(2, 8),
                            y=torch.tensor(1), num_nodes=3)
    norms = {}
    for clip in (None, 1e-3):
        torch.manual_seed(0)
        model = bt.Model3EndToEnd(76 + 6)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        criterion = nn.CrossEntropyLoss()
        fit.train_batch(model, Batch.from_data_list([graph, graph]), criterion, criterion, optimizer, torch.device('cpu'),
                        grad_clip=clip)
        norms[clip] = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9))
    assert norms[None] > 1e-3 and norms[1e-3] == pytest.approx(1e-3, rel=1e-3)


# the arm ----------------------------------------------------------------------------------------------------------------------


def test_the_arm_row_name_group_and_recipe():
    row = train_b.ARM_TABLE[ARM]
    assert train_b.ARMS.index(ARM) == 5 and train_b.ARM_DIR[ARM].name == 'b_sig500_v3' and row.sources == ('main-fix', 'repo')
    assert (row.sampler, row.num_blocks, row.features) == ('signal', 500, 'e2e') and not row.legacy
    assert train_b.ARM_HP[ARM] == {**train_b.HP, 'grad_clip': 1.0, 'gpu_cap_gb': 12} == train_b.HP_E2E
    assert train_b.HP_E2E['lr'] == 1e-3 and 'node_budget' not in train_b.HP_E2E and train_b.BLOCK_LEN == 64
    assert all('grad_clip' not in train_b.ARM_HP[a] for a in train_b.ARMS if a != ARM)
    cfg = tp.team_config(2, 1, ARM)
    assert cfg.name == f'kfold2_x86_gnn_s1_team-{ARM}' and cfg.group == team_group(ARM) == 'kfold_x86_gnn_team-sig500v3-e2e'
    assert dict(gates.load_stage('team').arms)[E2E] == cfg.group


def test_the_arm_refuses_another_input_another_block_length_and_a_non_e2e_arm_refuses_a_block_length(tmp_path):
    with pytest.raises(SystemExit, match='trains on --features e2e, not feat'):
        train_b.run(ARM, 0, 0, smoke=2, device='cpu', features='feat', log=QUIET)
    with pytest.raises(SystemExit, match='trains on --features emb, not e2e'):
        train_b.run('sig500v3-emb', 0, 0, smoke=2, device='cpu', features='e2e', log=QUIET)
    with pytest.raises(SystemExit, match='trains with --block-len 64, not 32'):
        train_b.run(ARM, 0, 0, smoke=2, device='cpu', block_len=32, log=QUIET)
    with pytest.raises(SystemExit, match='reads no instructions'):
        train_b.run('sig500v3-emb', 0, 0, smoke=2, device='cpu', block_len=64, log=QUIET)
    with pytest.raises(SystemExit):
        train_b.main(['--fold', '0', '--arm', ARM, '--block-len', '32', '--smoke', '2', '--device', 'cpu'])
    with pytest.raises(SystemExit):
        train_b.main(['--fold', '0', '--arm', ARM, '--features', 'feat', '--smoke', '2', '--device', 'cpu'])
    for old in ('sig500v3', 'sig500v3-feat', 'sigwhole-emb'):
        with pytest.raises(SystemExit, match='trains on --features'):
            train_b.run(old, 0, 0, smoke=2, device='cpu', features='e2e', log=QUIET)


def test_the_arm_refuses_a_build_of_another_sampler(tmp_path):
    from test_team_v3 import settings
    walk = settings(tmp_path / 'walk', source='main-fix', sample='walk', num_blocks=500)
    with pytest.raises(SystemExit, match='sampler walk'):
        train_b.run(ARM, 0, 0, smoke=2, device='cpu', b_dir=walk, log=QUIET)
    whole = settings(tmp_path / 'whole', source='main-fix', sample='signal', num_blocks=1000000)
    with pytest.raises(SystemExit, match='num_blocks 500'):
        train_b.run(ARM, 0, 0, smoke=2, device='cpu', b_dir=whole, log=QUIET)


def test_smoke_on_a_synthetic_build_writes_only_a_temp_dir(tmp_path, monkeypatch, tiny):          # noqa: F811
    rows = tp.sample_rows()
    folder = synthetic_build(tmp_path / 'b', sorted(rows), rows)
    monkeypatch.setattr(fit, 'RUNS', tmp_path / 'runs')
    monkeypatch.setattr(fit, 'write_run', lambda *a, **k: pytest.fail('a smoke run must not call write_run'))
    lines = []
    out = train_b.run(ARM, 0, 0, smoke=4, device='cpu', b_dir=folder, log=lines.append)
    assert any('features e2e, in_dim 64, edge_dim 8' in l for l in lines) and len(out['history']) == 2
    assert any(l.startswith('re-pool check') for l in lines) and not (tmp_path / 'runs').exists()
    assert Path(out['dir']).parent == Path(tempfile.gettempdir()) and (Path(out['dir']) / 'history.csv').is_file()
    assert all(h['loss'] == h['loss'] for h in out['history'])                                    # not NaN
    __import__('shutil').rmtree(out['dir'])


def test_five_synthetic_folds_are_read_by_stats_report_and_the_team_gates(tmp_path, monkeypatch, tiny):   # noqa: F811
    rows = tp.sample_rows()
    folder = synthetic_build(tmp_path / 'b', sorted(rows), rows)
    runs = tmp_path / 'runs'
    runs.mkdir()
    write_tfidf(runs)
    monkeypatch.setattr(fit, 'RUNS', runs)
    monkeypatch.setitem(train_b.ARM_HP, ARM, {**train_b.HP_E2E, 'epochs': 2})
    for fold in range(5):
        out = train_b.run(ARM, fold, 0, device='cpu', b_dir=folder, log=QUIET)
        assert out['dir'] == runs / f'kfold{fold}_x86_gnn_s0_team-{ARM}'
    d = runs / f'kfold0_x86_gnn_s0_team-{ARM}'
    metrics = json.loads((d / 'metrics.json').read_text(encoding='utf-8'))
    assert metrics['features'] == 'e2e' and metrics['in_dim'] == 64 and metrics['edge_dim'] == 8 and metrics['block_len'] == 64
    assert metrics['encoder'] == 'Model1TokenSequence d64 h4 L2' and metrics['gat'] == 'FullBlockGAT hidden128 heads4 layers2'
    counts = metrics['n_parameters']
    assert set(counts) == {'encoder', 'gat', 'total', 'unused_classifier'} and counts['unused_classifier'] == 4290
    assert counts['total'] == counts['encoder'] + counts['gat'] and counts['encoder'] > 0
    cut = metrics['truncated_blocks']
    assert cut['test_count'] == 0 and cut['test_share'] == 0.0 and cut['test_blocks'] > 0           # the synthetic blocks are short
    assert 0 < metrics['tokens_per_binary'] <= metrics['instructions_per_binary']
    assert metrics['recipe']['grad_clip'] == 1.0 and metrics['recipe']['lr'] == 1e-3 and metrics['recipe']['arm'] == ARM
    assert metrics['build_source'] == 'main-fix' and metrics['build_binaries'] == 40 and 0.0 <= metrics['test']['auc'] <= 1.0
    assert (d / 'history.csv').is_file() and (d / 'best.pt').is_file() and (d / 'predictions_auc.csv').is_file()
    monkeypatch.setattr(stats, 'RUNS', runs)
    complete, partial = stats.load_group(team_group(ARM), 'f1')
    assert list(complete) == [0] and not partial
    monkeypatch.setattr(report, 'RUNS', runs)
    monkeypatch.setattr(report, 'all_labels', lambda: {r['sample_id']: r['label'] for r in tiny.rows})
    report.main(['--out', str(tmp_path / 'RESULTS.md')])
    assert 'team-sig500v3-e2e' in (tmp_path / 'RESULTS.md').read_text(encoding='utf-8')
    gate_tables.pooled.cache_clear()
    text = '\n'.join(gates.run_stage('team', 20))
    gate_tables.pooled.cache_clear()
    assert f'| {E2E} | kfold_x86_gnn_team-sig500v3-e2e |' in text
    row = next(l for l in text.splitlines() if l.startswith(f'| {E2E} |') and l.count('|') == 12)
    assert 'missing' not in row


def test_the_gates_pair_the_arm_with_emb_r7_and_tfidf_only():
    stage = gates.load_stage('team')
    emb = 'team sig500v3 emb (Model 3)'
    assert dict(stage.arms)[E2E] == team_group(ARM)
    assert [a for a, b in stage.pairs if b == E2E] == ['TF-IDF', 'R7 typed, whole graph', emb]
    assert not [b for a, b in stage.pairs if a == E2E] and len(stage.pairs) == len(set(stage.pairs))
