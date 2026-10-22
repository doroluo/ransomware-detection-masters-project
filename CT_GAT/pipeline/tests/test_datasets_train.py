"""python -m pytest tests/test_datasets_train.py: the trainers of Models 1 to 3 on any dataset (plan v6, steps B1 to B3).

What is checked, on the second synthetic dataset `tiny_v3` (tests/tiny_cache.py) and synthetic pipeline-B builds of it
(`test_team_v3.synthetic_build`, which pools the block embeddings so that `check_repool` holds):
  * B1, `train_b` / `fit.py` / the queue verb: the refusals of a build of another dataset in both directions, of the legacy and the
    uncapped arms, the default build directory of each (protocol, arch), the 2-epoch runs on kfold all, kfold_v3 all and matched (run
    names, manifests, excluded ids in no split), the architecture-balance weights against `cell_weights` and the loss against
    `CrossEntropyLoss(weight=)`, the API remap from the training ids only, `queue --dry` names.
The tests that train need torch_geometric and skip with a reason without it; the refusals, names and weights do not.
"""
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from experiments import queue as q
from experiments.team_pipelines import common as tp, fit, train_b
from s04_evaluation.metrics import cell_weights
from test_team_pipelines import fresh_team_caches, needs_pyg, pinned_team_modules  # noqa: F401
from test_team_v3 import settings, synthetic_build
from tiny_cache import tiny_v3  # noqa: F401  (a fixture)

WHY = 'num_instructions > 4000000: load_cfg needs about 3 KB per instruction (31 GB RAM)'
TRAINED = [('kfold', 'all'), ('kfold_v3', 'all'), ('kfold_v3', 'matched')]
quiet = lambda *_: None                                                                   # noqa: E731


# ---- B1: the refusals and the build directories ----------------------------------------------------------------------------

ARMS_ON_A_BUILD = ('sig500v3', 'sig500v3-feat', 'sig500v3-emb', 'sig500v3-e2e')


def test_a_run_and_a_build_of_different_datasets_are_refused_in_both_directions(tmp_path):
    x86 = settings(tmp_path / 'x86', source='repo', sample='signal', num_blocks=500)                       # no protocol, no arch
    mendeley_all = settings(tmp_path / 'ma', source='repo', sample='signal', num_blocks=500, protocol='kfold', arch='all')
    combined = settings(tmp_path / 'c', source='repo', sample='signal', num_blocks=500, protocol='kfold_v3', arch='all')
    matched = settings(tmp_path / 'm', source='repo', sample='signal', num_blocks=500, protocol='kfold_v3', arch='matched')
    for arm in ARMS_ON_A_BUILD:
        assert train_b.check_build(arm, x86) == train_b.check_build(arm, x86, None, 'kfold', 'x86') == 'repo'
        assert train_b.check_build(arm, mendeley_all, None, 'kfold', 'all') == 'repo'
        assert train_b.check_build(arm, combined, None, 'kfold_v3', 'all') == 'repo'
        assert train_b.check_build(arm, combined, None, 'kfold_v3', 'matched') == 'repo'                    # matched reads the all build
        for build, protocol, arch, said in (
                (x86, 'kfold', 'all', 'was built for kfold x86, this run asks for kfold all'),
                (x86, 'kfold_v3', 'all', 'was built for kfold x86, this run asks for kfold_v3 all'),
                (x86, 'kfold_v3', 'matched', 'was built for kfold x86, this run asks for kfold_v3 matched'),
                (mendeley_all, 'kfold', 'x86', 'was built for kfold all, this run asks for kfold x86'),
                (mendeley_all, 'kfold_v3', 'all', 'was built for kfold all, this run asks for kfold_v3 all'),
                (combined, 'kfold', 'x86', 'was built for kfold_v3 all, this run asks for kfold x86'),
                (combined, 'kfold', 'all', 'was built for kfold_v3 all, this run asks for kfold all'),
                (matched, 'kfold_v3', 'all', 'matched is a training-time filter of the all build'),
                (matched, 'kfold_v3', 'matched', 'matched is a training-time filter of the all build')):
            with pytest.raises(SystemExit, match=said):
                train_b.check_build(arm, build, None, protocol, arch)
    with pytest.raises(SystemExit, match='was built for kfold x86, this run asks for kfold_v3 matched'):
        train_b.run('sig500v3-emb', 0, 0, smoke=2, device='cpu', b_dir=x86, log=quiet, protocol='kfold_v3', arch='matched')
    with pytest.raises(SystemExit, match='was built for kfold_v3 all, this run asks for kfold x86'):
        train_b.run('sig500v3-emb', 0, 0, smoke=2, device='cpu', b_dir=combined, log=quiet)                # today's call, a combined build
    zipped = tmp_path / 'zip'
    (zipped / 'build_info').mkdir(parents=True)
    (zipped / 'build_info' / 'build_settings.json').write_text(
        json.dumps({'source': 'repo', 'sample': 'signal', 'num_blocks': 500, 'protocol': 'kfold_v3', 'arch': 'all'}), encoding='utf-8')
    assert train_b.check_build('sig500v3-emb', zipped, None, 'kfold_v3', 'all') == 'repo'                  # the Drive zip layout says it too
    with pytest.raises(SystemExit, match='was built for kfold_v3 all, this run asks for kfold x86'):
        train_b.check_build('sig500v3-emb', zipped)


def test_the_legacy_arms_run_on_kfold_x86_only_and_the_uncapped_ones_on_kfold_only(tmp_path):
    for arm, name in (('sig500', 'b_sig500'), ('sig500v2', 'b_sig500_v2')):
        for protocol, arch in (('kfold', 'all'), ('kfold_v3', 'all'), ('kfold_v3', 'matched')):
            message = f'arm {arm} reads the x86 build {name}; use sig500v3-\\* on another dataset'
            with pytest.raises(SystemExit, match=message):
                train_b.run(arm, 0, 0, smoke=2, device='cpu', b_dir=tmp_path, log=quiet, protocol=protocol, arch=arch)
            with pytest.raises(SystemExit, match=message):
                train_b.check_build(arm, tmp_path, None, protocol, arch)
        train_b.check_arm_dataset(arm)                                                         # the x86 path is untouched
    for arm in ('sigwhole', 'sigwhole-feat', 'sigwhole-emb'):
        train_b.check_arm_dataset(arm, 'kfold', 'all')
        for arch in ('all', 'matched'):
            with pytest.raises(SystemExit, match='uncapped build, which is made for kfold only'):
                train_b.check_arm_dataset(arm, 'kfold_v3', arch)
    for protocol, arch, said in (('kfold_v3', 'x86', "--protocol kfold_v3 runs on --arch all or matched, not 'x86'"),
                                 ('kfold', 'matched', "--protocol kfold runs on --arch x86 or all, not 'matched'"),
                                 ('bundle', 'x86', "protocol 'bundle'")):
        with pytest.raises(SystemExit, match=said):
            train_b.check_arm_dataset('sig500v3', protocol, arch)
    with pytest.raises(SystemExit, match='needs --arch all or matched'):
        train_b.check_arm_dataset('sig500v3', 'kfold', 'x86', arch_balance=True)
    for protocol, arch in TRAINED:
        train_b.check_arm_dataset('sig500v3-e2e', protocol, arch, arch_balance=True)


def test_the_default_build_directory_follows_the_dataset(tmp_path, monkeypatch):
    team = tmp_path / 'team'
    monkeypatch.setattr(train_b, 'TEAM_RUNS', team)
    for name in ('b_sig500_mendeley_all', 'b_callwalk250_mendeley_all', 'b_sigwhole_mendeley_all', 'b_sig500_combined_v3',
                 'b_callwalk500_combined_v3', 'b_sig250_combined_v3'):
        (team / name).mkdir(parents=True)
    family = {'sig500v3': 'sig500', 'sig500v3-feat': 'sig500', 'sig500v3-emb': 'sig500', 'sig500v3-e2e': 'sig500',
              'sig250-feat': 'sig250', 'sig250-emb': 'sig250', 'sig250-e2e': 'sig250',
              'callwalk500-feat': 'callwalk500', 'callwalk500-emb': 'callwalk500', 'callwalk500-e2e': 'callwalk500',
              'callwalk250-feat': 'callwalk250', 'callwalk250-emb': 'callwalk250', 'callwalk250-e2e': 'callwalk250',
              'sigwhole': 'sigwhole', 'sigwhole-feat': 'sigwhole', 'sigwhole-emb': 'sigwhole'}
    for arm, name in family.items():
        assert train_b.build_family(arm) == name
    assert train_b.default_build_dir('sig500v3-emb', 'kfold', 'all') == team / 'b_sig500_mendeley_all'
    assert train_b.default_build_dir('callwalk250-e2e', 'kfold', 'all') == team / 'b_callwalk250_mendeley_all'
    assert train_b.default_build_dir('sigwhole-feat', 'kfold', 'all') == team / 'b_sigwhole_mendeley_all'
    for arch in ('all', 'matched'):
        assert train_b.default_build_dir('sig500v3-feat', 'kfold_v3', arch) == team / 'b_sig500_combined_v3'
        assert train_b.default_build_dir('callwalk500-e2e', 'kfold_v3', arch) == team / 'b_callwalk500_combined_v3'
        assert train_b.default_build_dir('sig250-emb', 'kfold_v3', arch) == team / 'b_sig250_combined_v3'
    # kfold x86 is today's: ARM_DIR, whatever the other directories hold
    for arm in train_b.ARMS:
        x86_dir = train_b.ARM_DIR[arm]
        assert not x86_dir.name.endswith(('_mendeley_all', '_combined_v3'))
        monkeypatch.setitem(train_b.ARM_DIR, arm, tmp_path)                       # an existing directory: the arm's own wins
        assert train_b.default_build_dir(arm) == train_b.default_build_dir(arm, 'kfold', 'x86') == tmp_path
        monkeypatch.setitem(train_b.ARM_DIR, arm, x86_dir)
    # a missing directory falls back to the team_b_build source, else says where the build was expected
    monkeypatch.delenv('GT_TEAM_B_BUILD', raising=False)
    monkeypatch.setattr(train_b, 'source_for', lambda name: (_ for _ in ()).throw(train_b.MissingSource('none')))
    with pytest.raises(SystemExit, match='b_callwalk500_mendeley_all does not exist and neither does the team_b_build source'):
        train_b.default_build_dir('callwalk500-feat', 'kfold', 'all')
    here = tmp_path / 'unzipped'
    here.mkdir()
    monkeypatch.setattr(train_b, 'source_for', lambda name: here)
    assert train_b.default_build_dir('callwalk500-feat', 'kfold', 'all') == here


# ---- B1: the queue verb ------------------------------------------------------------------------------------------------------

def test_a_queue_line_names_the_run_of_its_dataset():
    line = 'train_b --fold 3 --seed 1 --arm sig500v3-feat --protocol kfold_v3 --arch matched --arch-balance'
    step = q.parse_line(line)
    assert step.label == 'v3kfold3_matched_gnn-ab_s1_team-sig500v3-feat' == tp.team_config(
        3, 1, 'sig500v3-feat', 'kfold_v3', 'matched', True).name
    assert step.done_files == (f'{step.label}/metrics.json',) and step.module == 'experiments.team_pipelines.train_b'
    assert step.module_args == tuple(line.split()[1:])
    names = {
        'train_b --fold 0 --arm sig500v3-emb': 'kfold0_x86_gnn_s0_team-sig500v3-emb',
        'train_b --fold 0 --arm sig500v3-emb --protocol kfold --arch x86': 'kfold0_x86_gnn_s0_team-sig500v3-emb',
        'train_b --fold 1 --arm sig500v3-emb --arch all': 'kfold1_all_gnn_s0_team-sig500v3-emb',
        'train_b --fold 1 --arm sig500v3-emb --arch all --arch-balance': 'kfold1_all_gnn-ab_s0_team-sig500v3-emb',
        'train_b --fold 2 --seed 2 --arm callwalk500-e2e --protocol kfold_v3': 'v3kfold2_all_gnn_s2_team-callwalk500-e2e',
        'train_b --fold 2 --arm sig500v3 --protocol kfold_v3 --arch all --arch-balance': 'v3kfold2_all_gnn-ab_s0_team-sig500v3',
        'train_b --fold 4 --arm sig250-feat --protocol kfold_v3 --arch matched': 'v3kfold4_matched_gnn_s0_team-sig250-feat',
        'train_b --fold 4 --arm sigwhole --arch all --build RUNS/team/b_sigwhole_mendeley_all': 'kfold4_all_gnn_s0_team-sigwhole'}
    for text, name in names.items():
        assert q.parse_line(text).label == name
    out = io.StringIO()
    assert q.Queue(q.parse_plan(line + '\n'), Path('unused'), dry=True, out=out).run() == 0
    assert out.getvalue().splitlines() == [f'run v3kfold3_matched_gnn-ab_s1_team-sig500v3-feat train_b {line.split(" ", 1)[1]}']


@pytest.mark.parametrize('text, message', [
    ('train_b --fold 0 --arm sig500v3 --protocol kfold_v3 --arch x86\n', "runs on --arch all or matched, not 'x86'"),
    ('train_b --fold 0 --arm sig500v3 --protocol kfold --arch matched\n', "runs on --arch x86 or all, not 'matched'"),
    ('train_b --fold 0 --arm sig500v3 --protocol bundle\n', 'invalid choice'),
    ('train_b --fold 0 --arm sig500v3 --arch-balance\n', 'needs --arch all or matched'),
    ('train_b --fold 0 --arm sig500 --arch all\n', 'reads the x86 build b_sig500; use sig500v3-\\* on another dataset'),
    ('train_b --fold 0 --arm sigwhole --protocol kfold_v3\n', 'uncapped build, which is made for kfold only'),
])
def test_a_queue_line_for_a_dataset_the_arm_cannot_run_on_is_refused_with_its_line_number(text, message):
    with pytest.raises(q.PlanError, match=message) as e:
        q.parse_plan('# first\n' + text, 'p.txt')
    assert 'p.txt:2' in str(e.value)


def test_the_command_line_of_train_b_refuses_a_dataset_the_arm_cannot_run_on():
    for argv in (['--arm', 'sig500v3', '--protocol', 'kfold_v3', '--arch', 'x86'], ['--arm', 'sig500v3', '--arch-balance'],
                 ['--arm', 'sig500', '--protocol', 'kfold_v3'], ['--arm', 'sigwhole', '--protocol', 'kfold_v3']):
        with pytest.raises(SystemExit):
            train_b.main(['--fold', '0', '--smoke', '2', '--device', 'cpu', *argv])


# ---- B1: the weights and the loss ---------------------------------------------------------------------------------------------

@pytest.fixture
def v3_training(tiny_v3, tmp_path):                                                           # noqa: F811
    """The training graphs of fold 2 of kfold_v3 all, from a synthetic build of the dataset."""
    from s03b_cfg_feature.gat import graph_full_loader as gl
    rows = tp.sample_rows('kfold_v3', 'all')
    folder = synthetic_build(tmp_path / 'b', sorted(rows), rows, dataset=('kfold_v3', 'all'))
    pp, embedder, _ = train_b.build_state(folder)
    manifest = tp.fold_manifest(2, 0, 'kfold_v3', 'all')
    ids = [r['sample_id'] for r in manifest.train]
    graphs = {i: train_b.make_graph(pp, gl, embedder, train_b.load_slim(folder, i, 'kfold_v3', 'all'), None, 'emb') for i in ids}
    return type('Training', (), {'graphs': graphs, 'ids': ids, 'rows': rows, 'manifest': manifest})


def first_batch(graphs, ids, n=16):
    from torch_geometric.data import Batch
    return Batch.from_data_list([graphs[i] for i in ids[:n]])


def small_gat(dropout=0.0):
    from s03b_cfg_feature.gat import gat_full_model as gm
    torch.manual_seed(0)
    return gm.FullBlockGAT(in_dim=64, edge_dim=8, dropout=dropout)


@needs_pyg
def test_the_cell_weights_of_a_run_are_the_cell_weights_of_the_training_manifest(v3_training):
    ids, rows = v3_training.ids, v3_training.rows
    by_id, named = fit.cell_weight_table(rows, ids)
    labels, arches = [rows[i]['label'] for i in ids], [rows[i]['arch'] for i in ids]
    expect = cell_weights(labels, arches)
    assert [by_id[i] for i in ids] == expect.tolist() and abs(float(expect.mean()) - 1) < 1e-12
    assert set(named) == {fit.cell_name(l, a) for l, a in zip(labels, arches)} and len(named) == 4
    for (l, a), w in zip(zip(labels, arches), expect):
        assert named[fit.cell_name(l, a)] == pytest.approx(w)
    for cell, weight in named.items():
        count = sum(fit.cell_name(l, a) == cell for l, a in zip(labels, arches))
        assert weight == pytest.approx(len(ids) / (len(named) * count))                          # N / (C n_cell)
    assert fit.cell_name(1, 'x64') == 'R_x64' and fit.cell_name(0, 'x86') == 'G_x86'
    outside = sorted({r['sample_id'] for r in v3_training.manifest.val + v3_training.manifest.test})
    assert outside and not set(outside) & set(ids)                                            # the weights saw the training ids only
    assert fit.cell_weight_table(rows, ids + outside)[1] != named                             # and would differ had val and test counted


@needs_pyg
@pytest.mark.parametrize('budget', [None, 40])
def test_with_the_flag_the_loss_is_the_cell_weighted_mean_and_the_node_budget_does_not_change_it(v3_training, budget):
    graphs, ids, rows = v3_training.graphs, v3_training.ids, v3_training.rows
    batch = first_batch(graphs, ids)
    by_id, _ = fit.cell_weight_table(rows, ids)
    weight = lambda b: torch.tensor([by_id[i] for i in b.sample_id], dtype=torch.float32)       # noqa: E731
    model = small_gat().eval()
    with torch.no_grad():
        logits = model(batch)
    y, w = batch.y.view(-1), weight(batch)
    by_hand = float((nn.functional.cross_entropy(logits, y, reduction='none') * w).sum() / w.sum())
    assert float(fit.weighted_mean(logits, y, w)) == pytest.approx(by_hand, abs=1e-7)
    class_weights = torch.tensor([1.0, 3.0])             # a weight that is a function of the class is CrossEntropyLoss(weight=)
    plain = float(nn.CrossEntropyLoss(weight=class_weights)(logits, y))
    assert float(fit.weighted_mean(logits, y, class_weights[y])) == pytest.approx(plain, abs=1e-6)
    seen = {}
    for name, node_budget in (('whole', None), ('parts', budget)):
        m = small_gat().eval()
        optimizer = torch.optim.SGD(m.parameters(), lr=0.0)
        loss, parts = fit.train_batch(m, batch, nn.CrossEntropyLoss(weight=class_weights),
                                      nn.CrossEntropyLoss(weight=class_weights, reduction='sum'), optimizer, torch.device('cpu'),
                                      node_budget, None, weight)
        seen[name] = (loss, parts, [p.grad.clone() for p in m.parameters() if p.grad is not None])
    assert seen['whole'][0] == pytest.approx(by_hand, abs=1e-6) and seen['whole'][1] == 1
    assert (seen['parts'][1] > 1) == (budget is not None)
    assert seen['parts'][0] == pytest.approx(by_hand, abs=1e-5)                                # the parts add up to the whole batch's
    for a, b in zip(seen['whole'][2], seen['parts'][2]):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)


@needs_pyg
@pytest.mark.parametrize('budget', [None, 40])
def test_without_the_flag_the_first_step_is_the_class_weighted_cross_entropy_with_it_the_cell_weighted_mean(v3_training, budget):
    """The first optimiser step of `fit.fit` on the batch it draws: its loss against `CrossEntropyLoss(weight=)` of the class
    weights (no flag) and against the cell-weighted mean (flag), each to 1e-6 (1e-5 when the node budget cuts the batch)."""
    graphs, ids, rows = v3_training.graphs, v3_training.ids, v3_training.rows
    val_ids = [r['sample_id'] for r in v3_training.manifest.val]
    graphs = {**graphs, **{i: graphs[ids[0]] for i in val_ids}}                                # the epoch's evaluation needs some graphs
    labels = torch.tensor([int(graphs[i].y) for i in ids])
    counts = torch.bincount(labels, minlength=2).float().clamp(min=1)
    by_id, _ = fit.cell_weight_table(rows, ids)
    hp = {**train_b.HP, 'epochs': 1, 'batch_size': 16, **({} if budget is None else {'node_budget': budget})}
    first, real = {}, fit.train_batch
    for flag in (False, True):
        calls = []

        def spy(model, batch, criterion, criterion_sum, optimizer, device, budget=None, grad_clip=None, sample_weight=None,
                calls=calls):
            before = None
            if not calls:                                                       # the loss this batch has under the model before the step
                with torch.no_grad():
                    logits, y = model(batch), batch.y.view(-1)
                    w = torch.tensor([by_id[i] for i in batch.sample_id], dtype=torch.float32)
                    before = (float(nn.CrossEntropyLoss(weight=counts.sum() / counts)(logits, y)),
                              float((nn.functional.cross_entropy(logits, y, reduction='none') * w).sum() / w.sum()))
            loss, parts = real(model, batch, criterion, criterion_sum, optimizer, device, budget, grad_clip, sample_weight)
            if before:
                calls.append((loss, *before, parts))
            return loss, parts
        fit.train_batch = spy
        try:
            result = fit.fit(small_gat(), ids, val_ids, graphs, hp=hp, seed=0, device=torch.device('cpu'), log=quiet,
                             protocol='kfold_v3', arch='all', arch_balance=flag)
        finally:
            fit.train_batch = real
        first[flag] = (calls[0], result)
    tol = 1e-6 if budget is None else 1e-5
    (loss, todays, cells, parts), result = first[False]
    assert loss == pytest.approx(todays, abs=tol) and 'cell_weights' not in result and (parts > 1) == (budget is not None)
    (loss, todays, cells, parts), result = first[True]
    assert loss == pytest.approx(cells, abs=tol) and loss != pytest.approx(todays, abs=1e-4)
    assert set(result['cell_weights']) == {'R_x86', 'R_x64', 'G_x86', 'G_x64'}


# ---- B1: two epochs on each dataset ------------------------------------------------------------------------------------------

NAMES = {('kfold', 'all', False): 'kfold2_all_gnn_s1_team-sig500v3-emb', ('kfold', 'all', True): 'kfold2_all_gnn-ab_s1_team-sig500v3-emb',
         ('kfold_v3', 'all', False): 'v3kfold2_all_gnn_s1_team-sig500v3-emb',
         ('kfold_v3', 'all', True): 'v3kfold2_all_gnn-ab_s1_team-sig500v3-emb',
         ('kfold_v3', 'matched', False): 'v3kfold2_matched_gnn_s1_team-sig500v3-emb',
         ('kfold_v3', 'matched', True): 'v3kfold2_matched_gnn-ab_s1_team-sig500v3-emb'}


@needs_pyg
@pytest.mark.parametrize('arch_balance', [False, True])
@pytest.mark.parametrize('protocol, arch', TRAINED)
def test_a_two_epoch_run_on_each_dataset_writes_the_named_run_and_leaves_the_excluded_ids_out(
        tiny_v3, tmp_path, monkeypatch, protocol, arch, arch_balance):                        # noqa: F811
    rows = tp.sample_rows(protocol, 'all')
    manifest = tp.fold_manifest(2, 1, protocol, arch)
    victims = {manifest.train[0]['sample_id'], manifest.test[0]['sample_id']}                  # one training and one test binary
    folder = synthetic_build(tmp_path / 'b', sorted(set(rows) - victims), rows, dataset=(protocol, 'all'),
                             excluded={v: WHY for v in sorted(victims)})
    monkeypatch.setattr(fit, 'RUNS', tmp_path / 'runs')
    monkeypatch.setitem(train_b.ARM_HP, 'sig500v3-emb', {**train_b.HP, 'epochs': 2})
    taken, real_fit = {}, fit.fit

    def spy(model, train_ids, val_ids, graphs, **kw):
        taken.update(train=list(train_ids), val=list(val_ids), graphs=set(graphs), kw=kw)
        return real_fit(model, train_ids, val_ids, graphs, **kw)
    monkeypatch.setattr(fit, 'fit', spy)
    out = train_b.run('sig500v3-emb', 2, 1, device='cpu', b_dir=folder, log=quiet, protocol=protocol, arch=arch,
                      arch_balance=arch_balance)
    name = NAMES[protocol, arch, arch_balance]
    assert out['dir'] == tmp_path / 'runs' / name == tmp_path / 'runs' / tp.team_config(2, 1, 'sig500v3-emb', protocol, arch, arch_balance).name
    assert taken['kw']['protocol'] == protocol and taken['kw']['arch'] == arch and taken['kw']['arch_balance'] == arch_balance
    # excluded ids are in no split of the trainer, and not in the predictions
    assert not victims & (set(taken['train']) | set(taken['val']) | set(taken['graphs']))
    for file in ('predictions.csv', 'predictions_auc.csv'):
        text = (out['dir'] / file).read_text(encoding='utf-8')
        assert not any(v in text for v in victims)
    metrics = json.loads((out['dir'] / 'metrics.json').read_text(encoding='utf-8'))
    assert metrics['excluded'] == sorted(victims) and metrics['excluded_reason'] == WHY and metrics['run'] == name
    assert metrics['build_binaries'] == len(rows) - 2 and metrics['features'] == 'emb' and metrics['recipe']['epochs'] == 2
    saved = json.loads((out['dir'] / 'manifest.json').read_text(encoding='utf-8'))             # the fold's own manifest
    assert {s: [r['sample_id'] for r in getattr(manifest, s)] for s in ('train', 'val', 'test')} == {s: saved[s] for s in ('train', 'val', 'test')}
    if arch == 'matched':
        matched = {r['sample_id'] for r in tp._index(protocol) if str(r['arch_matched']) == '1'}
        everyone = {i for s in ('train', 'val', 'test') for i in saved[s]}
        assert everyone and everyone <= matched and everyone < set(rows)                       # all arch_matched, a proper subset of all
        assert all(str(rows[i]['arch_matched']) == '1' for i in everyone)
    train_rows = [r for r in manifest.train if r['sample_id'] not in victims]
    if arch_balance:
        expect = cell_weights([r['label'] for r in train_rows], [r['arch'] for r in train_rows])
        named = {fit.cell_name(r['label'], r['arch']): w for r, w in zip(train_rows, expect)}
        assert metrics['recipe']['arch_balance'] is True and set(metrics['recipe']['cell_weights']) == set(named)
        for cell, w in named.items():
            assert metrics['recipe']['cell_weights'][cell] == pytest.approx(w)                  # the cell weights of the training manifest
        everything = [r for r in manifest.train + manifest.val + manifest.test if r['sample_id'] not in victims]
        assert fit.cell_weight_table({r['sample_id']: r for r in everything}, [r['sample_id'] for r in everything])[1] != named
    else:
        assert 'arch_balance' not in metrics['recipe'] and 'cell_weights' not in metrics['recipe']
    assert 0.0 <= metrics['test']['auc'] <= 1.0


@needs_pyg
@pytest.mark.parametrize('protocol, arch', TRAINED)
def test_a_smoke_run_on_each_dataset_writes_no_run_directory(tiny_v3, tmp_path, monkeypatch, protocol, arch):   # noqa: F811
    rows = tp.sample_rows(protocol, 'all')
    folder = synthetic_build(tmp_path / 'b', sorted(rows), rows, dataset=(protocol, 'all'))
    monkeypatch.setattr(fit, 'RUNS', tmp_path / 'runs')
    monkeypatch.setattr(fit, 'write_run', lambda *a, **k: pytest.fail('a smoke run must not call write_run'))
    lines = []
    out = train_b.run('sig500v3-feat', 2, 1, smoke=8, device='cpu', b_dir=folder, log=lines.append, protocol=protocol, arch=arch,
                      arch_balance=True)
    expected = tp.team_config(2, 1, 'sig500v3-feat', protocol, arch, True).name
    assert any(l.startswith(f'smoke {expected}: ') for l in lines) and len(out['history']) == 2
    assert not (tmp_path / 'runs').exists()
    __import__('shutil').rmtree(out['dir'])


@needs_pyg
def test_a_run_refuses_a_build_of_another_dataset_before_it_reads_a_payload(tiny_v3, tmp_path):   # noqa: F811
    rows = tp.sample_rows('kfold_v3', 'all')
    folder = synthetic_build(tmp_path / 'b', sorted(rows), rows, dataset=('kfold_v3', 'all'))
    for protocol, arch, said in (('kfold', 'x86', 'was built for kfold_v3 all, this run asks for kfold x86'),
                                 ('kfold', 'all', 'was built for kfold_v3 all, this run asks for kfold all')):
        with pytest.raises(SystemExit, match=said):
            train_b.run('sig500v3-emb', 2, 1, smoke=8, device='cpu', b_dir=folder, log=quiet, protocol=protocol, arch=arch)
    x86 = synthetic_build(tmp_path / 'x', sorted(rows), rows)                                  # a build with no dataset keys: kfold x86
    with pytest.raises(SystemExit, match='was built for kfold x86, this run asks for kfold_v3 matched'):
        train_b.run('sig500v3-emb', 2, 1, smoke=8, device='cpu', b_dir=x86, log=quiet, protocol='kfold_v3', arch='matched')


# ---- B1: leakage ---------------------------------------------------------------------------------------------------------------

@needs_pyg
@pytest.mark.parametrize('protocol, arch', TRAINED)
def test_the_api_remap_and_its_unseen_set_come_from_the_training_ids_only(tiny_v3, tmp_path, monkeypatch, protocol, arch):   # noqa: F811
    """Training binaries draw their API ids from a few, validation and test binaries from all of them: the ids only they use are
    mapped to UNK_API, and the run's counts are those of the training ids, not of all the ids it reads."""
    pp = tp.import_team('preprocess_full', source='repo')
    first = pp.tok.FIRST_API_ID
    rows = tp.sample_rows(protocol, 'all')
    manifest = tp.fold_manifest(2, 1, protocol, arch)
    train = {r['sample_id'] for r in manifest.train}
    narrow = [pp.tok.NO_API, pp.tok.UNK_API, first, first + 1, first + 2]
    folder = synthetic_build(tmp_path / 'b', sorted(rows), rows, dataset=(protocol, 'all'),
                             kinds_of=lambda i: narrow if i in train else narrow + [first + 3, first + 4, first + 5])
    monkeypatch.setattr(fit, 'RUNS', tmp_path / 'runs')
    monkeypatch.setitem(train_b.ARM_HP, 'sig500v3-emb', {**train_b.HP, 'epochs': 2})
    lines = []
    out = train_b.run('sig500v3-emb', 2, 1, device='cpu', b_dir=folder, log=lines.append, protocol=protocol, arch=arch)
    metrics = json.loads((out['dir'] / 'metrics.json').read_text(encoding='utf-8'))
    used = {i: set(train_b.payload_api_ids(pp.tok, train_b.load_slim(folder, i, protocol, arch)).tolist())
            for i in (r['sample_id'] for rows_ in (manifest.train, manifest.val, manifest.test) for r in rows_)}
    from_train = set().union(*(used[i] for i in train))
    from_all = set().union(*used.values())
    assert from_train <= {first, first + 1, first + 2}
    assert from_all - from_train, 'the fixture must hold ids that only validation or test binaries use'
    assert metrics['n_api_ids_train'] == len(from_train) < len(from_all)
    assert metrics['n_api_ids_unseen_mapped'] == len(from_all - from_train) > 0
    assert any(f'unseen in val/test binaries mapped to UNK_API: {len(from_all - from_train)} of {len(from_all)} ids used' in l for l in lines)
    only_later = from_all - from_train
    seen = torch.tensor(sorted(from_train), dtype=torch.long)
    for i in used:
        remapped = train_b.remap_api(pp.tok, train_b.load_slim(folder, i, protocol, arch)['instruction_ids'], seen)[:, 3]
        assert not set(remapped[remapped >= first].tolist()) & only_later and set(remapped[remapped >= first].tolist()) <= from_train
