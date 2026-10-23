"""python -m pytest tests/test_datasets_train.py: the trainers of Models 1 to 3 on any dataset (plan v6, steps B1 to B3).

What is checked, on the second synthetic dataset `tiny_v3` (tests/tiny_cache.py) and synthetic pipeline-B builds of it
(`test_team_v3.synthetic_build`, which pools the block embeddings so that `check_repool` holds):
  * B1, `train_b` / `fit.py` / the queue verb: the refusals of a build of another dataset in both directions, of the legacy and the
    uncapped arms, the default build directory of each (protocol, arch), the 2-epoch runs on kfold all, kfold_v3 all and matched (run
    names, manifests, excluded ids in no split), the architecture-balance weights against `cell_weights` and the loss against
    `CrossEntropyLoss(weight=)`, the API remap from the training ids only, `queue --dry` names;
  * B2, `fivechannel`: the four (protocol, arch) pairs, the `-ab` name, the build check, the cell weights and their loss;
  * B3, `package_build`: the zip name and the README text of the three labels, and the recorded x86 README text;
  * the (protocol, arch) tables of the modules name the same datasets, and (`real_data`) `BUILD_TOTAL` is the number of ids of each;
  * `real_data` (read only): the published builds are kfold x86 builds that only the x86 arms read, no run directory has a name of this
    step, the real folds give the four cell weights from the training manifest.
The tests that train need torch_geometric and skip with a reason without it; the refusals, names and weights do not.
"""
import io
import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

import experiments.train as experiments_train
from common import manifests
from common.config import COMBINED_CACHE, DEFAULT_CACHE, FIVECHANNEL_DATASETS, RUNS, RunConfig
from common.manifests import load_index, make_manifest
from experiments import queue as q
from experiments.team_pipelines import common as tp, fit, package_build as pb, train_b
from s03a_transformer.five_channel import train as fc
from s04_evaluation import gate_tables, report
from s04_evaluation.metrics import cell_weights
from test_five_channel import fivechannel, write_build
from test_team_pipelines import fresh_team_caches, needs_pyg, pinned_team_modules  # noqa: F401
from test_team_v3 import settings, synthetic_build
from tiny_cache import tiny_v3  # noqa: F401  (a fixture)

HERE = Path(__file__).resolve().parent
BASE = 'processed/cfg_subgraphs/'
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


# ---- B2: fivechannel on a dataset --------------------------------------------------------------------------------------------

FIVE = [('kfold', 'x86'), ('kfold', 'all'), ('kfold_v3', 'all'), ('kfold_v3', 'matched')]


def five_run(fold=0, protocol='kfold', arch='x86', **kw):
    return fivechannel(fold, protocol=protocol, arch=arch, **kw)


def test_fivechannel_names_carry_the_dataset_and_the_ab_twin():
    assert [five_run(0, p, a).name for p, a in FIVE] == [
        'kfold0_x86_fivechannel_s0', 'kfold0_all_fivechannel_s0', 'v3kfold0_all_fivechannel_s0', 'v3kfold0_matched_fivechannel_s0']
    assert [five_run(3, p, a, seed=2, max_instructions=512).name for p, a in FIVE] == [
        'kfold3_x86_fivechannel-L512_s2', 'kfold3_all_fivechannel-L512_s2', 'v3kfold3_all_fivechannel-L512_s2',
        'v3kfold3_matched_fivechannel-L512_s2']
    ab = [five_run(1, p, a, arch_balance=True) for p, a in FIVE[1:]]
    assert [c.name for c in ab] == ['kfold1_all_fivechannel-ab_s0', 'v3kfold1_all_fivechannel-ab_s0', 'v3kfold1_matched_fivechannel-ab_s0']
    assert [c.group for c in ab] == ['kfold_all_fivechannel-ab', 'kfold_v3_all_fivechannel-ab', 'kfold_v3_matched_fivechannel-ab']
    assert five_run(1, 'kfold', 'all', arch_balance=True, max_instructions=512, threshold_from='oof').name == \
        'kfold1_all_fivechannel-L512-ab-thoof_s0'
    names = {five_run(f, p, a, seed=s, max_instructions=m, arch_balance=b, tag=t).name
             for p, a in FIVE for f in range(5) for s in range(3) for m in (128, 512) for b in (False, True) for t in ('', 'whole')
             if not (b and a == 'x86')}
    assert len(names) == 3 * 5 * 3 * 2 * 2 * 2 + 5 * 3 * 2 * 2          # distinct: the dataset, L512, ab and the tag all show
    recorded = {r['name'] for r in json.loads((HERE / 'fixtures' / 'run_names.json').read_text(encoding='utf-8'))}
    recorded |= {r['name'] for r in json.loads((HERE / 'fixtures' / 'run_names_grid.json').read_text(encoding='utf-8'))}
    assert not names & recorded                                              # no recorded name is a name of this step
    # the plain x86 name is today's; the gnn arms get no new letters from the fivechannel branch
    assert five_run().name == 'kfold0_x86_fivechannel_s0' and RunConfig(protocol='kfold', fold=0, arch_balance=True).name == 'kfold0_x86_gnn-ab_s0'


@pytest.mark.parametrize('kw', [dict(protocol='kfold_v3', arch='x86'), dict(protocol='kfold', arch='matched'), dict(arch_balance=True),
                                dict(protocol='kfold_v3', arch='all', val_seed=1), dict(protocol='kfold', arch='all', control='chain'),
                                dict(protocol='kfold', arch='all', select_on='cell'), dict(protocol='kfold_v3', arch='matched', mask_blocks=0.1)])
def test_fivechannel_refuses_a_dataset_without_a_build_and_the_options_that_stay_refused(kw):
    with pytest.raises(ValueError):
        fivechannel(0, **kw)


def test_arch_balance_stays_refused_by_the_other_sequence_model_and_the_name_of_the_gnn_is_unchanged():
    with pytest.raises(ValueError, match='arch_balance'):
        RunConfig(protocol='kfold', fold=0, arch='all', model='tokenseq', arch_balance=True)
    for p, a in FIVE[1:]:
        assert RunConfig(protocol=p, fold=0, arch=a, model='gnn', arch_balance=True).variant == 'gnn-ab'


def test_the_command_lines_take_the_dataset_flags():
    cfg, out, options = experiments_train.parse_args(['--model', 'fivechannel', '--protocol', 'kfold_v3', '--arch', 'matched', '--fold', '4',
                                                      '--arch-balance', '--build', 'b', '--device', 'cpu', '--smoke', '8'])
    assert cfg.name == 'v3kfold4_matched_fivechannel-ab_s0' and options == {'build': 'b', 'device': 'cpu', 'smoke': 8}
    assert experiments_train.parse_args(['--model', 'fivechannel', '--protocol', 'kfold', '--arch', 'all', '--fold', '1'])[0].name == \
        'kfold1_all_fivechannel_s0'
    for argv in (['--protocol', 'kfold_v3', '--fold', '0'], ['--protocol', 'kfold', '--fold', '0', '--arch-balance'],
                 ['--protocol', 'kfold', '--fold', '0', '--arch', 'matched']):          # arch x86 by default, or a pair with no build
        with pytest.raises(SystemExit):
            experiments_train.parse_args(['--model', 'fivechannel', *argv])
    cfg, opts = fc.parse_config(['--fold', '3', '--max-instructions', '512', '--build', 'b', '--smoke', '4'])        # today's command line
    assert cfg.name == 'kfold3_x86_fivechannel-L512_s0' and opts == {'build': 'b', 'device': None, 'smoke': 4}
    assert fc.parse_config(['--fold', '2', '--protocol', 'kfold_v3'])[0].name == 'v3kfold2_all_fivechannel_s0'      # arch: all by default
    assert fc.parse_config(['--fold', '2', '--protocol', 'kfold_v3', '--arch', 'matched', '--arch-balance'])[0].name == \
        'v3kfold2_matched_fivechannel-ab_s0'
    assert fc.parse_config(['--fold', '2', '--protocol', 'kfold', '--arch', 'all'])[0].name == 'kfold2_all_fivechannel_s0'
    for argv in (['--protocol', 'kfold', '--arch', 'matched'], ['--arch-balance'], ['--protocol', 'kfold_v3', '--arch', 'x86']):
        with pytest.raises(SystemExit):
            fc.parse_config(['--fold', '2', *argv])


def test_the_default_build_hint_names_the_dataset(tmp_path, monkeypatch):
    monkeypatch.delenv('GT_TEAM_B_BUILD', raising=False)
    monkeypatch.setattr(fc, 'source_for', lambda name: (_ for _ in ()).throw(fc.MissingSource('none')))
    for (protocol, arch), words in {('kfold', 'x86'): 'unzip Mendeley_x86_cfg_subgraphs_signal_500blocks_v3.zip at CT_GAT/',
                                    ('kfold', 'all'): 'unzip Mendeley_all_cfg_subgraphs_signal_500blocks_v3.zip at CT_GAT/',
                                    ('kfold_v3', 'all'): 'unzip combined_v3_cfg_subgraphs_signal_500blocks_v3.zip at CT_GAT/',
                                    ('kfold_v3', 'matched'): 'unzip combined_v3_cfg_subgraphs_signal_500blocks_v3.zip at CT_GAT/'}.items():
        with pytest.raises(FileNotFoundError, match=words):
            fc.resolve_build(None, protocol, arch)
    assert fc.fill_build() == fc.FILL_BUILD
    here = tmp_path / 'there'
    here.mkdir()
    monkeypatch.setattr(fc, 'source_for', lambda name: here)
    with pytest.raises(FileNotFoundError, match='combined_v3_cfg_subgraphs_signal_500blocks_v3.zip'):
        fc.resolve_build(None, 'kfold_v3', 'all')                      # an empty folder: the hint of the dataset


def five_build(root, rows, dataset=None, excluded=None, skip=()):
    """A `write_build` folder of the rows with the settings.json a build_b folder of `dataset` has."""
    write_build(root, rows, skip=skip)
    body = {'excluded': excluded or {}}
    if dataset is not None:
        body.update(protocol=dataset[0], arch=dataset[1])
    (root / 'settings.json').write_text(json.dumps(body), encoding='utf-8')
    return root


def test_a_run_refuses_a_build_of_another_dataset_in_both_directions(tiny_v3, tmp_path):
    rows = tiny_v3.rows
    builds = {None: five_build(tmp_path / 'x86', rows), ('kfold', 'all'): five_build(tmp_path / 'ma', rows, ('kfold', 'all')),
              ('kfold_v3', 'all'): five_build(tmp_path / 'c', rows, ('kfold_v3', 'all'))}
    ok = {None: [('kfold', 'x86')], ('kfold', 'all'): [('kfold', 'all')], ('kfold_v3', 'all'): [('kfold_v3', 'all'), ('kfold_v3', 'matched')]}
    for key, folder in builds.items():
        for protocol, arch in FIVE:
            cfg = five_run(0, protocol, arch, max_instructions=16)
            if (protocol, arch) in ok[key]:
                continue
            have = ('kfold', 'x86') if key is None else key
            with pytest.raises(SystemExit, match=f'was built for {have[0]} {have[1]}, this run asks for {protocol} {arch}'):
                fc.run(cfg, folder, 'cpu', 6, log=quiet)
    # a refused run creates no run directory
    with pytest.raises(SystemExit):
        fc.run(five_run(0, 'kfold_v3', 'all'), builds[None], 'cpu', None, tmp_path / 'runs', log=quiet)
    assert not (tmp_path / 'runs').exists()
    bare = write_build(tmp_path / 'bare', rows)                                  # a folder of .pt files with no settings: kfold x86
    with pytest.raises(SystemExit, match='was built for kfold x86, this run asks for kfold all'):
        fc.run(five_run(0, 'kfold', 'all'), bare, 'cpu', 6, log=quiet)


@pytest.mark.parametrize('arch_balance', [False, True])
@pytest.mark.parametrize('protocol, arch', FIVE[1:])
def test_a_two_epoch_run_on_each_dataset(tiny_v3, tmp_path, protocol, arch, arch_balance):
    rows = tiny_v3.rows
    cfg = five_run(2, protocol, arch, seed=1, max_instructions=16, arch_balance=arch_balance, epochs=2)
    manifest = make_manifest(cfg, load_index())
    victim = manifest.test[0]['sample_id']
    left_in_train = manifest.train[0]['sample_id']
    build = five_build(tmp_path / 'b', rows, (protocol, 'all'), {victim: 'listing too large', left_in_train: 'listing too large'},
                       skip={victim, left_in_train})
    runs = tmp_path / 'runs'
    metrics = fc.run(cfg, build, 'cpu', None, runs, log=quiet)
    d = runs / cfg.name
    assert cfg.name == {(False, 'kfold', 'all'): 'kfold2_all_fivechannel-L16_s1', (True, 'kfold', 'all'): 'kfold2_all_fivechannel-L16-ab_s1',
                        (False, 'kfold_v3', 'all'): 'v3kfold2_all_fivechannel-L16_s1',
                        (True, 'kfold_v3', 'all'): 'v3kfold2_all_fivechannel-L16-ab_s1',
                        (False, 'kfold_v3', 'matched'): 'v3kfold2_matched_fivechannel-L16_s1',
                        (True, 'kfold_v3', 'matched'): 'v3kfold2_matched_fivechannel-L16-ab_s1'}[arch_balance, protocol, arch]
    assert RunConfig.load(d / 'config.json') == cfg and metrics['run'] == d.name and metrics['epochs_run'] == 2
    saved = json.loads((d / 'manifest.json').read_text(encoding='utf-8'))
    assert saved['train'] == [r['sample_id'] for r in manifest.train] and saved['test'] == [r['sample_id'] for r in manifest.test]
    assert metrics['excluded'] == sorted([victim, left_in_train])               # excluded ids are in no split of the run
    for name in ('predictions.csv', 'predictions_auc.csv'):
        text = (d / name).read_text(encoding='utf-8')
        assert victim not in text and left_in_train not in text
    if arch == 'matched':
        index = {r['sample_id']: r for r in load_index()}
        assert all(str(index[i]['arch_matched']) == '1' for s in ('train', 'val', 'test') for i in saved[s])
        assert len({i for s in ('train', 'val', 'test') for i in saved[s]}) < len(rows)
    train_rows = [r for r in manifest.train if r['sample_id'] not in {victim, left_in_train}]
    if arch_balance:
        expect = cell_weights([r['label'] for r in train_rows], [r['arch'] for r in train_rows])
        named = {fit.cell_name(r['label'], r['arch']): w for r, w in zip(train_rows, expect)}
        assert metrics['arch_balance'] is True and set(metrics['cell_weights']) == set(named)
        for cell, w in named.items():
            assert metrics['cell_weights'][cell] == pytest.approx(w)             # the cell weights of the training manifest only
    else:
        assert 'arch_balance' not in metrics and 'cell_weights' not in metrics
    assert metrics['build_dir'] == str(build)


def test_the_loss_is_the_class_weighted_mean_of_today_and_with_the_flag_the_cell_weighted_mean(tiny_v3, tmp_path):
    rows = tiny_v3.rows
    build = five_build(tmp_path / 'b', rows, ('kfold_v3', 'all'))
    for flag in (False, True):
        cfg = five_run(2, 'kfold_v3', 'all', max_instructions=16, arch_balance=flag)
        manifest = make_manifest(cfg, load_index())
        train_rows = manifest.train[:20]                                       # one batch of 20 (< BATCH_SIZE)
        ids = [r['sample_id'] for r in train_rows]
        binaries = fc.load_binaries(build, {r['sample_id']: r['label'] for r in train_rows}, 16)
        items = fc.make_items(binaries, ids, fc.seen_api_ids(binaries, ids))
        weight = fc.loss_weights(cfg, train_rows, torch.device('cpu'))
        assert isinstance(weight, dict) == flag
        if flag:
            expect = cell_weights([r['label'] for r in train_rows], [r['arch'] for r in train_rows])
            assert [weight[i] for i in ids] == pytest.approx(expect.tolist())
        else:
            assert torch.equal(weight, fc.class_weights([r['label'] for r in train_rows], torch.device('cpu')))
        net = fc.new_net(cfg, fc.api_vocab_size(build, binaries), torch.device('cpu'))
        reference = fc.new_net(cfg, fc.api_vocab_size(build, binaries), torch.device('cpu'))
        reference.load_state_dict(net.state_dict())
        order = np.random.default_rng((0, 1, 7919)).permutation(len(ids))
        batch = fc.to_device(fc.collate(items, [ids[i] for i in order], cfg), torch.device('cpu'))
        reference.train()
        torch.manual_seed(5)
        per_binary = torch.nn.functional.cross_entropy(fc.forward(reference, batch).float().detach(), batch['label'], reduction='none')
        if flag:
            w = torch.tensor([weight[i] for i in batch['sample_ids']], dtype=torch.float32)
            expected = float((per_binary * w).sum() / w.sum())
        else:
            expected = float((per_binary * weight[batch['label']]).mean())            # today's loss
        torch.manual_seed(5)
        loss = fc.train_epoch(net, torch.optim.AdamW(net.parameters(), lr=0.0), items, ids, weight, cfg, 1, torch.device('cpu'), 0)
        assert loss == pytest.approx(expected, abs=1e-6)


# ---- B3: package_build -------------------------------------------------------------------------------------------------------

DATASET_SETTINGS = {'Mendeley_x86': {}, 'Mendeley_all': {'protocol': 'kfold', 'arch': 'all'},
                    'combined_v3': {'protocol': 'kfold_v3', 'arch': 'all'}}


@pytest.mark.parametrize('label, sample, blocks, name', [
    ('Mendeley_x86', 'signal', 500, 'Mendeley_x86_cfg_subgraphs_signal_500blocks_v3.zip'),
    ('Mendeley_x86', 'callwalk', 250, 'Mendeley_x86_cfg_subgraphs_callwalk_250blocks_v3.zip'),
    ('Mendeley_x86', 'signal', train_b.UNCAPPED_BLOCKS, 'Mendeley_x86_cfg_subgraphs_uncapped_v3.zip'),
    ('Mendeley_all', 'signal', 500, 'Mendeley_all_cfg_subgraphs_signal_500blocks_v3.zip'),
    ('Mendeley_all', 'signal', 250, 'Mendeley_all_cfg_subgraphs_signal_250blocks_v3.zip'),
    ('Mendeley_all', 'callwalk', 500, 'Mendeley_all_cfg_subgraphs_callwalk_500blocks_v3.zip'),
    ('Mendeley_all', 'callwalk', 250, 'Mendeley_all_cfg_subgraphs_callwalk_250blocks_v3.zip'),
    ('Mendeley_all', 'signal', train_b.UNCAPPED_BLOCKS, 'Mendeley_all_cfg_subgraphs_uncapped_v3.zip'),
    ('Mendeley_all', 'callwalk', 0, 'Mendeley_all_cfg_subgraphs_uncapped_v3.zip'),
    ('combined_v3', 'signal', 500, 'combined_v3_cfg_subgraphs_signal_500blocks_v3.zip'),
    ('combined_v3', 'callwalk', 500, 'combined_v3_cfg_subgraphs_callwalk_500blocks_v3.zip')])
def test_the_zip_name_carries_the_dataset(label, sample, blocks, name):
    assert pb.zip_name({'sample': sample, 'num_blocks': blocks, **DATASET_SETTINGS[label]}) == name
    if label == 'Mendeley_x86':        # the keys a kfold x86 build never wrote, or wrote with their default values
        assert pb.zip_name({'sample': sample, 'num_blocks': blocks, 'protocol': 'kfold', 'arch': 'x86'}) == name


def test_a_build_for_a_dataset_that_is_never_packaged_is_refused():
    for protocol, arch in (('kfold_v3', 'matched'), ('kfold_v3', 'x86'), ('kfold', 'matched'), ('bundle', 'all')):
        with pytest.raises(SystemExit, match=f'a build for {protocol} {arch} is not packaged'):
            pb.zip_name({'sample': 'signal', 'num_blocks': 500, 'protocol': protocol, 'arch': arch})
    with pytest.raises(SystemExit, match='only kfold x86, kfold all, kfold_v3 all'):
        pb.readme_text({'sample': 'signal', 'num_blocks': 500, 'protocol': 'kfold_v3', 'arch': 'matched'}, {'processed': 0, 'total': 0}, [])


def test_the_readme_text_of_an_x86_build_is_the_recorded_one():
    """tests/fixtures/package_readme_x86.json was written by `readme_text` and `zip_name` before the dataset labels existed (c159512)
    for three settings / summary pairs (capped with pins and seeded rows, callwalk with an excluded id, uncapped)."""
    recorded = json.loads((HERE / 'fixtures' / 'package_readme_x86.json').read_text(encoding='utf-8'))
    assert [c['name'] for c in recorded['cases']] == ['signal-250-pins-seeded', 'callwalk-500-no-pins-excluded', 'uncapped']
    for case in recorded['cases']:
        assert pb.zip_name(case['settings']) == case['zip_name'] and case['zip_name'].startswith('Mendeley_x86_cfg_subgraphs_')
        assert pb.readme_text(case['settings'], case['summary'], recorded['rows'], case['manifests']) == case['readme'], case['name']
        explicit = {**case['settings'], 'protocol': 'kfold', 'arch': 'x86'}
        assert pb.readme_text(explicit, case['summary'], recorded['rows'], case['manifests']) == case['readme']
    assert 'Per class and architecture' not in recorded['cases'][1]['readme'] and 'Excluded on purpose' not in recorded['cases'][1]['readme']


def dataset_rows(n_x86=6, n_x64=3):
    return [{'sample_id': f'{i:064x}', 'label': int(i % 2), 'family': f'f{i % 3}', 'arch': 'x86' if i < n_x86 else 'x64', 'fold': i % 5,
             'n_instructions': 10 + i, 'match_rate': 0.9, 'n_blocks_total': 5, 'n_blocks_kept': 4} for i in range(n_x86 + n_x64)]


def test_the_readme_of_a_mendeley_all_build_says_what_the_extension_added():
    pins = {'CT_GAT/pipeline/s02_preprocess/preprocess_full.py': 'a' * 64, 'CT_GAT/pipeline/s02_preprocess/tokenize_asm_gz.py': 'b' * 64}
    settings = {'sample': 'signal', 'num_blocks': 500, 'seed': 0, 'source': 'main-fix', 'pins': {'old.py': 'c' * 64}, 'protocol': 'kfold',
                'arch': 'all', 'extensions': [{'arch': 'all', 'ids_added': 3, 'source': 'repo', 'pins': pins,
                                               'embedder_rows': 'seeded-per-id', 'date': '2026-10-23'}]}
    rows = dataset_rows()
    text = pb.readme_text(settings, {'processed': 9, 'total': 9, 'excluded': {}}, rows, manifests=False)
    assert text.startswith('# Mendeley all-architecture (x86 and x64) CFG subgraphs (signal, 500 blocks), pipeline B of the team pipelines')
    assert 'the 9 Mendeley binaries (x86 and x64) of our 5-fold family-holdout set `kfold` (`folds_mendeley.csv`) (4 ransomware,' in text
    assert 'Per corpus' not in text                                                  # no corpus map: no line
    assert 'Per class and architecture: R x86 3, R x64 1, G x86 3, G x64 2.' in text
    assert ('its 6 x86 samples are byte-identical to the ones in `Mendeley_x86_cfg_subgraphs_signal_500blocks_v3.zip`' in text
            and 'the 3 x64 samples were added to it with `build_b --extend` (code source `repo`, files of the repository when extended:' in text)
    assert f"`CT_GAT/pipeline/s02_preprocess/preprocess_full.py`: sha256 `{'a' * 64}`" in text and 'drawn per (seed, API id)' in text
    assert 'for `--protocol kfold --arch all` only' in text and 'CFGs come from the Mendeley bundle' in text
    assert 'our kfold fold,' in text and 'Excluded on purpose' not in text and "source `main-fix`" in text      # the build's own source stays
    assert text.count(f"sha256 `{'c' * 64}`") == 1                                                            # the build's own pins stay too
    for forbidden in ('C:\\', 'C:/', '/Users/', 'Downloads', 'AppData'):
        assert forbidden not in text
    plain = pb.readme_text({k: v for k, v in settings.items() if k != 'extensions'}, {'processed': 9, 'total': 9}, rows)
    assert 'byte-identical' not in plain                                                                      # an unextended all build: no note


def test_the_readme_of_a_combined_build_lists_the_excluded_binaries_with_their_reason():
    settings = {'sample': 'callwalk', 'num_blocks': 500, 'seed': 0, 'source': 'repo', 'protocol': 'kfold_v3', 'arch': 'all'}
    rows = dataset_rows(6, 6)
    excluded = {f'{99 - i:064x}': WHY for i in range(3)}
    corpora = {r['sample_id']: ('mendeley' if i < 4 else 'ransomware_balanced' if i < 8 else 'balanced_goodware') for i, r in enumerate(rows)}
    text = pb.readme_text(settings, {'processed': 12, 'total': 15, 'excluded': excluded}, rows, corpora=corpora)
    assert 'Per corpus: balanced_goodware 4, mendeley 4, ransomware_balanced 4.' in text
    assert text.startswith('# combined_v3 (all architectures) CFG subgraphs (callwalk, 500 blocks), pipeline B of the team pipelines')
    assert 'the 15 binaries (x86 and x64) of the combined_v3 dataset that have a fold of our 5-fold family-holdout set `kfold_v3`' in text
    assert '3 excluded on purpose' in text and 'Per class and architecture: R x86 3, R x64 3, G x86 3, G x64 3.' in text
    assert ('CFGs and listings come from the collections Mendeley_Ransomware, Mendeley_Goodware, Ransomware_Balanced and Balanced_Goodware'
            in text and 'our kfold_v3 fold,' in text and 'a walk that grows' in text)
    assert 'Excluded on purpose (3; `excluded` of build_info/build_settings.json)' in text
    for sample_id in excluded:                                                            # every id with its reason, sorted
        assert f'\n- `{sample_id}`: {WHY}' in text
    assert text.index(f'`{sorted(excluded)[0]}`') < text.index(f'`{sorted(excluded)[-1]}`')
    assert 'The `--arch matched` runs of the trainers read this archive too' in text and 'byte-identical' not in text
    assert 'Excluded on purpose' not in pb.readme_text(settings, {'processed': 12, 'total': 12, 'excluded': {}}, rows)
    assert 'no exclusions' in pb.readme_text(settings, {'processed': 12, 'total': 12, 'excluded': {}}, rows)


def finished_dataset_build(root, label, extra=None, excluded=None):
    """A synthetic finished build of a dataset other than Mendeley x86: payloads with `dataset_metadata` carrying the arch of the row
    and the fold of the dataset's protocol, the summary of `build_b`, and the extension record of `build_b --extend` when asked."""
    protocol, arch = ('kfold_v3', 'all') if label == 'combined_v3' else ('kfold', 'all')
    rows = tp.sample_rows(protocol, arch)
    left_out = dict(excluded or {})
    ids = sorted(set(rows) - set(left_out))
    folder = synthetic_build(root, ids, rows, source='repo', sample='signal', num_blocks=500, dataset=(protocol, arch), excluded=left_out)
    for i in ids:
        payload = torch.load(folder / f'{i}.pt', weights_only=False)
        payload['dataset_metadata'] = {'sample_id': i, 'label': rows[i]['label'], 'arch': rows[i]['arch'], 'family': rows[i]['family'],
                                       'fold': rows[i]['fold']}
        torch.save(payload, folder / f'{i}.pt')
        (folder / f'{i}.manifest.json').write_text(json.dumps({'sample_id': i}), encoding='utf-8')
    (folder / 'build_summary.json').write_text(json.dumps({'processed': len(ids), 'total': len(rows), 'excluded': left_out}),
                                              encoding='utf-8')
    if extra:
        body = json.loads((folder / 'settings.json').read_text(encoding='utf-8'))
        (folder / 'settings.json').write_text(json.dumps({**body, **extra}), encoding='utf-8')
    return folder, rows, ids


@needs_pyg
@pytest.mark.parametrize('label', ['Mendeley_all', 'combined_v3'])
def test_a_packaged_build_of_another_dataset_unzips_and_the_trainers_read_it(tiny_v3, tmp_path, monkeypatch, label):   # noqa: F811
    import zipfile
    protocol = 'kfold_v3' if label == 'combined_v3' else 'kfold'
    victims = {sorted(tp.sample_rows(protocol, 'all'))[0]: WHY} if label == 'combined_v3' else None
    extra = None
    if label == 'Mendeley_all':
        n_x64 = sum(r['arch'] == 'x64' for r in tp.sample_rows('kfold', 'all').values())
        extra = {'extensions': [{'arch': 'all', 'ids_added': n_x64, 'source': 'repo', 'pins': {'p.py': 'a' * 64},
                                 'embedder_rows': 'seeded-per-id', 'date': '2026-10-23'}]}
    build, rows, ids = finished_dataset_build(tmp_path / 'b', label, extra, victims)
    monkeypatch.setattr(pb, 'cache_for', lambda protocol: tiny_v3.root)
    zpath, digest = pb.package(build, tmp_path / 'out', manifests=False)
    assert zpath.name == f'{label}_cfg_subgraphs_signal_500blocks_v3.zip'
    text = (tmp_path / 'out' / f'{zpath.stem}.README.md').read_text(encoding='utf-8')
    assert (f'the {len(rows)} ' in text) and (('Excluded on purpose (1;' in text) == (victims is not None))
    assert ('byte-identical' in text) == (extra is not None)
    per_corpus = {}
    for r in tiny_v3.rows:
        if r['sample_id'] in ids:
            per_corpus[r['corpus']] = per_corpus.get(r['corpus'], 0) + 1
    assert f"Per corpus: {', '.join(f'{c} {n}' for c, n in sorted(per_corpus.items()))}." in text      # from the cache index
    with zipfile.ZipFile(zpath) as z:
        assert z.read(f'README_{zpath.stem}.md').decode('utf-8') == text
        z.extractall(tmp_path / 'CT_GAT')
        index = z.read(BASE + 'build_info/index.csv').decode('utf-8').splitlines()
    unzipped = tmp_path / 'CT_GAT' / 'processed' / 'cfg_subgraphs'
    assert [l.split(',')[0] for l in index[1:]] == ids
    fold = {r['sample_id']: str(r['fold']) for r in rows.values()}                       # the fold of the build's protocol
    assert [l.split(',')[4] for l in index[1:]] == [fold[i] for i in ids]
    assert {l.split(',')[3] for l in index[1:]} == {'x86', 'x64'}
    assert train_b.read_excluded(unzipped) == (victims or {})
    arches = ('all',) if label == 'Mendeley_all' else ('all', 'matched')
    for arch in arches:                                                                    # the settings in the zip say what it serves
        assert train_b.check_build('sig500v3-emb', unzipped, None, protocol, arch) == 'repo'
    with pytest.raises(SystemExit, match=f'was built for {protocol} all, this run asks for kfold x86'):
        train_b.check_build('sig500v3-emb', unzipped)
    from s03a_transformer.five_channel import train as fc_train
    fc_train.check_build_dataset(unzipped, protocol, 'all')


# ---- real data (read only): the published builds and runs, the real folds ----------------------------------------------------

real_data = pytest.mark.real_data
needs_runs = pytest.mark.skipif(not (RUNS / 'team').is_dir(), reason='needs the run directory of this machine (GT_RUNS / datasets.json)')
needs_v2 = pytest.mark.skipif(not (Path(DEFAULT_CACHE) / 'index.csv').exists(), reason='needs the mendeley_x86 cache')
needs_combined = pytest.mark.skipif(not (Path(COMBINED_CACHE) / 'index.csv').exists(), reason='needs the combined cache')
PUBLISHED = {'b_sig500_v3': 'sig500v3-feat', 'b_sig250_main': 'sig250-feat', 'b_callwalk500_main': 'callwalk500-feat',
             'b_callwalk250_main': 'callwalk250-feat', 'b_sigwhole_v3': 'sigwhole-feat', 'b_sig500': 'sig500', 'b_sig500_v2': 'sig500v2'}


@real_data
@needs_runs
@pytest.mark.parametrize('name, arm', PUBLISHED.items())
def test_a_published_build_is_a_kfold_x86_build_that_the_x86_arm_reads_and_the_other_datasets_refuse(name, arm):
    build = RUNS / 'team' / name
    if not build.is_dir():
        pytest.skip(f'{build} is not on this machine')
    settings_on_disk = train_b.read_settings(build)
    assert 'protocol' not in settings_on_disk and 'arch' not in settings_on_disk                 # published builds are left as they were
    assert train_b.dataset_of(settings_on_disk) == ('kfold', 'x86')
    assert train_b.check_build(arm, build) == train_b.check_build(arm, build, None, 'kfold', 'x86') == train_b.read_source(build)
    assert train_b.ARM_DIR[arm] == build and train_b.default_build_dir(arm) == build == train_b.default_build_dir(arm, 'kfold', 'x86')
    for protocol, arch in (('kfold', 'all'), ('kfold_v3', 'all'), ('kfold_v3', 'matched')):
        with pytest.raises(SystemExit):                                                          # the legacy arms: off x86; the others: wrong build
            train_b.check_build(arm, build, None, protocol, arch)


@real_data
@needs_runs
def test_no_existing_fivechannel_run_carries_the_ab_suffix():
    """The only naming change of this step is the -ab suffix on fivechannel runs under arch_balance; no existing run has it."""
    existing = {p.name for p in RUNS.iterdir() if p.is_dir()}
    assert not [n for n in existing if 'fivechannel' in n and re.search(r'fivechannel(-L\d+)?-ab', n)]


@real_data
@pytest.mark.parametrize('protocol, arch', [('kfold', 'all'), ('kfold_v3', 'all'), ('kfold_v3', 'matched')])
def test_the_real_folds_give_four_cell_weights_from_the_training_manifest(monkeypatch, protocol, arch):
    cache = Path(DEFAULT_CACHE if protocol == 'kfold' else COMBINED_CACHE)
    if not (cache / 'index.csv').exists():
        pytest.skip(f'needs the {protocol} cache')
    monkeypatch.setattr(manifests, 'CACHE', cache)
    for fold in (0, 2):
        manifest = tp.fold_manifest(fold, 0, protocol, arch)
        rows = tp.sample_rows(protocol, 'all')
        ids = [r['sample_id'] for r in manifest.train]
        by_id, named = fit.cell_weight_table(rows, ids)
        assert len(named) == 4 and abs(sum(by_id.values()) / len(ids) - 1) < 1e-9
        assert fit.cell_weight_table(rows, ids + [r['sample_id'] for r in manifest.test])[1] != named      # the test split is not in them
        for cell, w in named.items():
            n_cell = sum(fit.cell_name(rows[i]['label'], rows[i]['arch']) == cell for i in ids)
            assert w == pytest.approx(len(ids) / (4 * n_cell))                                              # N / (C n_cell)
        if arch == 'matched':
            assert all(str(r['arch_matched']) == '1' for r in manifest.train + manifest.val + manifest.test)


# ---- the (protocol, arch) tables name the same datasets ----------------------------------------------------------------------

def pairs_of(table):
    """{protocol: (arch, ...)} as the flat list of (protocol, arch)."""
    return [(protocol, arch) for protocol, archs in table.items() for arch in archs]


def test_the_protocol_and_arch_tables_of_the_modules_name_the_same_datasets():
    built, run = pairs_of(tp.BUILD_ARCHES), pairs_of(train_b.RUN_ARCHES)
    assert list(FIVECHANNEL_DATASETS) == run                                                   # config's pairs are the trainers' run pairs, in order
    assert set(run) - set(built) == {('kfold_v3', 'matched')} and set(built) < set(run)          # a run adds `matched`, a filter of the all build
    assert set(pb.DATASETS) == set(built)                                                      # every built dataset is packaged, no other
    assert set(fc.FILL_BUILD_OF) == set(built)                                                 # fill_build maps `matched` to `all` before the lookup
    assert set(tp.DATASET_OF) == set(tp.DEFAULT_ARCH) == set(tp.BUILD_ARCHES) == set(train_b.RUN_ARCHES)
    assert all(tp.DEFAULT_ARCH[protocol] in archs for protocol, archs in tp.BUILD_ARCHES.items())
    assert set(gate_tables.BUILD_TOTAL) == set(run)                                            # one total per scope a stage can name
    assert gate_tables.BUILD_TOTAL['kfold_v3', 'matched'] == gate_tables.BUILD_TOTAL['kfold_v3', 'all']      # a matched run reads the all build
    assert set(report.CELL_SCOPES) == {pair for pair in run if pair[1] != 'x86'}                 # the datasets whose test sets hold both architectures
    for pair in built:
        assert fc.fill_build(*pair) is fc.FILL_BUILD_OF[pair]
    assert fc.fill_build('kfold_v3', 'matched') is fc.FILL_BUILD_OF['kfold_v3', 'all']


@real_data
@pytest.mark.parametrize('pair', [pytest.param(('kfold', 'x86'), marks=needs_v2), pytest.param(('kfold', 'all'), marks=needs_v2),
                                  pytest.param(('kfold_v3', 'all'), marks=needs_combined),
                                  pytest.param(('kfold_v3', 'matched'), marks=needs_combined)])
def test_real_build_total_is_the_number_of_ids_of_the_dataset(pair, monkeypatch):
    monkeypatch.setattr(manifests, 'CACHE', Path(DEFAULT_CACHE if pair[0] == 'kfold' else COMBINED_CACHE))      # each dataset on its own cache
    ids = tp.all_ids(*pair)
    if pair == ('kfold_v3', 'matched'):                                       # the build is the all build; the matched ids are a subset of it
        assert gate_tables.BUILD_TOTAL[pair] == len(tp.all_ids('kfold_v3', 'all')) > len(ids) == 4850
    else:
        assert gate_tables.BUILD_TOTAL[pair] == len(ids)
