"""python -m pytest tests/test_predict.py: `experiments.predict`, the command that scores binaries with the weights of a saved run.

A module fixture trains a 2-epoch run of every kind on the synthetic cache and builds of the other tests (`tiny_cache.build`,
`test_team_v3.synthetic_build`): Model 1 (L32 and the per-block `-pb`), the arms `sig500v3` (all), `-feat`, `-emb`, `-e2e` and
`-m1frozen`, and on the second dataset an `-ab` run of kfold_v3 matched. The tests then check that rescoring a run's own binaries
reproduces its saved predictions (both checkpoints) to 1e-6, that the table has the saved columns, that no file of the run folder is
touched, the `pred` thresholds, scoring by id and by split, and every refusal. CPU only; the tests that train need torch_geometric and
skip with a reason without it."""
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from common import manifests
from common.config import RunConfig
from experiments import predict
from experiments.team_pipelines import common as tp, fit, train_b
from s03a_transformer.five_channel import train as fc
from test_five_channel import fivechannel
from test_team_pipelines import NO_PYG, clear_team_caches, fresh_team_caches  # noqa: F401
from test_team_v3 import synthetic_build
from tiny_cache import build as build_cache

ROOT = Path(__file__).resolve().parents[1]
quiet = lambda *_: None                                                                   # noqa: E731
ARMS_X86 = ('sig500v3', 'sig500v3-feat', 'sig500v3-emb', 'sig500v3-e2e')
PB = 'kfold0_x86_fivechannel-L64-pb_s0'
RUNS_X86 = {'model1': 'kfold0_x86_fivechannel-L32_s0', 'model1-pb': PB,
            **{arm: f'kfold0_x86_gnn_s0_team-{arm}' for arm in ARMS_X86},
            'm1frozen': 'kfold0_x86_gnn_s0_team-sig500v3-m1frozen'}
V3_NAME = 'v3kfold2_matched_gnn-ab_s1_team-sig500v3-feat'
EXACT = 1e-6                    # the saved probabilities have 6 decimals


def snapshot(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(folder).iterdir())}


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    """Trained runs: {'runs': folder of the x86 runs, 'build': their build, 'v3_runs', 'v3_build', 'tmp'}."""
    if NO_PYG is not None:
        pytest.skip(NO_PYG)
    root = tmp_path_factory.mktemp('predict')
    out = {'tmp': root}
    with pytest.MonkeyPatch.context() as mp:
        for arm in (*ARMS_X86, 'sig500v3-m1frozen'):
            mp.setitem(train_b.ARM_HP, arm, {**train_b.ARM_HP[arm], 'epochs': 2})
        # kfold x86
        info = build_cache(root / 'cache86')
        mp.setattr(manifests, 'CACHE', root / 'cache86')
        clear_team_caches()
        rows = tp.sample_rows()
        folder = synthetic_build(root / 'b86', sorted(rows), rows)
        runs = root / 'runs86'
        mp.setattr(fit, 'RUNS', runs)
        fc.run(fivechannel(0, max_instructions=32), folder, 'cpu', None, runs, log=quiet)
        fc.run(fivechannel(0, max_instructions=64, per_block=True), folder, 'cpu', None, runs, log=quiet)
        for arm in ARMS_X86:
            train_b.run(arm, 0, 0, device='cpu', b_dir=folder, log=quiet)
        train_b.run('sig500v3-m1frozen', 0, 0, device='cpu', b_dir=folder, log=quiet, model1_run=runs / PB)
        out.update(runs=runs, build=folder, info=info)
        # kfold_v3 matched, architecture balance
        build_cache(root / 'cache_v3', v3=True)
        mp.setattr(manifests, 'CACHE', root / 'cache_v3')
        clear_team_caches()
        rows3 = tp.sample_rows('kfold_v3', 'all')
        folder3 = synthetic_build(root / 'b_v3', sorted(rows3), rows3, dataset=('kfold_v3', 'all'))
        runs3 = root / 'runs_v3'
        mp.setattr(fit, 'RUNS', runs3)
        train_b.run('sig500v3-feat', 2, 1, device='cpu', b_dir=folder3, log=quiet, protocol='kfold_v3', arch='matched', arch_balance=True)
        out.update(v3_runs=runs3, v3_build=folder3)
    clear_team_caches()
    return type('World', (), out)


def run_of(world, kind):
    return world.v3_runs / V3_NAME if kind == 'v3-matched-ab' else world.runs / RUNS_X86[kind]


def build_of(world, kind):
    return world.v3_build if kind == 'v3-matched-ab' else world.build


KINDS = [*RUNS_X86, 'v3-matched-ab']


def go(world, kind, tmp_path, **kw):
    """predict.predict on a kind of run with its build; (result, log lines)."""
    lines = []
    kw.setdefault('build', build_of(world, kind))
    kw.setdefault('out', tmp_path / f'{kind}.csv')
    if kind == 'm1frozen':
        kw.setdefault('model1_run', world.runs / PB)
    result = predict.predict(run_of(world, kind), log=lines.append, **kw)
    return result, lines


def read(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


# ---- reproduction -------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('checkpoint, saved', [('auc', 'predictions_auc.csv'), ('f1', 'predictions.csv')])
@pytest.mark.parametrize('kind', KINDS)
def test_rescoring_the_runs_own_binaries_reproduces_its_saved_predictions_to_1e_6(world, tmp_path, kind, checkpoint, saved):
    folder = run_of(world, kind)
    before = snapshot(folder)
    result, lines = go(world, kind, tmp_path, checkpoint=checkpoint, check=True)
    old, new = read(folder / saved), read(result['out'])
    assert [r['sample_id'] for r in new] == [r['sample_id'] for r in old] and len(old) > 5
    assert list(new[0]) == list(old[0])                                                       # the columns of the saved predictions
    for a, b in zip(old, new):
        assert (a['split'], a['label'], a['arch'], a['family'], a['fold']) == (b['split'], b['label'], b['arch'], b['family'], b['fold'])
        assert abs(float(a['prob']) - float(b['prob'])) <= EXACT
        assert (a['n_segments_total'], a['n_blocks_kept'], a['n_blocks_total']) == (b['n_segments_total'], b['n_blocks_kept'], b['n_blocks_total'])
    check = result['check']
    assert check['max_dprob'] <= EXACT and not check['problems'] and check['pred_agreement'] == 1.0
    assert snapshot(folder) == before                                                          # the run folder is never written to
    assert any('CHECK PASSED' in l for l in lines)


@pytest.mark.parametrize('kind', ['model1', 'sig500v3-e2e', 'v3-matched-ab'])
def test_the_pred_column_uses_the_threshold_of_the_checkpoint_and_says_which(world, tmp_path, kind):
    folder = run_of(world, kind)
    metrics = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
    cfg = RunConfig.load(folder / 'config.json')
    for checkpoint, saved, expect in (('auc', 'predictions_auc.csv', metrics['selection']['auc']['threshold_val']),
                                      ('f1', 'predictions.csv', cfg.threshold)):
        result, lines = go(world, kind, tmp_path, checkpoint=checkpoint, check=True)
        assert result['threshold'] == expect
        for r in read(result['out']):
            assert int(r['pred']) == int(float(r['prob']) >= expect)
        assert any(l.startswith('threshold for pred:') and (str(expect) in l) for l in lines)
    val = metrics['selection']['f1']['threshold_val']
    result, lines = go(world, kind, tmp_path, checkpoint='f1', threshold='val')
    assert result['threshold'] == val and any('selection.f1.threshold_val' in l for l in lines)
    assert go(world, kind, tmp_path, threshold='0.3')[0]['threshold'] == 0.3


# ---- which binaries, where the table goes ---------------------------------------------------------------------------------------

def test_a_split_and_ids_score_the_same_binaries_as_the_whole_table(world, tmp_path):
    kind = 'sig500v3-emb'
    everything = go(world, kind, tmp_path, split='all')[0]
    folder = run_of(world, kind)
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    test = go(world, kind, tmp_path, split='test')[0]
    assert test['ids'] == manifest['test'] and everything['ids'] == manifest['val'] + manifest['test']
    pick = [manifest['test'][3], manifest['val'][1], manifest['train'][0]]
    result, lines = go(world, kind, tmp_path, ids=pick)
    assert result['ids'] == pick
    by_id = dict(zip(everything['ids'], everything['probs']))
    for i, p in zip(pick[:2], result['probs'][:2]):
        assert abs(by_id[i] - p) <= 1e-6                                                     # batch composition does not matter
    rows = {r['sample_id']: r for r in read(result['out'])}
    assert [rows[i]['split'] for i in pick] == ['test', 'val', 'train']                      # the split the run's manifest gives
    assert all(rows[i]['label'] in ('0', '1') for i in pick)


def test_default_output_is_a_file_named_after_the_run_and_checkpoint_in_the_current_directory(world, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    kind = 'sig500v3-feat'
    result = predict.predict(run_of(world, kind), 'f1', build=build_of(world, kind), split='test', log=quiet)
    assert result['out'] == tmp_path / f'{RUNS_X86[kind]}_f1_scores.csv' and result['out'].is_file()
    with pytest.raises(SystemExit, match='inside the run folder'):
        predict.predict(run_of(world, kind), build=build_of(world, kind), out=run_of(world, kind) / 'x.csv', log=quiet)
    with pytest.raises(SystemExit, match='inside the run folder'):
        predict.predict(run_of(world, kind), build=build_of(world, kind), out=run_of(world, kind) / 'sub' / 'x.csv', log=quiet)
    assert not (run_of(world, kind) / 'x.csv').exists()


def test_the_command_line_scores_and_checks(world, tmp_path, capsys):
    kind = 'sig500v3-feat'
    out = tmp_path / 'cli.csv'
    predict.main(['--run', str(run_of(world, kind)), '--build', str(build_of(world, kind)), '--split', 'test', '--out', str(out)])
    text = capsys.readouterr().out
    assert 'threshold for pred:' in text and 'AUC' in text and out.is_file()
    predict.main(['--run', str(run_of(world, kind)), '--build', str(build_of(world, kind)), '--check', '--out', str(out)])
    assert 'CHECK PASSED' in capsys.readouterr().out
    with pytest.raises(SystemExit, match='takes no --ids'):
        predict.main(['--run', str(run_of(world, kind)), '--check', '--ids', 'a'])


def test_a_failed_check_exits_with_the_reason(world, tmp_path):
    kind = 'sig500v3-feat'
    copy = tmp_path / 'copy'
    shutil.copytree(run_of(world, kind), copy)
    lines = read(copy / 'predictions_auc.csv')
    lines[0]['prob'] = f'{1 - float(lines[0]["prob"]):.6f}'                                     # one saved probability that is not the model's
    with (copy / 'predictions_auc.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(lines[0]))
        w.writeheader()
        w.writerows(lines)
    with pytest.raises(SystemExit, match=r'CHECK FAILED: max \|dprob\|'):
        predict.predict(copy, build=build_of(world, kind), check=True, out=tmp_path / 'o.csv', log=quiet)
    with pytest.raises(SystemExit):
        predict.main(['--run', str(copy), '--build', str(build_of(world, kind)), '--check', '--out', str(tmp_path / 'o.csv')])


def test_check_prints_the_auc_difference_and_allows_1e_3(world, tmp_path):
    """|dAUC| = |now - saved| <= 1e-3 (an explicit bound, not 3-decimal rounding); the difference is printed for every run."""
    kind = 'sig500v3-feat'
    assert predict.AUC_TOLERANCE == 1e-3
    for shift, passes in ((0.0, True), (9e-4, True), (-9e-4, True), (1.2e-3, False), (-1.2e-3, False)):
        copy = tmp_path / f'auc{shift}'
        shutil.copytree(run_of(world, kind), copy)
        metrics = json.loads((copy / 'metrics.json').read_text(encoding='utf-8'))
        metrics['selection']['auc']['test']['auc'] += shift
        (copy / 'metrics.json').write_text(json.dumps(metrics), encoding='utf-8')
        lines = []
        if passes:
            result = predict.predict(copy, build=build_of(world, kind), check=True, out=tmp_path / 'o.csv', log=lines.append)
            assert abs(result['check']['dauc'] + shift) < 1e-9 and not result['check']['problems']      # now - saved
        else:
            with pytest.raises(SystemExit, match=r'CHECK FAILED: test AUC .* differs from the saved .* by .*\(> 0.001\)'):
                predict.predict(copy, build=build_of(world, kind), check=True, out=tmp_path / 'o.csv', log=lines.append)
        assert any(re.search(r'dAUC \(now - saved\) [+-]\d\.\d\de[+-]\d\d \(tolerance 0.001\)', l) for l in lines)


@pytest.mark.parametrize('kind', KINDS)
def test_every_check_prints_dauc(world, tmp_path, kind):
    result, lines = go(world, kind, tmp_path, check=True)
    if 'auc_now' in result['check']:
        assert abs(result['check']['dauc']) <= predict.AUC_TOLERANCE
        assert sum('dAUC (now - saved)' in l for l in lines) == 1


# ---- refusals ------------------------------------------------------------------------------------------------------------------

def test_a_folder_that_is_not_a_finished_run_with_weights_is_refused(world, tmp_path):
    kind = 'sig500v3-feat'
    src = run_of(world, kind)
    for gone, said in (('best_auc.pt', 'has no best_auc.pt'), ('metrics.json', 'no metrics.json'), ('config.json', 'no config.json'),
                       ('manifest.json', 'no manifest.json')):
        copy = tmp_path / f'no-{gone}'
        shutil.copytree(src, copy)
        (copy / gone).unlink()
        with pytest.raises(SystemExit, match=said):
            predict.predict(copy, build=build_of(world, kind), out=tmp_path / 'o.csv', log=quiet)
    copy = tmp_path / 'no-f1'
    shutil.copytree(src, copy)
    (copy / 'best.pt').unlink()
    with pytest.raises(SystemExit, match='has no best.pt'):
        predict.predict(copy, 'f1', build=build_of(world, kind), out=tmp_path / 'o.csv', log=quiet)
    with pytest.raises(SystemExit, match='not a directory'):
        predict.predict(tmp_path / 'nothing', log=quiet)


def test_a_run_of_another_model_is_refused(world, tmp_path):
    src = run_of(world, 'sig500v3-feat')
    other = tmp_path / 'gnn'
    shutil.copytree(src, other)
    config = json.loads((other / 'config.json').read_text(encoding='utf-8'))
    (other / 'config.json').write_text(json.dumps({**config, 'tag': ''}), encoding='utf-8')
    with pytest.raises(SystemExit, match='not scored by this command'):
        predict.predict(other, build=world.build, out=tmp_path / 'o.csv', log=quiet)
    (other / 'config.json').write_text(json.dumps({**config, 'tag': 'team-cb500'}), encoding='utf-8')
    with pytest.raises(SystemExit, match="arm 'cb500' is not one of train_b.ARMS"):
        predict.predict(other, build=world.build, out=tmp_path / 'o.csv', log=quiet)


@pytest.mark.parametrize('kind', ['sig500v3-feat', 'sig500v3-e2e', 'model1'])
def test_a_build_of_another_dataset_sampler_or_budget_is_refused_naming_both_sides(world, tmp_path, kind):
    def variant(name, **changes):
        copy = tmp_path / name
        shutil.copytree(build_of(world, kind), copy)
        settings = json.loads((copy / 'settings.json').read_text(encoding='utf-8'))
        (copy / 'settings.json').write_text(json.dumps({**settings, **changes}), encoding='utf-8')
        return copy
    dataset = variant('other-dataset', protocol='kfold_v3', arch='all')
    with pytest.raises(SystemExit, match='was built for kfold_v3 all, this run asks for kfold x86'):
        go(world, kind, tmp_path, build=dataset, check=True)
    if kind == 'model1':                                                                      # no arm: the sampler and budget of the run's own build
        with pytest.raises(SystemExit, match="was built with sample walk; .* with signal"):
            go(world, kind, tmp_path, build=variant('walk', sample='walk'), check=True)
        with pytest.raises(SystemExit, match='was built with num_blocks 250; .* with 500'):
            go(world, kind, tmp_path, build=variant('b250', num_blocks=250), check=True)
    else:
        with pytest.raises(SystemExit, match='sampler walk'):
            go(world, kind, tmp_path, build=variant('walk', sample='walk'), check=True)
        with pytest.raises(SystemExit, match='num_blocks 250'):
            go(world, kind, tmp_path, build=variant('b250', num_blocks=250), check=True)
        with pytest.raises(SystemExit, match='source'):
            go(world, kind, tmp_path, build=variant('src', source='gnn-331ce0b'), check=True)


def test_a_build_with_another_api_numbering_or_fewer_ids_than_the_checkpoint_is_refused(world, tmp_path):
    kind = 'sig500v3-e2e'
    names = json.loads((world.build / 'api_vocab.json').read_text(encoding='utf-8'))
    shuffled = tmp_path / 'shuffled'
    shutil.copytree(world.build, shuffled)
    (shuffled / 'api_vocab.json').write_text(json.dumps(names[::-1]), encoding='utf-8')
    with pytest.raises(SystemExit, match='numbers the API names differently'):
        go(world, kind, tmp_path, build=shuffled, check=True)
    short = tmp_path / 'short'
    shutil.copytree(world.build, short)
    (short / 'api_vocab.json').write_text(json.dumps(names[:2]), encoding='utf-8')
    with pytest.raises(SystemExit, match='embeds .* ids'):
        go(world, kind, tmp_path, build=short, check=True)
    extended = tmp_path / 'extended'                                                          # the run's build extended is accepted
    shutil.copytree(world.build, extended)
    (extended / 'api_vocab.json').write_text(json.dumps(names + ['lib.dll!More']), encoding='utf-8')
    assert not go(world, 'model1', tmp_path, build=extended, check=True)[0]['check']['problems']
    with pytest.raises(SystemExit, match='numbers the API names differently'):
        go(world, 'model1', tmp_path, build=shuffled, check=True)


def test_without_the_runs_own_build_a_foreign_build_is_found_out_by_rescoring_the_runs_own_binaries(world, tmp_path):
    kind = 'sig500v3-emb'
    copy = tmp_path / 'run'
    shutil.copytree(run_of(world, kind), copy)
    metrics = json.loads((copy / 'metrics.json').read_text(encoding='utf-8'))
    metrics['build_dir'] = str(tmp_path / 'gone')                                              # the machine of a teammate
    (copy / 'metrics.json').write_text(json.dumps(metrics), encoding='utf-8')
    lines = []
    predict.predict(copy, build=world.build, out=tmp_path / 'o.csv', log=lines.append)
    assert any(l.startswith('build identity:') for l in lines)
    foreign = tmp_path / 'foreign'                                                            # the same ids, other payloads
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(manifests, 'CACHE', world.tmp / 'cache86')
        clear_team_caches()
        rows = tp.sample_rows()
        synthetic_build(foreign, sorted(rows), rows, seed=5)
        clear_team_caches()
    with pytest.raises(SystemExit, match='not the same build|does not give the run'):
        predict.predict(copy, build=foreign, out=tmp_path / 'o.csv', log=quiet)


def test_ids_the_build_does_not_hold_are_refused_not_skipped(world, tmp_path):
    kind = 'sig500v3-feat'
    folder = run_of(world, kind)
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    with pytest.raises(SystemExit, match=r'1 of the 2 binaries asked for are not in the build .*e.g. \[\'f{64}\'\]'):
        go(world, kind, tmp_path, ids=[manifest['test'][0], 'f' * 64])
    thin = tmp_path / 'thin'                                                                   # a build that lacks some of the run's test binaries
    shutil.copytree(world.build, thin)
    (thin / f'{manifest["test"][0]}.pt').unlink()
    checkpoint = json.loads((thin / 'checkpoint.json').read_text(encoding='utf-8'))
    checkpoint['done'] = [i for i in checkpoint['done'] if i != manifest['test'][0]]
    (thin / 'checkpoint.json').write_text(json.dumps(checkpoint), encoding='utf-8')
    with pytest.raises(SystemExit, match='are not in the build'):
        go(world, kind, tmp_path, build=thin, split='test')
    model1 = 'model1'
    thin1 = tmp_path / 'thin1'
    shutil.copytree(world.build, thin1)
    (thin1 / f'{manifest["test"][0]}.pt').unlink()
    (thin1 / 'checkpoint.json').write_text(json.dumps(checkpoint), encoding='utf-8')
    with pytest.raises(SystemExit):
        go(world, model1, tmp_path, build=thin1, split='test')
    with pytest.raises(SystemExit, match='are not in the build'):
        go(world, model1, tmp_path, build=thin1, ids=[manifest['test'][0]])


def test_a_frozen_run_needs_its_model1_folder_and_the_same_vocabulary(world, tmp_path):
    kind = 'm1frozen'
    with pytest.raises(SystemExit, match='is a -m1frozen run: .*that folder is missing'):
        predict.predict(run_of(world, kind), build=world.build, out=tmp_path / 'o.csv', model1_run=tmp_path / 'absent', log=quiet)
    copy = tmp_path / 'run'
    shutil.copytree(run_of(world, kind), copy)
    metrics = json.loads((copy / 'metrics.json').read_text(encoding='utf-8'))
    metrics['model1_run'] = str(tmp_path / 'not-here-either')
    (copy / 'metrics.json').write_text(json.dumps(metrics), encoding='utf-8')
    with pytest.raises(SystemExit, match='is a -m1frozen run: .*that folder is missing'):
        predict.predict(copy, build=world.build, out=tmp_path / 'o.csv', log=quiet)
    # the recorded folder is found when it exists, a Model 1 folder of another fold is refused by the trainer's own check
    metrics['model1_run'] = str(world.runs / PB)
    (copy / 'metrics.json').write_text(json.dumps(metrics), encoding='utf-8')
    assert not predict.predict(copy, build=world.build, check=True, out=tmp_path / 'o.csv', log=quiet)['check']['problems']
    # a Model 1 that was trained on a build whose vocabulary differs
    pb = tmp_path / 'pb'
    shutil.copytree(world.runs / PB, pb)
    other = tmp_path / 'other-build'
    shutil.copytree(world.build, other)
    names = json.loads((other / 'api_vocab.json').read_text(encoding='utf-8'))
    (other / 'api_vocab.json').write_text(json.dumps(names[::-1]), encoding='utf-8')
    theirs = json.loads((pb / 'metrics.json').read_text(encoding='utf-8'))
    theirs['build_dir'] = str(other)
    (pb / 'metrics.json').write_text(json.dumps(theirs), encoding='utf-8')
    with pytest.raises(SystemExit, match=r'trained on the build .*other-build.*API ids are numbered per build'):             # the trainer's guard
        predict.predict(run_of(world, kind), build=world.build, out=tmp_path / 'o.csv', model1_run=pb, log=quiet)


def frozen_copy(world, tmp_path, name, run_metrics=None, model1_metrics=None):
    """(a copy of the -m1frozen run, a copy of its per-block Model 1), their metrics.json updated by the dicts (None deletes a key)."""
    out = []
    for source, extra, label in ((run_of(world, 'm1frozen'), run_metrics, 'run'), (world.runs / PB, model1_metrics, 'pb')):
        copy = tmp_path / f'{name}-{label}'
        shutil.copytree(source, copy)
        metrics = json.loads((copy / 'metrics.json').read_text(encoding='utf-8'))
        for key, value in (extra or {}).items():
            metrics.pop(key, None) if value is None else metrics.__setitem__(key, value)
        (copy / 'metrics.json').write_text(json.dumps(metrics), encoding='utf-8')
        out.append(copy)
    return out


def test_a_frozen_run_whose_model1_build_is_not_on_this_machine_uses_the_scored_build_it_shares_the_vocabulary_of(world, tmp_path, monkeypatch):
    """A teammate has the run, its Model 1 and the build, but the Model 1's metrics.json names a path of the training machine."""
    monkeypatch.setattr(train_b, 'TEAM_RUNS', tmp_path / 'no-team')
    gone = str(tmp_path / 'trained-elsewhere' / 'b_sig500_v3')
    run, pb = frozen_copy(world, tmp_path, 'gone', model1_metrics={'build_dir': gone})
    assert json.loads((run / 'metrics.json').read_text(encoding='utf-8')).get('model1_build')        # the run passed the guard when it trained
    result = predict.predict(run, build=world.build, check=True, model1_run=pb, out=tmp_path / 'o.csv', log=quiet)
    assert not result['check']['problems'] and result['check']['max_dprob'] <= EXACT
    # a run that did not record the Model 1 build it passed the guard with was never guarded: the guard's refusal, with the fix
    bare_run, bare_pb = frozen_copy(world, tmp_path, 'bare', run_metrics={'model1_build': None}, model1_metrics={'build_dir': gone})
    with pytest.raises(SystemExit, match=r'Model 1 build .*trained-elsewhere.*--model1-build DIR'):
        predict.predict(bare_run, build=world.build, check=True, model1_run=bare_pb, out=tmp_path / 'o.csv', log=quiet)
    # --model1-build names the Model 1's build explicitly, and a wrong one is the guard's refusal
    assert not predict.predict(bare_run, build=world.build, check=True, model1_run=bare_pb, model1_build=world.build,
                               out=tmp_path / 'o.csv', log=quiet)['check']['problems']
    other = tmp_path / 'other-vocabulary'
    shutil.copytree(world.build, other)
    names = json.loads((other / 'api_vocab.json').read_text(encoding='utf-8'))
    (other / 'api_vocab.json').write_text(json.dumps(names[::-1]), encoding='utf-8')
    with pytest.raises(SystemExit, match='API ids are numbered per build'):
        predict.predict(bare_run, build=world.build, check=True, model1_run=bare_pb, model1_build=other, out=tmp_path / 'o.csv', log=quiet)
    predict.main(['--run', str(bare_run), '--build', str(world.build), '--check', '--model1-run', str(bare_pb), '--model1-build',
                  str(world.build), '--out', str(tmp_path / 'cli.csv')])


def test_with_no_build_of_the_run_on_this_machine_the_rescoring_finds_a_scored_build_of_another_vocabulary(world, tmp_path, monkeypatch):
    """A teammate's machine: the run's `build_dir` and its Model 1's are both paths of the training machine. The scored build then
    stands in for the Model 1's, so the guard cannot tell the vocabulary; `--check` (the rescoring of the run's own binaries) does."""
    monkeypatch.setattr(train_b, 'TEAM_RUNS', tmp_path / 'no-team')
    gone = str(tmp_path / 'trained-elsewhere' / 'b_sig500_v3')
    run, pb = frozen_copy(world, tmp_path, 'noref', run_metrics={'build_dir': gone}, model1_metrics={'build_dir': gone})
    assert predict.reference_build(predict.load_run(run)) is None
    result = predict.predict(run, build=world.build, check=True, model1_run=pb, out=tmp_path / 'o.csv', log=quiet)
    assert not result['check']['problems'] and result['check']['max_dprob'] <= EXACT
    foreign = tmp_path / 'foreign'                                                            # the same ids, other payloads (as in the test above)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(manifests, 'CACHE', world.tmp / 'cache86')
        clear_team_caches()
        rows = tp.sample_rows()
        synthetic_build(foreign, sorted(rows), rows, seed=5)
        clear_team_caches()
    for check in (True, False):                                                                # the probe, and --check
        with pytest.raises(SystemExit, match='not the same build|does not give the run|CHECK FAILED'):
            predict.predict(run, build=foreign, check=check, model1_run=pb, out=tmp_path / 'o.csv', log=quiet)


def test_a_stand_in_build_is_probed_even_when_the_runs_build_is_here_and_a_wrong_model1_is_refused(world, tmp_path, monkeypatch):
    """The guard compares vocabularies, not weights: with the Model 1's build stood in by the scored one, a Model 1 whose API
    embedding rows are permuted passes it. Plain scoring (no --check) must still refuse it, before it scores anything."""
    from s02_preprocess.tokenize_asm_gz import FIRST_API_ID
    monkeypatch.setattr(train_b, 'TEAM_RUNS', tmp_path / 'no-team')
    gone = str(tmp_path / 'trained-elsewhere' / 'b_sig500_v3')
    run, pb = frozen_copy(world, tmp_path, 'wrong', model1_metrics={'build_dir': gone})
    assert predict.reference_build(predict.load_run(run)) is not None                         # the run's build is on this machine
    lines = []
    result = predict.predict(run, build=world.build, model1_run=pb, out=tmp_path / 'ok.csv', log=lines.append)    # the right Model 1
    assert any(l.startswith('build identity:') for l in lines)
    assert float(np.abs(result['probs'] - np.array([float(r['prob']) for r in predict.saved_rows(predict.load_run(run))])).max()) <= EXACT
    state = torch.load(pb / 'best_auc.pt', map_location='cpu')
    key = 'embedder.api_embedding.weight'
    state[key] = torch.cat([state[key][:FIRST_API_ID], state[key][FIRST_API_ID:].flip(0)])
    assert not torch.equal(state[key], torch.load(world.runs / PB / 'best_auc.pt', map_location='cpu')[key])
    torch.save(state, pb / 'best_auc.pt')
    out = tmp_path / 'wrong.csv'
    for kw in ({}, {'ids': predict.load_run(run).ids('val')[:2]}):
        with pytest.raises(SystemExit, match=r'stands in for the build of the frozen Model 1.*max \|dprob\|'):
            predict.predict(run, build=world.build, model1_run=pb, out=out, log=quiet, **kw)
    assert not out.exists()
    with pytest.raises(SystemExit, match='stands in'):                                          # --check is refused by the probe as well
        predict.predict(run, build=world.build, model1_run=pb, check=True, out=out, log=quiet)
    # an explicit --model1-build is the user's statement, not the stand-in: the probe is not forced on it
    lines.clear()
    predict.predict(run, build=world.build, model1_run=world.runs / PB, model1_build=world.build, out=tmp_path / 'e.csv', log=lines.append)
    assert not any(l.startswith('build identity:') for l in lines)


def test_a_frozen_run_finds_its_model1_build_in_runs_team_before_it_falls_back_to_the_scored_build(world, tmp_path, monkeypatch):
    team = tmp_path / 'team'
    shutil.copytree(world.build, team / 'b_sig500_v3')
    monkeypatch.setattr(train_b, 'TEAM_RUNS', team)
    run, pb = frozen_copy(world, tmp_path, 'team', model1_metrics={'build_dir': str(tmp_path / 'trained-elsewhere' / 'b_sig500_v3')})
    scorer = predict.make_scorer(predict.load_run(run), world.build, torch.device('cpu'), pb, quiet)
    assert scorer.model1_build(pb, None) is None                                           # the guard finds it by itself
    (team / 'b_sig500_v3' / 'api_vocab.json').write_text(json.dumps(['x.dll!A']), encoding='utf-8')      # and compares it with the scored build
    with pytest.raises(SystemExit, match='trained on the build'):
        predict.predict(run, build=world.build, check=True, model1_run=pb, out=tmp_path / 'o.csv', log=quiet)


def test_the_scored_build_stands_in_for_the_model1_build_only_with_the_vocabulary_of_the_runs_own_build(world, tmp_path, monkeypatch):
    """A build that extends the run's (allowed for scoring) is not the run's: the Model 1's missing build is then the guard's refusal."""
    monkeypatch.setattr(train_b, 'TEAM_RUNS', tmp_path / 'no-team')
    run, pb = frozen_copy(world, tmp_path, 'ext', model1_metrics={'build_dir': str(tmp_path / 'trained-elsewhere' / 'b')})
    extended = tmp_path / 'extended'
    shutil.copytree(world.build, extended)
    names = json.loads((extended / 'api_vocab.json').read_text(encoding='utf-8'))
    (extended / 'api_vocab.json').write_text(json.dumps(names + ['extra.dll!Z']), encoding='utf-8')
    scorer = predict.make_scorer(predict.load_run(run), world.build, torch.device('cpu'), pb, quiet)
    assert scorer.model1_build(pb, None) == scorer.build                                   # the run's own build
    scorer.build = extended
    assert scorer.model1_build(pb, None) is None                                           # so the guard refuses the missing build
    with pytest.raises(SystemExit, match='Model 1 build .*trained-elsewhere.*--model1-build DIR'):
        train_b.locate_model1_build(pb)


def test_the_trainers_still_build_what_predict_loads(world):
    """The network `train_b.new_model` builds is the one a run of each arm saved (the strict load_state_dict of predict is the
    guard; this names the shapes)."""
    for arm in ARMS_X86:
        state = torch.load(world.runs / RUNS_X86[arm] / 'best_auc.pt')
        metrics = json.loads((world.runs / RUNS_X86[arm] / 'metrics.json').read_text(encoding='utf-8'))
        rows = int(state['encoder.embedder.api_embedding.weight'].shape[0]) if arm.endswith('e2e') else None
        model = train_b.new_model(train_b.ARM_FEATURES[arm], metrics['in_dim'], metrics['edge_dim'], rows)
        model.load_state_dict(state)


# ---- numerics ------------------------------------------------------------------------------------------------------------------

def test_the_bf16_emulation_switches_the_trainers_autocast_on_and_restores_everything(world, tmp_path):
    from s03a_transformer.five_channel import train as fc_train
    from s03c_cfg_transformer import block_transformer as bt
    fast = torch.backends.mha.get_fastpath_enabled()
    assert not fc_train.AUTOCAST_ON_CPU and not bt.AUTOCAST_ON_CPU
    with predict.numerics(True):
        assert fc_train.AUTOCAST_ON_CPU and bt.AUTOCAST_ON_CPU and not torch.backends.mha.get_fastpath_enabled()
    with predict.numerics(False):
        assert not fc_train.AUTOCAST_ON_CPU
    assert not fc_train.AUTOCAST_ON_CPU and not bt.AUTOCAST_ON_CPU and torch.backends.mha.get_fastpath_enabled() == fast
    cpu = torch.device('cpu')
    for kind in ('model1', 'sig500v3-e2e', 'sig500v3-feat'):                                    # trained on cpu: fp32 by default
        run = predict.load_run(run_of(world, kind))
        assert not predict.bf16_emulated(run, cpu, 'auto') and not predict.trained_in_bf16(run)
    assert predict.bf16_emulated(predict.load_run(run_of(world, 'model1')), cpu, 'bf16')                  # forced
    assert not predict.bf16_emulated(predict.load_run(run_of(world, 'sig500v3-feat')), cpu, 'bf16')      # no autocast part
    assert not predict.bf16_emulated(predict.load_run(run_of(world, 'model1')), cpu, 'off')
    with pytest.raises(SystemExit, match='--autocast nope'):
        predict.bf16_emulated(predict.load_run(run_of(world, 'model1')), cpu, 'nope')
    # a run that says it was trained on cuda is rescored in emulated bf16 by default and with the looser tolerance
    for kind in ('model1', 'sig500v3-e2e'):
        copy = tmp_path / kind
        shutil.copytree(run_of(world, kind), copy)
        metrics = json.loads((copy / 'metrics.json').read_text(encoding='utf-8'))
        (copy / 'metrics.json').write_text(json.dumps({**metrics, 'device': 'cuda'}), encoding='utf-8')
        run = predict.load_run(copy)
        assert predict.trained_in_bf16(run) and predict.bf16_emulated(run, cpu, 'auto') and not predict.bf16_emulated(run, cpu, 'off')
        assert predict.default_tolerance(run, True) == predict.TOLERANCE_BF16 < predict.default_tolerance(run, False)
        lines = []
        out = predict.predict(copy, build=world.build, out=tmp_path / f'{kind}.csv', log=lines.append, autocast='bf16')
        assert any(l.startswith('numerics: bf16 autocast on the cpu') for l in lines)
        assert not fc_train.AUTOCAST_ON_CPU and not bt.AUTOCAST_ON_CPU
        old = {r['sample_id']: float(r['prob']) for r in read(copy / 'predictions_auc.csv')}
        assert max(abs(old[i] - p) for i, p in zip(out['ids'], out['probs'])) < 0.05             # bf16 rounding, not another model
    assert predict.default_tolerance(predict.load_run(run_of(world, 'sig500v3-feat')), False) == predict.TOLERANCE_FP32
    assert (predict.TOLERANCE_FP32, predict.TOLERANCE_BF16, predict.TOLERANCE_FP32_OF_BF16) == (1e-4, 5e-3, 6e-2)   # measured worst 5.2e-2
    assert predict.default_tolerance(predict.load_run(copy), False) == 6e-2


# ---- real data (read only) ---------------------------------------------------------------------------------------------------------

REAL_RUNS = ['kfold0_x86_gnn_s0_team-sig500v3-feat', 'kfold0_x86_gnn_s0_team-sig500v3-emb', 'kfold0_x86_gnn_s0_team-sig500v3-e2e',
             'kfold0_x86_gnn_s0_team-sigwhole-emb', 'kfold0_x86_fivechannel_s0', 'kfold0_x86_fivechannel-L512_s0',
             'kfold0_x86_fivechannel-L512_s0_whole', 'kfold0_all_gnn_s0_team-sig500v3-e2e', 'kfold0_all_gnn-ab_s0_team-sig500v3-feat',
             'kfold0_all_fivechannel-L512-ab_s0']


@pytest.mark.real_data
@pytest.mark.parametrize('name', REAL_RUNS)
def test_real_runs_are_reproduced_from_their_checkpoints_on_cpu(name, tmp_path):
    """`--check` on the published runs, with the build their metrics.json names, on cpu. Reads the run directory and the build only;
    the numbers are printed (pytest -s) for the report."""
    from common.config import RUNS
    folder = RUNS / name
    if not (folder / 'metrics.json').is_file() or not (folder / 'best_auc.pt').is_file():
        pytest.skip(f'{name} is not in {RUNS}')
    if not Path(json.loads((folder / 'metrics.json').read_text(encoding='utf-8')).get('build_dir', '')).is_dir():
        pytest.skip('the build of the run is not on this machine')
    before = snapshot(folder)
    result = predict.predict(folder, out=tmp_path / 'o.csv', check=True)
    c = result['check']
    print(f'{name}: max |dprob| {c["max_dprob"]:.3g} (tolerance {c["tolerance"]:g}), test AUC saved {c["auc_saved"]:.6f} now '
          f'{c["auc_now"]:.6f}, dAUC {c["dauc"]:+.2e} (tolerance {predict.AUC_TOLERANCE:g}), pred agreement {c["pred_agreement"]:.4f}, n {c["n"]}')
    assert not c['problems'] and snapshot(folder) == before


def test_a_run_whose_build_is_nowhere_stops_with_a_message(world, tmp_path, monkeypatch):
    copy = tmp_path / 'run'
    shutil.copytree(run_of(world, 'model1'), copy)
    metrics = json.loads((copy / 'metrics.json').read_text(encoding='utf-8'))
    (copy / 'metrics.json').write_text(json.dumps({**metrics, 'build_dir': str(tmp_path / 'gone')}), encoding='utf-8')
    monkeypatch.setenv('GT_TEAM_B_BUILD', str(tmp_path / 'nowhere'))
    with pytest.raises(SystemExit, match='is not a directory'):
        predict.main(['--run', str(copy), '--out', str(tmp_path / 'o.csv')])


def test_the_module_imports_without_torch_geometric(tmp_path):
    fake = tmp_path / 'fake'
    (fake / 'torch_geometric').mkdir(parents=True)
    (fake / 'torch_geometric' / '__init__.py').write_text('raise ImportError("torch_geometric is not installed")', encoding='utf-8')
    env = {**os.environ, 'GT_DATASETS': str(tmp_path / 'no-datasets.json'), 'PYTHONPATH': os.pathsep.join([str(fake), str(ROOT)]),
           'CUDA_VISIBLE_DEVICES': '-1'}
    r = subprocess.run([sys.executable, '-c', 'import sys, experiments.predict as p; assert "torch_geometric" not in sys.modules; print("ok")'],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == 'ok', r.stderr
