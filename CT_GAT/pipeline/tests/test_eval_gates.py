"""python -m pytest tests/test_eval_gates.py: the gates engine (`s04_evaluation.gates`) and the stage definitions of
`experiments/stages/`: the stage list, every stage on an empty run directory, the group names the stages read against RunConfig
(rows that read published runs of the earlier models and baselines need no model code), and a stage that exists only in the test."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import experiments.stages as stages_package
from experiments.stages import team as team_stage
from common.config import RunConfig
from eval_synth import build_fake
from s04_evaluation import gate_tables, gates, stats
from s04_evaluation.stage import LABEL, Stage, beats, non_inferior

ROOT = Path(__file__).resolve().parents[1]
PAIRED = ('team', 'model1_variants')


def stage(name):
    return gates.load_stage(name)


# ---- the engine ------------------------------------------------------------------------------------------------------


def test_the_listed_stages_are_the_files_of_experiments_stages_and_each_names_itself():
    files = {p.stem for p in Path(stages_package.__file__).parent.glob('*.py')} - {'__init__', 'common'}
    assert files == set(stages_package.STAGES) == set(gates.stage_names()) and len(files) == 2
    assert set(stages_package.STAGES) == set(PAIRED)
    for name in stages_package.STAGES:
        s = stage(name)
        assert s.name == name and s.pairs
        assert len(dict(s.arms)) == len(s.arms)                                  # an arm label is used once


def test_an_unknown_stage_is_refused():
    with pytest.raises(ValueError, match='unknown stage'):
        gates.load_stage('day2')
    r = subprocess.run([sys.executable, '-m', 's04_evaluation.gates', '--stage', 'day2'], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and 'invalid choice' in r.stderr


def test_the_command_line_lists_the_stages_and_runs_one(tmp_path):
    r = subprocess.run([sys.executable, '-m', 's04_evaluation.gates', '--help'], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and all(name in r.stdout for name in stages_package.STAGES)
    env = {**os.environ, 'GT_RUNS': str(tmp_path), 'PYTHONIOENCODING': 'utf-8'}
    r = subprocess.run([sys.executable, '-m', 's04_evaluation.gates', '--stage', 'team', '--resamples', '20'], cwd=ROOT,
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith('# Gate summary, stage team') and r.stdout.count('missing') >= 9


def test_a_stage_is_data_a_new_one_needs_no_engine_change(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    gate_tables.pooled.cache_clear()
    build_fake(tmp_path, {(('control', 'no_edges'),): (1.2, range(5)), (): (1.6, range(5)), (('control', 'shuffle'),): (0.8, [0, 1])})
    seen = []

    def rule(first, second, result, info):
        seen.append((first, second, info['seeds']))
        return 'beats' if beats(result) else 'does not beat'
    toy = Stage('toy', intro=lambda resamples: f'a stage defined in the test ({resamples} resamples)',
                arms=(('A', 'kfold_x86_gnn-noedges'), ('B', 'kfold_x86_gnn'), ('C', 'kfold_x86_gnn-shuffle')),
                pairs=(('A', 'B'), ('A', 'C')), rule=rule)
    text = '\n'.join(gates.render(toy, 40))
    gate_tables.pooled.cache_clear()
    assert text.startswith(f'# Gate summary, stage toy ({LABEL})') and 'a stage defined in the test (40 resamples)' in text
    assert '| A | kfold_x86_gnn-noedges | 0.' in text
    assert '| C | kfold_x86_gnn-shuffle | missing (seed 0 folds present [0, 1] of 0-4)' in text
    assert '| first | second | dAUC [95% CI] | dmacro-F1 [95% CI] | rule |' in text
    assert '| A | C | missing | missing | missing |' in text and seen == [('A', 'B', [0])]
    assert '### kfold_x86_gnn minus kfold_x86_gnn-noedges (selection auc, 40 resamples)' in text


@pytest.mark.parametrize('name', stages_package.STAGES)
def test_every_stage_renders_on_an_empty_run_directory(name, tmp_path, monkeypatch):
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    gate_tables.pooled.cache_clear()
    text = '\n'.join(gates.run_stage(name, 20))
    gate_tables.pooled.cache_clear()
    assert text.startswith(f'# Gate summary, stage {name} ({LABEL})') and 'Traceback' not in text
    assert text.count('missing') >= 5


def test_the_resamples_reach_the_text(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    gate_tables.pooled.cache_clear()
    for n in (20, 30):
        assert f'({n} resamples)' in '\n'.join(gates.run_stage('model1_variants', n))
    gate_tables.pooled.cache_clear()


# ---- the stages: groups follow the run configs, arms print as missing ---------------------------------------------------


@pytest.mark.parametrize('name', PAIRED)
def test_paired_stages_print_missing_arms_and_pairs(name, tmp_path, monkeypatch):
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    gate_tables.pooled.cache_clear()
    s = stage(name)
    text = '\n'.join(gates.render(s, 50))
    gate_tables.pooled.cache_clear()
    assert text.startswith(f'# Gate summary, stage {name}')
    for arm, group in s.arms:
        assert f'| {arm} | {group} | missing' in text
    for a, b in s.pairs:
        assert f'| {a} | {b} | missing | missing |' in text
        assert f'**{b} vs {a}** (selection auc): missing:' in text
    assert (' rule |' in text) == (s.rule is not None)


def test_team_groups_are_the_names_the_trainers_write():
    s = stage('team')
    arms = dict(s.arms)
    assert arms['team cb500'] == RunConfig(protocol='kfold', fold=3, arch='x86', seed=1, model='gnn', tag='team-cb500').group
    assert arms['team cb500'] == 'kfold_x86_gnn_team-cb500'
    assert arms['ours b500 typed'] == 'kfold_x86_gnn-mblk15-sg-cap500_b500'
    assert arms['ours b500 no-edges'] == 'kfold_x86_gnn-noedges-mblk15-sg-cap500_b500'
    assert arms['R7 typed, whole graph'] == 'kfold_x86_gnn-mblk15'
    assert arms['R2 no-edges, whole graph'] == 'kfold_x86_gnn-noedges-mblk15'
    assert arms['path 2: typed, CodeBERT vectors'] == 'kfold_x86_gnn-mblk15-bv-ours_cap0_codebert'
    assert arms['header-imports LR'] == 'imports-header' and arms['TF-IDF'] == 'tfidf'
    pairs = s.pairs
    first = ('team gnn500', 'team cb500', 'team sig500', 'team sig500v2')                 # the arms of the first four builds
    for ref in ('TF-IDF', 'R7 typed, whole graph', 'ours b500 typed'):
        v2 = ['team sig500v2'] if ref != 'ours b500 typed' else []                  # sig500v2: against sig500, R7 and TF-IDF
        assert [b for a, b in pairs if a == ref and b in first] == ['team gnn500', 'team cb500', 'team sig500'] + v2
    assert ('team sig500', 'team sig500v2') in pairs
    assert ('path 2: typed, CodeBERT vectors', 'team cb500') in pairs and ('R7 typed, whole graph', 'ours b500 typed') in pairs
    assert len(pairs) == len(set(pairs)) == 31 and s.extra == 'operating_points' and s.seeds == (0, 1, 2)
    assert s.options['match_rate_builds'] == (('b_sig500_v3', 'team/b_sig500_v3'), ('b_sigwhole_v3', 'team/b_sigwhole_v3'))


def test_team_gates_v3_and_whole_rows_and_pairs():
    s = stage('team')
    arms = dict(s.arms)
    for label, arm in (('team sig500v3', 'sig500v3'), ('team sig500v3 feat (Model 2)', 'sig500v3-feat'),
                       ('team sig500v3 emb (Model 3)', 'sig500v3-emb'), ('team sigwhole', 'sigwhole'),
                       ('team sigwhole feat (Model 2)', 'sigwhole-feat'), ('team sigwhole emb (Model 3)', 'sigwhole-emb')):
        assert arms[label] == RunConfig(protocol='kfold', fold=0, arch='x86', model='gnn', tag=f'team-{arm}').group
    # main's Model 1 on the uncapped build: written by `--model fivechannel --max-instructions 512 --tag whole`
    assert arms['fivechannel-L512 whole'] == RunConfig(protocol='kfold', fold=0, arch='x86', model='fivechannel', max_instructions=512,
                                                       tag='whole').group == 'kfold_x86_fivechannel-L512_whole'
    pairs = s.pairs
    v3, feat, emb = team_stage.V3
    for ref in ('R7 typed, whole graph', 'ours b500 no-edges'):
        assert [b for a, b in pairs if a == ref and b in team_stage.V3] == [v3, feat, emb]
    assert [a for a, b in pairs if b == v3] == ['team sig500v2', 'team sig500', 'R7 typed, whole graph', 'ours b500 no-edges']
    assert (feat, emb) in pairs and not any(b in team_stage.V3 for a, b in pairs if a in ('TF-IDF', 'ours b500 typed'))
    for before, after in zip(team_stage.V3, team_stage.WHOLE):
        assert (before, after) in pairs
    assert [b for a, b in pairs if a == 'R7 typed, whole graph' and b in team_stage.WHOLE_ROWS] == list(team_stage.WHOLE_ROWS)
    assert (team_stage.WHOLE[1], team_stage.WHOLE[2]) in pairs
    assert not set(team_stage.V3 + team_stage.WHOLE) & set(team_stage.TEAM_OF)
    assert not any(b in team_stage.WHOLE_ROWS for a, b in pairs if a in ('TF-IDF', 'ours b500 typed'))


def test_team_gates_pair_sig500v2_with_sig500_r7_and_tfidf():
    s = stage('team')
    assert dict(s.arms)['team sig500v2'] == 'kfold_x86_gnn_team-sig500v2'
    v2 = [(a, b) for a, b in s.pairs if b == 'team sig500v2']
    assert sorted(v2) == sorted([('team sig500', 'team sig500v2'), ('R7 typed, whole graph', 'team sig500v2'),
                                 ('TF-IDF', 'team sig500v2')])


def test_team_gates_print_the_operating_point_table_and_the_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    gate_tables.pooled.cache_clear()
    text = '\n'.join(gates.run_stage('team', 50))
    gate_tables.pooled.cache_clear()
    for arm, group in stage('team').arms:
        assert f'| {arm} | missing' in text
    for word in ('accuracy', 'precision', 'recall', 'TP', 'FN', 'FP', 'TN', 'Pre-registered rule', 'non-inferiority'):
        assert word in text


def test_team_rule_reads_the_cap_cost_as_non_inferiority():
    rule = stage('team').rule
    ok = {'auc': {'diff': -0.001, 'lo': -0.005, 'hi': 0.003}, 'macro_f1': {'diff': 0.0, 'lo': -0.01, 'hi': 0.01}}
    bad = {'auc': {'diff': -0.03, 'lo': -0.05, 'hi': -0.01}, 'macro_f1': {'diff': 0.0, 'lo': -0.01, 'hi': 0.01}}
    cost = ('R7 typed, whole graph', 'ours b500 typed')
    assert rule(*cost, ok, {'seeds': [0, 1, 2]}) == 'non-inferior' and non_inferior(ok) and not non_inferior(bad)
    assert rule(*cost, bad, {'seeds': [0, 1, 2]}) == 'not shown non-inferior'
    assert rule('TF-IDF', 'team gnn500', ok, {'seeds': [0]}) == 'does not beat (seeds [0] only: a direction, not the test)'


def test_model1_variants_gate_groups_and_pairs_follow_the_run_configs():
    s = stage('model1_variants')
    arms = dict(s.arms)
    assert list(arms) == ['TF-IDF', 'header-imports LR', 'R7 typed', 'R2 no-edges', 'Model 1 tokenseq',
                          'Model 1 tokenseq, out-of-fold threshold', 'Model 1 five-channel L128', 'Model 1 five-channel L512',
                          'team sig500v3']
    kfold = dict(protocol='kfold', fold=0, arch='x86')
    assert arms['Model 1 tokenseq'] == RunConfig(**kfold, model='tokenseq').group == 'kfold_x86_tokenseq'
    assert arms['Model 1 tokenseq, out-of-fold threshold'] == 'kfold_x86_tokenseq-thoof'
    assert arms['Model 1 five-channel L128'] == 'kfold_x86_fivechannel' and arms['Model 1 five-channel L512'] == 'kfold_x86_fivechannel-L512'
    assert arms['team sig500v3'] == RunConfig(**kfold, model='gnn', tag='team-sig500v3').group == 'kfold_x86_gnn_team-sig500v3'
    assert arms['R7 typed'] == 'kfold_x86_gnn-mblk15' and arms['R2 no-edges'] == 'kfold_x86_gnn-noedges-mblk15'
    variants = ['Model 1 tokenseq, out-of-fold threshold', 'Model 1 five-channel L128', 'Model 1 five-channel L512']
    assert s.pairs == tuple((ref, v) for v in variants for ref in ('Model 1 tokenseq', 'TF-IDF')) and s.seeds == (0, 1, 2)


def test_stats_and_gates_from_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    gate_tables.pooled.cache_clear()
    build_fake(tmp_path, {(('control', 'no_edges'),): (1.2, range(5)), (): (1.6, range(5)),
                          (('control', 'shuffle'),): (0.8, [0, 1, 2])})
    result, info = stats.compare('kfold_x86_gnn-noedges', 'kfold_x86_gnn-noedges', 'f1', 100)
    assert info['seeds'] == [0] and result['auc']['diff'] == 0 and result['macro_f1']['hi'] == 0
    result, info = stats.compare('kfold_x86_gnn-noedges', 'kfold_x86_gnn', 'auc', 100)
    assert result['auc']['diff'] > 0 and result['auc']['lo'] < result['auc']['hi']
    swapped, _ = stats.compare('kfold_x86_gnn', 'kfold_x86_gnn-noedges', 'auc', 100)
    assert swapped['auc']['diff'] == pytest.approx(-result['auc']['diff'])
    with pytest.raises(stats.MissingGroup):
        stats.compare('kfold_x86_gnn-noedges', 'kfold_x86_gnn-shuffle', 'f1', 50)
    gate_tables.pooled.cache_clear()
