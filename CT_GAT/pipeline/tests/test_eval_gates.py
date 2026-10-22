"""python -m pytest tests/test_eval_gates.py: the gates engine (`s04_evaluation.gates`) and the stage definitions of
`experiments/stages/`: the stage list, every stage on an empty run directory, the four bespoke layouts (day1, day1b, v3,
subgraph) on synthetic runs, the group names the stages read against RunConfig, and a stage that exists only in the test."""
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
from eval_synth import (CHAIN, FAKE_FLOORS, V3_MASKED, build_fake, build_subgraph_fake, build_v3_ab_fake, build_v3_fake,  # noqa: F401
                        gnn, write_baseline, write_truncated_v3_run, write_v3_run)
from s04_evaluation import gate_layouts, gate_tables, gates, report, stats
from s04_evaluation.metrics import cell_balanced
from s04_evaluation.stage import LABEL, Stage, beats, non_inferior

ROOT = Path(__file__).resolve().parents[1]
PAIRED = ('paths', 'paths_codebert', 'vocab', 'imports', 'noapi', 'team', 'model1', 'model1_variants')
BESPOKE = ('day1', 'day1b', 'v3', 'subgraph')


def stage(name):
    return gates.load_stage(name)


# ---- the engine ------------------------------------------------------------------------------------------------------

def test_the_listed_stages_are_the_files_of_experiments_stages_and_each_names_itself():
    files = {p.stem for p in Path(stages_package.__file__).parent.glob('*.py')} - {'__init__', 'common'}
    assert files == set(stages_package.STAGES) == set(gates.stage_names()) and len(files) == 12
    assert set(stages_package.STAGES) == set(BESPOKE) | set(PAIRED)
    for name in stages_package.STAGES:
        s = stage(name)
        assert s.name == name and s.layout == ('paired' if name in PAIRED else name)
        assert s.pairs or s.layout in ('day1b', 'subgraph')            # these two compare against a reference in their layout
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
    r = subprocess.run([sys.executable, '-m', 's04_evaluation.gates', '--stage', 'noapi', '--resamples', '20'], cwd=ROOT,
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith('# Gate summary, stage noapi') and r.stdout.count('missing') >= 9


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
    monkeypatch.setattr(report, 'v3_floors', lambda arch, seed=0: (FAKE_FLOORS, [FAKE_FLOORS] * 5))
    gate_tables.pooled.cache_clear()
    gate_layouts.v3_cell.cache_clear()
    text = '\n'.join(gates.run_stage(name, 20))
    gate_tables.pooled.cache_clear()
    gate_layouts.v3_cell.cache_clear()
    assert text.startswith(f'# Gate summary, stage {name} ({LABEL})') and 'Traceback' not in text
    assert text.count('missing') >= 5


def test_the_resamples_reach_the_text(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    gate_tables.pooled.cache_clear()
    for n in (20, 30):
        assert f'({n} resamples)' in '\n'.join(gates.run_stage('vocab', n))
    gate_tables.pooled.cache_clear()


# ---- the paired stages: groups follow the run configs, arms print as missing --------------------------------------------

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


def test_noapi_gate_groups_and_pairs_follow_the_run_configs():
    s = stage('noapi')
    arms = dict(s.arms)
    assert list(arms) == ['TF-IDF', 'TF-IDF noapi', 'header-imports LR', 'R7 typed', 'R7 noapi']
    assert arms['TF-IDF'] == 'tfidf' and arms['TF-IDF noapi'] == 'tfidf-noapi' and arms['header-imports LR'] == 'imports-header'
    base = dict(protocol='kfold', fold=0, arch='x86', model='gnn', mask_blocks=0.15)
    assert arms['R7 typed'] == RunConfig(**base).group == 'kfold_x86_gnn-mblk15'
    assert arms['R7 noapi'] == RunConfig(**base, vocab='noapi').group == 'kfold_x86_gnn-mblk15-voc-noapi'
    assert s.pairs == (('R7 typed', 'R7 noapi'), ('TF-IDF', 'TF-IDF noapi'), ('TF-IDF noapi', 'R7 noapi'),
                       ('header-imports LR', 'R7 noapi')) and s.seeds == (0, 1, 2)


def test_noapi_rule_reads_the_ablation_pairs_as_api_loss():
    rule = stage('noapi').rule

    def res(auc, f1):
        return {'auc': {'diff': auc, 'lo': auc - 0.02, 'hi': auc + 0.001},
                'macro_f1': {'diff': f1, 'lo': f1 - 0.02, 'hi': f1 + 0.001}}
    assert rule('R7 typed', 'R7 noapi', res(-0.05, -0.05), {'seeds': [0, 1, 2]}) == 'api loss shown'
    assert rule('R7 typed', 'R7 noapi', res(-0.001, -0.05), {'seeds': [0, 1, 2]}) == 'api loss not shown'
    assert rule('TF-IDF', 'TF-IDF noapi', res(-0.05, -0.05), {'seeds': [0]}) == (
        'api loss shown (seed 0 only: a direction, not the test)')
    assert rule('header-imports LR', 'R7 noapi', res(0.05, 0.05), {'seeds': [0, 1]}) == (
        'beats (seeds [0, 1] only: a direction, not the test)')
    assert rule('header-imports LR', 'R7 noapi', res(0.0, 0.0), {'seeds': [0, 1, 2]}) == 'does not beat'


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


def test_paths_group_names_follow_the_run_configs():
    arms = dict(stage('paths').arms)
    assert arms['R7 + TF-IDF stack'] == RunConfig(model='gnn', protocol='kfold', fold=0, stack_tfidf=True, mask_blocks=0.15).group
    assert arms['R7 typed, tokens'] == RunConfig(model='gnn', protocol='kfold', fold=0, mask_blocks=0.15).group
    assert arms['R2 no-edges, tokens'] == RunConfig(model='gnn', protocol='kfold', fold=0, mask_blocks=0.15,
                                                    control='no_edges').group
    assert arms['TF-IDF'] == 'tfidf' and stats.baseline_method('tfidf')


def test_paths_gates_print_the_group_names(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    gate_tables.pooled.cache_clear()
    text = '\n'.join(gates.run_stage('paths', 50))
    gate_tables.pooled.cache_clear()
    assert 'kfold_x86_gnn-stack-mblk15' in text and 'gnn-mblk15-bv-ours_cap0_ref' in text


def test_paths_codebert_pairs_and_groups():
    s = stage('paths_codebert')
    assert s.pairs[0] == ('path 1: no-edges, CodeBERT vectors', 'path 2: typed, CodeBERT vectors') and s.seeds == (0, 1, 2)
    text = ' '.join(group for _, group in s.arms)
    assert 'gnn-mblk15-bv-ours_cap0_codebert' in text and 'gnn-noedges-mblk15-bv-ours_cap0_codebert' in text


def test_model1_gate_groups_and_pairs_follow_the_run_configs():
    s = stage('model1')
    arms = dict(s.arms)
    assert list(arms) == ['TF-IDF', 'header-imports LR', 'R7 typed', 'R2 no-edges', 'Model 1 tokenseq']
    base = dict(protocol='kfold', fold=0, arch='x86', model='gnn', mask_blocks=0.15)
    assert arms['TF-IDF'] == 'tfidf' and arms['header-imports LR'] == 'imports-header'
    assert arms['R7 typed'] == RunConfig(**base).group == 'kfold_x86_gnn-mblk15'
    assert arms['R2 no-edges'] == RunConfig(**base, control='no_edges').group == 'kfold_x86_gnn-noedges-mblk15'
    assert arms['Model 1 tokenseq'] == RunConfig(protocol='kfold', fold=0, arch='x86', model='tokenseq').group
    assert arms['Model 1 tokenseq'].replace('-L32-w8', '') == 'kfold_x86_tokenseq'
    assert s.pairs == (('TF-IDF', 'Model 1 tokenseq'), ('header-imports LR', 'Model 1 tokenseq'),
                       ('R7 typed', 'Model 1 tokenseq'), ('R2 no-edges', 'Model 1 tokenseq')) and s.seeds == (0, 1, 2)


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


def test_model1_gates_say_what_the_rule_is(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    gate_tables.pooled.cache_clear()
    text = '\n'.join(gates.run_stage('model1', 20))
    gate_tables.pooled.cache_clear()
    assert 'pre-registered' in text and 'dAUC >= 0.01' in text and 'dmacro-F1 >= 0.02' in text


def test_the_vocab_and_imports_stages_pair_what_their_docstrings_say():
    vocab, imports = stage('vocab'), stage('imports')
    assert dict(vocab.arms)['R2 no-edges, yanping'] == 'kfold_x86_gnn-noedges-mblk15-voc-yanping'
    assert dict(vocab.arms)['TF-IDF, hybrid'] == 'tfidf-hybrid' and len(vocab.pairs) == 7
    assert dict(imports.arms)['R7 + header imports'] == 'kfold_x86_gnn-mblk15-imph'
    assert imports.pairs == (('R7 typed', 'R7 + header imports'), ('header imports, LR', 'R7 + header imports'),
                             ('TF-IDF', 'R7 + header imports'))


def test_day1_and_day1b_arms_are_run_names():
    day1, day1b = stage('day1'), stage('day1b')
    arms = dict(day1.arms)
    assert arms['R7 typed'] == RunConfig(model='gnn', protocol='kfold', fold=0).group == 'kfold_x86_gnn'
    assert arms['R5 untyped'] == RunConfig(model='gnn', protocol='kfold', fold=0, untyped=True).group
    assert day1.options['gate_a'] == ('R0 TF-IDF', 'R2 tfidf') and len(day1.pairs) == 5
    assert dict(day1b.arms) == {'R7 typed': 'kfold_x86_gnn', 'R2 tfidf': 'kfold_x86_gnn-noedges'}
    assert len(day1b.options['new_arms']) == 13 and all(g.startswith('kfold_x86_gnn-') for g in day1b.options['new_arms'])


def test_subgraph_gate_arms_are_run_names():
    sg, v3 = stage('subgraph'), stage('v3')
    assert sg.options['reference'][1] == 'gnn-mblk15' and sg.options['noedges'] == 'gnn-noedges-mblk15'
    assert [v for _, v in sg.arms] == ['gnn-mblk15-sg-cap20k', 'gnn-mblk15-sg-cap5k']
    assert [v for _, v in sg.options['weightings']] == [f'gnn-noedges-mblk15-enc-{e}' for e in ('raw', 'presence', 'rawlog')]
    assert [v for _, v in v3.arms[1:]] == ['gnn-noedges-mblk15-selcell', 'gnn-shuffle-mblk15-selcell', 'gnn-mblk15-selcell']
    assert [v for _, v in v3.options['ab_arms'][1:]] == ['gnn-noedges-mblk15-ab-selcell', 'gnn-shuffle-mblk15-ab-selcell',
                                                         'gnn-mblk15-ab-selcell']
    assert v3.options['arches'] == ('all', 'matched')
    assert v3.options['ab_pairs'] == (('R2 no-edges ab', 'R7 typed ab'), ('R4 shuffle ab', 'R7 typed ab'),
                                      ('R0 TF-IDF ab', 'R2 no-edges ab'))
    assert v3.options['twins'] == tuple((arm, f'{arm} ab') for arm, _ in v3.arms)


# ---- the bespoke layouts on synthetic runs -------------------------------------------------------------------------------

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
    text = '\n'.join(gates.run_stage('day1', 50))
    assert LABEL in text
    assert 'kfold_x86_gnn-shuffle' in text and 'missing (seed 0 folds present [0, 1, 2] of 0-4)' in text
    assert '**R2 tfidf vs R7 typed**' in text and '**R2 tfidf vs R4 shuffle** (selection f1): missing' in text
    gate_tables.pooled.cache_clear()


def test_day1b_gates_from_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    gate_tables.pooled.cache_clear()
    build_fake(tmp_path, {(('control', 'no_edges'),): (1.2, range(5)), (): (1.6, range(5)),
                          (('direction', 'forward'),): (1.5, range(5)), (('train_fraction', 0.5),): (1.3, range(5)),
                          (('drop_edges', 0.2),): (1.4, [0, 1, 4])})
    for fold in range(5):
        write_baseline(tmp_path, fold, 1.0)
        write_baseline(tmp_path, fold, 0.8, tf=50)
    assert gate_layouts.gnn_groups() == ['kfold_x86_gnn', 'kfold_x86_gnn-dedge20', 'kfold_x86_gnn-fwd',
                                  'kfold_x86_gnn-noedges', 'kfold_x86_gnn-tf50']
    text = '\n'.join(gates.run_stage('day1b', 50))
    assert 'stage day1b' in text and LABEL in text
    table = text.split('## Paired differences')[0]
    assert '| kfold_x86_gnn-dedge20 | missing (seed 0 folds present [0, 1, 4] of 0-4)' in table
    assert '| kfold_x86_gnn-fwd | 0.' in table and 'AUC checkpoint' in table
    assert '**kfold_x86_gnn-fwd vs R7 typed**' in text and '**kfold_x86_gnn-fwd vs R2 tfidf**' in text
    assert 'selection auc' in text and 'kfold_x86_gnn-fwd minus kfold_x86_gnn ' in text
    assert '**kfold_x86_gnn-dedge20 vs R7 typed** (selection auc): missing: kfold_x86_gnn-dedge20' in text
    assert '**kfold_x86_gnn-bwd vs R7 typed** (selection auc): missing' in text
    curve = text.split('## Learning curve')[1]
    rows = [l for l in curve.splitlines() if l.startswith('| ') and 'train fraction' not in l]
    assert [l.split(' | ')[:3] for l in rows] == [
        ['| 0.25', 'R7 typed', 'kfold_x86_gnn-tf25'], ['| 0.25', 'TF-IDF', 'tfidf-tf25'],
        ['| 0.5', 'R7 typed', 'kfold_x86_gnn-tf50'], ['| 0.5', 'TF-IDF', 'tfidf-tf50'],
        ['| 1.0', 'R7 typed', 'kfold_x86_gnn'], ['| 1.0', 'TF-IDF', 'tfidf']]
    assert 'missing' in rows[0] and 'missing' in rows[1] and 'missing' not in ''.join(rows[2:])
    gate_tables.pooled.cache_clear()


def test_v3_gates_from_disk(tmp_path, monkeypatch):
    build_v3_fake(tmp_path, monkeypatch)
    text = '\n'.join(gates.run_stage('v3', 50))
    assert 'stage v3' in text and LABEL in text
    all_part, matched_part = text.split('## Arch matched: pooled table')
    head = all_part.split('### Arch all')[0]
    row = {l.split(' | ')[0].lstrip('| '): l for l in head.splitlines() if l.startswith('| ')}
    assert 'R x86 recall' in head and 'AUC ckpt, val thr AUC' in head and 'F1 ckpt, 0.5 cell-bal AUC' in head
    assert 'AUC ckpt, val thr cell-bal macro-F1' in head
    assert row['R0 TF-IDF'].split(' | ')[1] == 'tfidf' and 'missing' not in row['R0 TF-IDF']
    assert 'missing (seed 0 folds present [0, 1, 2] of 0-4)' in row['R4 shuffle']
    assert 'R5 untyped' not in head and 'selcell' in row['R7 typed'].split(' | ')[1] and 'mblk15' in row['R7 typed']
    cells = row['R7 typed'].split(' | ')
    assert len(cells) == 16 and 'missing' not in row['R7 typed']
    assert all(0 <= float(c.strip(' |')) <= 1 for c in cells[2:])
    assert '| floor: majority | - | 0.500 | 0.600 | 0.400 | 0.500 | 0.600 | 0.400 | 1.000 | 1.000 | 0.000 | 0.000 '            '| 0.500 | 0.330 | 0.500 | 0.330 |' in head
    assert '| floor: x86_rule | - | 0.700 | 0.800 | 0.690 | 0.700 | 0.800 | 0.690 | 1.000 | 0.000 | 0.000 | 1.000 '            '| 0.500 | 0.500 | 0.500 | 0.500 |' in head
    assert 'missing' in row['R7 typed ab'] and 'kfold_v3_all_gnn-mblk15-ab-selcell' in row['R7 typed ab']
    assert 'block-masked' in head and 'validation cell-balanced AUC' in head
    assert 'floor: x86_rule' in head and 'x86-rule macro-F1 per fold: 0.690, 0.690, 0.690, 0.690, 0.690' in head
    for pair in ('R2 no-edges vs R7 typed', 'R4 shuffle vs R7 typed', 'R0 TF-IDF vs R2 no-edges'):
        assert f'**{pair}**' in all_part and f'**{pair}**' in matched_part
    assert '(selection auc): missing: kfold_v3_all_gnn-shuffle-mblk15-selcell' in all_part
    assert 'kfold_v3_all_gnn-noedges-mblk15-selcell minus tfidf' in all_part
    assert 'kfold_v3_matched_gnn-noedges-mblk15-selcell minus tfidf' in matched_part
    assert 'kfold_v3_all_gnn-mblk15-selcell minus kfold_v3_all_gnn-noedges-mblk15-selcell' in all_part
    assert all_part.count('Cell-balanced: every test sample weighted') >= 1
    lines = [l for l in text.split('## Verdict')[1].splitlines() if l.startswith('- Arch')]
    assert len(lines) == 8 and all(LABEL in l for l in lines)
    assert lines[0].startswith('- Arch all: cannot assess, missing: R4 shuffle')
    assert lines[4].startswith('- Arch matched: R7 does not beat both R2 and R4')
    assert lines[5].startswith('- Arch matched: cell-balanced: R7 does not beat both R2 and R4')
    assert lines[6].startswith('- Arch matched: cannot assess, missing: R2 no-edges ab')
    assert lines[7].startswith('- Arch matched: cell-balanced: cannot assess, missing: R2 no-edges ab')


def test_v3_verdict_rule():
    def res(auc, f1, lo=0.001):
        return {'auc': {'diff': auc, 'lo': lo, 'hi': auc + 0.1}, 'macro_f1': {'diff': f1, 'lo': lo, 'hi': f1 + 0.1}}
    assert beats(res(0.01, 0.02)) and not beats(res(0.009, 0.05)) and not beats(res(0.05, 0.019))
    assert not beats(res(0.05, 0.05, lo=-0.001)) and not beats(res(0.05, 0.05, lo=0.0))
    both = {('R2 no-edges', 'R7 typed'): res(0.05, 0.05), ('R4 shuffle', 'R7 typed'): res(0.05, 0.05)}
    assert 'R7 beats both R2 and R4: beats R2 no-edges' in gate_layouts.v3_verdict('all', both)
    one = {**both, ('R4 shuffle', 'R7 typed'): res(0.0, 0.0, lo=-0.1)}
    line = gate_layouts.v3_verdict('matched', one)
    assert 'does not beat both' in line and 'does not beat R4 shuffle' in line and 'seed 0 only' in line


def test_v3_gates_beats_both_on_strong_signal(tmp_path, monkeypatch):
    build_v3_fake(tmp_path, monkeypatch)
    for fold in (3, 4):
        write_v3_run(tmp_path, 'all', fold, 1.2, control='shuffle', **V3_MASKED)
    gate_layouts.v3_cell.cache_clear()
    text = '\n'.join(gates.run_stage('v3', 100))
    line = next(l for l in text.split('## Verdict')[1].splitlines() if l.startswith('- Arch all'))
    assert line.startswith('- Arch all: R7 beats both R2 and R4: beats R2 no-edges') and 'beats R4 shuffle' in line


def test_v3_gates_truncation_line(tmp_path, monkeypatch):
    build_v3_fake(tmp_path, monkeypatch)
    for d in tmp_path.glob('v3kfold*_matched_gnn-mblk15-selcell_s0'):
        shutil.rmtree(d)
    cut = {f'f{fold}_{i}' for fold in range(5) for i in range(0, 120, 10)}
    write_truncated_v3_run(tmp_path, 'matched', cut)
    gate_layouts.v3_cell.cache_clear()
    text = '\n'.join(gates.run_stage('v3', 30))
    lines = {l.split(':')[0]: l for l in text.splitlines() if 'truncat' in l and l.startswith('- Arch')}
    assert 'truncation not assessed, R7 predictions lack n_blocks_kept' in lines['- Arch all']
    line = lines['- Arch matched']
    assert f'R7 truncated {5 * 12} of 600 test binaries' in line and 'cap 500000' in line
    assert 'truncated: R recall' in line and '; untruncated: R recall' in line
    p = stats.load_group('kfold_v3_matched_gnn-mblk15-selcell', 'auc', ('kfold_v3', 'matched'))[0][0]
    m = p.truncated == 1
    assert m.sum() == 60 and f"G recall {np.mean(p.pred[m & (p.y == 0)] == 0):.3f} (n={int((m & (p.y == 0)).sum())})" in line


def test_gates_v3_cell_balanced_columns_ab_arms_and_twins(tmp_path, monkeypatch):
    build_v3_ab_fake(tmp_path, monkeypatch)
    text = '\n'.join(gates.run_stage('v3', 50))
    all_part = text.split('## Arch matched: pooled table')[0]
    row = {l.split(' | ')[0].lstrip('| '): l for l in all_part.splitlines() if l.startswith('| ')}
    for arm in ('R7 typed', 'R7 typed ab', 'R2 no-edges ab', 'R4 shuffle ab', 'R0 TF-IDF ab'):
        cells = row[arm].split(' | ')
        assert len(cells) == 16 and 'missing' not in row[arm], arm
        assert all(0 <= float(c.strip(' |')) <= 1 for c in cells[2:])
    assert 'R5 untyped' not in text
    got = gate_layouts.v3_cell('kfold_v3_all_gnn-mblk15-selcell', 'f1', 'all')
    q = stats.load_group('kfold_v3_all_gnn-mblk15-selcell', 'f1', ('kfold_v3', 'all'))[0][0]
    want = cell_balanced(q.y, q.prob, q.pred, q.arch)
    assert got[7] == pytest.approx(want['auc']) and got[8] == pytest.approx(want['macro_f1'])
    assert '### Arch all: paired cell-balanced differences, second minus first' in all_part
    assert 'Cell-balanced: every test sample weighted' in all_part
    twin = all_part.split('each arch-balanced arm minus its unweighted twin')[1]
    assert 'kfold_v3_all_gnn-mblk15-ab-selcell minus kfold_v3_all_gnn-mblk15-selcell' in twin
    assert 'tfidf-ab minus tfidf' in twin
    assert 'kfold_v3_all_gnn-noedges-mblk15-ab-selcell minus kfold_v3_all_gnn-noedges-mblk15-selcell' in twin
    ab_pairs = all_part.split('arch-balanced arms, paired cell-balanced differences')[1].split('### Arch all: each')[0]
    assert 'kfold_v3_all_gnn-mblk15-ab-selcell minus kfold_v3_all_gnn-noedges-mblk15-ab-selcell' in ab_pairs
    assert 'kfold_v3_all_gnn-noedges-mblk15-ab-selcell minus tfidf-ab' in ab_pairs
    plain_ab = all_part.split('arch-balanced arms, paired differences, second minus first')[1].split('### Arch all: arch')[0]
    assert 'kfold_v3_all_gnn-mblk15-ab-selcell minus kfold_v3_all_gnn-shuffle-mblk15-ab-selcell' in plain_ab
    assert 'Cell-balanced: every' not in plain_ab
    verdicts = [l for l in text.split('## Verdict')[1].splitlines() if l.startswith('- Arch all')]
    assert len(verdicts) == 4 and verdicts[3].startswith('- Arch all: cell-balanced: R7 ab beats both R2 and R4')
    assert verdicts[2].startswith('- Arch all: R7 ab beats both R2 and R4')


def test_subgraph_gates_from_disk(tmp_path, monkeypatch):
    build_subgraph_fake(tmp_path, monkeypatch, CHAIN)
    text = '\n'.join(gates.run_stage('subgraph', 100))
    assert 'stage subgraph' in text and LABEL in text
    noninf = text.split('## Subgraph policies')[1].split('## Chain merge')[0]
    ni = {l.split(' | ')[0].lstrip('| '): l for l in noninf.splitlines() if l.startswith('| R7')}
    assert ni['R7 typed, cap 20k'].endswith('| yes |') and ni['R7 typed, cap 5k'].endswith('| no |')
    assert set(ni) == {'R7 typed, cap 20k', 'R7 typed, cap 5k'}
    assert 'kfold_x86_gnn-mblk15-sg-cap5k' in ni['R7 typed, cap 5k'] and 'kfold_x86_gnn-mblk15 ' in noninf
    res, _ = stats.compare('kfold_x86_gnn-mblk15', 'kfold_x86_gnn-mblk15-sg-cap20k', 'auc', 100, seeds=[0])
    assert non_inferior(res) and f"{res['auc']['diff'] + 0.0:+.3f} [{res['auc']['lo'] + 0.0:+.3f}" in ni[
        'R7 typed, cap 20k']
    assert '- Chain merge (kfold x86): 41000 of 10000000 blocks in 1800 binaries could merge' in text
    assert 'reduction of 0.41%' in text and 'below the 5% bar, so no chain-merge policy is built' in text
    weights = text.split('## Block-level counting weightings')[1]
    w = {l.split(' | ')[0].lstrip('| '): l for l in weights.splitlines() if l.startswith('| R2')}
    assert set(w) == {'R2 no-edges, raw', 'R2 no-edges, presence', 'R2 no-edges, rawlog'}
    assert 'missing' in w['R2 no-edges, rawlog'] and 'missing' not in w['R2 no-edges, raw']
    assert 'kfold_x86_gnn-noedges-mblk15-enc-presence' in w['R2 no-edges, presence']
    assert 'kfold_x86_gnn-noedges-mblk15 minus' not in weights and 'kfold_x86_gnn-noedges-mblk15-enc-raw minus' in weights
    pooled = text.split('## Subgraph policies')[0]
    assert '| R7 typed, whole graph | kfold_x86_gnn-mblk15 |' in pooled and 'R2 no-edges, tfidf' in pooled


def test_subgraph_gates_without_runs_or_chain_file(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    gate_tables.pooled.cache_clear()
    text = '\n'.join(gates.run_stage('subgraph', 30))
    assert 'Chain merge: not measured (subgraph_chain.json is missing' in text
    assert text.count('missing') >= 10 and 'Traceback' not in text
    chain = dict(CHAIN, reduction=0.06, n_chain_blocks=600_000)
    (tmp_path / 'subgraph_chain.json').write_text(json.dumps(chain))
    assert 'a chain-merge policy is worth building' in gate_layouts.chain_line(0.05)
