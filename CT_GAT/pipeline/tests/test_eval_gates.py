"""python -m pytest tests/test_eval_gates.py: the gates engine (`s04_evaluation.gates`) and the stage definitions of
`experiments/stages/`: the stage list, every stage on an empty run directory, the group names the stages read against RunConfig
(rows that read published runs of the earlier models and baselines need no model code), the stages on the whole Mendeley dataset
and on the combined dataset (`scope`, `balanced`, the `cells` table), and a stage that exists only in the test."""
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import experiments.stages as stages_package
from experiments import queue as q
from experiments.stages import team as team_stage
from common.config import RunConfig
from eval_synth import build_fake, build_v3_ab_fake, write_v3_baseline, write_v3_run
from experiments.team_pipelines import common as tp
from s04_evaluation import gate_tables, gates, report, stats
from s04_evaluation.stage import DEFAULT_SCOPE, LABEL, Stage, beats, non_inferior
from test_team_v2 import fake_build, fake_payload

ROOT = Path(__file__).resolve().parents[1]
PAIRED = ('team', 'model1_variants', 'team_mendeley_all', 'team_v3_all', 'team_v3_matched')
DATASET_STAGES = {'team_mendeley_all': ('kfold', 'all'), 'team_v3_all': ('kfold_v3', 'all'), 'team_v3_matched': ('kfold_v3', 'matched')}


def stage(name):
    return gates.load_stage(name)


# ---- the engine ------------------------------------------------------------------------------------------------------


def test_the_listed_stages_are_the_files_of_experiments_stages_and_each_names_itself():
    files = {p.stem for p in Path(stages_package.__file__).parent.glob('*.py')} - {'__init__', 'common'}
    assert files == set(stages_package.STAGES) == set(gates.stage_names()) and len(files) == 5
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
    assert len(pairs) == len(set(pairs)) == 34 + 12 and s.extra == 'operating_points' and s.seeds == (0, 1, 2)    # 12: callwalk / 250 pairs
    assert not [p for p in pairs[:34] if 'sig250' in ' '.join(p) or 'callwalk' in ' '.join(p)]                 # the first 34 are as before
    assert s.options['match_rate_builds'][:2] == (('b_sig500_v3', 'team/b_sig500_v3'), ('b_sigwhole_v3', 'team/b_sigwhole_v3'))
    assert [b for b, _ in s.options['match_rate_builds'][2:]] == ['b_sig250_main', 'b_callwalk500_main', 'b_callwalk250_main']


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


# ---- the stages on the whole Mendeley dataset and on the combined dataset -----------------------------------------------------


def group_of(run_name):
    """RunConfig.group of a run directory name: kfold0_all_gnn_s2_team-sig500v3-feat -> kfold_all_gnn_team-sig500v3-feat."""
    m = re.fullmatch(r'(v3kfold|kfold)\d_(x86|all|matched)_(.+?)_s\d(_.+)?', run_name)
    return f"{'kfold_v3' if m[1] == 'v3kfold' else 'kfold'}_{m[2]}_{m[3]}{m[4] or ''}"


def plan_groups(plan):
    """{group: set of (fold, seed)} of the train and train_b lines of a plan."""
    found = {}
    for step in q.load_plan(plan):
        if step.verb in q.TRAIN:
            fold, seed = (int(re.search(r'kfold(\d)_', step.label)[1]), int(re.search(r'_s(\d)(_|$)', step.label)[1]))
            found.setdefault(group_of(step.label), set()).add((fold, seed))
    return found


@pytest.mark.parametrize('name,scope', DATASET_STAGES.items())
def test_the_dataset_stages_name_their_scope_and_read_it_everywhere(name, scope):
    s = stage(name)
    assert tuple(s.options['scope']) == scope and s.options['balanced'] is True and s.extra == 'cells' and s.seeds == (0, 1, 2)
    assert s.rule is not None and len(set(s.pairs)) == len(s.pairs) and len(dict(s.arms)) == len(s.arms)
    protocol, arch = scope
    for label, group in s.arms:
        if group in ('tfidf', 'tfidf-ab'):
            continue
        assert group.startswith(f'{protocol}_{arch}_'), (label, group)                  # every run group carries the stage's dataset
    for a, b in s.pairs:
        assert a in dict(s.arms) and b in dict(s.arms)
    assert stage('team').options.get('scope', DEFAULT_SCOPE) == DEFAULT_SCOPE and 'balanced' not in stage('team').options


def test_the_dataset_stage_groups_are_the_names_the_trainers_write():
    for (name, scope), arch_balance in itertools.product(DATASET_STAGES.items(), (False, True)):
        arms = dict(stage(name).arms)
        label = 'team sig500v3 emb (Model 3)' + (' ab' if arch_balance else '')
        assert arms[label] == tp.team_config(1, 2, 'sig500v3-emb', *scope, arch_balance=arch_balance).group
        assert arms[label].count('gnn-ab_team-') == int(arch_balance)
        l512 = f"Model 1 five-channel L512{' ab' if arch_balance else ''}"
        assert arms[l512] == RunConfig(protocol=scope[0], fold=0, arch=scope[1], model='fivechannel', max_instructions=512,
                                       arch_balance=arch_balance).group
    mendeley = dict(stage('team_mendeley_all').arms)
    assert mendeley['team sig500v3 feat (Model 2)'] == 'kfold_all_gnn_team-sig500v3-feat'
    assert mendeley['team sigwhole emb (Model 3)'] == 'kfold_all_gnn_team-sigwhole-emb'
    assert mendeley['Model 1 five-channel L512 whole'] == 'kfold_all_fivechannel-L512_whole' and mendeley['TF-IDF'] == 'tfidf'
    assert dict(stage('team_v3_matched').arms)['team callwalk500 e2e (Model 3, trained encoder)'] == 'kfold_v3_matched_gnn_team-callwalk500-e2e'
    published = dict(stage('team_v3_all').arms)
    assert published['R7 typed ab'] == 'kfold_v3_all_gnn-mblk15-ab-selcell' == RunConfig(
        protocol='kfold_v3', fold=0, arch='all', model='gnn', mask_blocks=0.15, select_on='cell', arch_balance=True).group
    assert published['R2 no-edges ab'] == 'kfold_v3_all_gnn-noedges-mblk15-ab-selcell'
    assert published['R4 shuffle ab'] == 'kfold_v3_all_gnn-shuffle-mblk15-ab-selcell' and published['R0 TF-IDF ab'] == 'tfidf-ab'


def test_the_dataset_stage_pairs_are_the_comparisons_of_the_plan():
    pairs = set(stage('team_mendeley_all').pairs)
    inputs = ('feat (Model 2)', 'emb (Model 3)', 'e2e (Model 3, trained encoder)')
    for name in inputs:
        assert ('TF-IDF', f'team sig250 {name}') in pairs
        assert (f'team sig500v3 {name}', f'team callwalk500 {name}') in pairs and (f'team sig250 {name}', f'team callwalk250 {name}') in pairs
        assert (f'team sig500v3 {name}', f'team sig250 {name}') in pairs and (f'team callwalk500 {name}', f'team callwalk250 {name}') in pairs
        assert (f'team sig500v3 {name}', f'team sig500v3 {name} ab') in pairs
    assert ('team sig500v3 feat (Model 2)', 'team sig500v3 emb (Model 3)') in pairs        # feat -> emb
    assert ('team sig500v3 emb (Model 3)', 'team sig500v3 e2e (Model 3, trained encoder)') in pairs        # emb -> e2e
    assert ('team sig500v3 feat (Model 2)', 'team sigwhole feat (Model 2)') in pairs        # 500 blocks -> uncapped
    assert ('Model 1 five-channel L512', 'Model 1 five-channel L512 whole') in pairs and ('Model 1 five-channel L128', 'Model 1 five-channel L512 ab'
                                                                                          ) not in pairs
    assert ('Model 1 five-channel L512', 'Model 1 five-channel L512 ab') in pairs and ('TF-IDF', 'Model 1 five-channel L128 ab') in pairs
    v3 = set(stage('team_v3_all').pairs)
    for reference in ('R7 typed ab', 'R2 no-edges ab', 'R0 TF-IDF ab'):
        for arm in ('team sig500v3 emb (Model 3)', 'team callwalk500 e2e (Model 3, trained encoder)', 'Model 1 five-channel L512',
                    'team sig500v3 feat (Model 2) ab'):
            assert (reference, arm) in v3
    assert not [p for p in v3 if p[0] == 'R7 typed' or p[1] == 'R7 typed']                  # the plain GNN arms are not on disk: no pair
    assert set(stage('team_v3_matched').pairs) == v3 and [a for a, _ in stage('team_v3_matched').arms] == [a for a, _ in stage('team_v3_all').arms]


@pytest.mark.parametrize('plan,stage_names', [('team_mendeley_all', ('team_mendeley_all',)), ('team_v3', ('team_v3_all', 'team_v3_matched'))])
def test_a_dataset_plan_trains_fifteen_runs_of_each_group_its_stages_read(plan, stage_names):
    groups = plan_groups(plan)
    assert all(runs == {(f, s) for f in range(5) for s in range(3)} for runs in groups.values())
    read = {g for name in stage_names for _, g in stage(name).arms}
    assert set(groups) <= read                                                      # every trained group is a row of a stage
    unread = read - set(groups)
    if plan == 'team_mendeley_all':
        assert unread == {'tfidf'} and len(groups) == 23
    else:                                                                           # the published rows, and the 250-block builds that are not part of the plan
        assert all(g in ('tfidf', 'tfidf-ab') or '-mblk15' in g or 'sig250' in g or 'callwalk250' in g for g in unread)
        assert len(groups) == 26 and {g.split('_')[2] for g in groups} == {'all', 'matched'}
    assert not [g for g in groups if g.startswith('kfold_x86')]                       # no run of a plan lands on a name of the x86 runs


def synthetic_v3_runs(tmp_path, monkeypatch):
    """The published v3 rows (build_v3_ab_fake), plus two team arms of arch all: sig500v3-emb stronger than the ab R7 arm and
    sig500v3-feat weaker, five folds of seed 0 each."""
    build_v3_ab_fake(tmp_path, monkeypatch)
    for fold in range(5):
        write_v3_run(tmp_path, 'all', fold, 2.2, tag='team-sig500v3-emb')
        write_v3_run(tmp_path, 'all', fold, 0.9, tag='team-sig500v3-feat')
    gate_tables.pooled.cache_clear()


def table_rows(text, heading, until):
    """{(first cells...)} of the table lines between two headings, keyed by their first two cells."""
    part = text.split(heading)[1].split(until)[0]
    return {tuple(c.strip() for c in l.split('|')[1:3]): [c.strip() for c in l.split('|')[3:-1]] for l in part.splitlines() if l.startswith('| ')}


def test_the_cells_table_the_scope_and_the_balanced_pairs_on_a_synthetic_run_directory(tmp_path, monkeypatch):
    synthetic_v3_runs(tmp_path, monkeypatch)
    text = '\n'.join(gates.render(stage('team_v3_all'), 40))
    gate_tables.pooled.cache_clear()
    emb, ab = 'team sig500v3 emb (Model 3)', 'R7 typed ab'
    pooled_rows = table_rows(text, '## Pooled, AUC checkpoint', '## Per-architecture')
    assert not pooled_rows[('R0 TF-IDF', 'tfidf')][0].startswith('missing')               # scope kfold_v3 all reaches the baseline rows
    assert not pooled_rows[(emb, 'kfold_v3_all_gnn_team-sig500v3-emb')][0].startswith('missing')
    assert pooled_rows[('team sig250 emb (Model 3)', 'kfold_v3_all_gnn_team-sig250-emb')][0].startswith('missing')
    assert gate_tables.pooled('tfidf', 'auc', ('kfold_v3', 'all'))[0] > 0.5
    assert gate_tables.pooled('tfidf', 'auc').startswith('missing')                 # the default scope is kfold x86, whose baselines are not here
    gate_tables.pooled.cache_clear()
    cells = table_rows(text, '## Per-architecture recalls', '## Paired differences, second minus first')
    header = next(l for l in text.splitlines() if l.startswith('| arm | group | AUC | PR-AUC | macro-F1 | R x86 recall'))
    assert header.endswith('| G x64 recall | cell-bal AUC | cell-bal macro-F1 |')
    row = cells[(emb, 'kfold_v3_all_gnn_team-sig500v3-emb')]
    complete, _ = stats.load_group('kfold_v3_all_gnn_team-sig500v3-emb', 'auc', ('kfold_v3', 'all'))
    p = complete[0]
    expected = [float(np.mean(p.pred[(p.y == label) & (p.arch == arch)] == label)) for label, arch in ((1, 'x86'), (1, 'x64'), (0, 'x86'), (0, 'x64'))]
    assert len(row) == 9 and row[3:7] == [f'{v:.3f}' for v in expected] and all(not c.startswith('missing') for c in row)
    assert cells[('floor: x86_rule', '-')][3:7] == ['1.000', '0.000', '0.000', '1.000'] and ('floor: majority', '-') in cells
    assert 'x86-rule macro-F1 per fold: 0.690, 0.690, 0.690, 0.690, 0.690 (pooled 0.690).' in text
    ordinary = table_rows(text, '## Paired differences, second minus first', '## Paired differences, cell-balanced')
    balanced = table_rows(text, '## Paired differences, cell-balanced', '### Cell-balanced differences, detail')
    key = (ab, emb)
    assert ordinary[key][0] != 'missing' and balanced[key][0] != 'missing' and ordinary[key] != balanced[key]
    assert ordinary[key][-1].startswith(('beats', 'does not beat')) and balanced[key][-1].startswith(('beats', 'does not beat'))
    assert ordinary[('R0 TF-IDF ab', emb)][0] != 'missing'                         # a baseline in a pair is read for the stage's dataset
    assert ordinary[(ab, 'team sig500v3 feat (Model 2)')][0] != 'missing' and ordinary[(emb, 'team sig500v3 e2e (Model 3, trained encoder)')][0] == 'missing'
    assert text.count('Cell-balanced: every test sample weighted') >= 2 and 'excluded from both arms' not in text


def test_a_stage_without_scope_or_balance_prints_neither_extra_table(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    gate_tables.pooled.cache_clear()
    text = '\n'.join(gates.render(stage('team'), 20))
    gate_tables.pooled.cache_clear()
    assert '## Paired differences, cell-balanced' not in text and '## Per-architecture' not in text


def test_the_cells_table_prints_missing_floors_when_the_cache_is_not_built(tmp_path, monkeypatch):
    def no_cache(*args, **kwargs):
        raise FileNotFoundError('no cache')
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    monkeypatch.setattr(report, 'manifest_floors', no_cache)
    for cached in (gate_tables.pooled, report.v3_floors, report.kfold_floors):
        cached.cache_clear()
    for name in DATASET_STAGES:
        text = '\n'.join(gates.render(stage(name), 20))
        assert f'Floors: missing (no cache for protocol {DATASET_STAGES[name][0]}).' in text and 'Traceback' not in text
    for cached in (gate_tables.pooled, report.v3_floors, report.kfold_floors):
        cached.cache_clear()


EXCLUDED = {f'x{i}': 'num_instructions > 4000000' for i in range(34)}
MATCH_CASES = [(('kfold_v3', 'all'), {'processed': 2, 'total': 4, 'excluded': {'x': 'r', 'y': 'r'}}, 4, True),
               (('kfold_v3', 'all'), {'processed': 2, 'total': 4, 'excluded': {'x': 'r'}}, 4, False),
               (('kfold', 'all'), {'processed': 4, 'total': 4}, 4, True),
               (DEFAULT_SCOPE, {'processed': 4, 'total': 4}, None, False),                           # the x86 builds hold 1810
               (('kfold', 'all'), {'processed': 2411, 'total': 2411}, None, True),
               (('kfold', 'all'), {'processed': 1810, 'total': 1810}, None, False),                 # the summary of the x86 build a --extend is still extending
               (('kfold_v3', 'matched'), {'processed': 5618, 'total': 5652, 'excluded': EXCLUDED}, None, True)]      # a matched run reads the all build


@pytest.mark.parametrize('scope,summary,total,complete', MATCH_CASES)
def test_the_match_rate_line_counts_the_total_of_the_builds_dataset(tmp_path, scope, summary, total, complete):
    fake_build(tmp_path / 'b', {'r1': fake_payload([1, 1, 1, 0], 1), 'g1': fake_payload([1, 1, 1, 1], 0)})
    (tmp_path / 'b' / 'build_summary.json').write_text(json.dumps(summary), encoding='utf-8')
    line = gate_tables.match_rate_line(tmp_path / 'b', 'b_test', scope, total)
    assert line.startswith('Match rates of `b_test` (share of') == complete
    if complete:
        assert 'ransomware 1 binaries, mean 0.750' in line and 'goodware 1 binaries, mean 1.000' in line
    else:
        assert 'missing (build incomplete' in line


def test_a_tfidf_run_without_a_validation_threshold_gives_the_same_reason_in_the_pooled_and_the_cells_table(tmp_path, monkeypatch):
    synthetic_v3_runs(tmp_path, monkeypatch)
    for fold in range(5):                       # the kfold arch all baselines on disk: no threshold_val, so no AUC-checkpoint scores
        write_v3_baseline(tmp_path / 'other', 'all', fold, 1.3, threshold=None)
        (tmp_path / 'other' / 'baselines' / f'v3kfold{fold}_all_s0').rename(tmp_path / 'baselines' / f'kfold{fold}_all_s0')
    gate_tables.pooled.cache_clear()
    reason = 'missing (baseline has no validation threshold)'
    assert gate_tables.pooled('tfidf', 'auc', ('kfold', 'all')) == reason
    assert gate_tables.cell_row('tfidf', ('kfold', 'all')) == reason
    assert not isinstance(gate_tables.cell_row('tfidf', ('kfold_v3', 'all')), str)           # the kfold_v3 baselines have one
    gate_tables.pooled.cache_clear()
