"""python -m pytest tests/test_team_builds.py: the arms of the three builds of this repository's own files (`sig250`, `callwalk500`,
`callwalk250`, each with `-feat`, `-emb` and `-e2e`), their settings check, run names, the rows and pairs of `gates --stage team`,
and the plan `experiments/plans/team_builds.txt`.

The arm table, the refusals, the gates and the plan need no torch_geometric; the smoke runs on synthetic builds do and skip with a
reason without it."""
import json
import re
from pathlib import Path

import pytest

from experiments import queue as q
from experiments.stages import team as team_stage
from experiments.stages.team import team_group
from experiments.team_pipelines import common as tp, fit, train_b
from s04_evaluation import gate_tables, gates, stats
from test_team_pipelines import fake_group, fresh_team_caches, needs_pyg, pinned_team_modules  # noqa: F401
from test_team_v2 import fake_build, fake_payload
from test_team_v3 import settings, synthetic_build
from tiny_cache import tiny  # noqa: F401  (a fixture)

# arm family -> (build folder, sampler, block budget)
BUILDS = {'sig250': ('b_sig250_main', 'signal', 250), 'callwalk500': ('b_callwalk500_main', 'callwalk', 500),
          'callwalk250': ('b_callwalk250_main', 'callwalk', 250)}
INPUTS = {'feat': 'feat', 'emb': 'emb', 'e2e': 'e2e'}
NEW_ARMS = tuple(f'{family}-{suffix}' for family in BUILDS for suffix in INPUTS)
QUIET = lambda *_: None                                                                       # noqa: E731
HERE = Path(__file__).resolve().parent


def test_the_arm_rows_name_build_sampler_budget_source_input_and_recipe():
    assert len(NEW_ARMS) == 9 and set(NEW_ARMS) <= set(train_b.ARMS)
    for family, (folder, sampler, blocks) in BUILDS.items():
        for suffix, features in INPUTS.items():
            row = train_b.ARM_TABLE[f'{family}-{suffix}']
            assert train_b.ARM_DIR[f'{family}-{suffix}'].name == folder and train_b.ARM_DIR[f'{family}-{suffix}'].parent.name == 'team'
            assert (row.sampler, row.num_blocks, row.features, row.sources, row.legacy) == (sampler, blocks, features, ('repo',), False)
            assert row.hp is (train_b.HP_E2E if suffix == 'e2e' else train_b.HP)
            assert train_b.ARM_FEATURES[f'{family}-{suffix}'] == features
    assert 'sig250' not in train_b.ARMS and 'callwalk500' not in train_b.ARMS            # no `all` arm for the new builds
    assert train_b.ARMS.index('sig500v3-e2e') == 5 and train_b.ARMS[-3:] == ('sigwhole', 'sigwhole-feat', 'sigwhole-emb')


def test_the_new_run_names_collide_with_nothing_known():
    names = {tp.team_config(f, s, arm).name for arm in NEW_ARMS for f in range(5) for s in range(3)}
    assert len(names) == 135 and all(re.fullmatch(r'kfold\d_x86_gnn_s\d_team-(sig250|callwalk500|callwalk250)-(feat|emb|e2e)', n) for n in names)
    groups = {team_group(a) for a in NEW_ARMS} | {team_group(a) for a in train_b.ARMS if a not in NEW_ARMS}
    assert len(groups) == len(train_b.ARMS)                                                # no group shared with an older arm
    for fixture in (HERE / 'fixtures').rglob('*'):
        if fixture.is_file() and fixture.name != 'team_builds.txt' and fixture.suffix in ('.txt', '.json', '.csv', '.md'):
            text = fixture.read_text(encoding='utf-8', errors='replace')
            assert not [n for n in ('team-sig250', 'team-callwalk') if n in text], fixture
    runs = stats.RUNS
    if Path(runs).is_dir():
        assert not names & {p.name for p in Path(runs).iterdir()}


def test_each_new_arm_refuses_a_build_of_another_sampler_budget_or_source(tmp_path):
    def refused(arm, d, message):
        with pytest.raises(SystemExit, match=message) as e:
            train_b.run(arm, 0, 0, smoke=2, device='cpu', b_dir=d, log=QUIET)
        return str(e.value)
    for family, (_, sampler, blocks) in BUILDS.items():
        other = 'callwalk' if sampler == 'signal' else 'signal'
        for suffix in INPUTS:
            arm = f'{family}-{suffix}'
            good = settings(tmp_path / arm, source='repo', sample=sampler, num_blocks=blocks)
            assert train_b.check_build(arm, good) == 'repo'
            message = refused(arm, settings(tmp_path / f'{arm}-s', source='repo', sample=other, num_blocks=blocks), f'sampler {other}')
            assert f'sampler {sampler}, num_blocks {blocks}, source repo' in message             # all three named
            refused(arm, settings(tmp_path / f'{arm}-n', source='repo', sample=sampler, num_blocks=250 if blocks == 500 else 500),
                    f'num_blocks {250 if blocks == 500 else 500}')
            refused(arm, settings(tmp_path / f'{arm}-u', source='repo', sample=sampler, num_blocks=1000000), 'num_blocks 1000000')
            refused(arm, settings(tmp_path / f'{arm}-m', source='main-fix', sample=sampler, num_blocks=blocks), 'was built with main-fix')
            refused(arm, settings(tmp_path / f'{arm}-k', sample=sampler, num_blocks=blocks), 'gnn-331ce0b')
            with pytest.raises(SystemExit, match='reads builds of repo, not --source main-fix'):
                train_b.run(arm, 0, 0, smoke=2, device='cpu', b_dir=good, source='main-fix', log=QUIET)
    assert train_b.check_build('sig500v3-emb', settings(tmp_path / 'v3', source='repo', sample='signal', num_blocks=500)) == 'repo'
    with pytest.raises(SystemExit, match='sampler signal, num_blocks 500'):                     # the old arms are not widened
        train_b.check_build('sig500v3-emb', settings(tmp_path / 'cw', source='repo', sample='callwalk', num_blocks=500))


def test_the_e2e_arms_take_only_their_block_length_and_the_others_none():
    with pytest.raises(SystemExit, match='reads no instructions'):
        train_b.run('callwalk500-emb', 0, 0, smoke=2, device='cpu', block_len=64, log=QUIET)
    with pytest.raises(SystemExit, match='trains with --block-len 64, not 32'):
        train_b.run('sig250-e2e', 0, 0, smoke=2, device='cpu', block_len=32, log=QUIET)
    with pytest.raises(SystemExit, match='trains on --features e2e, not emb'):
        train_b.run('callwalk250-e2e', 0, 0, smoke=2, device='cpu', features='emb', log=QUIET)


def test_the_gate_rows_pairs_and_match_rate_builds_of_the_new_arms():
    stage = gates.load_stage('team')
    arms = dict(stage.arms)
    names = [f'team {family} {name}' for family in BUILDS for name in ('feat (Model 2)', 'emb (Model 3)', 'e2e (Model 3, trained encoder)')]
    for name, arm in zip(names, NEW_ARMS):
        assert arms[name] == team_group(arm)
    assert names[0] == 'team sig250 feat (Model 2)' and set(names) <= set(arms)
    pairs = set(stage.pairs)
    for sampler_pair in (('500', 'sig500v3'), ('250', 'sig250')):
        for name in ('feat (Model 2)', 'emb (Model 3)', 'e2e (Model 3, trained encoder)'):
            assert (f'team {sampler_pair[1]} {name}', f'team callwalk{sampler_pair[0]} {name}') in pairs
    for first, second in (('sig500v3', 'sig250'), ('callwalk500', 'callwalk250')):
        for name in ('feat (Model 2)', 'emb (Model 3)', 'e2e (Model 3, trained encoder)'):
            assert (f'team {first} {name}', f'team {second} {name}') in pairs
    assert len(stage.pairs) == len(pairs) and len(team_stage.SAMPLER) == 6 and len(team_stage.BUDGET) == 6
    assert not [p for p in pairs if (p[0] in names or p[1] in names) and p not in team_stage.SAMPLER + team_stage.BUDGET]   # only those twelve
    assert [m[0] for m in stage.options['match_rate_builds']] == ['b_sig500_v3', 'b_sigwhole_v3', 'b_sig250_main', 'b_callwalk500_main',
                                                                   'b_callwalk250_main']
    assert [m[1] for m in stage.options['match_rate_builds']][2:] == ['team/b_sig250_main', 'team/b_callwalk500_main', 'team/b_callwalk250_main']


def test_the_existing_gate_rows_and_pairs_are_unchanged():
    stage = gates.load_stage('team')
    old_arms = [(n, g) for n, g in stage.arms if n not in team_stage.NEW]
    assert [n for n, _ in old_arms][:17] == [
        'TF-IDF', 'header-imports LR', 'R7 typed, whole graph', 'R2 no-edges, whole graph', 'ours b500 typed', 'ours b500 no-edges',
        'team gnn500', 'team cb500', 'team sig500', 'team sig500v2', 'team sig500v3', 'team sig500v3 feat (Model 2)',
        'team sig500v3 emb (Model 3)', 'team sig500v3 e2e (Model 3, trained encoder)', 'team sigwhole', 'team sigwhole feat (Model 2)',
        'team sigwhole emb (Model 3)'] and len(old_arms) == 19
    old_pairs = [p for p in stage.pairs if p not in team_stage.SAMPLER + team_stage.BUDGET]
    assert len(old_pairs) == len(stage.pairs) - 12 == 34 and old_pairs == list(stage.pairs)[:34]      # the 34 pairs of before, in order
    assert not [p for p in old_pairs if p[0] in team_stage.NEW or p[1] in team_stage.NEW]


def test_team_gates_pair_callwalk_with_signal_and_250_with_500(tiny, tmp_path, monkeypatch):     # noqa: F811
    runs = tmp_path / 'runs'
    runs.mkdir()
    for arm, signal in (('sig500v3-emb', 1.0), ('sig250-emb', 0.8), ('callwalk500-emb', 1.2), ('callwalk250-emb', 1.0)):
        fake_group(runs, lambda f, s, arm=arm: tp.team_config(f, s, arm), signal, seeds=(0, 1, 2))
    monkeypatch.setattr(stats, 'RUNS', runs)
    gate_tables.pooled.cache_clear()
    (runs / 'team').mkdir()
    build = runs / 'team' / 'b_callwalk500_main'
    fake_build(build, {'r1': fake_payload([0] * 10, 1), 'g1': fake_payload([1] * 8 + [0] * 2, 0)})
    (build / 'build_summary.json').write_text(json.dumps({'processed': 1810, 'total': 1810}), encoding='utf-8')
    text = '\n'.join(gates.run_stage('team', 40))
    gate_tables.pooled.cache_clear()
    pair = {(l.split('|')[1].strip(), l.split('|')[2].strip()): l.split('|')[-2].strip()
            for l in text.splitlines() if l.startswith('| ') and l.count('|') == 6}
    emb = 'emb (Model 3)'
    assert pair[(f'team sig500v3 {emb}', f'team callwalk500 {emb}')] in ('beats', 'does not beat')
    assert pair[(f'team sig250 {emb}', f'team callwalk250 {emb}')] in ('beats', 'does not beat')
    assert pair[(f'team sig500v3 {emb}', f'team sig250 {emb}')] != 'missing'
    assert pair[(f'team callwalk500 {emb}', f'team callwalk250 {emb}')] != 'missing'
    lines = [l for l in text.splitlines() if l.startswith('Match rates of `b_')]
    assert [l.split('`')[1] for l in lines] == ['b_sig500_v3', 'b_sigwhole_v3', 'b_sig250_main', 'b_callwalk500_main', 'b_callwalk250_main']
    assert 'ransomware 1 binaries, mean 0.000' in lines[3] and 'goodware 1 binaries, mean 0.800' in lines[3]
    assert lines[2].startswith('Match rates of `b_sig250_main`: missing (no build_summary.json')


def test_the_plan_has_the_train_b_lines_of_the_nine_arms_then_report_and_gates():
    steps = q.load_plan('team_builds')
    train = [s for s in steps if s.verb == 'train_b']
    assert len(train) == 135 and [s.verb for s in steps[-2:]] == ['report', 'gates'] and len(steps) == 137
    seen = {}
    for s in train:
        args = list(s.args)
        arm, fold, seed = args[args.index('--arm') + 1], int(args[args.index('--fold') + 1]), int(args[args.index('--seed') + 1])
        assert '--build' not in args and s.label == tp.team_config(fold, seed, arm).name
        seen.setdefault(arm, set()).add((fold, seed))
    assert set(seen) == set(NEW_ARMS) and all(v == {(f, s) for f in range(5) for s in range(3)} for v in seen.values())
    assert [dict(zip(('b', 'i'), (s.label.split('team-')[1].rsplit('-', 1)[0], s.label.split('-')[-1]))) for s in train][::15] == [
        {'b': family, 'i': suffix} for family in BUILDS for suffix in INPUTS]                          # build by build, 15 runs each


# smoke runs on synthetic builds ---------------------------------------------------------------------------------------------


@needs_pyg
@pytest.mark.parametrize('arm,sampler,blocks,in_dim', [('sig250-feat', 'signal', 250, 77), ('callwalk500-emb', 'callwalk', 500, 64),
                                                       ('callwalk250-e2e', 'callwalk', 250, 64)])
def test_a_new_arm_smoke_runs_on_a_synthetic_build_of_its_sampler(tmp_path, monkeypatch, tiny, arm, sampler, blocks, in_dim):   # noqa: F811
    rows = tp.sample_rows()
    folder = synthetic_build(tmp_path / 'b', sorted(rows), rows, source='repo', sample=sampler, num_blocks=blocks)
    monkeypatch.setattr(fit, 'RUNS', tmp_path / 'runs')
    monkeypatch.setattr(fit, 'write_run', lambda *a, **k: pytest.fail('a smoke run must not call write_run'))
    lines = []
    out = train_b.run(arm, 2, 1, smoke=6, device='cpu', b_dir=folder, log=lines.append)
    assert any(f'in_dim {in_dim}, edge_dim 8' in l for l in lines) and len(out['history']) == 2
    other = 'signal' if sampler == 'callwalk' else 'callwalk'
    with pytest.raises(SystemExit, match=f'sampler {other}'):
        train_b.run(arm, 2, 1, smoke=6, device='cpu', b_dir=synthetic_build(tmp_path / 'c', sorted(rows), rows, source='repo',
                                                                              sample=other, num_blocks=blocks), log=QUIET)
    __import__('shutil').rmtree(out['dir'])
