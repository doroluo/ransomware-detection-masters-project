"""python -m pytest tests/test_team_m1frozen.py: the build guard of the `-m1frozen` arms (`train_b.check_same_api_vocab`,
`load_frozen_model1`): API ids are numbered per build, so the per-block Model 1 that is frozen must have trained on the arm's build.
Synthetic builds and Model 1 runs only. Below the guard: the plans `team_m1frozen.txt` and `team_v3_m1frozen.txt` (run names, order,
builds, collisions) and the gate rows and pairs of the four stages that read them.

The guard needs torch and Janice's `Model1TokenSequence`, not torch_geometric; the one test that runs the arm end to end on a
synthetic build does and skips with a reason without it."""
import json
import re
import shutil
from pathlib import Path

import pytest
import torch

from common.config import RunConfig
from common.manifests import Manifest
from experiments import queue as q
from experiments.team_pipelines import common as tp, fit, train_b
from experiments.team_pipelines.build_b import resolve_dir
from s04_evaluation import gates
from s03a_transformer.five_channel.model import Model1TokenSequence, tok
from test_team_pipelines import needs_pyg
from test_team_v3 import N_API, synthetic_build
from tiny_cache import tiny  # noqa: F401  (a fixture)

QUIET = lambda *_: None                                                                       # noqa: E731
NAMES = [f'lib.dll!Fn{k}' for k in range(7)]
IDS = {'train': ['a', 'b', 'c'], 'val': ['d'], 'test': ['e', 'f']}


def manifest():
    return Manifest(*[[{'sample_id': i} for i in IDS[s]] for s in ('train', 'val', 'test')], ['fam'])


def vocab_build(root, names):
    root.mkdir(parents=True, exist_ok=True)
    (root / 'api_vocab.json').write_text(json.dumps(names), encoding='utf-8')
    return root


def model1_run(root, build, rows=None, names=NAMES, build_dir=True, per_block=True, splits=IDS):
    """A finished per-block Model 1 run: config.json, manifest.json, metrics.json (`build_dir`), best_auc.pt with `rows` API rows."""
    root.mkdir(parents=True, exist_ok=True)
    rows = tok.FIRST_API_ID + len(names) if rows is None else rows
    (root / 'config.json').write_text(json.dumps({'per_block': per_block, 'max_instructions': train_b.BLOCK_LEN}), encoding='utf-8')
    (root / 'manifest.json').write_text(json.dumps({**splits, 'val_families': ['fam']}), encoding='utf-8')
    (root / 'metrics.json').write_text(json.dumps({'build_dir': str(build)} if build_dir else {}), encoding='utf-8')
    torch.save(Model1TokenSequence(api_vocab_size=rows, max_seq_len=train_b.BLOCK_LEN).state_dict(), root / 'best_auc.pt')
    return root


def refused(run, arm_build, match, smoke=None):
    with pytest.raises(SystemExit, match=match) as e:
        train_b.load_frozen_model1(run, manifest(), smoke, QUIET, b_dir=arm_build)
    return str(e.value)


def test_a_model1_of_the_same_build_loads_frozen(tmp_path):
    build = vocab_build(tmp_path / 'b', NAMES)
    run = model1_run(tmp_path / 'm1', build)
    net, source = train_b.load_frozen_model1(run, manifest(), None, QUIET, b_dir=build)
    assert source == str(run / 'best_auc.pt') and not net.training and not any(p.requires_grad for p in net.parameters())
    assert net.embedder.api_embedding.weight.shape[0] == tok.FIRST_API_ID + len(NAMES)
    assert train_b.check_same_api_vocab(run, build) == (build, tok.FIRST_API_ID + len(NAMES))


def test_a_copy_of_the_build_is_the_same_build_because_the_vocabulary_is_equal(tmp_path):
    """The build is compared by the content of its api_vocab.json, not its path: a build moved or copied (the Drive zip) is the same."""
    run = model1_run(tmp_path / 'm1', vocab_build(tmp_path / 'one', NAMES))
    net, source = train_b.load_frozen_model1(run, manifest(), None, QUIET, b_dir=vocab_build(tmp_path / 'two', list(NAMES)))
    assert source and not net.training


def test_a_run_name_does_not_say_the_build_so_another_vocabulary_is_refused_naming_both_builds_and_the_fix(tmp_path):
    arm_build = vocab_build(tmp_path / 'sig500', NAMES)
    m1_build = vocab_build(tmp_path / 'sig250', NAMES[:3] + ['other.dll!X'] + NAMES[4:])         # 6 of the 7 positions agree
    run = model1_run(tmp_path / 'm1', m1_build)
    message = refused(run, arm_build, 'trained on the build')
    assert str(m1_build) in message and str(arm_build) in message                           # both builds
    assert '--model fivechannel --per-block --max-instructions 64' in message and f'--build {arm_build}' in message
    assert '--model1-run DIR' in message and '6 of the 7 positions' in message
    assert 'so the frozen encoder would read the wrong API embeddings' in message and 'extends' not in message


def test_a_vocabulary_in_another_order_is_refused(tmp_path):
    arm_build = vocab_build(tmp_path / 'a', NAMES)
    run = model1_run(tmp_path / 'm1', vocab_build(tmp_path / 'b', NAMES[::-1]))
    refused(run, arm_build, 'API ids are numbered per build')


@pytest.mark.parametrize('longer_is_the_arm', [True, False])
def test_a_vocabulary_that_only_extends_the_other_is_refused(tmp_path, longer_is_the_arm):
    """`build_b --extend` appends names, so the old ids agree. The Model 1 of the shorter build has no row for the new ids (an index
    error on the first x64 id), and a Model 1 of the longer one holds rows the arm's build does not: both are refused. The
    extended build is another dataset (kfold all against x86) whose folds `check_same_splits` refuses anyway."""
    short, long = vocab_build(tmp_path / 'short', NAMES[:4]), vocab_build(tmp_path / 'long', NAMES)
    arm_build, m1_build = (long, short) if longer_is_the_arm else (short, long)
    message = refused(model1_run(tmp_path / 'm1', m1_build, names=json.loads((m1_build / 'api_vocab.json').read_text())), arm_build,
                      'trained on the build')
    n_theirs, n_ours = (4, 7) if longer_is_the_arm else (7, 4)
    assert f"one build's vocabulary extends the other's ({n_theirs} vs {n_ours} names)" in message
    assert "trained on another dataset's build" in message and 'positions they share' not in message
    assert 'wrong API embeddings' not in message
    assert f'--build {arm_build}' in message and '--model1-run DIR' in message                    # the fix advice is kept


def test_a_model1_run_without_a_recorded_build_is_refused(tmp_path):
    build = vocab_build(tmp_path / 'b', NAMES)
    message = refused(model1_run(tmp_path / 'm1', build, build_dir=False), build, 'does not record the build it trained on')
    assert 'no build_dir in its metrics.json' in message and '--build <the arm' in message
    run = model1_run(tmp_path / 'm2', build)
    (run / 'metrics.json').unlink()                                                      # no metrics.json at all
    refused(run, build, 'does not record the build it trained on')


def test_a_build_that_is_missing_or_has_no_vocabulary_cannot_be_compared_and_is_refused(tmp_path):
    build = vocab_build(tmp_path / 'b', NAMES)
    refused(model1_run(tmp_path / 'gone', tmp_path / 'not-here'), build, r'Model 1 build .*not-here.* has no api_vocab.json')
    bare = tmp_path / 'bare'
    bare.mkdir()
    refused(model1_run(tmp_path / 'm1', build), bare, r'arm build .*bare.* has no api_vocab.json')
    refused(model1_run(tmp_path / 'm2', bare), build, r'Model 1 build .*bare.* has no api_vocab.json')


def test_a_checkpoint_with_more_api_rows_than_the_arm_build_can_hold_is_refused(tmp_path):
    build = vocab_build(tmp_path / 'b', NAMES)
    run = model1_run(tmp_path / 'm1', build, rows=tok.FIRST_API_ID + len(NAMES) + 1)       # same vocabulary file, one id beyond it
    message = refused(run, build, 'more than the')
    assert f'{tok.FIRST_API_ID + len(NAMES) + 1} API embedding rows' in message and f'the {tok.FIRST_API_ID + len(NAMES)} ids' in message
    train_b.load_frozen_model1(model1_run(tmp_path / 'ok', build), manifest(), None, QUIET, b_dir=build)      # exactly the capacity loads


def test_the_older_checks_still_come_first(tmp_path):
    build = vocab_build(tmp_path / 'b', NAMES)
    refused(model1_run(tmp_path / 'pb', build, per_block=False), build, 'is not a per-block Model 1 at 64')
    refused(model1_run(tmp_path / 'sp', build, splits={**IDS, 'test': ['e']}), build, "its test split \\(1 ids\\) is not this run's")
    with pytest.raises(SystemExit, match='does not exist: train the per-block Model 1'):
        train_b.load_frozen_model1(tmp_path / 'empty', manifest(), None, QUIET, b_dir=build)


def test_the_smoke_path_without_a_checkpoint_is_unchanged_random_weights_and_no_build_check(tmp_path):
    lines = []
    net, source = train_b.load_frozen_model1(tmp_path / 'none', manifest(), 8, lines.append, b_dir=tmp_path / 'no-such-build')
    assert source is None and not net.training and any('random weights' in l for l in lines)


@needs_pyg
def test_the_arm_runs_on_a_synthetic_build_with_its_model1_and_refuses_another_vocabulary(tmp_path, monkeypatch, tiny):   # noqa: F811
    rows = tp.sample_rows()
    folder = synthetic_build(tmp_path / 'b', sorted(rows), rows, source='main-fix', sample='signal', num_blocks=500)
    names = json.loads((folder / 'api_vocab.json').read_text())
    assert len(names) == N_API
    full = tp.fold_manifest(2, 1, 'kfold', 'x86')
    run = model1_run(tmp_path / 'm1', folder, names=names)
    full.save(run / 'manifest.json')
    monkeypatch.setattr(fit, 'RUNS', tmp_path / 'runs')
    monkeypatch.setattr(fit, 'write_run', lambda *a, **k: pytest.fail('a smoke run must not call write_run'))
    lines = []
    out = train_b.run('sig500v3-m1frozen', 2, 1, smoke=6, device='cpu', b_dir=folder, log=lines.append, model1_run=run)
    assert any(f'frozen Model 1 encoder: {run / "best_auc.pt"}' in l for l in lines) and len(out['history']) == 2
    shutil.rmtree(out['dir'])
    other = vocab_build(tmp_path / 'other', names[::-1])                                  # a build of another vocabulary
    run_other = model1_run(tmp_path / 'm1-other', other, names=names)
    full.save(run_other / 'manifest.json')
    with pytest.raises(SystemExit, match='API ids are numbered per build') as e:
        train_b.run('sig500v3-m1frozen', 2, 1, smoke=6, device='cpu', b_dir=folder, log=QUIET, model1_run=run_other)
    assert str(other) in str(e.value) and str(folder) in str(e.value)


# the plans ------------------------------------------------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
PLAN_DATASETS = {      # plan -> [(protocol, arch, arch_balance)] in plan order
    'team_m1frozen': [('kfold', 'x86', False), ('kfold', 'all', False), ('kfold', 'all', True)],
    'team_v3_m1frozen': [('kfold_v3', 'all', False), ('kfold_v3', 'all', True), ('kfold_v3', 'matched', False),
                         ('kfold_v3', 'matched', True)]}
ARM = 'sig500v3-m1frozen'
FROZEN = 'team sig500v3 m1frozen (Model 3, frozen Model 1)'
BLOCK = 'Model 1 five-channel L64 per block'
STAGE_SIZES = {'team': (28, 46), 'team_mendeley_all': (24, 53), 'team_v3_all': (27, 85), 'team_v3_matched': (27, 85)}   # arms, pairs before


def model1_cfg(fold, seed, protocol, arch, ab):
    return RunConfig(protocol=protocol, fold=fold, arch=arch, seed=seed, model='fivechannel', max_instructions=64, per_block=True,
                     arch_balance=ab)


def expected_names(plan):
    """[(per-block Model 1 run name, arm run name)] in plan order: per dataset, seed by seed, fold by fold."""
    return [(model1_cfg(f, s, *dataset).name, tp.team_config(f, s, ARM, *dataset).name)
            for dataset in PLAN_DATASETS[plan] for s in range(3) for f in range(5)]


def all_labels(plan):
    return [s.label for s in q.load_plan(plan) if s.verb in q.TRAIN]


def group_of(name):
    """The group of a run name (the fold and seed stripped): kfold0_all_gnn-ab_s1_team-x -> kfold_all_gnn-ab_team-x."""
    return re.sub(r'^(v3)?kfold\d_', lambda m: 'kfold_v3_' if m[1] else 'kfold_', re.sub(r'_s\d(_|$)', r'\1', name))


def arg(step, flag, default=None):
    return step.args[step.args.index(flag) + 1] if flag in step.args else default


@pytest.mark.parametrize('plan,groups,runs', [('team_m1frozen', 6, 90), ('team_v3_m1frozen', 8, 120)])
def test_a_plan_trains_each_folds_per_block_model1_right_before_its_frozen_arm(plan, groups, runs):
    steps = q.load_plan(plan)
    train = [s for s in steps if s.verb in q.TRAIN]
    assert len(train) == runs and len({s.label for s in train}) == runs and len(PLAN_DATASETS[plan]) * 2 == groups
    assert [s.label for s in train] == [n for pair in expected_names(plan) for n in pair]       # Model 1, then its arm, fold by fold
    assert len({group_of(s.label) for s in train}) == groups
    for model1, arm in zip(train[::2], train[1::2]):
        assert model1.verb == 'train' and arm.verb == 'train_b'
        assert '--per-block' in model1.args and arg(model1, '--max-instructions') == '64' and arg(model1, '--model') == 'fivechannel'
        assert arg(arm, '--arm') == ARM and '--build' not in arm.args
        protocol = arg(model1, '--protocol')
        arch = arg(model1, '--arch', tp.DEFAULT_ARCH[protocol])
        assert train_b.model1_run_dir(int(arg(model1, '--fold')), int(arg(model1, '--seed')), protocol, arch,
                                      '--arch-balance' in model1.args).name == model1.label       # the run the arm looks for by name
        assert [arg(arm, f) for f in ('--fold', '--seed')] == [arg(model1, f) for f in ('--fold', '--seed')]
        assert ('--arch-balance' in arm.args) == ('--arch-balance' in model1.args) and arg(arm, '--arch') == arg(model1, '--arch')


@pytest.mark.parametrize('plan', PLAN_DATASETS)
def test_the_model1_of_a_plan_line_trains_on_the_build_its_arm_reads_by_default(plan):
    """The guard compares the Model 1's build with the arm's: the `--build` of each Model 1 line is the default build of its arm."""
    folder = {('kfold', 'x86'): 'b_sig500_v3', ('kfold', 'all'): 'b_sig500_mendeley_all', ('kfold_v3', 'all'): 'b_sig500_combined_v3',
              ('kfold_v3', 'matched'): 'b_sig500_combined_v3'}
    for step in (s for s in q.load_plan(plan) if s.verb == 'train'):
        protocol = arg(step, '--protocol')
        arch = arg(step, '--arch', tp.DEFAULT_ARCH[protocol])
        arm_dir = train_b.ARM_DIR[ARM] if (protocol, arch) == ('kfold', 'x86') else \
            train_b.TEAM_RUNS / f'b_{train_b.build_family(ARM)}_{train_b.BUILD_SUFFIX[protocol]}'
        assert resolve_dir(Path(arg(step, '--build'))) == arm_dir and arm_dir.name == folder[protocol, arch]


def test_the_plans_end_in_the_summaries_of_the_stages_that_read_them():
    ends = lambda plan: [(s.verb, s.args) for s in q.load_plan(plan)][-3:]                      # noqa: E731
    assert ends('team_m1frozen') == [('report', ()), ('gates', ('team',)), ('gates', ('team_mendeley_all',))]
    assert ends('team_v3_m1frozen') == [('report', ()), ('gates', ('team_v3_all',)), ('gates', ('team_v3_matched',))]


def test_the_plan_recordings_are_the_dry_runs_and_the_plans_collide_with_no_other_run_name():
    """Full run names, not substrings: the x86 names of `team_m1frozen` sit next to the names of the new datasets."""
    names = {plan: set(all_labels(plan)) for plan in PLAN_DATASETS}
    mine = names['team_m1frozen'] | names['team_v3_m1frozen']
    assert len(names['team_m1frozen']) == 90 and len(names['team_v3_m1frozen']) == 120 and len(mine) == 210
    token = re.compile(r'[\w-]*kfold[\w-]*')                  # whole tokens of the fixtures of the other plans and tests
    others = set()
    for fixture in (HERE / 'fixtures').rglob('*'):
        if fixture.is_file() and fixture.suffix in ('.txt', '.json', '.csv', '.md') and fixture.stem not in PLAN_DATASETS:
            others |= set(token.findall(fixture.read_text(encoding='utf-8', errors='replace')))
    assert not mine & others, sorted(mine & others)[:3]
    for plan in PLAN_DATASETS:
        lines = (HERE / 'fixtures' / 'plans_dry' / f'{plan}.txt').read_text(encoding='utf-8').splitlines()
        assert [l.split()[1] for l in lines] == all_labels(plan) and all(l.startswith('run ') for l in lines)
    for plan in (p.stem for p in q.PLANS.glob('*.txt') if p.stem not in PLAN_DATASETS):          # the other plans, run by run
        assert not set(all_labels(plan)) & mine, plan
    datasets = {d for ds in PLAN_DATASETS.values() for d in ds}                                  # every other arm on the same datasets
    other_arms = {tp.team_config(f, s, arm, p, a, ab).name for arm in train_b.ARMS if not arm.endswith('-m1frozen')
                  for p, a, ab in datasets for f in range(5) for s in range(3)
                  if not (ab and a == 'x86') and not (arm.startswith('sigwhole') and p != 'kfold')}
    assert other_arms and not other_arms & mine
    assert all(re.search(r'team-sig500v3-m1frozen$|fivechannel-L64-pb(-ab)?_s\d$', n) for n in mine)


# the stages -----------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize('name,scope', [('team', ('kfold', 'x86')), ('team_mendeley_all', ('kfold', 'all')),
                                        ('team_v3_all', ('kfold_v3', 'all')), ('team_v3_matched', ('kfold_v3', 'matched'))])
def test_each_stage_gains_the_rows_and_pairs_of_the_frozen_arm_and_keeps_the_old_ones_first(name, scope):
    stage = gates.load_stage(name)
    arms, pairs, rows = dict(stage.arms), list(stage.pairs), [label for label, _ in stage.arms]
    variants = [False] if name == 'team' else [False, True]               # the x86 stage has no -ab twins
    old_arms, old_pairs = STAGE_SIZES[name]
    added = ['Model 1 five-channel L128', BLOCK, FROZEN] if name == 'team' else [BLOCK, FROZEN, BLOCK + ' ab', FROZEN + ' ab']
    assert len(rows) == old_arms + len(added) and rows[old_arms:] == added                # appended after the old rows
    assert len(pairs) == old_pairs + 3 * len(variants) and len(set(pairs)) == len(pairs)
    assert not [p for p in pairs[:old_pairs] if {FROZEN, BLOCK, FROZEN + ' ab', BLOCK + ' ab'} & set(p)]      # the old pairs are the first ones
    assert not [l for l in rows[:old_arms] if 'm1frozen' in l or 'per block' in l]
    for ab in variants:
        suffix = ' ab' if ab else ''
        pb_group = model1_cfg(0, 0, *scope, ab).group
        assert arms[BLOCK + suffix] == pb_group and pb_group.endswith('fivechannel-L64-pb' + ('-ab' if ab else ''))
        assert arms[FROZEN + suffix] == tp.team_config(1, 2, ARM, *scope, ab).group
        emb, e2e = f'team sig500v3 emb (Model 3){suffix}', f'team sig500v3 e2e (Model 3, trained encoder){suffix}'
        l128 = 'Model 1 five-channel L128' + suffix
        assert pairs[old_pairs + 3 * variants.index(ab):][:3] == [(emb, FROZEN + suffix), (FROZEN + suffix, e2e), (l128, BLOCK + suffix)]
        assert arms[l128] == RunConfig(protocol=scope[0], fold=0, arch=scope[1], model='fivechannel', arch_balance=ab).group
    assert 'frozen Model 1' in stage.intro_text(40)


def test_every_group_the_two_plans_train_is_a_row_of_a_stage_that_reads_them():
    read = {'team_m1frozen': {g for n in ('team', 'team_mendeley_all') for _, g in gates.load_stage(n).arms},
            'team_v3_m1frozen': {g for n in ('team_v3_all', 'team_v3_matched') for _, g in gates.load_stage(n).arms}}
    for plan, groups in (('team_m1frozen', 6), ('team_v3_m1frozen', 8)):
        trained = {group_of(label) for label in all_labels(plan)}
        assert len(trained) == groups and trained <= read[plan], sorted(trained - read[plan])
