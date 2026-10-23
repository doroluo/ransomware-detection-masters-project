"""python -m pytest tests/test_team_m1frozen.py: the build guard of the `-m1frozen` arms (`train_b.check_same_api_vocab`,
`load_frozen_model1`): API ids are numbered per build, so the per-block Model 1 that is frozen must have trained on the arm's build.
Synthetic builds and Model 1 runs only; the run-name, plan and gate tests of the arm are further down.

The guard needs torch and Janice's `Model1TokenSequence`, not torch_geometric; the one test that runs the arm end to end on a
synthetic build does and skips with a reason without it."""
import json
import shutil

import pytest
import torch

from common.manifests import Manifest
from experiments.team_pipelines import common as tp, fit, train_b
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
    refused(model1_run(tmp_path / 'm1', m1_build, names=json.loads((m1_build / 'api_vocab.json').read_text())), arm_build,
            'API ids are numbered per build')


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
