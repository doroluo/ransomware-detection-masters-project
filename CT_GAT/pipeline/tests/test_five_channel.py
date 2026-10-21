"""python -m pytest tests/test_five_channel.py: Janice Park's five-channel Model 1 (`s03a_transformer/five_channel`), its registry entry
`fivechannel`, its run names and a 2-epoch run on a build of 40 synthetic `.pt` files. CPU only; the folds come from the small
synthetic cache of tests/tiny_cache.py, the build is written here in the layout of `experiments/team_pipelines/build_b.py`."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from tiny_cache import tiny  # noqa: F401  (a fixture)

import experiments.train as experiments_train
from common.config import MAX_INSTRUCTIONS, RunConfig
from common.manifests import load_index, make_manifest
from common.paths import source_for
from common.registry import MODELS, spec_for
from s03a_transformer.five_channel import model as five
from s03a_transformer.five_channel import train as fc
from s03a_transformer.five_channel.spec import FiveChannelSpec
from s04_evaluation import gate_tables, report, stats

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]
tok = five.tok


def fivechannel(fold=0, **kw):
    return RunConfig(**{'protocol': 'kfold', 'fold': fold, 'arch': 'x86', 'model': 'fivechannel', 'epochs': 2,
                        'num_workers': 0, **kw})


def write_build(root, rows, seed=0, n_api=6, skip=()):
    """`<sample_id>.pt` in the layout of a pipeline-B build for each row of the tiny cache: instruction_ids [N, 5] (opcode,
    operand 1, operand 2, api, behaviour), label and file_id. The API ids are NO_API, UNK_API or FIRST_API_ID + 0..n_api - 1."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    for r in rows:
        if r['sample_id'] in skip:
            continue
        n = int(rng.integers(5, 300))
        ids = np.stack([rng.integers(1, 256, n), rng.integers(0, tok.FIRST_API_ID, n), rng.integers(0, tok.FIRST_API_ID, n),
                        rng.choice([tok.NO_API, tok.UNK_API] + [tok.FIRST_API_ID + k for k in range(n_api)], n),
                        rng.integers(0, tok.BEHAVIOR_VOCAB_SIZE, n)], axis=1)
        ids[:, 0] = np.where(r['label'] == 1, ids[:, 0] % 100 + 1, ids[:, 0] % 100 + 100)       # a signal for the model to find
        torch.save({'file_id': r['sample_id'], 'label': int(r['label']), 'instruction_ids': torch.as_tensor(ids, dtype=torch.long)},
                   root / f"{r['sample_id']}.pt")
    return root


@pytest.fixture(scope='module')
def build(tiny, tmp_path_factory):  # noqa: F811
    return write_build(tmp_path_factory.mktemp('build') / 'b', tiny.rows)


# ---- her file ----------------------------------------------------------------------------------------------------------

def test_janices_classes_are_unchanged_and_the_demo_is_still_there():
    for name in ('FiveChannelEmbedder', 'Model1TokenSequence', 'Model1Dataset', 'model1_collate_fn'):
        assert hasattr(five, name)
    text = Path(five.__file__).read_text(encoding='utf-8')
    assert 'Janice Park' in text and 'if __name__ == "__main__":' in text
    net = five.Model1TokenSequence(api_vocab_size=tok.FIRST_API_ID + 4, d_model=16, nhead=2, num_layers=1, max_seq_len=10)
    batch = five.model1_collate_fn([{'instruction_ids': torch.randint(0, 5, (7, 5)), 'label': torch.tensor(1)},
                                    {'instruction_ids': torch.randint(0, 5, (12, 5)), 'label': torch.tensor(0)}], max_instructions=10)
    assert batch['instruction_ids'].shape == (2, 10, 5) and batch['padding_mask'].sum(1).tolist() == [3, 0]
    assert net(batch['instruction_ids'], padding_mask=batch['padding_mask']).shape == (2, 2)


def test_the_module_imports_stage_2_through_its_own_folder():
    assert Path(tok.__file__).resolve() == (ROOT / 's02_preprocess' / 'tokenize_asm_gz.py').resolve()
    assert five.OPCODE_VOCAB_SIZE == 256 and five.OPERAND_VOCAB_SIZE == tok.FIRST_API_ID


# ---- the registry and the names ----------------------------------------------------------------------------------------------

def test_fivechannel_is_registered():
    assert MODELS['fivechannel'] == 's03a_transformer.five_channel.spec:FiveChannelSpec'
    assert isinstance(spec_for(fivechannel()), FiveChannelSpec)
    assert 'fivechannel' in experiments_train.MODEL_OPTIONS and 'fivechannel' in MODELS


def test_fivechannel_names_carry_the_non_defaults_only():
    assert fivechannel().name == 'kfold0_x86_fivechannel_s0' and fivechannel().group == 'kfold_x86_fivechannel'
    assert fivechannel(max_instructions=MAX_INSTRUCTIONS).name == fivechannel().name
    assert fivechannel(2, max_instructions=512, seed=1).name == 'kfold2_x86_fivechannel-L512_s1'
    assert fivechannel(tag='t').name == 'kfold0_x86_fivechannel_s0_t'


@pytest.mark.parametrize('kw', [dict(seq_len=256), dict(max_windows=8), dict(mask_blocks=0.1), dict(control='no_edges'),
                                dict(val_seed=3), dict(protocol='kfold_v3'), dict(arch='all'), dict(max_instructions=0)])
def test_fivechannel_refuses_what_does_not_apply(kw):
    with pytest.raises(ValueError):
        fivechannel(**kw)


def test_max_instructions_belongs_to_fivechannel():
    with pytest.raises(ValueError, match="max_instructions needs model='fivechannel'"):
        RunConfig(protocol='kfold', fold=0, model='tokenseq', max_instructions=256)
    with pytest.raises(ValueError, match="max_instructions needs model='fivechannel'"):
        RunConfig(protocol='kfold', fold=0, model='gnn', max_instructions=256)


def test_the_command_line_takes_the_flags_of_the_model():
    cfg, out, options = experiments_train.parse_args(['--model', 'fivechannel', '--protocol', 'kfold', '--fold', '2', '--seed', '1',
                                                      '--max-instructions', '512', '--build', 'b', '--device', 'cpu', '--smoke', '8'])
    assert cfg == fivechannel(2, seed=1, max_instructions=512, epochs=40, num_workers=2) and out is None
    assert options == {'build': 'b', 'device': 'cpu', 'smoke': 8}
    for argv in (['--export', 'x'], ['--window-budget', '4']):
        with pytest.raises(SystemExit):
            experiments_train.parse_args(['--model', 'fivechannel', '--protocol', 'kfold', '--fold', '0', *argv])
    for model in ('gnn', 'tokenseq'):
        with pytest.raises(SystemExit):
            experiments_train.parse_args(['--model', model, '--protocol', 'kfold', '--fold', '0', '--build', 'b'])


def test_the_build_comes_from_the_flag_then_the_environment_then_datasets_json(tmp_path, monkeypatch):
    monkeypatch.delenv('GT_TEAM_B_BUILD', raising=False)
    monkeypatch.setenv('GT_DATASETS', str(tmp_path / 'none.json'))
    with pytest.raises(FileNotFoundError, match='GT_TEAM_B_BUILD'):
        fc.resolve_build()
    with pytest.raises(FileNotFoundError, match='not a directory'):
        fc.resolve_build(tmp_path / 'nowhere')
    (tmp_path / 'b1').mkdir()
    assert fc.resolve_build(tmp_path / 'b1') == tmp_path / 'b1'
    monkeypatch.setenv('GT_TEAM_B_BUILD', str(tmp_path / 'b1'))
    assert fc.resolve_build() == tmp_path / 'b1' == source_for('team_b_build')
    monkeypatch.delenv('GT_TEAM_B_BUILD')
    datasets = tmp_path / 'datasets.json'
    datasets.write_text(json.dumps({'mine': {'cache': 'c', 'protocols': ['kfold'], 'sources': {'team_b_build': 'b1'}}}), encoding='utf-8')
    monkeypatch.setenv('GT_DATASETS', str(datasets))
    assert source_for('team_b_build') == tmp_path / 'b1'


# ---- the API vocabulary of a fold ----------------------------------------------------------------------------------------------

def binary(api, label=0, n_extra=3):
    """A Binary whose instructions use the API ids `api` (in the fourth channel)."""
    ids = torch.zeros((len(api) + n_extra, 5), dtype=torch.long)
    ids[:len(api), 3] = torch.tensor(api, dtype=torch.long)
    ids[:, 0] = 7
    return fc.Binary(ids, torch.unique(ids[:, 3][ids[:, 3] >= tok.FIRST_API_ID]), len(ids), label)


def test_the_api_remap_touches_only_ids_unseen_in_training():
    first = tok.FIRST_API_ID
    binaries = {'a': binary([first, first + 1, tok.UNK_API]), 'b': binary([first + 1, tok.NO_API]),
                'v': binary([first, first + 5, tok.UNK_API, tok.NO_API]), 't': binary([first + 1, first + 9])}
    seen = fc.seen_api_ids(binaries, ['a', 'b'])
    assert seen.tolist() == [first, first + 1]
    items = fc.make_items(binaries, ['a', 'b', 'v', 't'], seen)
    got = {i: items[i]['instruction_ids'][:, 3].tolist() for i in items}
    assert got['a'] == binaries['a'].head[:, 3].tolist() and got['b'] == binaries['b'].head[:, 3].tolist()      # training: untouched
    assert got['v'][:4] == [first, tok.UNK_API, tok.UNK_API, tok.NO_API]             # first + 5 was not in training
    assert got['t'][:2] == [first + 1, tok.UNK_API]
    for i in items:                                                    # no other channel and no other row changes
        assert torch.equal(items[i]['instruction_ids'][:, [0, 1, 2, 4]], binaries[i].head[:, [0, 1, 2, 4]])
        assert items[i]['instruction_ids'].shape == binaries[i].head.shape
    assert binaries['v'].head[1, 3] == first + 5                       # the loaded tensor itself is not edited


def test_the_remap_is_the_one_of_train_b():
    from experiments.team_pipelines.train_b import remap_api, train_api_ids
    binaries = {'a': binary([80, 81, 76]), 'b': binary([81, 90])}
    seen = fc.seen_api_ids(binaries, ['a'])
    payloads = {i: {'instruction_ids': b.head} for i, b in binaries.items()}
    assert torch.equal(seen, train_api_ids(tok, payloads, ['a']))
    assert torch.equal(fc.make_items(binaries, ['b'], seen)['b']['instruction_ids'], remap_api(tok, binaries['b'].head, seen))


def test_load_binaries_reads_the_build_through_her_dataset(build, tiny):  # noqa: F811
    labels = {r['sample_id']: r['label'] for r in tiny.rows[:6]}
    got = fc.load_binaries(Path(build), labels, 16)
    assert set(got) == set(labels) and all(len(b.head) <= 16 and b.head.shape[1] == 5 for b in got.values())
    sid = tiny.rows[0]['sample_id']
    full = torch.load(Path(build) / f'{sid}.pt', weights_only=False)['instruction_ids']
    assert got[sid].n_instructions == len(full) and torch.equal(got[sid].head, full[:16])
    assert set(got[sid].api.tolist()) == set(full[:, 3][full[:, 3] >= tok.FIRST_API_ID].tolist())
    with pytest.raises(ValueError, match='label'):
        fc.load_binaries(Path(build), {sid: 1 - labels[sid]}, 16)
    with pytest.raises(FileNotFoundError, match='no <sample_id>.pt'):
        fc.load_binaries(Path(build), {'f' * 64: 0}, 16)


def test_an_unfinished_build_is_refused(tiny, tmp_path):  # noqa: F811
    write_build(tmp_path, tiny.rows[:4])
    (tmp_path / 'checkpoint.json').write_text(json.dumps({'done': [tiny.rows[0]['sample_id']]}), encoding='utf-8')
    with pytest.raises(FileNotFoundError, match='checkpoint.json'):
        fc.load_binaries(tmp_path, {r['sample_id']: r['label'] for r in tiny.rows[:2]}, 16)


# ---- a run -----------------------------------------------------------------------------------------------------------------------

def test_a_two_epoch_run_writes_the_run_directory_the_harness_reads(tiny, build, tmp_path, monkeypatch):  # noqa: F811
    runs = tmp_path / 'runs'
    for fold in range(5):
        metrics = fc.run(fivechannel(fold, max_instructions=32), build, 'cpu', None, runs, log=lambda *_: None)
    cfg = fivechannel(0, max_instructions=32)
    metrics = json.loads((runs / cfg.name / 'metrics.json').read_text(encoding='utf-8'))
    d = runs / 'kfold0_x86_fivechannel-L32_s0'
    for name in ('config.json', 'manifest.json', 'history.csv', 'best.pt', 'best_auc.pt', 'predictions.csv', 'predictions_auc.csv',
                 'metrics.json'):
        assert (d / name).is_file(), name
    saved = RunConfig.load(d / 'config.json')
    assert saved == cfg and saved.name == d.name
    on_disk = json.loads((d / 'metrics.json').read_text(encoding='utf-8'))
    assert on_disk['run'] == d.name and on_disk['epochs_run'] == 2 and set(on_disk['selection']) == {'f1', 'auc'}
    assert on_disk['max_instructions'] == 32 and set(on_disk['truncated_instructions']) == {'train', 'val', 'test'}
    # the folds are ours: the manifest of make_manifest for this config, the predictions cover its validation and test binaries
    want = make_manifest(cfg, load_index())
    manifest = json.loads((d / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['train'] == [r['sample_id'] for r in want.train] and manifest['test'] == [r['sample_id'] for r in want.test]
    with (d / 'predictions.csv').open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert {r['sample_id'] for r in rows if r['split'] == 'test'} == set(manifest['test'])
    assert {r['sample_id'] for r in rows if r['split'] == 'val'} == set(manifest['val'])
    thr = on_disk['selection']['auc']['threshold_val']
    with (d / 'predictions_auc.csv').open(encoding='utf-8') as f:
        assert all(int(r['pred']) == int(float(r['prob']) >= thr) for r in csv.DictReader(f))
    # stats.py, report.py and the gate tables read it
    monkeypatch.setattr(stats, 'RUNS', runs)
    monkeypatch.setattr(report, 'RUNS', runs)
    for selection in ('f1', 'auc'):
        complete, partial = stats.load_group(cfg.group, selection)
        assert sorted(complete) == [0] and not partial and len(complete[0].y) == 40
        result, info = stats.compare(cfg.group, cfg.group, selection, resamples=20)
        assert info['seeds'] == [0] and result['auc']['diff'] == 0
    gate_tables.pooled.cache_clear()
    auc, _, macro_f1 = gate_tables.pooled(cfg.group, 'auc')
    gate_tables.pooled.cache_clear()
    assert 0 <= auc <= 1 and 0 <= macro_f1 <= 1
    report.main(['--out', str(tmp_path / 'RESULTS.md')])
    text = (tmp_path / 'RESULTS.md').read_text(encoding='utf-8')
    line = next(l for l in text.splitlines() if l.startswith('| Model 1 five-channel L32 '))
    assert '| kfold | x86 | 1 |' in line and "'max_instructions'" not in line and metrics['run'] == d.name


def test_the_out_of_fold_threshold_changes_the_threshold_and_nothing_else(tiny, build, tmp_path):  # noqa: F811
    runs = tmp_path / 'runs'
    default = fc.run(fivechannel(1, max_instructions=24, epochs=3), build, 'cpu', None, runs, log=lambda *_: None)
    oof = fc.run(fivechannel(1, max_instructions=24, epochs=3, threshold_from='oof'), build, 'cpu', None, runs, log=lambda *_: None)
    d, o = runs / 'kfold1_x86_fivechannel-L24_s0', runs / 'kfold1_x86_fivechannel-L24-thoof_s0'
    assert 'threshold_rule' not in default
    rule = oof['threshold_rule']
    assert rule['rule'] == 'oof' and rule['k'] == 3 and rule['refit_epochs'] == {'f1': oof['best_epoch'], 'auc': oof['auc_best_epoch']}
    assert rule['threshold_val']['auc'] == default['selection']['auc']['threshold_val']
    assert oof['selection']['auc']['threshold_val'] == rule['threshold']['auc'] and 'out-of-fold' in oof['selection']['auc']['rule']
    key = lambda path: [(r['sample_id'], r['split'], r['prob']) for r in csv.DictReader(path.open(encoding='utf-8'))]
    assert key(d / 'predictions.csv') == key(o / 'predictions.csv') and key(d / 'predictions_auc.csv') == key(o / 'predictions_auc.csv')
    thr = rule['threshold']['auc']
    with (o / 'predictions_auc.csv').open(encoding='utf-8') as f:
        assert all(int(r['pred']) == int(float(r['prob']) >= thr) for r in csv.DictReader(f))


def test_the_registry_hands_the_run_over_and_names_it(tiny, build, tmp_path, monkeypatch):  # noqa: F811
    from s03a_transformer.five_channel import spec as spec_module
    monkeypatch.setattr(fc, 'RUNS', tmp_path)
    monkeypatch.setattr(spec_module, 'RUNS', tmp_path)
    monkeypatch.setattr(experiments_train, 'RUNS', tmp_path)
    cfg = fivechannel(1, max_instructions=24)
    experiments_train.run(cfg, 'named', build=str(build), device='cpu')
    assert (tmp_path / 'named' / 'metrics.json').is_file() and not list(tmp_path.glob('.stage-*'))
    assert RunConfig.load(tmp_path / 'named' / 'config.json') == cfg
    with pytest.raises(TypeError, match='fivechannel takes'):
        FiveChannelSpec().run(cfg, None, export='x')


def test_a_smoke_run_writes_only_a_temporary_directory(tiny, build, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(fc, 'RUNS', tmp_path / 'runs')
    metrics = fc.run(fivechannel(2, max_instructions=16), build, 'cpu', 6, log=lambda *_: None)
    assert metrics['smoke'] == 6 and metrics['epochs_run'] == 2 and not (tmp_path / 'runs').exists()
    with pytest.raises(ValueError, match='never writes under'):
        fc.run(fivechannel(2), build, 'cpu', 6, tmp_path / 'runs', log=lambda *_: None)


def test_a_binary_the_build_left_out_is_out_of_every_split_and_recorded(tiny, tmp_path):  # noqa: F811
    cfg = fivechannel(0, max_instructions=16)
    manifest = make_manifest(cfg, load_index())
    left_out = manifest.test[0]['sample_id']
    root = write_build(tmp_path / 'b', tiny.rows, skip={left_out})
    (root / 'settings.json').write_text(json.dumps({'excluded': {left_out: 'listing too large'}}), encoding='utf-8')
    metrics = fc.run(cfg, root, 'cpu', None, tmp_path / 'runs', log=lambda *_: None)
    assert metrics['excluded'] == [left_out] and metrics['excluded_reason'] == 'listing too large'
    with (tmp_path / 'runs' / cfg.name / 'predictions.csv').open(encoding='utf-8') as f:
        assert left_out not in {r['sample_id'] for r in csv.DictReader(f)}


def test_the_command_line_of_the_module_runs_a_smoke(tiny, build):  # noqa: F811
    cfg, opts = fc.parse_config(['--fold', '3', '--max-instructions', '512', '--build', 'b', '--smoke', '4'])
    assert cfg.name == 'kfold3_x86_fivechannel-L512_s0' and opts == {'build': 'b', 'device': None, 'smoke': 4}
    with pytest.raises(SystemExit):
        fc.parse_config(['--fold', '9'])
    r = subprocess.run([sys.executable, '-m', 's03a_transformer.five_channel.train', '--help'], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and '--max-instructions' in r.stdout
