"""python -m pytest tests/test_harness_prep.py: step 23, harness prep for Model 1 (plan S1): write_run lifted to
evaluation/run_io.py, the GT_RUNS / GT_EXPORT overrides of common/paths.py, and RunConfig model 'tokenseq'.

CPU only; no test needs the real caches (the existing team tests cover write_run on the real folds)."""
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from common import paths
from common.config import MAX_WINDOWS, SEQ_LEN, RunConfig
from evaluation import run_io
from experiments.team_pipelines import common as team
from tiny_cache import tiny  # noqa: F401  (a fixture)
from token_sequence import train as ts_train
from training.train import check_supported, parse_config

ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = [p for p in ROOT.rglob('*.py') if not {'tests', '.git'} & set(p.relative_to(ROOT).parts)]


# ---- run_io ----------------------------------------------------------------------------------------------------------

def test_write_run_lives_in_run_io_and_the_team_module_re_exports_it():
    assert team.write_run is run_io.write_run and team._check_cover is run_io._check_cover
    assert run_io.write_run.__module__ == 'evaluation.run_io'
    assert 'def write_run' not in (ROOT / 'experiments' / 'team_pipelines' / 'common.py').read_text(encoding='utf-8')
    for name in ('evaluate', 'val_threshold_of', 'truncated_count', 'write_predictions', 'AUC_RULES', 'fold_manifest'):
        assert hasattr(team, name), name                        # what the team trainers and their tests import from here


def fake_rows(manifest, rng):
    out = {}
    for split in ('val', 'test'):
        rows = getattr(manifest, split)
        probs = np.clip(0.5 + 0.8 * (np.array([r['label'] for r in rows]) - 0.5) + rng.normal(0, 0.2, len(rows)), 0.01, 0.99)
        out[split] = ([r['sample_id'] for r in rows], probs, [(3, 5, 7)] * len(rows))
    return out


def test_write_run_with_the_default_manifest_equals_the_one_passed_in(tiny, tmp_path):  # noqa: F811
    from common.manifests import load_index, make_manifest
    cfg = RunConfig(protocol='kfold', fold=1, seed=2, model='gnn', mask_blocks=0.15, tag='t')
    manifest = make_manifest(cfg, load_index())
    rows = fake_rows(manifest, np.random.default_rng(0))
    extra = {'best_epoch': 2, 'epochs_run': 5, 'auc_best_epoch': 3, 'val_auc': 0.9, 'val_macro_f1': 0.8}
    a = run_io.write_run(tmp_path / 'a' / cfg.name, cfg, rows, rows, 0.4, 0.45, extra)
    b = run_io.write_run(tmp_path / 'b' / cfg.name, cfg, rows, rows, 0.4, 0.45, extra, manifest=manifest)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)                  # (NaN != NaN, so not ==)
    for name in ('config.json', 'manifest.json', 'metrics.json', 'predictions.csv', 'predictions_auc.csv'):
        assert (tmp_path / 'a' / cfg.name / name).read_bytes() == (tmp_path / 'b' / cfg.name / name).read_bytes()
    with pytest.raises(ValueError, match='RunConfig name'):
        run_io.write_run(tmp_path / 'wrong', cfg, rows, rows, 0.4, 0.45, extra)
    with pytest.raises(AssertionError):
        run_io.write_run(tmp_path / 'c' / cfg.name, cfg, {**rows, 'test': tuple(x[:-1] for x in rows['test'])}, rows, 0.4,
                         0.45, extra)
    with pytest.raises(KeyError):
        run_io.write_run(tmp_path / 'd' / cfg.name, cfg, rows, rows, 0.4, 0.45, {'best_epoch': 1})


# ---- GT_RUNS and GT_EXPORT -------------------------------------------------------------------------------------------

def probe(env, code):
    """Run `code` in a fresh interpreter with GT_RUNS / GT_EXPORT / GT_CACHE as in `env` (and no others)."""
    clean = {k: v for k, v in os.environ.items() if k not in ('GT_RUNS', 'GT_EXPORT', 'GT_CACHE')}
    r = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, env={**clean, **env})
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


PROBE = '''
import json
from common import paths
from common.config import EXPORT, RUNS
from evaluation import report, stats
from experiments.team_pipelines import common as team
from transformer_paths import block_export as be
print(json.dumps({"runs": str(paths.RUNS), "export": str(paths.EXPORT), "config_runs": str(RUNS), "config_export": str(EXPORT),
                  "stats": str(stats.RUNS), "report": str(report.RUNS), "team": str(team.TEAM_RUNS),
                  "be_export": str(be.EXPORT), "be_out": str(be.resolve_out("RUNS/export/x")),
                  "cache": str(paths.CACHE)}))
'''


def test_defaults_are_unchanged():
    got = probe({}, PROBE)
    assert got['runs'] == got['config_runs'] == got['stats'] == got['report'] == str(Path('C:/Users/chaoa/Downloads/graph_transformer_runs'))
    assert got['export'] == got['config_export'] == got['be_export'] == str(Path(got['runs']) / 'export')
    assert got['team'] == str(Path(got['runs']) / 'team') and got['cache'] == str(paths.DEFAULT_CACHE)
    assert paths.runs_for({}) == paths.DEFAULT_RUNS
    assert paths.export_for({}) == paths.runs_for({}) / 'export'


def test_gt_runs_moves_the_runs_and_the_default_export(tmp_path):
    got = probe({'GT_RUNS': str(tmp_path / 'r')}, PROBE)
    assert Path(got['runs']) == Path(got['config_runs']) == Path(got['stats']) == Path(got['report']) == tmp_path / 'r'
    assert Path(got['export']) == Path(got['config_export']) == Path(got['be_export']) == tmp_path / 'r' / 'export'
    assert Path(got['team']) == tmp_path / 'r' / 'team'
    assert Path(got['be_out']) == (tmp_path / 'r' / 'export' / 'x').resolve()
    assert got['cache'] == str(paths.DEFAULT_CACHE)                    # GT_CACHE is a separate switch


def test_gt_export_moves_the_exports_only(tmp_path):
    got = probe({'GT_EXPORT': str(tmp_path / 'e')}, PROBE)
    assert Path(got['export']) == Path(got['config_export']) == Path(got['be_export']) == tmp_path / 'e'
    assert Path(got['runs']) == paths.runs_for({}) and Path(got['be_out']) == (tmp_path / 'e' / 'x').resolve()
    both = probe({'GT_RUNS': str(tmp_path / 'r'), 'GT_EXPORT': str(tmp_path / 'e')}, PROBE)
    assert Path(both['runs']) == tmp_path / 'r' and Path(both['export']) == tmp_path / 'e'


def test_the_env_functions_take_an_explicit_environment(tmp_path):
    assert paths.runs_for({'GT_RUNS': str(tmp_path)}) == tmp_path and paths.runs_for({'GT_RUNS': ''}) == paths.DEFAULT_RUNS
    assert paths.export_for({'GT_RUNS': str(tmp_path)}) == tmp_path / 'export'
    assert paths.export_for({'GT_RUNS': str(tmp_path), 'GT_EXPORT': str(tmp_path / 'x')}) == tmp_path / 'x'
    assert paths.cache_for('kfold', {'GT_RUNS': str(tmp_path)}) == paths.DEFAULT_CACHE


def test_resolve_out_follows_gt_export(tmp_path):
    out = probe({'GT_EXPORT': str(tmp_path / 'e')}, '''
import json
from transformer_paths import block_export as be
refused = []
for bad in ("RUNS/other/x", "RUNS/export", "x"):
    try:
        be.resolve_out(bad)
    except ValueError:
        refused.append(bad)
print(json.dumps({"ok": str(be.resolve_out("RUNS/export/ours_cap0")), "refused": refused}))
''')
    assert Path(out['ok']) == (tmp_path / 'e' / 'ours_cap0').resolve() and out['refused'] == ['RUNS/other/x', 'RUNS/export', 'x']


def test_no_module_spells_the_runs_or_export_path_itself():
    """Only common/paths.py names the runs directory; no module builds `RUNS / 'export'`."""
    literal = [p.relative_to(ROOT).as_posix() for p in CODE_FILES
               if "Path('C:/Users/chaoa/Downloads/graph_transformer_runs" in p.read_text(encoding='utf-8')]
    assert literal == ['common/paths.py']
    assert [p.relative_to(ROOT).as_posix() for p in CODE_FILES if "RUNS / 'export'" in p.read_text(encoding='utf-8')] == []


def test_the_trainer_reads_its_run_directory_from_gt_runs(tmp_path):
    """token_sequence.train writes under RUNS; with GT_RUNS set that is the temporary directory."""
    out = probe({'GT_RUNS': str(tmp_path)}, '''
import json
from token_sequence import train
from common.config import RUNS
print(json.dumps({"runs": str(RUNS), "module": str(train.RUNS)}))
''')
    assert Path(out['runs']) == Path(out['module']) == tmp_path


# ---- RunConfig model 'tokenseq' --------------------------------------------------------------------------------------

def tok(**kw):
    return RunConfig(**{'protocol': 'kfold', 'fold': 0, 'arch': 'x86', 'model': 'tokenseq', **kw})


def test_tokenseq_names():
    assert tok().name == 'kfold0_x86_tokenseq_s0' and tok().group == 'kfold_x86_tokenseq'
    assert [tok(fold=f, seed=s).name for f, s in ((0, 0), (4, 1), (2, 2))] == [
        'kfold0_x86_tokenseq_s0', 'kfold4_x86_tokenseq_s1', 'kfold2_x86_tokenseq_s2']
    assert tok(seq_len=256).name == 'kfold0_x86_tokenseq-L256_s0'
    assert tok(max_windows=32).name == 'kfold0_x86_tokenseq-w32_s0'
    assert tok(seq_len=256, max_windows=32, seed=2).name == 'kfold0_x86_tokenseq-L256-w32_s2'
    assert tok(seq_len=SEQ_LEN, max_windows=MAX_WINDOWS).name == tok().name == 'kfold0_x86_tokenseq_s0'       # defaults: no suffix
    assert (SEQ_LEN, MAX_WINDOWS) == (512, 64)
    assert tok(tag='t').name == 'kfold0_x86_tokenseq_s0_t'


def test_tokenseq_val_seed_is_forced_to_the_export_carve_and_never_printed():
    for seed in (0, 1, 2):
        cfg = tok(seed=seed)
        assert cfg.val_seed == 0 and 'vs' not in cfg.name and 'vs' not in cfg.group      # seeds 1, 2 would print -vs0 for a gnn
        assert tok(seed=seed, val_seed=0).name == cfg.name
    with pytest.raises(ValueError, match='validation carve'):
        tok(seed=1, val_seed=1)
    with pytest.raises(ValueError, match='validation carve'):
        tok(val_seed=2)
    assert RunConfig(protocol='kfold', fold=0, model='gnn', seed=1, val_seed=0).name == 'kfold0_x86_gnn-vs0_s1'    # others unchanged


def test_tokenseq_needs_a_fold_of_an_export():
    with pytest.raises(ValueError):
        RunConfig(protocol='bundle', model='tokenseq')
    with pytest.raises(ValueError):
        tok(fold=None)


GNN_ONLY = [dict(control='no_edges'), dict(control='shuffle'), dict(mask_blocks=0.15), dict(mask_tokens=0.1),
            dict(drop_functions=0.1), dict(drop_library=0.1), dict(aux_recon=0.1), dict(drop_edges=0.1), dict(flip_edges=0.1),
            dict(block_input='vec', block_vec='x'), dict(encoder='mean'), dict(readout='gated'), dict(untyped=True),
            dict(no_call_mp=True), dict(stack_tfidf=True), dict(direction='forward'), dict(subgraph='cap', budget=5000),
            dict(vocab='noapi'), dict(imports='header'), dict(gnn_fn_layers=4), dict(gnn_bin_layers=1),
            dict(train_fraction=0.5), dict(arch_balance=True), dict(select_on='cell'), dict(fixed_epochs=3)]


@pytest.mark.parametrize('option', GNN_ONLY, ids=lambda o: next(iter(o)))
def test_tokenseq_rejects_the_gnn_only_options(option):
    with pytest.raises(ValueError, match='do not apply'):
        tok(**option)


def test_check_supported_rejects_tokenseq_with_gnn_options_however_it_was_built():
    ok = tok()
    ts_train.check_supported(ok)
    bad = tok()
    object.__setattr__(bad, 'control', 'no_edges')                       # bypasses __post_init__, as an edited config.json would not
    with pytest.raises(ValueError, match='control'):
        ts_train.check_supported(bad)
    with pytest.raises(ValueError, match='tokenseq'):
        ts_train.check_supported(RunConfig(protocol='kfold', fold=0, model='gnn'))
    with pytest.raises(NotImplementedError, match='token_sequence.train'):
        check_supported(ok)                                              # training/train.py does not train it
    check_supported(RunConfig(protocol='kfold', fold=0, model='gnn'))


def test_seq_len_and_max_windows_belong_to_tokenseq_only():
    with pytest.raises(ValueError, match="model='tokenseq'"):
        RunConfig(protocol='kfold', fold=0, model='gnn', seq_len=256)
    with pytest.raises(ValueError, match="model='tokenseq'"):
        RunConfig(protocol='kfold', fold=0, model='gnn', max_windows=8)
    for bad in (dict(seq_len=0), dict(max_windows=0)):
        with pytest.raises(ValueError):
            tok(**bad)


def test_tokenseq_config_round_trips_and_gnn_flags_exist(tmp_path):
    cfg = tok(seed=1, seq_len=256, max_windows=16, epochs=3, patience=2)
    cfg.save(tmp_path / 'config.json')
    again = RunConfig.load(tmp_path / 'config.json')
    assert again == cfg and again.name == cfg.name and again.val_seed == 0
    assert json.loads((tmp_path / 'config.json').read_text(encoding='utf-8'))['val_seed'] == 0
    old = {k: v for k, v in dataclasses.asdict(RunConfig(protocol='kfold', fold=0, model='gnn')).items()
           if k not in ('seq_len', 'max_windows')}
    (tmp_path / 'old.json').write_text(json.dumps(old), encoding='utf-8')                  # a config.json from before this step
    assert RunConfig.load(tmp_path / 'old.json') == RunConfig(protocol='kfold', fold=0, model='gnn')
    parsed, _ = parse_config(['--protocol', 'kfold', '--fold', '0', '--model', 'gnn'])
    assert (parsed.seq_len, parsed.max_windows) == (512, 64)


def test_existing_run_names_stay_the_same():
    """Names written before this step: a spread of the variants (the test on the runs directory is test_gnn_config's)."""
    gnn = lambda **kw: RunConfig(**{'protocol': 'kfold', 'fold': 1, 'model': 'gnn', **kw})
    assert gnn(mask_blocks=0.15).name == 'kfold1_x86_gnn-mblk15_s0'
    assert gnn(control='no_edges', mask_blocks=0.15, seed=2).name == 'kfold1_x86_gnn-noedges-mblk15_s2'
    assert gnn(seed=1, val_seed=0).name == 'kfold1_x86_gnn-vs0_s1'
    assert gnn(vocab='noapi', mask_blocks=0.15).name == 'kfold1_x86_gnn-mblk15-voc-noapi_s0'
    assert gnn(tag='team-sig500v2').name == 'kfold1_x86_gnn_s0_team-sig500v2'
    assert RunConfig(protocol='kfold_v3', fold=3, arch='all', model='gnn', mask_blocks=0.15, select_on='cell').name == (
        'v3kfold3_all_gnn-mblk15-selcell_s0')
    assert RunConfig(protocol='kfold', fold=0, seed=2, val_seed=0).name == 'kfold0_x86_gnn-vs0_s2'
    assert RunConfig().name == 'bundle_x86_gnn_s0'
