"""python -m pytest tests/test_noapi.py: the api-token ablation, vocabulary `noapi` (mn, sig and beh tokens per block, no api
token): configuration and names, the vocabulary and block tokens against `ours` (3 real binaries and the tiny cache), masking
and the network, the TF-IDF baseline `--vocab noapi`, the run script, and the gate stage `noapi` on synthetic runs."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from common.config import VOCABS, RunConfig
from common.manifests import load_index, load_sample, make_manifest
from evaluation import baselines, gates, stats
from pipeline.batching import encode_binary, loader
from pipeline.s03_tokenizer import vocab as vocab_module
from pipeline.s03_tokenizer.block_tokens import token_table
from pipeline.s03_tokenizer.vocab import VOCAB_INSN_KINDS, VOCAB_KINDS, TokenVocab, uses_reader
from tiny_cache import tiny  # noqa: F401  (a fixture)
from training.train import build_model, parse_config

ROOT = Path(__file__).resolve().parents[1]
BASH = next((b for b in ('C:/Program Files/Git/bin/bash.exe', shutil.which('bash')) if b and Path(b).exists()), None)
needs_bash = pytest.mark.skipif(BASH is None, reason='needs bash')
FOLDS = range(5)


def gnn(**kw):
    return RunConfig(**{'model': 'gnn', 'num_workers': 0, **kw})


def fold(f=0, **kw):
    return gnn(protocol='kfold', fold=f, **kw)


def api_free(tokens, kind, tok_offsets, per_block, n_kinds):
    """(tokens, kind) of an `ours` table without its api tokens, and the offsets / sizes of the blocks that remain."""
    keep = kind != n_kinds
    block_of = np.repeat(np.arange(len(per_block)), per_block)
    sizes = per_block - np.bincount(block_of[~keep], minlength=len(per_block))
    return tokens[keep], kind[keep], np.cumsum(sizes) - sizes, sizes


# ---- configuration -------------------------------------------------------------------------------------------------

def test_noapi_is_a_vocab_with_its_own_suffix():
    assert 'noapi' in VOCABS and VOCABS[:3] == ('ours', 'yanping', 'hybrid')
    assert fold(0, mask_blocks=0.15, vocab='noapi').name == 'kfold0_x86_gnn-mblk15-voc-noapi_s0'
    assert fold(2, vocab='noapi', seed=1).name == 'kfold2_x86_gnn-voc-noapi_s1'
    assert fold(0, mask_blocks=0.15, vocab='noapi').variant == 'gnn-mblk15-voc-noapi'
    assert fold(0, vocab='noapi', imports='header').variant == 'gnn-voc-noapi-imph'      # header imports stay a separate switch
    for bad in (dict(model='transformer', vocab='noapi'), dict(vocab='noapi', block_input='vec', block_vec='x')):
        with pytest.raises(ValueError):
            RunConfig(protocol='kfold', fold=0, **bad)
    cfg, _ = parse_config(['--protocol', 'kfold', '--fold', '0', '--model', 'gnn', '--vocab', 'noapi'])
    assert cfg.vocab == 'noapi'


def test_vocabulary_kinds_are_explicit_and_the_old_ones_unchanged():
    assert VOCAB_KINDS == {'ours': ('mn', 'sig', 'beh', 'api'), 'noapi': ('mn', 'sig', 'beh'),
                           'yanping': ('y0', 'y1', 'y2', 'api'), 'hybrid': ('mn', 'beh', 'y1', 'y2', 'api')}
    assert VOCAB_INSN_KINDS['noapi'] == VOCAB_INSN_KINDS['ours'] == ('mn', 'sig', 'beh')
    assert [uses_reader(v) for v in ('ours', 'noapi', 'yanping', 'hybrid')] == [False, False, True, True]


# ---- the vocabulary and the block tokens ---------------------------------------------------------------------------

def train_rows(tiny, n=30):  # noqa: F811
    return tiny.rows[:n]


def test_noapi_vocabulary_is_ours_without_the_api_kind(tiny, monkeypatch):  # noqa: F811
    rows = train_rows(tiny)
    ours = TokenVocab(rows, fold(0))
    monkeypatch.setattr(vocab_module, 'load_ytok', lambda *a: pytest.fail('noapi must not read the reader side arrays'))
    new = TokenVocab(rows, fold(0, vocab='noapi'))
    assert new.kinds == new.insn_kinds == ('mn', 'sig', 'beh') and new.has_api is False and ours.has_api is True
    assert ours.kinds == ('mn', 'sig', 'beh', 'api') and ours.insn_kinds == ('mn', 'sig', 'beh')
    for table in (new.lut, new.size, new.offset, new.mask_id):
        assert 'api' not in table
    for k in new.kinds:                                                     # the surviving kinds are cut and numbered alike
        assert (new.lut[k] == ours.lut[k]).all() and new.size[k] == ours.size[k] and new.offset[k] == ours.offset[k]
    assert new.total == ours.total - ours.size['api'] - 1 and len(new.idf) == new.total
    assert new.mask_id == {k: new.total - 3 + j for j, k in enumerate(new.kinds)}
    assert new.idf[-3:].tolist() == [1.0] * 3
    assert (new.idf[:ours.offset['api']] == ours.idf[:ours.offset['api']]).all()


def test_noapi_block_tokens_are_ours_without_the_api_tokens(tiny):  # noqa: F811
    rows = train_rows(tiny)
    ours, new = TokenVocab(rows, fold(0)), TokenVocab(rows, fold(0, vocab='noapi'))
    for sid in tiny.ids[30:34]:
        z = load_sample(sid)
        blocks = np.arange(0, len(z['blk_insn_ptr']) - 1)
        t_o, k_o, off_o, per_o, ins_o, api_o = token_table(z, ours, blocks)
        t_n, k_n, off_n, per_n, ins_n, api_n = token_table(z, new, blocks)
        want_t, want_k, want_off, want_per = api_free(t_o, k_o, off_o, per_o, 3)
        assert len(api_o) > 0 and len(api_n) == 0 and len(ins_n) == 3
        assert (t_n == want_t).all() and (k_n == want_k).all() and (off_n == want_off).all() and (per_n == want_per).all()
        assert k_n.max() < 3 and t_n.max() < new.total - 3                  # no api kind, no mask id


@pytest.mark.parametrize('mask', [dict(), dict(mask_tokens=0.5), dict(mask_blocks=0.5)])
def test_noapi_items_masking_and_network(tiny, mask):  # noqa: F811
    rows = train_rows(tiny)
    cfg = fold(0, vocab='noapi', batch_size=8, **mask)
    v = TokenVocab(rows, cfg)
    z = load_sample(tiny.ids[2])
    out = encode_binary(z, v, cfg, np.random.default_rng(3) if mask else None, 's')
    masked = out['tokens'] >= v.total - len(v.kinds)
    assert (out['tokens'] < v.total).all() and masked.any() == bool(mask)
    assert len(out['tok_weights']) == len(out['tokens'])
    batch = next(iter(loader(tiny.rows[10:18], v, cfg, train=False)))
    model = build_model(cfg, v)
    logits = model(batch)
    assert logits.shape == (8,) and torch.isfinite(logits).all()
    logits.sum().backward()


# ---- the TF-IDF baseline --------------------------------------------------------------------------------------------

def run_noapi_baselines(tiny, min_df=2):  # noqa: F811
    sizes = {k: len(v) for k, v in json.loads((tiny.root / 'vocab.json').read_text(encoding='utf-8')).items()}
    for f in FOLDS:
        baselines.run(RunConfig(protocol='kfold', fold=f, arch='x86', seed=0, min_df=min_df), sizes,
                      tiny.runs / 'baselines', 'noapi')


def test_noapi_baseline_counts_mn_sig_beh_and_writes_its_directory(tiny, monkeypatch):  # noqa: F811
    monkeypatch.setattr(baselines, 'load_ytok', lambda *a: pytest.fail('noapi must not read the reader side arrays'))
    run_noapi_baselines(tiny)
    gates.pooled.cache_clear()
    for f in FOLDS:
        d = tiny.runs / 'baselines' / f'kfold{f}_x86_s0_voc-noapi'
        m = json.loads((d / 'metrics.json').read_text(encoding='utf-8'))
        assert list(m['methods']) == ['tfidf_noapi_lr', 'tfidf_sig_lr'] and m['vocab'] == 'noapi'
        assert 'threshold_val' in m['methods']['tfidf_noapi_lr']
        manifest = make_manifest(RunConfig(protocol='kfold', fold=f, arch='x86', seed=0), load_index())
        counts = {k: np.zeros(n, dtype=int) for k, n in (('mn', 30), ('sig', 50), ('beh', 9), ('api', 24))}
        for r in manifest.train:
            z = load_sample(r['sample_id'])
            for k in counts:
                counts[k][np.unique(z[k])] += 1
        assert m['tfidf_features'] == sum(int((counts[k] >= 2).sum()) for k in ('mn', 'sig', 'beh'))
        assert m['tfidf_features'] > int((counts['sig'] >= 2).sum())                    # more than sig alone: mn and beh count too
        rows = (d / 'predictions.csv').read_text(encoding='utf-8').splitlines()[1:]
        assert {r.split(',')[2] for r in rows} == {'tfidf_noapi_lr', 'tfidf_sig_lr'}
    assert not (tiny.runs / 'baselines' / 'kfold0_x86_s0').exists()                      # the plain directory is untouched
    folds = stats.folds_by_seed('tfidf-noapi')
    assert sorted(folds[0]) == [0, 1, 2, 3, 4] and stats.baseline_method('tfidf-noapi') == ('tfidf_noapi_lr', '_voc-noapi')
    r = gates.pooled('tfidf-noapi', 'auc')
    assert not isinstance(r, str) and 0 <= r[0] <= 1
    gates.pooled.cache_clear()


def test_the_default_baseline_is_unchanged(tiny):  # noqa: F811
    sizes = {k: len(v) for k, v in json.loads((tiny.root / 'vocab.json').read_text(encoding='utf-8')).items()}
    out = tiny.runs / 'plain_noapi_test'
    baselines.run(RunConfig(protocol='kfold', fold=1, arch='x86', seed=0), sizes, out)
    m = json.loads((out / 'kfold1_x86_s0' / 'metrics.json').read_text(encoding='utf-8'))
    assert list(m['methods']) == ['tfidf_sig_api_lr', 'structural_lr'] and 'vocab' not in m


# ---- the run script -------------------------------------------------------------------------------------------------

@needs_bash
def test_noapi_runs_dry_names_equal_run_config_names():
    r = subprocess.run([BASH, 'scripts/noapi_runs.sh', '--dry'], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert lines[0] == 'cpu baselines.py --protocol kfold --fold 0 1 2 3 4 --arch x86 --vocab noapi'
    assert lines[-1] == 'cpu gates.py --stage noapi > RUNS/gates_noapi.md'
    runs = [l.split()[1:] for l in lines if l.startswith('run ')]
    assert len(runs) == 15 and len(lines) == 17
    seen = []
    for name, *argv in runs:
        cfg, _ = parse_config(argv)
        assert cfg.name == name, (name, cfg.name)
        assert (cfg.protocol, cfg.arch, cfg.model, cfg.mask_blocks, cfg.vocab, cfg.imports, cfg.control, cfg.tag) == (
            'kfold', 'x86', 'gnn', 0.15, 'noapi', '', '', '')
        assert name == f'kfold{cfg.fold}_x86_gnn-mblk15-voc-noapi_s{cfg.seed}'
        seen.append((cfg.seed, cfg.fold))
    assert seen == [(s_, f) for s_ in (0, 1, 2) for f in FOLDS]


@needs_bash
def test_noapi_runs_skips_finished_runs_honours_stop_and_retries(tmp_path):
    """With a fake interpreter: a run with metrics.json is skipped, STOP stops, a failing run is retried once."""
    py = tmp_path / 'fakepy.sh'
    py.write_text('#!/bin/bash\n'
                  'echo "$@" >> "$GT_RUNS/calls.txt"\n'
                  'case "$1" in\n'
                  '  train.py) n=$(grep -c "^train.py" "$GT_RUNS/calls.txt"); [ "$n" = 2 ] && exit 1\n'
                  '            d=$GT_RUNS/kfold$5_x86_gnn-mblk15-voc-noapi_s$9; mkdir -p "$d"; echo {} > "$d/metrics.json";;\n'
                  '  baselines.py) for f in 0 1 2 3 4; do mkdir -p "$GT_RUNS/baselines/kfold${f}_x86_s0_voc-noapi";'
                  ' echo {} > "$GT_RUNS/baselines/kfold${f}_x86_s0_voc-noapi/metrics.json"; done;;\n'
                  '  gates.py) echo "# gates";;\n'
                  'esac\n', encoding='utf-8', newline='\n')
    runs = tmp_path / 'runs'
    runs.mkdir()
    env = {**os.environ, 'GT_PY': py.as_posix(), 'GT_RUNS': runs.as_posix()}
    (runs / 'kfold0_x86_gnn-mblk15-voc-noapi_s0').mkdir()
    (runs / 'kfold0_x86_gnn-mblk15-voc-noapi_s0' / 'metrics.json').write_text('{}')
    (runs / 'STOP').write_text('')
    r = subprocess.run([BASH, 'scripts/noapi_runs.sh'], cwd=ROOT, capture_output=True, text=True, env=env)
    assert 'skip kfold0_x86_gnn-mblk15-voc-noapi_s0' in r.stdout and 'stopped before kfold1_x86_gnn-mblk15-voc-noapi_s0' in r.stdout
    assert 'stopped before baselines noapi' in r.stdout and r.returncode != 0 and 'start kfold' not in r.stdout
    assert 'finished gates noapi rc=0, 14 of 15 runs without metrics.json' in r.stdout
    (runs / 'STOP').unlink()
    r = subprocess.run([BASH, 'scripts/noapi_runs.sh'], cwd=ROOT, capture_output=True, text=True, env=env)
    out = r.stdout.splitlines()
    assert r.returncode == 0, r.stdout + r.stderr
    assert out[0].startswith('start baselines noapi') and 'retry kfold2_x86_gnn-mblk15-voc-noapi_s0' in out
    assert [l for l in out if l.startswith('retry ')] == ['retry kfold2_x86_gnn-mblk15-voc-noapi_s0']
    assert sum(l.startswith('finished kfold') for l in out) == 14 and (runs / 'gates_noapi.md').read_text().startswith('# gates')
    assert (runs / 'logs' / 'kfold2_x86_gnn-mblk15-voc-noapi_s0.log.try1').exists()
    assert all(l.endswith('rc=0') or ' rc=0 ' in l + ' ' for l in out if l.startswith('finished kfold'))
    r = subprocess.run([BASH, 'scripts/noapi_runs.sh'], cwd=ROOT, capture_output=True, text=True, env=env)
    assert r.returncode == 0 and r.stdout.count('skip kfold') == 15 and 'skip baselines noapi' in r.stdout


# ---- the gate stage -------------------------------------------------------------------------------------------------

def test_noapi_gate_groups_and_pairs_follow_the_run_configs():
    arms = dict(gates.NOAPI_ARMS)
    assert list(arms) == ['TF-IDF', 'TF-IDF noapi', 'header-imports LR', 'R7 typed', 'R7 noapi']
    assert arms['TF-IDF'] == 'tfidf' and arms['TF-IDF noapi'] == 'tfidf-noapi' and arms['header-imports LR'] == 'imports-header'
    base = dict(protocol='kfold', fold=0, arch='x86', model='gnn', mask_blocks=0.15)
    assert arms['R7 typed'] == RunConfig(**base).group == 'kfold_x86_gnn-mblk15'
    assert arms['R7 noapi'] == RunConfig(**base, vocab='noapi').group == 'kfold_x86_gnn-mblk15-voc-noapi'
    assert gates.NOAPI_PAIRS == (('R7 typed', 'R7 noapi'), ('TF-IDF', 'TF-IDF noapi'), ('TF-IDF noapi', 'R7 noapi'),
                                 ('header-imports LR', 'R7 noapi'))


def test_noapi_gates_print_missing_arms(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    gates.pooled.cache_clear()
    text = '\n'.join(gates.noapi(40))
    gates.pooled.cache_clear()
    assert text.startswith('# Gate summary, stage noapi')
    for arm, group in gates.NOAPI_ARMS:
        assert f'| {arm} | {group} | missing' in text
    for a, b in gates.NOAPI_PAIRS:
        assert f'| {a} | {b} | missing | missing | missing |' in text
    assert gates.STAGES['noapi'] is gates.noapi


def test_gates_cli_knows_the_noapi_stage(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['gates.py', '--stage', 'noapi', '--resamples', '20'])
    gates.pooled.cache_clear()
    gates.main()
    gates.pooled.cache_clear()
    assert capsys.readouterr().out.startswith('# Gate summary, stage noapi')
