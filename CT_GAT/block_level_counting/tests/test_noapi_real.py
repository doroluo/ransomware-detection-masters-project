"""python -m pytest tests/test_noapi_real.py: the api-token ablation on the real v2 cache (3 real binaries) and the gate stage
`noapi` on synthetic runs next to the real TF-IDF baselines. Kept apart from tests/test_noapi.py, whose module-scoped tiny-cache
fixture points every module-level CACHE and RUNS at a synthetic cache for the rest of that module."""
import json
import shutil

import numpy as np
import pytest

from common.config import DEFAULT_CACHE, RUNS, RunConfig
from common.manifests import load_index, load_sample, make_manifest
from evaluation import gates, stats
from pipeline.s03_tokenizer.block_tokens import token_table
from pipeline.s03_tokenizer.vocab import TokenVocab

needs_real_data = pytest.mark.skipif(not (DEFAULT_CACHE / 'index.csv').exists(), reason='needs the v2 cache')
needs_baselines = pytest.mark.skipif(not (RUNS / 'baselines' / 'kfold0_x86_s0' / 'predictions.csv').exists(),
                                     reason='needs the tfidf kfold baselines')
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


# ---- 3 real binaries ------------------------------------------------------------------------------------------------

@needs_real_data
def test_real_binaries_noapi_table_is_ours_without_api():
    index = load_index()
    manifest = make_manifest(RunConfig(protocol='kfold', fold=0, arch='x86', model='gnn'), index)
    rows = sorted(manifest.train, key=lambda r: int(r['num_blocks']))[:80]
    ours, new = TokenVocab(rows, fold(0)), TokenVocab(rows, fold(0, vocab='noapi'))
    picks = [r for r in sorted(manifest.test, key=lambda r: int(r['num_blocks'])) if int(r['num_api']) > 0]
    picks = [picks[len(picks) // 4], picks[len(picks) // 2], picks[3 * len(picks) // 4]]
    assert len({r['sample_id'] for r in picks}) == 3
    for r in picks:
        z = load_sample(r['sample_id'])
        blocks = np.arange(len(z['blk_insn_ptr']) - 1)
        t_o, k_o, off_o, per_o, *_ = token_table(z, ours, blocks)
        t_n, k_n, off_n, per_n, *_ = token_table(z, new, blocks)
        want_t, want_k, want_off, want_per = api_free(t_o, k_o, off_o, per_o, 3)
        assert (k_o == 3).sum() > 0                                         # the binary has api tokens that were removed
        assert (t_n == want_t).all() and (k_n == want_k).all() and (off_n == want_off).all() and (per_n == want_per).all()
        assert not (k_n == 3).any()                                         # no api-kind id
        lo, hi = new.offset['beh'], new.offset['beh'] + new.size['beh']
        assert ((t_n >= 0) & (t_n < new.total - 3)).all() and (t_n[k_n == 2] >= lo).all() and (t_n[k_n == 2] < hi).all()


def fake_rows(manifest, rng, signal):
    out = {}
    for split in ('val', 'test'):
        rows = getattr(manifest, split)
        probs = np.clip(0.5 + signal * (np.array([r['label'] for r in rows]) - 0.5) + rng.normal(0, 0.2, len(rows)),
                        0.01, 0.99)
        out[split] = ([r['sample_id'] for r in rows], probs, [(10, 500, 700)] * len(rows))
    return out


def fake_group(runs, cfg_of, signal, seeds=(0, 1, 2)):
    from experiments.team_pipelines import common as tp
    from training.train import val_threshold_of
    for seed in seeds:
        rng = np.random.default_rng(100 * seed + int(10 * signal))
        for f in FOLDS:
            cfg, manifest = cfg_of(f, seed), tp.fold_manifest(f, seed)
            f1, auc = fake_rows(manifest, rng, signal), fake_rows(manifest, rng, signal)
            tp.write_run(runs / cfg.name, cfg, f1, auc, val_threshold_of(manifest.val, *f1['val'][:2]),
                         val_threshold_of(manifest.val, *auc['val'][:2]),
                         {'best_epoch': 2, 'epochs_run': 7, 'auc_best_epoch': 3, 'val_auc': 0.9, 'val_macro_f1': 0.8})


@needs_real_data
@needs_baselines
def test_noapi_gates_from_synthetic_runs(tmp_path, monkeypatch):
    runs = tmp_path / 'runs'
    (runs / 'baselines').mkdir(parents=True)
    for f in FOLDS:
        shutil.copytree(RUNS / 'baselines' / f'kfold{f}_x86_s0', runs / 'baselines' / f'kfold{f}_x86_s0')
        noapi_dir = runs / 'baselines' / f'kfold{f}_x86_s0_voc-noapi'                    # a TF-IDF noapi that is the plain one renamed
        shutil.copytree(runs / 'baselines' / f'kfold{f}_x86_s0', noapi_dir)
        m = json.loads((noapi_dir / 'metrics.json').read_text(encoding='utf-8'))
        m['methods'] = {'tfidf_noapi_lr': m['methods']['tfidf_sig_api_lr']}
        (noapi_dir / 'metrics.json').write_text(json.dumps(m), encoding='utf-8')
        lines = (noapi_dir / 'predictions.csv').read_text(encoding='utf-8').splitlines()
        (noapi_dir / 'predictions.csv').write_text('\n'.join([lines[0]] + [l.replace('tfidf_sig_api_lr', 'tfidf_noapi_lr')
                                                                         for l in lines[1:] if 'tfidf_sig_api_lr' in l]),
                                                   encoding='utf-8')
    base = dict(protocol='kfold', arch='x86', model='gnn', mask_blocks=0.15)
    fake_group(runs, lambda f, s: RunConfig(**base, fold=f, seed=s), 1.2)                              # R7, 3 seeds
    fake_group(runs, lambda f, s: RunConfig(**base, fold=f, seed=s, vocab='noapi'), 0.2)               # R7 noapi, much weaker
    monkeypatch.setattr(stats, 'RUNS', runs)
    gates.pooled.cache_clear()
    text = '\n'.join(gates.noapi(60))
    gates.pooled.cache_clear()
    pair = {(l.split('|')[1].strip(), l.split('|')[2].strip()): l.split('|')[-2].strip()
            for l in text.splitlines() if l.startswith('| ') and l.count('|') == 6
            and (l.split('|')[1].strip(), l.split('|')[2].strip()) in gates.NOAPI_PAIRS}
    assert len(pair) == 4
    assert pair[('R7 typed', 'R7 noapi')] == 'api loss shown'                                          # 3 seeds in both arms
    assert pair[('TF-IDF', 'TF-IDF noapi')].startswith('api loss not shown (seed 0 only: a direction, not the test)')
    assert pair[('TF-IDF noapi', 'R7 noapi')].endswith('(seed 0 only: a direction, not the test)')
    assert pair[('TF-IDF noapi', 'R7 noapi')].startswith('does not beat')
    assert pair[('header-imports LR', 'R7 noapi')] == 'missing'                                         # header-imports LR was not copied
    arms = dict(gates.NOAPI_ARMS)
    pooled_rows = {l.split('|')[1].strip(): l.split('|')[3].strip() for l in text.splitlines()
                   if l.startswith('| ') and l.count('|') == 6 and arms.get(l.split('|')[1].strip()) == l.split('|')[2].strip()}
    assert pooled_rows['TF-IDF'] == pooled_rows['TF-IDF noapi'] == '0.982'
    assert float(pooled_rows['R7 typed']) > float(pooled_rows['R7 noapi'])
    assert pooled_rows['header-imports LR'].startswith('missing')
    assert 'seed 0 only: a direction' in text and 'seeds [0, 1, 2]' in text
