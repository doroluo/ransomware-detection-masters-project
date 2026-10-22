"""python -m pytest tests/test_team_v3.py: the sig500v3 arms of `train_b` (Models 2 and 3), the arm table, the settings check, the two
layouts of a build (a `build_b` folder and an unzipped Drive zip), the `main-fix` / `repo` source labels and the v3 rows, pairs and
match-rate line of `gates --stage team`.

Everything here runs on CPU with the tiny cache of `tiny_cache.py` and synthetic builds written by `synthetic_build`, which pools the
block embeddings with the repository's own `FiveChannelEmbedder`, so `check_repool` holds on them as on a real build. The tests that
train need torch_geometric and skip with a reason without it; the arm table, the refusals and the layouts do not."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from common.config import RunConfig
from experiments.stages.team import team_group
from experiments.team_pipelines import common as tp, fit, train_b
from experiments.team_pipelines.build_b import load_vocab_embedder
from s04_evaluation import gate_tables, gates, stats
from test_team_pipelines import fake_group, fresh_team_caches, needs_pyg, pinned_team_modules  # noqa: F401
from test_team_v2 import fake_build, fake_payload
from tiny_cache import tiny  # noqa: F401  (a fixture)

V3_ARMS = (('sig500v3', 'all'), ('sig500v3-feat', 'feat'), ('sig500v3-emb', 'emb'))
S02 = 'CT_GAT/pipeline/s02_preprocess/'
FIX_PINS = {S02 + 'preprocess_full.py': 'a' * 64, S02 + 'tokenize_asm_gz.py': 'b' * 64}      # what a build from 145ed35 recorded, say
N_API = 6


def settings(path, **kw):
    path.mkdir(parents=True, exist_ok=True)
    (path / 'settings.json').write_text(json.dumps(kw), encoding='utf-8')
    return path


def synthetic_build(root, ids, rows, layout='build', source='main-fix', num_blocks=500, sample='signal', seed=0, excluded=None,
                    pins=None):
    """A pipeline-B build of synthetic payloads for `ids` (rows: sample_rows()) in `root`, and the folder to pass as `--build`.

    layout 'build': settings.json, checkpoint.json, api_vocab.json, build_log.csv, channel_embedder.pt and the samples in `root`.
    layout 'zip': what the Drive zip unpacks to under `root`: processed/api_vocab.json, processed/cfg_subgraphs/ with the samples,
    channel_embedder.pt and build_info/{build_settings.json, build_log.csv, build_summary.json, index.csv}; no checkpoint.json.
    The block embeddings are pooled from the instruction ids with the saved embedder and the api-kind columns of the block features
    are the counts of those ids, so `train_b.check_repool` holds."""
    pp = tp.import_team('preprocess_full', source='repo')
    torch.manual_seed(seed)
    embedder = pp.FiveChannelEmbedder(pp.tok.FIRST_API_ID + N_API).eval()
    vocab = pp.tok.ApiVocab()
    for k in range(N_API):
        vocab.id_for(f'lib.dll!Fn{k}')
    rng = np.random.default_rng(seed)
    folder = Path(root) / 'processed' / 'cfg_subgraphs' if layout == 'zip' else Path(root)
    folder.mkdir(parents=True, exist_ok=True)
    kinds = [pp.tok.NO_API, pp.tok.UNK_API] + list(range(pp.tok.FIRST_API_ID, pp.tok.FIRST_API_ID + N_API))
    log = ['sample_id,status,n_blocks_total,n_blocks_kept,seconds']
    for sample_id in ids:
        n_blocks, n_insn = int(rng.integers(2, 6)), int(rng.integers(8, 20))
        insn_to_block = torch.tensor(np.sort(np.concatenate([np.arange(n_blocks), rng.integers(0, n_blocks, n_insn - n_blocks)])))
        instruction_ids = torch.stack([torch.tensor(rng.integers(1, 50, n_insn)), torch.tensor(rng.integers(0, 30, n_insn)),
                                       torch.tensor(rng.integers(0, 30, n_insn)), torch.tensor(rng.choice(kinds, n_insn)),
                                       torch.tensor(rng.integers(0, pp.tok.BEHAVIOR_VOCAB_SIZE, n_insn))], 1).long()
        matched = torch.tensor(rng.random(n_insn) > 0.2)
        features = torch.tensor(rng.random((n_blocks, 77)) * 5, dtype=torch.float32)
        features[:, train_b.kind_slice(pp)] = train_b.block_counts(pp.tok, instruction_ids, matched, insn_to_block, n_blocks)
        n_edges = n_blocks
        torch.save({'file_id': sample_id, 'label': rows[sample_id]['label'], 'instruction_ids': instruction_ids,
                    'insn_to_block': insn_to_block, 'instruction_matched': matched, 'block_features': features,
                    'block_embeddings': pp.pool_block_embeddings(instruction_ids, insn_to_block, n_blocks, embedder),
                    'edge_index': torch.stack([torch.arange(n_edges), (torch.arange(n_edges) + 1) % n_blocks]),
                    'edge_attr': torch.nn.functional.one_hot(torch.tensor(rng.integers(0, 6, n_edges)), 7).float(),
                    'edge_back': torch.tensor(rng.random(n_edges) > 0.5), 'edge_type': torch.tensor(rng.integers(0, 6, n_edges))},
                   folder / f'{sample_id}.pt')
        log.append(f'{sample_id},ok,{n_blocks},{n_blocks},0.1')
    torch.save(embedder.state_dict(), folder / 'channel_embedder.pt')
    body = {'sample': sample, 'num_blocks': num_blocks, 'seed': seed, 'source': source, 'pins': pins or dict(FIX_PINS),
            'excluded': excluded or {}}
    if layout == 'zip':
        info = folder / 'build_info'
        info.mkdir()
        (info / 'build_settings.json').write_text(json.dumps(body), encoding='utf-8')
        (info / 'build_log.csv').write_text('\n'.join(log) + '\n', encoding='utf-8')
        (info / 'build_summary.json').write_text(json.dumps({'processed': len(ids), 'total': len(ids)}), encoding='utf-8')
        (info / 'index.csv').write_text('sample_id,label\n', encoding='utf-8')
        pp.tok.save_api_vocab(vocab, Path(root) / 'processed' / 'api_vocab.json')
    else:
        (folder / 'settings.json').write_text(json.dumps(body), encoding='utf-8')
        (folder / 'build_log.csv').write_text('\n'.join(log) + '\n', encoding='utf-8')
        (folder / 'checkpoint.json').write_text(json.dumps({'settings': {k: v for k, v in body.items() if k not in ('pins', 'excluded')},
                                                            'done': sorted(ids), 'api_names': N_API}), encoding='utf-8')
        pp.tok.save_api_vocab(vocab, folder / 'api_vocab.json')
    return folder


@pytest.fixture
def ids_of_fold(tiny):                                                                      # noqa: F811
    return sorted(tp.sample_rows())


# the arm table ------------------------------------------------------------------------------------------------------------


def test_v3_arms_names_sources_and_feature_modes():
    assert train_b.ARMS[:5] == ('sig500', 'sig500v2', 'sig500v3', 'sig500v3-feat', 'sig500v3-emb')
    for arm, features in V3_ARMS:
        row = train_b.ARM_TABLE[arm]
        assert train_b.ARM_DIR[arm].name == 'b_sig500_v3' and row.sources == ('main-fix', 'repo')
        assert (row.sampler, row.num_blocks, row.features, row.hp) == ('signal', 500, features, train_b.HP) and not row.legacy
        assert train_b.ARM_FEATURES[arm] == features and 'node_budget' not in train_b.ARM_HP[arm]
        cfg = tp.team_config(2, 1, arm)
        assert cfg.name == f'kfold2_x86_gnn_s1_team-{arm}' and cfg.group == team_group(arm)
    for arm, source, name in (('sig500', 'gnn-331ce0b', 'b_sig500'), ('sig500v2', 'main-c7899f0', 'b_sig500_v2')):
        row = train_b.ARM_TABLE[arm]
        assert row.sources == (source,) and row.legacy and row.features == 'all' and train_b.ARM_DIR[arm].name == name
    assert train_b.FEATURE_MODE == {'all': 'both', 'feat': 'features', 'emb': 'embeddings'}
    assert set(tp.SOURCE_CODE) == {'gnn-331ce0b', 'main-c7899f0', 'main-fix', 'repo'}
    assert tp.code_source('main-fix') == tp.code_source('repo') == 'repo' and tp.code_source('main-c7899f0') == 'main-c7899f0'
    assert tp.source_label_ok('main-fix') and tp.source_label_ok('repo', ('main-fix', 'repo')) and not tp.source_label_ok('x')
    assert not tp.source_label_ok('gnn-331ce0b', ('main-fix', 'repo'))
    with pytest.raises(KeyError):
        tp.code_source('x')


def test_train_b_refuses_a_contradicting_features_value_or_source_flag():
    quiet = lambda *_: None                                                                   # noqa: E731
    with pytest.raises(SystemExit, match='trains on --features feat, not all'):
        train_b.run('sig500v3-feat', 0, 0, smoke=2, device='cpu', features='all', log=quiet)
    with pytest.raises(SystemExit, match='trains on --features all, not emb'):
        train_b.run('sig500v3', 0, 0, smoke=2, device='cpu', features='emb', log=quiet)
    with pytest.raises(SystemExit, match='trains on --features all, not feat'):
        train_b.run('sig500', 0, 0, smoke=2, device='cpu', features='feat', log=quiet)
    with pytest.raises(ValueError):
        train_b.run('sig500v3', 0, 0, smoke=2, device='cpu', features='both', log=quiet)
    with pytest.raises(SystemExit, match='reads builds of main-fix or repo, not --source gnn-331ce0b'):
        train_b.run('sig500v3', 0, 0, smoke=2, device='cpu', source='gnn-331ce0b', log=quiet)
    with pytest.raises(SystemExit, match='not --source main-fix'):
        train_b.run('sig500v2', 0, 0, smoke=2, device='cpu', source='main-fix', log=quiet)
    with pytest.raises(SystemExit):
        train_b.main(['--fold', '0', '--arm', 'sig500v3', '--features', 'feat', '--smoke', '2', '--device', 'cpu'])


def test_train_b_refuses_a_build_that_contradicts_the_arm_in_sampler_blocks_or_source(tmp_path):
    quiet = lambda *_: None                                                                   # noqa: E731

    def refused(arm, d, message, **kw):
        with pytest.raises(SystemExit, match=message) as e:
            train_b.run(arm, 0, 0, smoke=2, device='cpu', b_dir=d, log=quiet, **kw)
        return str(e.value)
    for built_with in ('gnn-331ce0b', 'main-c7899f0'):
        d = settings(tmp_path / built_with, **({'source': built_with} if built_with != 'gnn-331ce0b' else {}))
        for arm, _ in V3_ARMS:
            refused(arm, d, f'was built with {built_with}')
    v3 = settings(tmp_path / 'v3', source='main-fix', sample='signal', num_blocks=500)
    refused('sig500v2', v3, 'built with main-fix')                                            # the older arms refuse a v3 build
    refused('sig500', v3, 'built with main-fix')
    refused('sig500v3', v3, 'was built with main-fix, not --source repo', source='repo')
    walk = settings(tmp_path / 'walk', source='main-fix', sample='walk', num_blocks=500)
    message = refused('sig500v3', walk, 'sampler walk')
    assert 'sampler signal' in message and 'num_blocks 500' in message and 'source main-fix or repo' in message     # all three named
    wrong = settings(tmp_path / 'wrong', source='main-fix', sample='signal', num_blocks=250)
    refused('sig500v3-emb', wrong, 'num_blocks 250')
    bare = settings(tmp_path / 'bare', source='main-fix')                                      # keys a v3 build always writes
    refused('sig500v3', bare, 'sampler None, num_blocks None')
    unknown = settings(tmp_path / 'unknown', source='elsewhere')
    with pytest.raises(SystemExit, match='unknown source'):
        train_b.read_source(unknown)
    whole = settings(tmp_path / 'whole', source='main-fix', sample='signal', num_blocks=1000000)
    capped = settings(tmp_path / 'capped', source='repo', sample='signal', num_blocks=500)
    for arm in ('sigwhole', 'sigwhole-feat', 'sigwhole-emb'):
        refused(arm, capped, 'num_blocks >= 1000000 \\(uncapped\\)')
        assert train_b.check_build(arm, whole) == 'main-fix'
    for arm, _ in V3_ARMS:
        refused(arm, whole, 'num_blocks 500, source')
        assert train_b.check_build(arm, capped) == 'repo' and train_b.check_build(arm, v3) == 'main-fix'
    # the two first builds wrote fewer keys: the first has no `source`, and neither is held to keys it lacks
    assert train_b.check_build('sig500', settings(tmp_path / 'first', sample='signal', num_blocks=500, seed=0)) == 'gnn-331ce0b'
    assert train_b.check_build('sig500', settings(tmp_path / 'first2')) == 'gnn-331ce0b'
    assert train_b.check_build('sig500v2', settings(tmp_path / 'second', source='main-c7899f0')) == 'main-c7899f0'
    refused('sig500v2', settings(tmp_path / 'second_walk', source='main-c7899f0', sample='walk'), 'sampler walk')


def test_a_run_without_a_build_names_the_arms_directory_and_the_unzip_target(tmp_path, monkeypatch):
    monkeypatch.setitem(train_b.ARM_DIR, 'sig500v3', tmp_path / 'nothing' / 'b_sig500_v3')
    monkeypatch.delenv('GT_TEAM_B_BUILD', raising=False)
    from common import paths
    monkeypatch.setattr(train_b, 'source_for', lambda name: (_ for _ in ()).throw(paths.MissingSource('none')))
    with pytest.raises(SystemExit, match='b_sig500_v3 does not exist and neither does the team_b_build source'):
        train_b.default_build_dir('sig500v3')
    here = tmp_path / 'there'
    here.mkdir()
    monkeypatch.setattr(train_b, 'source_for', lambda name: here)
    assert train_b.default_build_dir('sig500v3') == here                                      # a zip unpacked where the README says
    monkeypatch.setitem(train_b.ARM_DIR, 'sig500v3', tmp_path)
    assert train_b.default_build_dir('sig500v3') == tmp_path                                  # the arm's own directory wins when it exists


# the two layouts ------------------------------------------------------------------------------------------------------------


def test_build_layout_reads_a_build_folder_and_an_unzipped_drive_zip(tmp_path):
    folder = settings(tmp_path / 'b', source='repo', sample='signal', num_blocks=500, excluded={'x' * 64: 'why'})
    (folder / 'checkpoint.json').write_text(json.dumps({'done': ['a' * 64, 'b' * 64]}), encoding='utf-8')
    layout = train_b.build_layout(folder)
    assert (layout.kind, layout.settings, layout.build_log, layout.checkpoint, layout.vocab) == (
        'build', folder / 'settings.json', folder / 'build_log.csv', folder / 'checkpoint.json', None)          # no vocabulary file yet
    (folder / 'api_vocab.json').write_text('{}', encoding='utf-8')
    assert train_b.build_layout(folder).vocab == folder / 'api_vocab.json'
    assert train_b.done_ids(folder) == {'a' * 64, 'b' * 64} and train_b.read_excluded(folder) == {'x' * 64: 'why'}
    zip_root = tmp_path / 'zip'
    cfg = zip_root / 'processed' / 'cfg_subgraphs'
    (cfg / 'build_info').mkdir(parents=True)
    (cfg / 'build_info' / 'build_settings.json').write_text(json.dumps({'source': 'main-fix', 'sample': 'signal', 'num_blocks': 500}),
                                                            encoding='utf-8')
    for stem in ('c' * 64, 'd' * 64):
        (cfg / f'{stem}.pt').write_bytes(b'')
    (cfg / 'channel_embedder.pt').write_bytes(b'')
    (cfg / 'notasample.pt').write_bytes(b'')
    layout = train_b.build_layout(cfg)
    assert layout.kind == 'zip' and layout.settings == cfg / 'build_info' / 'build_settings.json'
    assert layout.build_log == cfg / 'build_info' / 'build_log.csv' and layout.checkpoint is None and layout.vocab is None
    assert train_b.done_ids(cfg) == {'c' * 64, 'd' * 64}                                       # no checkpoint: the sample files
    assert train_b.read_source(cfg) == 'main-fix' and train_b.check_build('sig500v3-emb', cfg) == 'main-fix'
    (zip_root / 'processed' / 'api_vocab.json').write_text('{}', encoding='utf-8')
    assert train_b.build_layout(cfg).vocab == zip_root / 'processed' / 'api_vocab.json'     # one level up
    (cfg / 'api_vocab.json').write_text('{}', encoding='utf-8')
    assert train_b.build_layout(cfg).vocab == cfg / 'api_vocab.json'                          # in the folder first
    (cfg / 'checkpoint.json').write_text(json.dumps({'done': ['e' * 64]}), encoding='utf-8')   # a zip that has one (the packager's next step)
    assert train_b.done_ids(cfg) == {'e' * 64}
    assert train_b.build_layout(tmp_path / 'empty').settings is None and train_b.read_settings(tmp_path / 'empty') == {}


@needs_pyg
@pytest.mark.parametrize('layout', ['build', 'zip'])
def test_the_embedder_is_loaded_with_the_vocabulary_one_level_up_or_without_one(tmp_path, tiny, layout):   # noqa: F811
    rows = tp.sample_rows()
    folder = synthetic_build(tmp_path / 'b', sorted(rows)[:3], rows, layout=layout)
    pp = tp.import_team('preprocess_full', source='repo')
    vocab_file = train_b.build_layout(folder).vocab
    assert vocab_file is not None and vocab_file.is_file() and (vocab_file.parent == folder) == (layout == 'build')
    with_vocab = load_vocab_embedder(folder, pp, vocab_file)
    assert len(with_vocab[0].names) == N_API
    vocab_file.unlink()
    assert train_b.build_layout(folder).vocab is None
    pp, embedder, done = train_b.build_state(folder)                                           # sized from channel_embedder.pt alone
    assert embedder.api_embedding.num_embeddings == pp.tok.FIRST_API_ID + N_API and len(done) == 3
    assert all(torch.equal(a, b) for a, b in zip(with_vocab[1].state_dict().values(), embedder.state_dict().values()))
    assert load_vocab_embedder(folder, pp)[0].names == []                                      # no file: an empty vocabulary


# training on synthetic builds -----------------------------------------------------------------------------------------------


@needs_pyg
@pytest.mark.parametrize('layout', ['build', 'zip'])
@pytest.mark.parametrize('arm,features,in_dim', [('sig500v3', 'all', 141), ('sig500v3-feat', 'feat', 77), ('sig500v3-emb', 'emb', 64)])
def test_v3_arm_smoke_on_a_synthetic_build_writes_no_run_dir(tmp_path, monkeypatch, tiny, layout, arm, features, in_dim):   # noqa: F811
    rows = tp.sample_rows()
    folder = synthetic_build(tmp_path / 'b', sorted(rows), rows, layout=layout)
    monkeypatch.setattr(fit, 'RUNS', tmp_path / 'runs')
    monkeypatch.setattr(fit, 'write_run', lambda *a, **k: pytest.fail('a smoke run must not call write_run'))
    lines = []
    out = train_b.run(arm, 2, 1, smoke=6, device='cpu', b_dir=folder, source='main-fix', features=features, log=lines.append)
    assert any(f'features {features}, in_dim {in_dim}, edge_dim 8' in l for l in lines)
    assert any(l.startswith('re-pool check') and 'max |difference| 0.00e+00' in l for l in lines) and len(out['history']) == 2
    assert not (tmp_path / 'runs').exists()
    assert Path(out['dir']).parent == Path(__import__('tempfile').gettempdir()) and (Path(out['dir']) / 'history.csv').is_file()
    __import__('shutil').rmtree(out['dir'])


@needs_pyg
def test_a_full_run_records_the_label_the_builds_pins_and_the_pins_of_the_code_it_imported(tmp_path, monkeypatch, tiny):   # noqa: F811
    rows = tp.sample_rows()
    folder = synthetic_build(tmp_path / 'b', sorted(rows), rows, source='main-fix')
    monkeypatch.setattr(fit, 'RUNS', tmp_path / 'runs')
    monkeypatch.setitem(train_b.ARM_HP, 'sig500v3-emb', {**train_b.HP, 'epochs': 2})
    out = train_b.run('sig500v3-emb', 2, 1, device='cpu', b_dir=folder, log=lambda *_: None)
    d = tmp_path / 'runs' / 'kfold2_x86_gnn_s1_team-sig500v3-emb'
    assert out['dir'] == d
    metrics = json.loads((d / 'metrics.json').read_text(encoding='utf-8'))
    assert metrics['build_source'] == 'main-fix' and metrics['features'] == 'emb' and metrics['in_dim'] == 64
    assert metrics['build_pins'] == FIX_PINS                                                    # the build's, not the files now in the repository
    repo_files = {rel: hashlib.sha256((tp.REPO_ROOT / rel).read_bytes()).hexdigest() for rel in FIX_PINS}
    assert metrics['code_pins'] == repo_files and metrics['code_pins'] != metrics['build_pins']
    assert metrics['build_binaries'] == 40 and 'excluded' not in metrics and metrics['build_dir'] == str(folder)
    assert 0.0 <= metrics['test']['auc'] <= 1.0 and metrics['recipe']['arm'] == 'sig500v3-emb'


@needs_pyg
def test_the_repo_label_is_accepted_and_a_build_without_pins_gets_the_pins_of_its_code(tmp_path, monkeypatch, tiny):   # noqa: F811
    rows = tp.sample_rows()
    folder = synthetic_build(tmp_path / 'b', sorted(rows), rows, source='repo')
    body = json.loads((folder / 'settings.json').read_text(encoding='utf-8'))
    body.pop('pins')
    (folder / 'settings.json').write_text(json.dumps(body), encoding='utf-8')
    monkeypatch.setattr(fit, 'RUNS', tmp_path / 'runs')
    monkeypatch.setitem(train_b.ARM_HP, 'sig500v3', {**train_b.HP, 'epochs': 2})
    train_b.run('sig500v3', 0, 0, device='cpu', b_dir=folder, source='repo', log=lambda *_: None)
    metrics = json.loads((tmp_path / 'runs' / 'kfold0_x86_gnn_s0_team-sig500v3' / 'metrics.json').read_text(encoding='utf-8'))
    assert metrics['build_source'] == 'repo' and metrics['build_pins'] == metrics['code_pins'] == dict(tp.source_pins('repo'))


@needs_pyg
def test_a_build_whose_stored_embeddings_do_not_match_the_ids_is_refused(tmp_path, tiny):             # noqa: F811
    rows = tp.sample_rows()
    folder = synthetic_build(tmp_path / 'b', sorted(rows), rows)
    for victim in rows:                                                                         # the one that is checked is a training id
        payload = torch.load(folder / f'{victim}.pt', weights_only=False)
        payload['block_embeddings'] = payload['block_embeddings'] + 1e-2
        torch.save(payload, folder / f'{victim}.pt')
    with pytest.raises(AssertionError, match='re-pooled block embeddings differ'):
        train_b.run('sig500v3-emb', 2, 1, smoke=40, device='cpu', b_dir=folder, log=lambda *_: None)   # the alias is guarded by this check


@needs_pyg
def test_feature_modes_slice_the_141_d_input_and_keep_the_edges(tmp_path, tiny):                        # noqa: F811
    from s03b_cfg_feature.gat import graph_full_loader as gl
    rows = tp.sample_rows()
    folder = synthetic_build(tmp_path / 'b', sorted(rows)[:5], rows)
    pp, embedder, _ = train_b.build_state(folder)
    for sample_id in sorted(rows)[:5]:
        payload = train_b.load_slim(folder, sample_id)
        api = payload['instruction_ids'][:, 3]
        real = torch.unique(api[api >= pp.tok.FIRST_API_ID])
        for seen in (None, real[: len(real) // 2], torch.zeros(0, dtype=torch.long)):         # stored ids, half the ids, none
            g = {m: train_b.make_graph(pp, gl, embedder, payload, seen, m) for m in train_b.FEATURES}
            assert (g['all'].x.shape[1], g['feat'].x.shape[1], g['emb'].x.shape[1]) == (141, 77, 64)
            assert torch.equal(g['feat'].x, g['all'].x[:, 64:]) and torch.equal(g['emb'].x, g['all'].x[:, :64])
            for m in ('feat', 'emb'):
                assert torch.equal(g[m].edge_index, g['all'].edge_index) and torch.equal(g[m].edge_attr, g['all'].edge_attr)
                assert torch.equal(g[m].edge_type, g['all'].edge_type) and g[m].y == g['all'].y and g[m].edge_attr.shape[1] == 8
    assert train_b.check_repool(pp, embedder, payload) <= 1e-5


# gates -----------------------------------------------------------------------------------------------------------------------


def test_team_gates_v3_match_rate_line(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, 'RUNS', tmp_path)
    gate_tables.pooled.cache_clear()
    absent = '\n'.join(gates.run_stage('team', 30))
    assert 'Match rates of `b_sig500_v3`: missing (no build_summary.json' in absent
    (tmp_path / 'team').mkdir()
    build = tmp_path / 'team' / 'b_sig500_v3'
    fake_build(build, {'r1': fake_payload([0] * 10, 1), 'r2': fake_payload([1] * 10, 1), 'g1': fake_payload([1] * 8 + [0] * 2, 0)})
    (build / 'build_summary.json').write_text(json.dumps({'processed': 1809, 'total': 1810}), encoding='utf-8')
    assert 'missing (build incomplete: 1809 of 1810' in '\n'.join(gates.run_stage('team', 30))
    (build / 'build_summary.json').write_text(json.dumps({'processed': 1810, 'total': 1810}), encoding='utf-8')
    text = '\n'.join(gates.run_stage('team', 30))
    gate_tables.pooled.cache_clear()
    line = [l for l in text.splitlines() if l.startswith('Match rates of `b_sig500_v3`')]
    assert len(line) == 1
    assert 'ransomware 2 binaries, mean 0.500, below 50% 1 (50.0%), below 90% 1 (50.0%), exactly 0 50.0%' in line[0]
    assert 'goodware 1 binaries, mean 0.800, below 50% 0 (0.0%), below 90% 1 (100.0%), exactly 0 0.0%' in line[0]
    for name in ('team sig500v3', 'team sig500v3 feat (Model 2)', 'team sig500v3 emb (Model 3)'):
        assert f'| {name} | kfold_x86_gnn_team-' in text and f'| {name} | missing' in text      # missing arms print as missing
    assert '| team sig500v3 feat (Model 2) | team sig500v3 emb (Model 3) | missing | missing | missing |' in text
    lines = text.splitlines()
    assert next(i for i, l in enumerate(lines) if l.startswith('| arm | AUC | PR-AUC | macro-F1 | accuracy')) < lines.index(line[0]) < \
        lines.index('## Paired differences, second minus first')                               # after the operating points


def test_team_gates_from_synthetic_v3_runs(tiny, tmp_path, monkeypatch):                          # noqa: F811
    runs = tmp_path / 'runs'
    runs.mkdir()
    base = dict(protocol='kfold', arch='x86', model='gnn')
    cap = dict(mask_blocks=0.15, subgraph='cap', budget=500, tag='b500')
    fake_group(runs, lambda f, s: RunConfig(**base, fold=f, seed=s, mask_blocks=0.15), 0.5)                          # R7
    fake_group(runs, lambda f, s: RunConfig(**base, fold=f, seed=s, control='no_edges', **cap), 0.4)                 # b500 no-edges
    fake_group(runs, lambda f, s: tp.team_config(f, s, 'sig500'), 0.7, seeds=(0, 1))
    fake_group(runs, lambda f, s: tp.team_config(f, s, 'sig500v2'), 0.9)
    fake_group(runs, lambda f, s: tp.team_config(f, s, 'sig500v3'), 1.2)
    fake_group(runs, lambda f, s: tp.team_config(f, s, 'sig500v3-feat'), 0.8)
    fake_group(runs, lambda f, s: tp.team_config(f, s, 'sig500v3-emb'), 1.0, seeds=(0, 1))
    monkeypatch.setattr(stats, 'RUNS', runs)
    gate_tables.pooled.cache_clear()
    text = '\n'.join(gates.run_stage('team', 40))
    gate_tables.pooled.cache_clear()
    pair = {(l.split('|')[1].strip(), l.split('|')[2].strip()): l.split('|')[-2].strip()
            for l in text.splitlines() if l.startswith('| ') and l.count('|') == 6}
    stage = gates.load_stage('team')
    v3, feat, emb = ('team sig500v3', 'team sig500v3 feat (Model 2)', 'team sig500v3 emb (Model 3)')
    for a, b in stage.pairs:
        if (b in (v3, feat, emb) or a in (v3, feat, emb)) and a != 'TF-IDF' and b not in ('team sigwhole', 'team sigwhole feat (Model 2)',
                                                                                          'team sigwhole emb (Model 3)'):
            assert pair[(a, b)] != 'missing', (a, b)
    assert 'seeds [0, 1] only' in pair[('team sig500', v3)] and pair[('team sig500v2', v3)] in ('beats', 'does not beat')
    assert 'seeds [0, 1] only' in pair[(feat, emb)] and 'seeds [0, 1] only' in pair[('R7 typed, whole graph', emb)]
    assert pair[('R7 typed, whole graph', v3)] == 'beats'
