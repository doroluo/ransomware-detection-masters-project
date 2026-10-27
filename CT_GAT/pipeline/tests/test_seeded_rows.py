"""`build_b --seeded-rows`: the API rows the embedder adds depend only on (seed, API id), so a build stopped and resumed equals one
run straight through. A build without the flag keeps the rows of the global torch generator and its settings stay as they were.

`process_sample` is replaced by a stand-in that grows the vocabulary by two names per sample, grows the embedder with the repository's
own `expand_api` and saves block embeddings pooled from the new rows; everything else (`load_state`, the checkpoint, the resume,
`save_state`) is `build_b`'s. The ids are those of the tiny cache."""
import json

import pytest
import torch

from experiments.team_pipelines import build_b, common as tp
from test_team_pipelines import fresh_team_caches, pinned_team_modules  # noqa: F401  (fixtures)
from tiny_cache import tiny  # noqa: F401  (a fixture)

N = 4
STOP_AT = 3          # the sample the first run fails on: the checkpoint (SAVE_EVERY 2) then holds two samples


class Result(dict):
    """What the stand-in `process_sample` returns: the dict `save_sample` reads, plus `.sample`."""


@pytest.fixture
def stand_in(tiny, tmp_path, monkeypatch):  # noqa: F811
    """(pp, ids, fail): `build_b` run on a stand-in `process_sample`; `fail['at']` names the sample index it raises on."""
    real = tp.import_team('preprocess_full', source='repo')
    ids = sorted(tp.sample_rows())[:N]
    fail = {'at': None}

    class Shim:
        def __getattr__(self, name):
            return getattr(real, name)

        @staticmethod
        def process_sample(asm_path, cfg_path, label, seed, api_vocab, channel_embedder, file_id, **kwargs):
            k = ids.index(file_id)
            if fail['at'] == k:
                raise RuntimeError('stopped on purpose')
            for j in (2 * k, 2 * k + 1):
                api_vocab.id_for(f'lib.dll!Fn{j}')
            channel_embedder.expand_api(real.api_table_size(api_vocab))
            api = torch.arange(real.tok.FIRST_API_ID, real.api_table_size(api_vocab))
            token_ids = torch.stack([torch.ones_like(api), torch.ones_like(api), torch.ones_like(api), api, torch.zeros_like(api)], 1)
            pooled = real.pool_block_embeddings(token_ids, torch.zeros(len(api), dtype=torch.long), 1, channel_embedder)
            sample = type('S', (), {'original_blocks': 1, 'sampled_blocks': 1})
            return Result(sample=sample, manifest={'sample_id': file_id, 'label': label}, models={'block_embeddings': pooled})

        @staticmethod
        def save_sample(result, out):
            torch.save({'file_id': result['manifest']['sample_id'], 'block_embeddings': result['models']['block_embeddings']},
                       out / f"{result['manifest']['sample_id']}.pt")

    monkeypatch.setattr(build_b, 'import_team', lambda *args, **kwargs: Shim())
    monkeypatch.setattr(build_b, 'paths_for', lambda sample_id: (None, None))
    monkeypatch.setattr(build_b, 'SAVE_EVERY', 2)
    return real, ids, fail


def run(out, ids, *flags):
    return build_b.main(['--out', str(out), '--source', 'repo', '--ids', *ids, *flags])


def state(out, ids):
    """(embedder state, {id: block_embeddings}) of a build folder."""
    payloads = {i: torch.load(out / f'{i}.pt', weights_only=False)['block_embeddings'] for i in ids}
    return torch.load(out / 'channel_embedder.pt'), payloads


def stopped_then_resumed(out, ids, fail, *flags):
    fail['at'] = STOP_AT
    with pytest.raises(SystemExit):
        run(out, ids, *flags)
    assert json.loads((out / 'checkpoint.json').read_text(encoding='utf-8'))['done'] == ids[:2]
    fail['at'] = None
    return run(out, ids, *flags)


def test_a_resumed_build_with_seeded_rows_equals_a_straight_one(stand_in, tmp_path):
    real, ids, fail = stand_in
    straight, resumed = tmp_path / 'straight', tmp_path / 'resumed'
    run(straight, ids, '--seeded-rows')
    summary = stopped_then_resumed(resumed, ids, fail, '--seeded-rows')
    assert summary['built_this_run'] == N - 2
    a, b = state(straight, ids), state(resumed, ids)
    assert a[0].keys() == b[0].keys()
    for key in a[0]:
        assert torch.equal(a[0][key], b[0][key]), key
    for i in ids:
        assert torch.equal(a[1][i], b[1][i])
    assert a[0]['api_embedding.weight'].shape[0] == real.tok.FIRST_API_ID + 2 * N
    for out in (straight, resumed):
        settings = json.loads((out / 'settings.json').read_text(encoding='utf-8'))
        assert settings['embedder_rows'] == 'seeded-per-id'
        assert json.loads((out / 'checkpoint.json').read_text(encoding='utf-8'))['settings']['embedder_rows'] == 'seeded-per-id'


def test_a_row_depends_on_the_seed_and_the_api_id_only(stand_in, tmp_path):
    real, ids, _ = stand_in
    run(tmp_path / 'a', ids[:2], '--seeded-rows')
    run(tmp_path / 'b', ids, '--seeded-rows')                          # a longer build: the same first rows
    first = real.tok.FIRST_API_ID
    a, b = state(tmp_path / 'a', ids[:2])[0], state(tmp_path / 'b', ids)[0]
    assert torch.equal(a['api_embedding.weight'][first:], b['api_embedding.weight'][first:first + 4])
    other = tmp_path / 'c'
    run(other, ids[:2], '--seeded-rows', '--seed', '1')
    assert not torch.equal(a['api_embedding.weight'][first:], state(other, ids[:2])[0]['api_embedding.weight'][first:])


def test_a_build_without_the_flag_keeps_the_global_generator_rows(stand_in, tmp_path):
    real, ids, fail = stand_in
    plain = tmp_path / 'plain'
    run(plain, ids)
    settings = json.loads((plain / 'settings.json').read_text(encoding='utf-8'))
    assert 'embedder_rows' not in settings
    assert json.loads((plain / 'checkpoint.json').read_text(encoding='utf-8'))['settings'] == {
        'sample': 'signal', 'num_blocks': 500, 'seed': 0, 'source': 'repo', 'pins': settings['pins']}
    resumed = tmp_path / 'resumed'
    stopped_then_resumed(resumed, ids, fail)                            # today's behaviour: the rows after the resume are other draws
    first = real.tok.FIRST_API_ID
    a, b = state(plain, ids)[0]['api_embedding.weight'], state(resumed, ids)[0]['api_embedding.weight']
    assert torch.equal(a[:first + 4], b[:first + 4]) and not torch.equal(a[first + 4:], b[first + 4:])


def test_the_resume_must_agree_with_the_build_on_the_flag(stand_in, tmp_path):
    _, ids, fail = stand_in
    keyed, plain = tmp_path / 'keyed', tmp_path / 'plain'
    for out, flags in ((keyed, ('--seeded-rows',)), (plain, ())):
        fail['at'] = STOP_AT
        with pytest.raises(SystemExit):
            run(out, ids, *flags)
    fail['at'] = None
    with pytest.raises(SystemExit, match='pass --seeded-rows exactly when'):
        run(plain, ids, '--seeded-rows')                                # a build without the key stays as it was
    with pytest.raises(SystemExit, match='pass --seeded-rows exactly when'):
        run(keyed, ids)
    assert 'embedder_rows' not in json.loads((plain / 'settings.json').read_text(encoding='utf-8'))
    assert run(plain, ids)['processed'] == N and run(keyed, ids, '--seeded-rows')['processed'] == N


def test_seeded_rows_refuse_code_whose_embedder_takes_no_seed(stand_in, monkeypatch, tmp_path):
    class Old:
        FiveChannelEmbedder = type('E', (), {'__init__': lambda self, size: None})
    monkeypatch.setattr(build_b, 'import_team', lambda *args, **kwargs: Old())
    with pytest.raises(SystemExit, match='takes a seed'):
        run(tmp_path / 'x', stand_in[1], '--seeded-rows')
