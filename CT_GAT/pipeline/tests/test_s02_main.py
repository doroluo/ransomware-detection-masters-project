"""`preprocess_full.main` end to end on tiny collection folders (see s02_collection.py)."""
import csv
import json
import sys
from pathlib import Path

import pytest
import torch

import s02_collection as col

STAGE = Path(__file__).resolve().parents[1] / 's02_preprocess'
sys.path.insert(0, str(STAGE))
import preprocess_full as pf  # noqa: E402
import tokenize_asm_gz as tok  # noqa: E402


def sha(n):
    return f'{n:064x}'


def run_status(tmp_path, roots, *extra, out='out', vocab='vocab.json', seed=0):
    """Run `main` on the roots; returns (exit status, output folder, vocabulary file)."""
    out_dir, vocab_path = tmp_path / out, tmp_path / vocab
    argv = [arg for root in roots for arg in ('--root', str(root))]
    argv += ['--num-blocks', '8', '--seed', str(seed), '--output', str(out_dir), '--vocab', str(vocab_path), *extra]
    return pf.main(argv), out_dir, vocab_path


def run(tmp_path, roots, *extra, **kwargs):
    """`run_status` for a run that must succeed; returns (output folder, vocabulary file)."""
    status, out_dir, vocab_path = run_status(tmp_path, roots, *extra, **kwargs)
    assert status == 0
    return out_dir, vocab_path


def api_column(out_dir, sample_id):
    payload = torch.load(out_dir / f'{sample_id}.pt', weights_only=False)
    return payload['instruction_ids'][:, 3].tolist()


# ---- 1. the vocabulary is fitted on training rows only -----------------------------------------------------------

def test_a_test_only_api_gets_unk_and_a_train_api_keeps_its_id(tmp_path):
    # root a holds only a test sample and is listed first; root b holds the training sample
    a = col.write_collection(tmp_path / 'a', [(sha(1), 'test', 1, ['CreateFileW', 'RegSetValueExW'])])
    b = col.write_collection(tmp_path / 'b', [(sha(2), 'train', 0, ['CreateFileW'])])
    out_dir, vocab_path = run(tmp_path, [a, b])
    assert json.loads(vocab_path.read_text(encoding='utf-8')) == ['CreateFileW']
    assert tok.FIRST_API_ID in api_column(out_dir, sha(2))
    test_apis = [value for value in api_column(out_dir, sha(1)) if value != tok.NO_API]
    assert sorted(test_apis) == [tok.UNK_API, tok.FIRST_API_ID]


def test_split_all_and_split_train_build_the_same_vocabulary(tmp_path):
    a = col.write_collection(tmp_path / 'a', [(sha(1), 'test', 1, ['RegSetValueExW']), (sha(3), 'train', 1, ['CloseHandle'])])
    _, vocab_all = run(tmp_path, [a], out='all', vocab='all.json')
    _, vocab_train = run(tmp_path, [a], '--split', 'train', out='train', vocab='train.json')
    assert vocab_all.read_text(encoding='utf-8') == vocab_train.read_text(encoding='utf-8')
    assert json.loads(vocab_all.read_text(encoding='utf-8')) == ['CloseHandle']


# ---- 2. the embedder is seeded -----------------------------------------------------------------------------------

def three_samples(tmp_path, name):
    return col.write_collection(tmp_path / name, [
        (sha(1), 'train', 1, ['CreateFileW']),
        (sha(2), 'train', 0, ['CreateFileW', 'RegSetValueExW']),
        (sha(3), 'test', 1, ['CloseHandle']),
    ])


def block_embeddings(out_dir):
    return {sample_id: torch.load(out_dir / f'{sample_id}.pt', weights_only=False)['block_embeddings']
            for sample_id in (sha(1), sha(2), sha(3))}


def test_two_fresh_runs_give_identical_block_embeddings(tmp_path):
    root = three_samples(tmp_path, 'c')
    first, _ = run(tmp_path, [root], out='first', vocab='v1.json')
    torch.manual_seed(999)                       # whatever the process did in between
    second, _ = run(tmp_path, [root], out='second', vocab='v2.json')
    a, b = block_embeddings(first), block_embeddings(second)
    for sample_id in a:
        assert a[sample_id].numel() and torch.equal(a[sample_id], b[sample_id])
    other, _ = run(tmp_path, [root], out='other', vocab='v3.json', seed=1)
    assert not torch.equal(block_embeddings(other)[sha(1)], a[sha(1)])


def test_new_api_rows_depend_on_the_seed_and_the_row_only():
    one = pf.FiveChannelEmbedder(tok.FIRST_API_ID, seed=7)
    one.expand_api(tok.FIRST_API_ID + 2)
    one.expand_api(tok.FIRST_API_ID + 5)
    torch.manual_seed(123)
    two = pf.FiveChannelEmbedder(tok.FIRST_API_ID, seed=7)
    two.expand_api(tok.FIRST_API_ID + 5)
    assert torch.equal(one.api_embedding.weight[tok.FIRST_API_ID:], two.api_embedding.weight[tok.FIRST_API_ID:])
    assert float(one.api_embedding.weight[0].detach().abs().sum()) == 0.0
    three = pf.FiveChannelEmbedder(tok.FIRST_API_ID, seed=8)
    three.expand_api(tok.FIRST_API_ID + 5)
    assert not torch.equal(one.api_embedding.weight[tok.FIRST_API_ID:], three.api_embedding.weight[tok.FIRST_API_ID:])


# ---- 3. a failed sample is recorded and fails the run --------------------------------------------------------------

def corrupt(path):
    col.write_gz(path, 'this is not json')


def collection_with_one_bad_sample(tmp_path):
    root = col.write_collection(tmp_path / 'c', [
        (sha(1), 'train', 1, ['CreateFileW']),
        (sha(2), 'train', 0, ['CloseHandle']),
        (sha(3), 'test', 1, ['CreateFileW']),
    ])
    corrupt(root / 'samples' / 'train' / 'fam' / f'{sha(1)}.jsonl.gz')
    return root


def test_a_failed_sample_is_listed_and_the_run_goes_on_and_exits_non_zero(tmp_path):
    root = collection_with_one_bad_sample(tmp_path)
    status, out_dir, _ = run_status(tmp_path, [root])
    assert status == 1
    failures = json.loads((out_dir / 'failures.json').read_text(encoding='utf-8'))
    assert [item['sample_id'] for item in failures] == [sha(1)]
    assert 'CfgSchemaError' in failures[0]['error'] and 'not JSON' in failures[0]['error']
    assert sorted(path.stem for path in out_dir.glob('*.pt') if path.stem != 'channel_embedder') == [sha(2), sha(3)]


def test_fail_fast_stops_at_the_first_failure(tmp_path):
    root = collection_with_one_bad_sample(tmp_path)
    status, out_dir, _ = run_status(tmp_path, [root], '--fail-fast')
    assert status == 1
    assert [item['sample_id'] for item in json.loads((out_dir / 'failures.json').read_text(encoding='utf-8'))] == [sha(1)]
    assert not list(out_dir.glob(f'{sha(2)}.pt')) and not list(out_dir.glob(f'{sha(3)}.pt'))


def test_a_clean_run_exits_zero_and_removes_an_old_failures_file(tmp_path):
    root = three_samples(tmp_path, 'ok')
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    (out_dir / 'failures.json').write_text('[]', encoding='utf-8')
    status, out_dir, _ = run_status(tmp_path, [root])
    assert status == 0 and not (out_dir / 'failures.json').exists()


# ---- 4. the size caps read the splits CSV and are off by default -------------------------------------------------

CAP_COLUMNS = ['asm_bytes', 'num_blocks', 'num_instructions']


def sized_collection(tmp_path):
    return col.write_collection(tmp_path / 'sized', [
        (sha(1), 'train', 1, ['CreateFileW'], {'asm_bytes': 900, 'num_blocks': 4, 'num_instructions': 40}),
        (sha(2), 'train', 0, ['CloseHandle'], {'asm_bytes': 100, 'num_blocks': 9, 'num_instructions': 40}),
        (sha(3), 'test', 1, ['CreateFileW'], {'asm_bytes': 100, 'num_blocks': 4, 'num_instructions': 40}),
    ], extra_columns=CAP_COLUMNS)


def built(out_dir):
    return sorted(path.stem for path in out_dir.glob('*.pt') if path.stem != 'channel_embedder')


def test_no_cap_is_set_by_default(tmp_path):
    out_dir, _ = run(tmp_path, [sized_collection(tmp_path)])
    assert built(out_dir) == [sha(1), sha(2), sha(3)] and not (out_dir / 'skipped.json').exists()


@pytest.mark.parametrize('flag, value, skipped', [
    ('--max-asm-bytes', '500', sha(1)),
    ('--max-cfg-blocks', '5', sha(2)),
    ('--max-cfg-instructions', '30', None),
])
def test_a_cap_skips_the_right_samples_and_records_them(tmp_path, flag, value, skipped):
    out_dir, _ = run(tmp_path, [sized_collection(tmp_path)], flag, value)
    recorded = json.loads((out_dir / 'skipped.json').read_text(encoding='utf-8'))
    expected = [sha(1), sha(2), sha(3)] if skipped is None else [skipped]
    assert [item['sample_id'] for item in recorded] == expected
    word = {'--max-asm-bytes': 'asm_bytes', '--max-cfg-blocks': 'blocks', '--max-cfg-instructions': 'instructions'}[flag]
    assert all(word in item['reason'] for item in recorded)
    assert built(out_dir) == [i for i in (sha(1), sha(2), sha(3)) if i not in expected]


# ---- 5. where the CFG is -------------------------------------------------------------------------------------------

def one_each(tmp_path, name, **kwargs):
    return col.write_collection(tmp_path / name, [(sha(1), 'train', 1, ['CreateFileW']), (sha(3), 'test', 0, ['CloseHandle'])],
                                **kwargs)


@pytest.mark.parametrize('with_cfg_path', [False, True])
def test_the_cfg_beside_the_listing_or_named_by_the_cfg_path_column(tmp_path, with_cfg_path):
    root = one_each(tmp_path, 'beside', with_cfg_path=with_cfg_path)
    out_dir, _ = run(tmp_path, [root])
    assert built(out_dir) == [sha(1), sha(3)]


@pytest.mark.parametrize('with_cfg_path', [False, True])
def test_cfg_root_points_at_a_separate_cfg_tree(tmp_path, with_cfg_path):
    cfg_tree = tmp_path / 'cfgs'
    root = one_each(tmp_path, 'apart', cfg_root=cfg_tree, with_cfg_path=with_cfg_path)
    assert not list(root.rglob('*.jsonl.gz'))
    status, out_dir, _ = run_status(tmp_path, [root])
    assert status == 1 and built(out_dir) == []        # without --cfg-root the CFG is missing
    assert 'missing CFG' in json.loads((out_dir / 'failures.json').read_text(encoding='utf-8'))[0]['error']
    out_dir, _ = run(tmp_path, [root], '--cfg-root', str(cfg_tree), out='with_root')
    assert built(out_dir) == [sha(1), sha(3)]


def test_a_cfg_path_that_leaves_the_root_is_refused(tmp_path):
    root = one_each(tmp_path, 'escape', with_cfg_path=True)
    for split in ('train', 'test'):
        path = root / 'splits' / f'{split}.csv'
        with path.open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row['cfg_path'] = '../' + row['cfg_path']
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    status, out_dir, _ = run_status(tmp_path, [root])
    assert status == 1 and 'unsafe CFG path' in json.loads((out_dir / 'failures.json').read_text(encoding='utf-8'))[0]['error']


def test_a_splits_csv_without_a_collection_column_uses_the_folder_name(tmp_path):
    root = one_each(tmp_path, 'nocollection', drop_columns=('collection',))
    rows = tok._selected_rows([root], 'all', None, None)
    assert {row['collection'] for row in rows} == {'nocollection'}
    out_dir, _ = run(tmp_path, [root])
    assert built(out_dir) == [sha(1), sha(3)]


# ---- 6. a folder keeps one set of settings -------------------------------------------------------------------------

def test_a_rerun_with_the_same_settings_goes_on_and_one_with_other_settings_is_refused(tmp_path):
    root = three_samples(tmp_path, 'guard')
    out_dir, _ = run(tmp_path, [root])
    settings = json.loads((out_dir / 'settings.json').read_text(encoding='utf-8'))
    assert settings['sample'] == 'walk' and settings['seed'] == 0 and settings['num_blocks'] == 8
    assert settings['include_library'] is False and set(settings['source_sha256']) == {'preprocess_full.py', 'tokenize_asm_gz.py'}
    run(tmp_path, [root], '--split', 'test')                       # another split of the same settings is fine
    for flag, value in (('--sample', 'signal'), ('--sample', 'callwalk'), ('--seed', '3'), ('--num-blocks', '9'), ('--max-instructions', '50')):
        options = {'--num-blocks': '8', '--seed': '0', '--output': str(out_dir), '--vocab': str(tmp_path / 'vocab.json'),
                   flag: value}
        with pytest.raises(SystemExit) as stop:
            pf.main(['--root', str(root), *[word for item in options.items() for word in item]])
        assert stop.value.code == 2
    with pytest.raises(SystemExit):
        run(tmp_path, [root], '--include-library')


def test_sample_callwalk_on_the_command_line_records_itself_and_its_regions(tmp_path):
    root = three_samples(tmp_path, 'callwalk')
    out_dir, _ = run(tmp_path, [root], '--sample', 'callwalk')
    assert json.loads((out_dir / 'settings.json').read_text(encoding='utf-8'))['sample'] == 'callwalk'
    for sample_id in (sha(1), sha(2), sha(3)):
        sampling = json.loads((out_dir / f'{sample_id}.manifest.json').read_text(encoding='utf-8'))['sampling']
        assert sampling['mode'] == 'callwalk' and sampling['regions'] == 1 and sampling['unresolved_call_targets'] == 0
        payload = torch.load(out_dir / f'{sample_id}.pt', weights_only=False)
        assert payload['edge_attr'].shape[1] == 7 and payload['edge_type_names'] == list(pf.EDGE_TYPE_NAMES)


def test_overwrite_rebuilds_with_the_new_settings(tmp_path):
    root = three_samples(tmp_path, 'over')
    out_dir, _ = run(tmp_path, [root])
    out_dir, _ = run(tmp_path, [root], '--overwrite', '--include-library')
    assert json.loads((out_dir / 'settings.json').read_text(encoding='utf-8'))['include_library'] is True


def test_a_folder_of_samples_without_settings_is_refused(tmp_path):
    root = three_samples(tmp_path, 'legacy')
    out_dir, _ = run(tmp_path, [root])
    (out_dir / 'settings.json').unlink()
    with pytest.raises(SystemExit):
        run(tmp_path, [root])
    run(tmp_path, [root], '--overwrite')


def test_a_changed_source_file_is_a_change_of_settings(tmp_path, monkeypatch):
    root = three_samples(tmp_path, 'source')
    run(tmp_path, [root])
    real = pf._run_settings

    def edited(args):
        settings = real(args)
        settings['source_sha256'] = {**settings['source_sha256'], 'preprocess_full.py': '0' * 64}
        return settings

    monkeypatch.setattr(pf, '_run_settings', edited)
    with pytest.raises(SystemExit):
        run(tmp_path, [root])


# ---- 7. one pass over the listing ----------------------------------------------------------------------------------

def five_samples(tmp_path):
    apis = [['CreateFileW'], ['CreateFileW', 'RegSetValueExW'], ['CloseHandle', 'CreateFileW', 'ReadFile'], [], ['WriteFile']]
    return col.write_collection(tmp_path / 'five', [(sha(i + 1), 'train', i % 2, names) for i, names in enumerate(apis)])


def test_the_collector_gives_the_rows_of_collect_ida_imports(tmp_path):
    root = five_samples(tmp_path)
    paths = sorted(root.rglob('*.asm.gz'))
    for path in paths:
        lines = list(pf.iter_text_lines(path))
        collector = tok.IdaImportCollector()
        for line in lines:
            collector.feed(line)
        assert collector.rows == tok.collect_ida_imports(lines)
    assert any(tok.collect_ida_imports(pf.iter_text_lines(path)) for path in paths)


def test_process_sample_reads_the_listing_once_and_matches_the_two_pass_result(tmp_path, monkeypatch):
    root = five_samples(tmp_path)
    reads = []
    real = pf.iter_text_lines
    monkeypatch.setattr(pf, 'iter_text_lines', lambda path: reads.append(path) or real(path))
    for index in range(5):
        sample_id = sha(index + 1)
        asm = root / 'samples' / 'train' / 'fam' / f'{sample_id}.asm.gz'
        cfg_path = asm.with_name(f'{sample_id}.jsonl.gz')
        reads.clear()
        result = pf.process_sample(asm, cfg_path, label=index % 2, seed=0, api_vocab=tok.ApiVocab(), num_blocks=8,
                                   file_id=sample_id)
        assert len(reads) == 1
        # the two-pass flow: align, then tokenize with a second scan of the listing for the imports
        cfg = pf.load_cfg(cfg_path)
        sample = pf.sample_connected_subgraph(cfg, seed=0, num_blocks=8)
        pf.align_sampled_blocks(cfg, sample.selected_keys, real(asm), [])
        inventory = pf.tokenize_selected_blocks(cfg, sample.selected_keys, real(asm), tok.ApiVocab())
        old_ids = [instruction.token_ids for key in sample.selected_keys for instruction in cfg.blocks[key].instructions]
        new_ids = result['models']['model1']['instruction_ids'].tolist()
        assert old_ids == new_ids
        assert inventory == result['manifest']['import_inventory']


# ---- 8. what a run saves, and when ---------------------------------------------------------------------------------

def test_limit_and_sample_id_runs_never_write_the_shared_vocabulary(tmp_path):
    root = three_samples(tmp_path, 'limit')
    vocab_path = tmp_path / 'shared.json'
    vocab_path.write_text(json.dumps(['Existing']), encoding='utf-8')
    run(tmp_path, [root], '--limit', '2', out='limited', vocab='shared.json')
    run(tmp_path, [root], '--sample-id', sha(2), out='one', vocab='shared.json')
    assert json.loads(vocab_path.read_text(encoding='utf-8')) == ['Existing']
    run(tmp_path, [root], out='full', vocab='shared.json')
    assert json.loads(vocab_path.read_text(encoding='utf-8'))[0] == 'Existing' and len(json.loads(vocab_path.read_text(encoding='utf-8'))) > 1


@pytest.mark.parametrize('every, writes', [(50, 1), (2, 2), (1, 3)])
def test_vocabulary_and_embedder_are_saved_at_checkpoints_and_once_at_the_end(tmp_path, monkeypatch, every, writes):
    root = three_samples(tmp_path, 'saves')
    calls = []
    real_vocab, real_torch_save = tok.save_api_vocab, torch.save
    monkeypatch.setattr(tok, 'save_api_vocab', lambda vocab, path: calls.append('vocab') or real_vocab(vocab, path))
    monkeypatch.setattr(pf.torch, 'save', lambda obj, path, *a, **k: calls.append(Path(path).name) or real_torch_save(obj, path, *a, **k))
    run(tmp_path, [root], '--checkpoint-every', str(every))
    assert calls.count('vocab') == writes and calls.count('channel_embedder.pt') == writes


def test_a_rerun_skips_the_samples_of_the_checkpoint_and_rebuilds_the_rest(tmp_path):
    import os
    root = three_samples(tmp_path, 'resume')
    out_dir, _ = run(tmp_path, [root])
    assert json.loads((out_dir / 'checkpoint.json').read_text(encoding='utf-8'))['done'] == sorted([sha(1), sha(2), sha(3)])
    for sample_id in (sha(1), sha(2), sha(3)):
        os.utime(out_dir / f'{sample_id}.pt', (1, 1))
    (out_dir / 'checkpoint.json').write_text(json.dumps({'done': [sha(1), sha(3)]}), encoding='utf-8')
    run(tmp_path, [root])
    stamps = {sample_id: (out_dir / f'{sample_id}.pt').stat().st_mtime for sample_id in (sha(1), sha(2), sha(3))}
    assert stamps[sha(1)] == 1 and stamps[sha(3)] == 1 and stamps[sha(2)] > 1


# ---- 7. --num-blocks 0 and --whole keep every block ----------------------------------------------------------------

def run_uncapped(tmp_path, root, out, *flags, vocab='vocab.json'):
    argv = ['--root', str(root), '--seed', '0', '--output', str(tmp_path / out), '--vocab', str(tmp_path / vocab), *flags]
    return pf.main(argv), tmp_path / out


def test_no_budget_at_all_is_refused_and_whole_conflicts_with_a_block_budget(tmp_path):
    root = three_samples(tmp_path, 'nobudget')
    with pytest.raises(SystemExit) as stop:
        run_uncapped(tmp_path, root, 'none')
    assert stop.value.code == 2
    with pytest.raises(SystemExit) as stop:
        run_uncapped(tmp_path, root, 'both', '--whole', '--num-blocks', '8')
    assert stop.value.code == 2


@pytest.mark.parametrize('sample_mode', ['walk', 'signal', 'callwalk'])
def test_whole_and_num_blocks_zero_equal_a_budget_larger_than_the_binary(tmp_path, sample_mode):
    root = three_samples(tmp_path, 'whole')
    outs = {}
    for name, flags in (('zero', ['--num-blocks', '0']), ('whole', ['--whole']), ('whole0', ['--whole', '--num-blocks', '0']),
                        ('huge', ['--num-blocks', str(10**9)])):
        status, outs[name] = run_uncapped(tmp_path, root, name, *flags, '--sample', sample_mode, vocab=f'{name}.json')
        assert status == 0
    reference = block_embeddings(outs['huge'])
    for name in ('zero', 'whole', 'whole0'):
        current = block_embeddings(outs[name])
        for sample_id in reference:
            assert reference[sample_id].numel() and torch.equal(reference[sample_id], current[sample_id])
    settings = {name: json.loads((out / 'settings.json').read_text(encoding='utf-8')) for name, out in outs.items()}
    assert settings['zero'] == settings['whole'] == settings['whole0']
    assert settings['whole']['num_blocks'] == 0 and settings['huge']['num_blocks'] == 10**9


def test_the_settings_guard_treats_uncapped_as_another_setting(tmp_path):
    root = three_samples(tmp_path, 'guardwhole')
    out_dir, _ = run(tmp_path, [root])                                  # --num-blocks 8
    for flags in (['--whole'], ['--num-blocks', '0']):
        with pytest.raises(SystemExit) as stop:
            run_uncapped(tmp_path, root, 'out', *flags)
        assert stop.value.code == 2
    status, out_dir = run_uncapped(tmp_path, root, 'out', '--whole', '--overwrite')
    assert status == 0 and json.loads((out_dir / 'settings.json').read_text(encoding='utf-8'))['num_blocks'] == 0
    status, _ = run_uncapped(tmp_path, root, 'out', '--num-blocks', '0')   # the same mode again goes on
    assert status == 0
    with pytest.raises(SystemExit) as stop:
        run_uncapped(tmp_path, root, 'out', '--num-blocks', '8')
    assert stop.value.code == 2


# ---- 7. a collection whose rows are all unassigned -----------------------------------------------------------------

def unassigned_collection(tmp_path, name):
    """A collection like Ransomware_Balanced: splits/train.csv and test.csv hold the header only, every row is in unassigned.csv."""
    root = three_samples(tmp_path, name)
    rows = []
    for split in ('train', 'test'):
        path = root / 'splits' / f'{split}.csv'
        with path.open(newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows += list(reader)
        path.write_text(','.join(fields) + '\n', encoding='utf-8')
    with (root / 'splits' / 'unassigned.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return root


def test_preprocess_full_on_a_collection_with_no_split_rows_exits_1_and_leaves_no_settings(tmp_path):
    root = unassigned_collection(tmp_path, 'Unassigned')
    out_dir = tmp_path / 'out'
    with pytest.raises(SystemExit) as stop:
        pf.main(['--root', str(root), '--num-blocks', '8', '--output', str(out_dir), '--vocab', str(tmp_path / 'vocab.json')])
    message = str(stop.value)
    assert stop.value.code == message                                     # a string code: printed on stderr, exit status 1
    assert str(root.resolve()) in message and 'splits/train.csv and splits/test.csv' in message
    assert 'The rows of Unassigned are in splits/unassigned.csv' in message and 'Mendeley collections fill' in message
    assert not (out_dir / 'settings.json').exists()
    assert not (tmp_path / 'vocab.json').exists()


def test_tokenize_asm_gz_on_a_collection_with_no_split_rows_exits_1_and_writes_nothing(tmp_path, monkeypatch):
    root = unassigned_collection(tmp_path, 'Unassigned')
    out_dir = tmp_path / 'tok'
    monkeypatch.setattr(sys, 'argv', ['tokenize_asm_gz.py', '--root', str(root), '--output', str(out_dir)])
    with pytest.raises(SystemExit) as stop:
        tok.main()
    assert 'splits/unassigned.csv' in str(stop.value) and not out_dir.exists()


def test_an_empty_selection_by_a_filter_does_not_blame_the_split_files(tmp_path):
    root = three_samples(tmp_path, 'Filtered')
    with pytest.raises(SystemExit) as stop:
        pf.main(['--root', str(root), '--num-blocks', '8', '--sample-id', sha(99), '--output', str(tmp_path / 'out'),
                 '--vocab', str(tmp_path / 'vocab.json')])
    message = str(stop.value)
    assert f'--sample-id {sha(99)}' in message and 'unassigned' not in message
    assert not (tmp_path / 'out' / 'settings.json').exists()
