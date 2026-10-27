"""python -m pytest tests/test_package_build.py: `experiments/team_pipelines/package_build.py` on a synthetic build.

The zip name from the build's sampler and block budget, the layout that unzips at `CT_GAT/` (read back by `train_b.build_layout` and
its settings check), the README generated from the build's settings, the .sha256, the options and the refusals."""
import hashlib
import json
import re
import zipfile

import pytest
import torch

from experiments.team_pipelines import common as tp, package_build as pb, train_b
from test_team_pipelines import fresh_team_caches, pinned_team_modules  # noqa: F401
from test_team_v3 import synthetic_build
from tiny_cache import tiny  # noqa: F401  (a fixture)

BASE = 'processed/cfg_subgraphs/'


def finished_build(root, rows, sample='callwalk', num_blocks=500, **kw):
    """A synthetic build folder as `build_b` leaves it when finished: the samples with `dataset_metadata`, their manifests and
    build_summary.json."""
    ids = sorted(rows)
    folder = synthetic_build(root, ids, rows, source='repo', sample=sample, num_blocks=num_blocks, **kw)
    for i in ids:
        payload = torch.load(folder / f'{i}.pt', weights_only=False)
        payload['dataset_metadata'] = {'sample_id': i, 'label': rows[i]['label'], 'arch': 'x86', 'family': rows[i]['family'],
                                       'fold': rows[i]['fold']}
        torch.save(payload, folder / f'{i}.pt')
        (folder / f'{i}.manifest.json').write_text(json.dumps({'sample_id': i}), encoding='utf-8')
    (folder / 'build_summary.json').write_text(json.dumps({'processed': len(ids), 'total': len(ids), 'excluded': {}}), encoding='utf-8')
    return folder


@pytest.mark.parametrize('sample,blocks,name', [
    ('signal', 250, 'Mendeley_x86_cfg_subgraphs_signal_250blocks_v3.zip'),
    ('signal', 500, 'Mendeley_x86_cfg_subgraphs_signal_500blocks_v3.zip'),
    ('callwalk', 250, 'Mendeley_x86_cfg_subgraphs_callwalk_250blocks_v3.zip'),
    ('callwalk', 500, 'Mendeley_x86_cfg_subgraphs_callwalk_500blocks_v3.zip'),
    ('callwalk', 999999, 'Mendeley_x86_cfg_subgraphs_callwalk_999999blocks_v3.zip'),
    ('signal', train_b.UNCAPPED_BLOCKS, 'Mendeley_x86_cfg_subgraphs_uncapped_v3.zip'),
    ('callwalk', 0, 'Mendeley_x86_cfg_subgraphs_uncapped_v3.zip'),
    ('callwalk', train_b.UNCAPPED_BLOCKS + 5, 'Mendeley_x86_cfg_subgraphs_uncapped_v3.zip')])
def test_the_zip_name_carries_the_sampler_and_the_block_count(sample, blocks, name):
    assert pb.zip_name({'sample': sample, 'num_blocks': blocks}) == name


def test_other_samplers_and_missing_keys_are_refused():
    for settings in ({'sample': 'walk', 'num_blocks': 500}, {'num_blocks': 500}, {'sample': 'signal'}, {'sample': 'signal', 'num_blocks': -1}):
        with pytest.raises(SystemExit):
            pb.zip_name(settings)


def test_the_layout_unzips_at_ct_gat_and_train_b_reads_it(tiny, tmp_path):   # noqa: F811
    rows = tp.sample_rows()
    build = finished_build(tmp_path / 'b', rows)
    before = {p.name: p.read_bytes() for p in build.iterdir()}
    zpath, digest = pb.package(build, tmp_path / 'out')
    assert zpath.name == 'Mendeley_x86_cfg_subgraphs_callwalk_500blocks_v3.zip'
    assert {p.name: p.read_bytes() for p in build.iterdir()} == before                                   # the build is read only
    assert (tmp_path / 'out' / f'{zpath.name}.sha256').read_bytes() == f'{digest}  {zpath.name}\n'.encode('ascii')
    assert digest == hashlib.sha256(zpath.read_bytes()).hexdigest()
    ids = sorted(rows)
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        assert z.testzip() is None
        assert names[0] == f'README_{zpath.stem}.md'
        assert sorted(n for n in names[1:] if not n.startswith(BASE)) == ['processed/api_vocab.json']
        assert sorted(n[len(BASE):] for n in names if n.startswith(BASE) and n.count('/') == 2) == sorted(
            ['channel_embedder.pt'] + [f'{i}.pt' for i in ids] + [f'{i}.manifest.json' for i in ids])
        assert sorted(n[len(BASE):] for n in names if n.startswith(BASE + 'build_info/')) == [
            'build_info/build_log.csv', 'build_info/build_settings.json', 'build_info/build_summary.json', 'build_info/index.csv']
        assert {z.getinfo(f'{BASE}{i}.pt').compress_type for i in ids} == {zipfile.ZIP_STORED}
        z.extractall(tmp_path / 'CT_GAT')
        index = z.read(BASE + 'build_info/index.csv').decode('utf-8').splitlines()
    unzipped = tmp_path / 'CT_GAT' / 'processed' / 'cfg_subgraphs'
    assert [l.split(',')[0] for l in index[1:]] == ids and index[0].split(',')[:3] == ['sample_id', 'label', 'family']
    layout = train_b.build_layout(unzipped)
    assert layout.kind == 'zip' and layout.vocab == tmp_path / 'CT_GAT' / 'processed' / 'api_vocab.json'
    assert train_b.check_build('callwalk500-emb', unzipped) == 'repo' and train_b.done_ids(unzipped) == set(ids)
    with pytest.raises(SystemExit, match='sampler callwalk'):
        train_b.check_build('sig250-feat', unzipped)
    first = torch.load(unzipped / f'{ids[0]}.pt', weights_only=False)
    assert torch.equal(first['block_embeddings'], torch.load(build / f'{ids[0]}.pt', weights_only=False)['block_embeddings'])


def test_the_readme_is_generated_from_the_settings_and_names_no_machine_path(tiny, tmp_path):   # noqa: F811
    rows = tp.sample_rows()
    build = finished_build(tmp_path / 'b', rows, sample='signal', num_blocks=250)
    zpath, _ = pb.package(build, tmp_path / 'out')
    text = (tmp_path / 'out' / f'{zpath.stem}.README.md').read_text(encoding='utf-8')
    with zipfile.ZipFile(zpath) as z:
        assert z.read(f'README_{zpath.stem}.md').decode('utf-8') == text
    ransomware = sum(r['label'] == 1 for r in rows.values())
    assert '`--sample signal`, budget 250 blocks' in text and f'({ransomware} ransomware,' in text and f'{len(rows) - ransomware} goodware' in text
    assert 'highest signal first' in text and 'source `repo`' in text and 'CT_GAT/pipeline/s02_preprocess/preprocess_full.py' in text
    assert 'both' not in text.split('How it was built')[1].split('Before training')[0]
    for forbidden in (str(tmp_path), 'C:\\', 'C:/', '/Users/', 'Downloads', 'AppData'):
        assert forbidden not in text and forbidden not in pb.readme_text({'sample': 'signal', 'num_blocks': 250}, {'processed': 0, 'total': 0}, [])
    walk = finished_build(tmp_path / 'c', rows, sample='callwalk', num_blocks=500)
    zpath2, _ = pb.package(walk, tmp_path / 'out')
    text2 = (tmp_path / 'out' / f'{zpath2.stem}.README.md').read_text(encoding='utf-8')
    assert '`--sample callwalk`, budget 500 blocks' in text2 and 'call edges in both directions' in text2


def test_the_uncapped_zip_says_it_serves_both_samplers(tiny, tmp_path):   # noqa: F811
    rows = tp.sample_rows()
    for sample in ('signal', 'callwalk'):
        build = finished_build(tmp_path / sample, rows, sample=sample, num_blocks=train_b.UNCAPPED_BLOCKS)
        zpath, _ = pb.package(build, tmp_path / 'out' / sample, manifests=False)
        assert zpath.name == 'Mendeley_x86_cfg_subgraphs_uncapped_v3.zip'
        text = (zpath.parent / f'{zpath.stem}.README.md').read_text(encoding='utf-8')
        assert 'uncapped build of both `--sample signal` and `--sample callwalk`' in text and 'no block budget' in text


def gnu_sha256sum_check(directory, sha_file):
    """`sha256sum -c <sha_file>` as GNU coreutils reads it: every line is `<64 lowercase hex>`, two spaces (text mode), then the name
    up to the newline. A CR belongs to the name, so a CRLF file names a file that does not exist and fails."""
    raw = sha_file.read_bytes()
    assert raw.endswith(b'\n')
    results = []
    for line in raw.split(b'\n')[:-1]:
        m = re.fullmatch(rb'([0-9a-f]{64})  (.+)', line)
        assert m, line
        target = directory / m.group(2).decode('utf-8')
        results.append(target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == m.group(1).decode('ascii'))
    return results


def test_the_sha256_and_readme_files_have_unix_line_ends_and_check_like_sha256sum(tiny, tmp_path):   # noqa: F811
    rows = tp.sample_rows()
    build = finished_build(tmp_path / 'b', rows)
    zpath, digest = pb.package(build, tmp_path / 'out')
    sha = tmp_path / 'out' / f'{zpath.name}.sha256'
    assert b'\r' not in sha.read_bytes() and sha.read_bytes() == f'{digest}  {zpath.name}\n'.encode('ascii')
    assert gnu_sha256sum_check(tmp_path / 'out', sha) == [True]
    readme = (tmp_path / 'out' / f'{zpath.stem}.README.md').read_bytes()
    assert b'\r' not in readme
    with zipfile.ZipFile(zpath) as z:
        assert z.read(f'README_{zpath.stem}.md') == readme and b'\r' not in z.read(BASE + 'build_info/index.csv')
    crlf = tmp_path / 'crlf.sha256'                                                                     # what write_text wrote on Windows
    crlf.write_bytes(sha.read_bytes().replace(b'\n', b'\r\n'))
    assert gnu_sha256sum_check(tmp_path / 'out', crlf) == [False]


def test_the_options_no_manifests_deflate_pt_and_out(tiny, tmp_path):   # noqa: F811
    rows = tp.sample_rows()
    build = finished_build(tmp_path / 'b', rows)
    out = tmp_path / 'new' / 'dir'
    assert pb.main([str(build), '--out', str(out), '--no-manifests', '--deflate-pt']) == 0
    zpath = out / 'Mendeley_x86_cfg_subgraphs_callwalk_500blocks_v3.zip'
    with zipfile.ZipFile(zpath) as z:
        assert not [n for n in z.namelist() if n.endswith('.manifest.json')]
        pts = [i for i in z.infolist() if i.filename.endswith('.pt') and 'channel_embedder' not in i.filename]
        assert len(pts) == len(rows) and {i.compress_type for i in pts} == {zipfile.ZIP_DEFLATED}
        readme = z.read(f'README_{zpath.stem}.md').decode('utf-8')
        assert 'left out of this archive' in readme and '.manifest.json' not in readme.split('```')[1]   # nor in the layout block
    default = pb.package(build, tmp_path / 'default')[0]
    with zipfile.ZipFile(default) as z:
        assert len([n for n in z.namelist() if n.endswith('.manifest.json')]) == len(rows)
        readme = z.read(f'README_{default.stem}.md').decode('utf-8')
        assert 'left out of this archive' not in readme and '<sha256>.manifest.json' in readme.split('```')[1]


def test_a_seeded_rows_build_says_so_in_its_readme(tiny, tmp_path):   # noqa: F811
    rows = tp.sample_rows()
    build = finished_build(tmp_path / 'b', rows)
    body = json.loads((build / 'settings.json').read_text(encoding='utf-8'))
    (build / 'settings.json').write_text(json.dumps({**body, 'embedder_rows': 'seeded-per-id'}), encoding='utf-8')
    zpath, _ = pb.package(build, tmp_path / 'out')
    assert '`--seeded-rows`' in (zpath.parent / f'{zpath.stem}.README.md').read_text(encoding='utf-8')


def test_an_unfinished_or_incomplete_build_is_refused(tiny, tmp_path):   # noqa: F811
    rows = tp.sample_rows()
    build = finished_build(tmp_path / 'b', rows)
    summary = build / 'build_summary.json'
    good = summary.read_text(encoding='utf-8')
    summary.write_text(json.dumps({'processed': len(rows) - 1, 'total': len(rows), 'excluded': {}}), encoding='utf-8')
    with pytest.raises(SystemExit, match='not finished'):
        pb.package(build, tmp_path / 'out')
    summary.write_text(json.dumps({'processed': len(rows) - 1, 'total': len(rows), 'excluded': {'x' * 64: 'too big'}}), encoding='utf-8')
    with pytest.raises(SystemExit, match='holds 40 sample files'):                                       # finished on paper, one file too many
        pb.package(build, tmp_path / 'out')
    summary.write_text(good, encoding='utf-8')
    (build / 'api_vocab.json').unlink()
    with pytest.raises(SystemExit, match='api_vocab.json is missing'):
        pb.package(build, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()                                                              # nothing written for a refused build
