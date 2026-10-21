"""The Mendeley corpus of build_cache: the bundle when one is configured, else the Mendeley_Ransomware and Mendeley_Goodware
collections under the `mendeley_collections` source. Synthetic folders only; the real-data comparison of the two routes is in
the step's build log, not in CI."""
import csv
import json
from pathlib import Path

import pytest

from common.paths import SOURCES, MissingSource, source_for
from s02_preprocess.cache import build_cache as bc

ROWS = [  # sample_id, label, family, representation, split
    ('a1', '1', 'conti', 'normal', 'train'), ('b2', '0', 'goodware', 'normal', 'test'), ('c3', '1', 'lockbit', 'upx_packed', 'train'),
    ('d4', '0', 'goodware', 'normal', ''), ('e5', '1', 'ryuk', 'normal', 'test')]
FIELDS = ['sample_id', 'label', 'family', 'representation', 'recommended_split', 'cfg_path']


def write_samples(folder, rows, cfg):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'samples.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({'sample_id': i, 'label': label, 'family': fam, 'representation': rep, 'recommended_split': split,
                     'cfg_path': cfg.format(sample=i, family=fam)} for i, label, fam, rep, split in rows)


@pytest.fixture
def folders(tmp_path, monkeypatch):
    """A bundle (all rows, cfg/<sample>.jsonl.gz, aliases csv) and the same rows split over two collections (cfg_path
    samples/<family>/<sample>.jsonl.gz, aliases json), as the real ones are."""
    for var in ('GT_BUNDLE', 'GT_MENDELEY_COLLECTIONS'):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv('GT_DATASETS', str(tmp_path / 'no-datasets.json'))
    bundle, coll = tmp_path / 'bundle', tmp_path / 'coll'
    write_samples(bundle, ROWS, 'cfg/{sample}.jsonl.gz')
    (bundle / 'provenance').mkdir()
    (bundle / 'provenance' / 'identity_aliases.csv').write_text('sha256,cohort_sha256,current_representation\n'
                                                                'a1,cohortA,normal\nb2,cohortB,normal\n', encoding='utf-8')
    for name, label, aliases in (('Mendeley_Ransomware', '1', [{'sha256': 'a1', 'cohort_sha256': 'cohortA'}]),
                                 ('Mendeley_Goodware', '0', [{'sha256': 'b2', 'cohort_sha256': 'cohortB'}])):
        write_samples(coll / name, [r for r in reversed(ROWS) if r[1] == label], 'samples/train/{family}/{sample}.jsonl.gz')
        (coll / name / 'provenance').mkdir()
        (coll / name / 'provenance' / 'identity_aliases.json').write_text(json.dumps(aliases), encoding='utf-8')
    return bundle, coll


def test_the_mendeley_source_has_an_environment_variable_and_no_default(folders):
    assert SOURCES['mendeley_collections'][0] == 'GT_MENDELEY_COLLECTIONS'
    with pytest.raises(MissingSource, match='GT_MENDELEY_COLLECTIONS'):
        source_for('mendeley_collections')


def test_without_a_bundle_both_collections_are_read_sorted_with_their_own_roots(folders, monkeypatch):
    bundle, coll = folders
    monkeypatch.setenv('GT_MENDELEY_COLLECTIONS', str(coll))
    assert bc.configured_bundle() is None
    assert bc.corpus_roots('mendeley') == [coll / 'Mendeley_Ransomware', coll / 'Mendeley_Goodware']
    rows = bc.corpus_samples('mendeley')
    assert [r['sample_id'] for r in rows] == ['a1', 'b2', 'c3', 'd4', 'e5']             # the bundle's order: by sample_id
    for r in rows:
        root = coll / ('Mendeley_Ransomware' if r['label'] == '1' else 'Mendeley_Goodware')
        assert Path(r['root']) == root and r['corpus'] == 'mendeley'
        assert r['cfg_path'] == f"samples/train/{r['family']}/{r['sample_id']}.jsonl.gz"       # relative to its own collection
        assert Path(r['root']) / r['cfg_path'] == root / 'samples' / 'train' / r['family'] / f"{r['sample_id']}.jsonl.gz"
    assert set(bc.corpus_aliases('mendeley')) == {'a1', 'b2'}                         # the json of both collections
    assert bc.corpus_aliases('mendeley')['b2']['cohort_sha256'] == 'cohortB'


def test_a_configured_bundle_is_read_as_before_and_beats_the_collections(folders, monkeypatch):
    bundle, coll = folders
    monkeypatch.setenv('GT_MENDELEY_COLLECTIONS', str(coll))
    monkeypatch.setenv('GT_BUNDLE', str(bundle))
    assert bc.corpus_roots('mendeley') == [bundle]
    rows = bc.corpus_samples('mendeley')
    assert [r['sample_id'] for r in rows] == [r[0] for r in ROWS] and {r['root'] for r in rows} == {str(bundle)}
    assert rows[0]['cfg_path'] == 'cfg/a1.jsonl.gz'
    assert set(bc.corpus_aliases('mendeley')) == {'a1', 'b2'}                         # the bundle's csv


def test_the_bundle_may_come_from_datasets_json(folders, monkeypatch, tmp_path):
    bundle, coll = folders
    registry = tmp_path / 'datasets.json'
    registry.write_text(json.dumps({'v2': {'cache': 'c', 'sources': {'bundle': str(bundle), 'mendeley_collections': str(coll)}}}),
                        encoding='utf-8')
    monkeypatch.setenv('GT_DATASETS', str(registry))
    assert bc.corpus_roots('mendeley') == [bundle]
    registry.write_text(json.dumps({'v2': {'cache': 'c', 'sources': {'mendeley_collections': str(coll)}}}), encoding='utf-8')
    assert bc.corpus_roots('mendeley') == [coll / 'Mendeley_Ransomware', coll / 'Mendeley_Goodware']


def test_both_routes_exclude_the_same_rows(folders, monkeypatch):
    bundle, coll = folders
    def reasons():
        aliases = bc.corpus_aliases('mendeley')
        return {r['sample_id']: bc.exclusion(r, aliases.get(r['sample_id'], {})) for r in bc.corpus_samples('mendeley')}
    monkeypatch.setenv('GT_BUNDLE', str(bundle))
    from_bundle = reasons()
    monkeypatch.delenv('GT_BUNDLE')
    monkeypatch.setenv('GT_MENDELEY_COLLECTIONS', str(coll))
    assert reasons() == from_bundle == {'a1': '', 'b2': '', 'c3': 'upx stub', 'd4': 'no recommended split', 'e5': ''}


def test_neither_route_found_names_both(folders, monkeypatch, tmp_path):
    with pytest.raises(MissingSource) as e:
        bc.corpus_roots('mendeley')
    assert 'GT_BUNDLE' in str(e.value) and 'GT_MENDELEY_COLLECTIONS' in str(e.value) and 'Mendeley_Goodware' in str(e.value)
    (tmp_path / 'half' / 'Mendeley_Ransomware').mkdir(parents=True)                  # one collection is not enough
    monkeypatch.setenv('GT_MENDELEY_COLLECTIONS', str(tmp_path / 'half'))
    with pytest.raises(MissingSource, match='half.*lacks one of them'):
        bc.corpus_roots('mendeley')


def test_other_corpora_keep_their_single_collection_root(folders, monkeypatch, tmp_path):
    monkeypatch.setenv('GT_COLLECTIONS', str(tmp_path / 'colls'))
    assert bc.corpus_roots('balanced_goodware') == [tmp_path / 'colls' / 'Balanced_Goodware']
    assert bc.corpus_roots('ransomware_balanced') == [tmp_path / 'colls' / 'Ransomware_Balanced']
