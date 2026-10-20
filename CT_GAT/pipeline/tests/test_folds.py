"""The family-holdout fold builder (s02_preprocess/cache/folds.py).

What is tested: the assignment is the algorithm that wrote tests/fixtures/folds_mendeley.csv (a copy of
results/family_holdout/folds_mendeley.csv of the LLM_Classification history). Fed that CSV's own `family` and `group`
columns it returns its `fold` column on every row. What is not reproduced: the CSV rebuilt from the bundle's samples.csv,
because the bundle lacks the goodware stream-hash groups and the original 2,509-file cohort; the last test records that as
a strict xfail so that it turns into a failure the day the inputs exist.
"""
import collections
import csv
from pathlib import Path

import pytest

from s02_preprocess.cache import folds

FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'folds_mendeley.csv'


def read_fixture():
    with FIXTURE.open(encoding='utf-8', newline='') as fh:
        return list(csv.DictReader(fh))


def pool_of(csv_rows):
    return [{'sha256': r['sha256'], 'label': int(r['label']), 'family': r['family'], 'group': r['group'],
             'arch': r['arch'], 'orig_set': r['orig_set'], 'corpus': r['corpus']} for r in csv_rows]


def test_fixture_is_the_committed_csv():
    rows = read_fixture()
    assert len(rows) == 2509 and set(rows[0]) == set(folds.FIELDS)
    assert collections.Counter((r['label'], r['arch']) for r in rows) == {('1', 'x86'): 1152, ('1', 'x64'): 114,
                                                                          ('0', 'x86'): 749, ('0', 'x64'): 494}
    assert len({r['family'] for r in rows if r['label'] == '1'}) == 38
    assert len({r['group'] for r in rows if r['label'] == '0'}) == 1024


def test_assign_folds_gives_back_the_committed_fold_column_from_the_csv_own_groups():
    """Not a regeneration: the families and goodware groups are taken from the CSV."""
    committed = read_fixture()
    rebuilt = folds.assign_folds(pool_of(committed))
    assert [(r['sha256'], str(r['fold'])) for r in rebuilt] == [(r['sha256'], r['fold']) for r in committed]


def test_written_csv_equals_the_committed_csv(tmp_path):
    committed = read_fixture()
    folds.write_csv(tmp_path / 'f.csv', folds.assign_folds(pool_of(committed)), 'mendeley')
    with (tmp_path / 'f.csv').open(encoding='utf-8', newline='') as fh:
        assert list(csv.DictReader(fh)) == committed


def test_committed_folds_keep_families_and_groups_whole():
    rows = folds.assign_folds(pool_of(read_fixture()))
    by_family = collections.defaultdict(set)
    by_group = collections.defaultdict(set)
    for r in rows:
        (by_family[r['family']] if r['label'] else by_group[r['group']]).add(r['fold'])
    assert all(len(v) == 1 for v in by_family.values()) and all(len(v) == 1 for v in by_group.values())
    assert sorted({f for v in by_family.values() for f in v}) == list(range(5))


def test_assign_greedy_largest_first_into_the_lightest_fold_ties_by_name_then_index():
    assert folds.assign_greedy({'a': 5, 'b': 4, 'c': 3, 'd': 2, 'e': 1, 'f': 1}, k=3) == {
        'a': 0, 'b': 1, 'c': 2, 'd': 2, 'e': 1, 'f': 0}
    assert folds.assign_greedy({'y': 2, 'x': 2, 'z': 2}, k=2) == {'x': 0, 'y': 1, 'z': 0}     # equal sizes: by name
    assert folds.assign_greedy({}) == {}
    assert sorted(set(folds.assign_greedy({str(i): 1 for i in range(10)}).values())) == list(range(5))


def test_normalise_family_and_aliases():
    assert folds.normalise_family('Lock-Bit 3.0') == 'lockbit30'
    assert folds.normalise_family('REvil') == 'revil' and folds.normalise_family('Sodinokibi') == 'revil'
    assert folds.normalise_family('Stop/Djvu') == 'stop'
    assert folds.normalise_family('MedusaLocker') != folds.normalise_family('Medusa')      # kept distinct on purpose
    assert folds.normalise_family(None) == 'none'


def test_check_rows_refuses_a_sha_with_two_labels_a_repeat_and_an_unknown_arch():
    ok = {'sha256': 'a', 'label': 1, 'family': 'f', 'group': 'f', 'arch': 'x86'}
    folds.check_rows([ok, {**ok, 'sha256': 'b', 'label': 0, 'arch': 'x64'}])
    for bad in ([ok, {**ok, 'label': 0}], [ok, ok], [{**ok, 'arch': 'arm'}]):
        with pytest.raises(ValueError):
            folds.check_rows(bad)


def write_bundle(root):
    (root / 'provenance').mkdir(parents=True)
    header = ['sample_id', 'label', 'original_split', 'recommended_split', 'architecture', 'family', 'identity_group',
              'representation']
    rows = [
        ['r1', 1, 'train', 'train', 'x86', 'Conti', 'r1', 'as_stored'],
        ['r2', 1, 'train', 'train', 'x64', 'conti', 'r2', 'as_stored'],
        ['r3', 1, 'test', 'test', 'x86', 'LockBit', 'r3', 'as_stored'],
        ['r4', 1, 'test', '', 'x86', 'LockBit', 'r4', 'as_stored'],            # no recommended split: out
        ['g1', 0, 'train', 'train', 'x86', '', 'g1', 'as_stored'],
        ['g2', 0, 'train', 'train', 'x64', '', 'g2', 'as_stored'],
        ['g3', 0, 'test', 'test', 'x86', '', 'g3', 'as_stored'],
        ['g4', 0, 'test', 'test', 'x86', '', 'g4', 'upx_packed'],             # UPX stub: out
    ]
    with (root / 'samples.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    (root / 'provenance' / 'identity_aliases.csv').write_text(
        'sha256,cohort_sha256,identity_group,corpus,original_split,identity_status,current_representation,matched_cohort_tag\n',
        encoding='utf-8')


def test_pool_rows_and_cli_on_a_synthetic_bundle(tmp_path, monkeypatch, capsys):
    bundle = tmp_path / 'bundle'
    write_bundle(bundle)
    monkeypatch.setenv('GT_BUNDLE', str(bundle))
    pool = folds.pool_rows()
    assert [r['sha256'] for r in pool] == ['r1', 'r2', 'r3', 'g1', 'g2', 'g3']
    assert {r['family'] for r in pool if r['label']} == {'conti', 'lockbit'}
    assert {r['family'] for r in pool if not r['label']} == {'goodware'}
    assert [r['group'] for r in pool][3:] == ['g1', 'g2', 'g3']
    grouped = folds.pool_rows(groups={'g1': 'mn:same', 'g2': 'mn:same'})
    assert [r['group'] for r in grouped][3:] == ['mn:same', 'mn:same', 'g3']
    groups_csv = tmp_path / 'groups.csv'
    groups_csv.write_text('sample_id,group\ng1,mn:same\ng2,mn:same\n', encoding='utf-8')
    assert folds.read_groups(groups_csv) == {'g1': 'mn:same', 'g2': 'mn:same'}

    out = tmp_path / 'out'
    folds.main(['--out', str(out), '--groups', str(groups_csv)])
    with (out / 'folds_mendeley.csv').open(encoding='utf-8', newline='') as fh:
        written = list(csv.DictReader(fh))
    by_id = {r['sha256']: r for r in written}
    assert by_id['g1']['fold'] == by_id['g2']['fold'] and by_id['r1']['fold'] == by_id['r2']['fold']   # whole groups and families
    assert by_id['r1']['dataset'] == 'mendeley' and len(written) == 6
    report = capsys.readouterr().out
    assert report.startswith('6 files') and 'fold 4:' in report and 'ransomware' in report and 'x64' in report


def test_cli_without_out_or_a_configured_folds_file_says_what_to_give(tmp_path, monkeypatch):
    write_bundle(tmp_path / 'bundle')
    monkeypatch.setenv('GT_BUNDLE', str(tmp_path / 'bundle'))
    monkeypatch.delenv('GT_FOLDS', raising=False)
    monkeypatch.setenv('GT_DATASETS', str(tmp_path / 'absent.json'))
    with pytest.raises(SystemExit, match='--out'):
        folds.main([])


def test_report_counts_labels_and_architectures_per_fold():
    rows = folds.assign_folds([
        {'sha256': f'r{i}', 'label': 1, 'family': f'f{i % 3}', 'group': f'f{i % 3}', 'arch': ('x86', 'x64')[i % 2],
         'orig_set': '', 'corpus': 'm'} for i in range(6)] + [
        {'sha256': f'g{i}', 'label': 0, 'family': 'goodware', 'group': f'g{i}', 'arch': 'x86', 'orig_set': '',
         'corpus': 'm'} for i in range(4)])
    lines = folds.report(rows).splitlines()
    assert lines[0] == '10 files' and len(lines) == 6
    assert sum(int(line.split()[2]) for line in lines[1:]) == 10


@pytest.mark.real_data
@pytest.mark.xfail(strict=True, reason='the bundle has neither the goodware stream-hash groups nor the 2,509-file cohort '
                                       'of folds_mendeley.csv (2,411 files join, 2,085 of them change fold)')
def test_regenerating_folds_mendeley_from_the_bundle_gives_the_committed_csv():
    committed = {r['sha256']: r for r in read_fixture()}
    from s02_preprocess.cache.build_cache import corpus_root, read_aliases
    aliases = read_aliases(corpus_root('mendeley'))
    rebuilt = folds.assign_folds(folds.pool_rows())
    assert len(rebuilt) == len(committed)
    for r in rebuilt:
        key = r['sha256'] if r['sha256'] in committed else aliases.get(r['sha256'], {}).get('cohort_sha256')
        assert str(r['fold']) == committed[key]['fold']
