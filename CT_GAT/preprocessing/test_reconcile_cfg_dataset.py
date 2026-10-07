import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reconcile_cfg_dataset import reconcile
from vm_ida_cfg import write_json


def row(digest, corpus='mendeley', split='mal_train'):
    return dict(sha256=digest, corpus=corpus, original_split=split, set=split, label='1',
                kind='candidate', source=corpus + '/' + split + '/' + digest, arch='x86')


def member(digest, corpus='mendeley', split='mal_train', included='1'):
    return dict(sha256=digest, corpus=corpus, set=split, label='1', in_cohort=included, tag='plain')


def proof(raw, unpacked):
    return dict(sha256=raw, observed_sha256=raw, matches_inventory=True,
                unverified_filename_candidates=[dict(sha256=unpacked)],
                upx_verification=dict(status='verified_exact_cohort_hash', returncode=0, unpacked_sha256=unpacked))


class ReconciliationTests(unittest.TestCase):
    def run_case(self, directory, rows, members, proofs=()):
        root = Path(directory)
        write_json(root / 'inventory.json', rows)
        write_json(root / 'cohort_membership.json', members)
        write_json(root / 'identity_evidence.json', dict(rows=list(proofs)))
        before = {name: (root / name).read_bytes() for name in ('inventory.json', 'cohort_membership.json')}
        result = reconcile(root)
        for name, contents in before.items():
            self.assertEqual((root / name).read_bytes(), contents)
        return result

    def test_verified_transform_resolves_membership_preserving_raw_hash(self):
        raw, unpacked = 'a' * 64, 'b' * 64
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_case(directory, [row(raw)], [member(unpacked)], [proof(raw, unpacked)])
            self.assertEqual(result['mendeley_identity_status'], {'verified_upx_transform': 1})
            text = (Path(directory) / 'reconciliation/train.csv').read_text()
            self.assertIn(raw, text)
            self.assertIn(unpacked, text)
            self.assertIn('upx_packed', text)

    def test_exact_cross_source_duplicate_reserved_for_test(self):
        digest = 'a' * 64
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_case(directory, [row(digest, split='mal_test'), row(digest, corpus='vs')],
                                   [member(digest, split='mal_test'), member(digest, corpus='vs')])
            self.assertEqual((result['selected_train'], result['selected_test']), (0, 1))
            self.assertEqual(result['exclusions']['identity_reserved_for_test'], 1)

    def test_transformed_cross_source_duplicate_also_blocked(self):
        raw, unpacked = 'a' * 64, 'b' * 64
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_case(directory, [row(raw, split='mal_test'), row(unpacked, corpus='vs')],
                                   [member(unpacked, split='mal_test'), member(unpacked, corpus='vs')], [proof(raw, unpacked)])
            self.assertEqual((result['selected_train'], result['selected_test']), (0, 1))

    def test_unverified_transform_cannot_restore_cohort_membership(self):
        raw, unpacked = 'a' * 64, 'b' * 64
        invalid = proof(raw, unpacked)
        invalid['upx_verification']['status'] = 'not_verified'
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_case(directory, [row(raw)], [member(unpacked)], [invalid])
            self.assertEqual(result['selected_train'], 0)
            self.assertEqual(result['exclusions']['unverified_cohort_membership'], 1)

    def test_unsupported_labels_and_original_exclusions_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_case(directory, [row('a' * 64), row('b' * 64, corpus='virusshare', split='')],
                                   [member('a' * 64, included='0')])
            self.assertEqual(result['selected_train'], 0)
            self.assertEqual(result['exclusions']['original_cohort_exclusion_or_conflict'], 1)
            self.assertEqual(result['exclusions']['no_original_experiment_split'], 1)

    def test_ready_manifest_updates_only_for_existing_successful_features(self):
        digest = 'a' * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_case(directory, [row(digest)], [member(digest)])
            (root / 'status').mkdir()
            write_json(root / 'status' / (digest + '.json'), dict(status='ok', graph_status='recovered'))
            self.assertEqual(reconcile(root)['ready_train'], 0)
            (root / 'features').mkdir()
            (root / 'features' / (digest + '.jsonl.gz')).write_bytes(b'validated upstream')
            self.assertEqual(reconcile(root)['ready_train'], 1)


if __name__ == '__main__':
    unittest.main()
