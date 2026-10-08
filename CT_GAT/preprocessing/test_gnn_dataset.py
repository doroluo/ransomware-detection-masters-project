"""Contract tests for topology, split identity, corruption checks and packaging."""
import argparse
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import build_gnn_bundle as builder
import gnn_dataset as reader


def function(fid=0):
    insn = dict(mnemonic='call', operands=[dict(category='memory')],
                is_call=True, import_ids=[0])
    return dict(record='function', id=fid, name='f', rva=4096,
                blocks=[dict(id=10, start_rva=4096, external=False, instructions=[insn]),
                        dict(id=30, start_rva=8192, external=True, instructions=[]),
                        dict(id=50, start_rva=4100, external=False, instructions=[])],
                edges=[[10, 30]])


IMPORTS = [dict(dll='KERNEL32.dll', name='Sleep', ordinal=None)]


def export(path, digest, records=None, complete=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = dict(record='header', sha256=digest, schema='ida-cfg-research/2.1',
                  pe=dict(imports=IMPORTS))
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        for record in [header] + (records if records is not None else [function()]) + [dict(record='footer', complete=complete)]:
            stream.write(json.dumps(record) + '\n')


class GraphTests(unittest.TestCase):
    def test_remaps_directed_edges_and_retains_isolated_external_nodes(self):
        g = reader.function_graph(function(), IMPORTS)
        self.assertEqual(g['num_nodes'], 3)
        self.assertEqual(g['edge_index'], [[0], [1]])
        self.assertEqual(g['node_tokens'], [['call'], [], []])
        self.assertEqual(g['node_is_external'], [False, True, False])
        self.assertEqual(g['node_api_references'][0], [['KERNEL32.dll!Sleep']])
        self.assertEqual(g['node_operand_categories'][0], [['memory']])
        self.assertEqual(len(g['x'][0]), 6)
        self.assertGreater(g['x'][0][5], 0)
        self.assertEqual(g['x'][2], [0, 0, 0, 0, 0, 0])

    def test_invalid_topology_fails(self):
        f = function(); f['edges'].append([10, 999])
        with self.assertRaisesRegex(ValueError, 'missing block'):
            reader.function_graph(f, IMPORTS)
        f = function(); f['blocks'][1]['id'] = 10
        with self.assertRaisesRegex(ValueError, 'duplicate block'):
            reader.function_graph(f, IMPORTS)

    def test_zero_edges_shape(self):
        f = function(); f['edges'] = []
        self.assertEqual(reader.function_graph(f, IMPORTS)['edge_index'], [[], []])

    def test_reader_identity_and_completion(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'sample.jsonl.gz'
            export(p, 'a' * 64)
            self.assertEqual(len(list(reader.iter_file_functions(p, 'a' * 64))), 1)
            with self.assertRaisesRegex(ValueError, 'identity'):
                list(reader.iter_file_functions(p, 'b' * 64))
            export(p, 'a' * 64, complete=False)
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                list(reader.iter_file_functions(p, 'a' * 64))


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.source, self.dest = root / 'raw', root / 'bundle'
        self.source.mkdir()
        (self.source / 'status').mkdir()
        self.rows = []
        for i, (label, split, status) in enumerate([(0, 'train', 'recovered'),
                (1, 'train', 'recovered'), (1, 'test', 'recovered'),
                (0, 'test', 'no_function_instructions')], 1):
            digest = ('%064x' % i)
            p = self.source / 'features' / (digest + '.jsonl.gz')
            export(p, digest, records=[function(0), function(9)] if status == 'recovered' else [])
            counts = dict(functions=2 if status == 'recovered' else 0,
                          blocks=6 if status == 'recovered' else 0,
                          edges=2 if status == 'recovered' else 0,
                          instructions=2 if status == 'recovered' else 0)
            (self.source / 'status' / (digest + '.json')).write_text(json.dumps(
                dict(status='ok', artifact_sha256=builder.sha(p), counts=counts)))
            name = 'goodware' if label == 0 else 'ransomware'
            self.rows.append(dict(sha256=digest, corpus='mendeley', group_folder='mendeley/' + name + '/' + split,
                original_split=name + '_' + split, source='original/path', selected='1', eligible='1',
                kind='candidate', arch='x64', graph_status=status, feature_ready='1', exclusion_reason='',
                selection_reason='', label=str(label), effective_split=split, family='test_family',
                identity_group=digest, current_representation='as_stored', matched_cohort_tag='plain'))
        # Preserve an alias occurrence, but do not duplicate the sample or split.
        duplicate = copy.deepcopy(self.rows[0]); duplicate['selected'] = '0'
        duplicate['source'] = 'original/duplicate'; self.rows.append(duplicate)
        builder.write_csv(self.source / 'reconciliation/mendeley_samples.csv', self.rows)
        builder.write_csv(self.source / 'reconciliation/identity_aliases.csv', [dict(corpus='mendeley', sha256=self.rows[0]['sha256'])])
        (self.source / 'policy.json').write_text('{}')
        guide = root / 'guide.md'; guide.write_text('# Test guide')
        self.args = argparse.Namespace(source=self.source, destination=self.dest, guide=guide, zip=True)

    def test_portable_bundle_splits_union_and_corruption(self):
        builder.build(self.args)
        self.assertEqual(reader.verify_bundle(self.dest), 16)
        ds = reader.MendeleyDataset(self.dest, 'train', architecture='x64')
        self.assertEqual(len(ds), 2)
        self.assertEqual(len(reader.MendeleyDataset(self.dest, 'test')), 1)
        self.assertEqual(len(reader.MendeleyDataset(self.dest, 'all')), 4)
        sample = ds[0]
        g = ds.sample_graph(sample)
        self.assertEqual(g['num_nodes'], 6)
        self.assertEqual(g['edge_index'], [[0, 3], [1, 4]])
        self.assertEqual(g['node_function_id'], [0, 0, 0, 9, 9, 9])
        self.assertEqual(sample['original_file_occurrences'], '2')
        self.assertEqual(len(reader.MendeleyDataset(self.dest, class_name='goodware')), 1)
        with zipfile.ZipFile(self.dest.with_suffix('.zip')) as z:
            self.assertIsNone(z.testzip())
        (self.dest / sample['cfg_path']).write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            reader.verify_bundle(self.dest)

    def test_cross_split_identity_is_rejected(self):
        self.rows[2]['identity_group'] = self.rows[0]['identity_group']
        builder.write_csv(self.source / 'reconciliation/mendeley_samples.csv', self.rows)
        with self.assertRaisesRegex(ValueError, 'identity leakage'):
            builder.build(self.args)

    def test_source_corruption_is_rejected(self):
        (self.source / 'features' / (self.rows[0]['sha256'] + '.jsonl.gz')).write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'unverified export'):
            builder.build(self.args)

    def test_policy_excludes_empty_graph_and_non_native(self):
        self.assertEqual(builder.recommendation(self.rows[0]), '')
        self.assertIn('no_function_instructions', builder.recommendation(self.rows[3]))
        r = dict(self.rows[0], kind='managed_metadata', arch='PowerPC')
        self.assertIn('managed_metadata', builder.recommendation(r))
        self.assertIn('unsupported_architecture', builder.recommendation(r))


if __name__ == '__main__':
    unittest.main()
