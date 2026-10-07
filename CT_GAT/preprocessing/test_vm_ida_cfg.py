"""Safety/schema and dataset bookkeeping regression tests; no samples required."""
import copy
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('vm_cfg', Path(__file__).with_name('vm_ida_cfg.py'))
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)
DIGEST = 'a' * 64


def graph():
    return dict(schema=cfg.SCHEMA, sha256=DIGEST, ida_version='9.4', arch='x64', truncated=False,
                functions=[dict(id=0, library=False, thunk=False,
                                blocks=[dict(id=0, type=0, mnemonics=['call', 'jne'], operand_types=[[7], [7]]),
                                        dict(id=1, type=2, mnemonics=['retn'], operand_types=[[]])],
                                edges=[[0, 1]], calls=[dict(block=0, instruction=0, targets=[0])])],
                counts=dict(functions=1, blocks=2, edges=1, instructions=3, calls=1))


class GraphTests(unittest.TestCase):
    def test_valid_cfg_and_ordinal_calls(self):
        self.assertEqual(cfg.validate(graph(), DIGEST)['instructions'], 3)

    def test_reject_raw_bytes_assembly_and_operand_values(self):
        for key in ('bytes', 'asm', 'operands', 'address', 'name'):
            data = graph()
            data['functions'][0]['blocks'][0][key] = 'forbidden'
            with self.assertRaises(ValueError):
                cfg.validate(data, DIGEST)

    def test_reject_bad_edges_calls_alignment_and_metadata(self):
        mutations = [lambda d: d['functions'][0]['edges'].append([0, 7]),
                     lambda d: d['functions'][0]['calls'][0]['targets'].append(7),
                     lambda d: d['functions'][0]['blocks'][0]['operand_types'].pop(),
                     lambda d: d['functions'][0]['blocks'][0]['mnemonics'].__setitem__(0, 'mov eax, 0x1234'),
                     lambda d: d['counts'].__setitem__('instructions', 8),
                     lambda d: d.__setitem__('sha256', 'b' * 64)]
        for mutate in mutations:
            data = copy.deepcopy(graph())
            mutate(data)
            with self.assertRaises(ValueError):
                cfg.validate(data, DIGEST)

    def test_python38_streaming_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'fixture'
            path.write_bytes(b'fixture' * 1000000)
            self.assertEqual(cfg.sha256(path), hashlib.sha256(path.read_bytes()).hexdigest())

    def test_archive_rejects_traversal_and_symlinks(self):
        for name, member_type in [('../bad', tarfile.REGTYPE), ('cfg/link', tarfile.SYMTYPE)]:
            with tempfile.TemporaryDirectory() as temp:
                archive_path = Path(temp) / 'bad.tar'
                with tarfile.open(archive_path, 'w') as archive:
                    info = tarfile.TarInfo(name)
                    info.type = member_type
                    archive.addfile(info)
                with self.assertRaises(ValueError):
                    cfg.verify_package(archive_path, Path(temp) / 'out')
                self.assertFalse((Path(temp) / 'out').exists())

    def test_package_groups_and_mendeley_split_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            graph_name = 'cfg/' + DIGEST + '.json.gz'
            header = 'sha256,group_folder,original_split,arch,status\n'
            manifest = (header + DIGEST + ',mendeley/ransomware/test/family,mal_test,x64,ok\n').encode()
            payload = {'manifest.csv': manifest, 'summary.json': b'{}', 'split_audit.json': b'[]',
                       'inventory.json': b'[]', 'cohort_membership.json': b'[]',
                       graph_name: gzip.compress(json.dumps(graph()).encode())}
            checksums = {name: hashlib.sha256(raw).hexdigest() for name, raw in payload.items()}
            payload['checksums.json'] = json.dumps(checksums).encode()
            archive_path = root / 'safe.tar'
            with tarfile.open(archive_path, 'w') as archive:
                for name, raw in payload.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(raw)
                    archive.addfile(info, io.BytesIO(raw))
            cfg.verify_package(archive_path, root / 'desktop')
            group = root / 'desktop/groups/mendeley/ransomware/test/family'
            self.assertTrue((group / (DIGEST + '.json.gz')).is_file())
            self.assertIn('mal_test', (group / 'manifest.csv').read_text())
            self.assertIn('x64', (group / 'manifest.csv').read_text())

    def test_archive_checksum_mismatch_publishes_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive_path = root / 'bad.tar'
            payload = {'manifest.csv': b'tampered', 'checksums.json': json.dumps({'manifest.csv': '0' * 64}).encode()}
            with tarfile.open(archive_path, 'w') as archive:
                for name, raw in payload.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(raw)
                    archive.addfile(info, io.BytesIO(raw))
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                cfg.verify_package(archive_path, root / 'desktop')
            self.assertFalse((root / 'desktop').exists())

    def test_mendeley_manifest_preserves_arch_split_and_exclusion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = dict(sha256=DIGEST, corpus='mendeley', group='mendeley_mal_test',
                       group_folder='mendeley/ransomware/test/family', original_split='mal_test',
                       set='mal_test', kind='managed_pe', arch='x86', label='1',
                       in_cohort='0', exclude_reason='dotnet',
                       cohort_rows=[dict(set='mal_test', arch='x86', tag='dotnet', family='family')])
            cfg.write_json(root / 'inventory.json', [row])
            cfg.write_json(root / 'jobs.json', [])
            summary = cfg.summarize(root)
            with (root / 'mendeley_samples.csv').open(newline='', encoding='utf-8') as stream:
                import csv
                saved = list(csv.DictReader(stream))[0]
            self.assertEqual((saved['original_split'], saved['arch'], saved['cohort_tag']), ('mal_test', 'x86', 'dotnet'))
            self.assertEqual((saved['in_cohort'], saved['exclude_reason'], saved['status']), ('0', 'dotnet', 'managed_pe'))
            self.assertEqual(summary['files'], 1)


if __name__ == '__main__':
    unittest.main()
