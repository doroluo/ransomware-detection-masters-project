import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_mendeley_asm as asm


class AssemblyTransferTests(unittest.TestCase):
    def package(self, root, text=b'; IDA listing\nstart: ret\n', corrupt=False, folder=None, extra=None):
        sha = 'a' * 64
        packed = gzip.compress(text)
        state = dict(sha256=sha, status='ok', asm_sha256=hashlib.sha256(text).hexdigest(),
                     asm_bytes=len(text), compressed_sha256=hashlib.sha256(packed).hexdigest())
        if corrupt:
            state['asm_sha256'] = '0' * 64
        rows = [dict(sha256=sha, corpus='mendeley', group='mendeley_good_' + split,
                     group_folder=folder or 'mendeley/goodware/' + split,
                     original_split=split, arch='x86', kind='candidate') for split in ('train', 'test')]
        members = {'inventory.json': json.dumps(rows).encode(), 'policy.json': b'{}',
                   'index.json': json.dumps(dict(states=[state], final=False)).encode(),
                   'asm/' + sha + '.asm.gz': packed}
        if extra:
            members[extra] = b'unexpected'
        path = root / 'package.tar'
        with tarfile.open(path, 'w') as archive:
            for name, data in members.items():
                entry = tarfile.TarInfo(name)
                entry.size = len(data)
                archive.addfile(entry, io.BytesIO(data))
        return path, sha

    def test_verified_plain_asm_and_both_original_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, sha = self.package(root)
            target = root / 'desktop'
            asm.accept(package, target)
            for split in ('train', 'test'):
                path = target / 'groups/mendeley/goodware' / split / (sha + '.asm')
                self.assertEqual(path.read_bytes(), b'; IDA listing\nstart: ret\n')
                self.assertTrue((path.parent / 'manifest.csv').exists())
            summary = json.loads((target / 'summary.json').read_text())
            self.assertEqual(summary['files'], 2)
            self.assertEqual(summary['unique_statuses'], {'ok': 1})

    def test_corrupt_listing_never_becomes_final_asm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, sha = self.package(root, corrupt=True)
            with self.assertRaisesRegex(ValueError, 'text_checksum_mismatch'):
                asm.accept(package, root / 'desktop')
            self.assertFalse((root / 'desktop/asm' / (sha + '.asm')).exists())

    def test_path_traversal_and_unexpected_members_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, _ = self.package(root, folder='mendeley/../../outside')
            with self.assertRaisesRegex(ValueError, 'unsafe_group_path'):
                asm.accept(package, root / 'desktop')
            package, _ = self.package(root, extra='sample.exe')
            with self.assertRaisesRegex(ValueError, 'unexpected_archive_member'):
                asm.accept(package, root / 'desktop')

    def test_binary_content_rejected(self):
        with self.assertRaisesRegex(ValueError, 'not_plain_text'):
            asm.text_digest(io.BytesIO(b'MZ\0payload'))

    def test_shared_export_publishes_once_and_records_cfg_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(asm.common, 'check_out', return_value=root), patch.object(
                    asm, 'export_open_database', return_value=dict(sha256='a'*64, status='ok')) as export:
                asm.export_current(root, 'a'*64, 'b'*64)
                asm.export_current(root, 'a'*64, 'b'*64)
                self.assertEqual(export.call_count, 1)
            status = json.loads((root/'status'/('a'*64+'.json')).read_text())
            self.assertEqual(status['analysis_mode'], 'shared_cfg_database')
            self.assertEqual(status['cfg_extractor_sha256'], 'b'*64)
            self.assertEqual(len(asm.untransferred(root)), 1)

    def test_shared_asm_failure_remains_eligible_for_backfill(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(asm.common, 'check_out', return_value=root), patch.object(
                    asm, 'export_open_database', side_effect=RuntimeError('ASM failure')):
                asm.export_current(root, 'a'*64, 'b'*64)
            self.assertFalse((root/'status'/('a'*64+'.json')).exists())
            self.assertTrue((root/'combined_errors'/('a'*64+'.json')).exists())

    def test_disk_deferral_removes_only_partial_and_prevents_automatic_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'asm').mkdir()
            (root/'status').mkdir()
            partial=root/'asm'/('a'*64+'.asm.gz.asm.tmp')
            partial.write_bytes(b'partial listing')
            preserved=root/'asm'/('b'*64+'.asm.gz')
            preserved.write_bytes(b'completed other sample')
            with patch.object(asm.common,'check_out',return_value=root), patch.object(asm,'export_open_database') as export:
                state=asm.defer_disk_limited_assembly(root,'a'*64)
                asm.export_current(root,'a'*64,'c'*64)
                export.assert_not_called()
            self.assertFalse(partial.exists())
            self.assertEqual(preserved.read_bytes(),b'completed other sample')
            self.assertEqual(state['status'],'deferred')
            self.assertEqual(state['incomplete_bytes_removed'],15)

    def test_disk_deferral_keeps_valid_cfg(self):
        import run_research_cfg as runner
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'asm').mkdir()
            (root/'status').mkdir()
            (root/'combined_asm.json').write_text(json.dumps(dict(out=str(root))))
            (root/'asm'/('a'*64+'.asm.gz.asm.tmp')).write_bytes(b'partial')
            cfg=root/'graph.jsonl.gz'
            cfg.write_bytes(b'validated fixture')
            with patch.object(asm.common,'check_out',return_value=root), patch.object(runner.rich,'validate_file') as validate, patch.object(runner.shutil,'disk_usage') as disk:
                disk.return_value.free=4*1024**3
                self.assertTrue(runner.recover_cfg_after_asm_disk_limit(root,cfg,'a'*64))
                validate.assert_called_once_with(cfg,'a'*64)
            self.assertEqual(cfg.read_bytes(),b'validated fixture')


if __name__ == '__main__':
    unittest.main()
