import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

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


if __name__ == '__main__':
    unittest.main()
