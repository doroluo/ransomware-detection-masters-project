import gzip
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_cfg_with_ida as rich
import run_research_cfg as runner
import vm_ida_cfg as common

DIGEST = 'a' * 64


def records():
    insn = dict(rva=4096, size=5, mnemonic='call', operands=[dict(type=7, category='code_target', dtype=2, width=4,
                    register='', immediate=None, target_rva=8192, displacement=None, phrase=None, processor_flags=[0, 0, 0, 0])],
                code_refs=[8192], data_refs=[12288], is_call=True, import_ids=[0], call_resolution='import')
    return [dict(record='header', schema=rich.SCHEMA, sha256=DIGEST, extractor_sha256='b' * 64,
                 arch='x64', input_kind='candidate', size=0, byte_histogram=[0] * 256, pe={}, ida_version='9.4', imagebase='0x140000000', analysis_options={}),
            dict(record='function', id=0, rva=4096, name='function_name', flags=0, library=False, thunk=False, chunks=[[4096, 4101]],
                 blocks=[dict(id=0, start_rva=4096, end_rva=4101, type=0, external=False, instructions=[insn]),
                         dict(id=1, start_rva=8192, end_rva=8193, type=6, external=True, instructions=[])], edges=[[0, 1]]),
            dict(record='string', rva=None, file_offset=100, encoding='ascii', text='<script>inert JSON string</script>'),
            dict(record='footer', counts=dict(functions=1, blocks=2, edges=1, instructions=1, calls=1, strings=1, orphan_instructions=0),
                 graph_status='recovered', complete=True, static_limitations=['no_execution'], quality={})]


def serialized(values):
    return ''.join(json.dumps(v) + '\n' for v in values)


class ResearchTests(unittest.TestCase):
    def test_rich_features_and_external_edges(self):
        result = rich.validate_stream(io.StringIO(serialized(records())), DIGEST)
        self.assertEqual(result['footer']['counts']['instructions'], 1)

    def test_incomplete_file_is_never_success(self):
        with self.assertRaisesRegex(ValueError, 'missing_completion'):
            rich.validate_stream(io.StringIO(serialized(records()[:-1])), DIGEST)

    def test_bad_edges_and_forbidden_raw_code_fields(self):
        values = records()
        values[1]['edges'] = [[0, 7]]
        with self.assertRaisesRegex(ValueError, 'dangling_edge'):
            rich.validate_stream(io.StringIO(serialized(values)), DIGEST)
        values = records()
        values[1]['blocks'][0]['instructions'][0]['bytes'] = '90'
        with self.assertRaisesRegex(ValueError, 'unexpected_fields'):
            rich.validate_stream(io.StringIO(serialized(values)), DIGEST)

    def test_strings_cross_large_chunks_and_utf16(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture'
            path.write_bytes(b'\0' + b'A' * 1048590 + b'\0\0' + 'ransomware'.encode('utf-16le') + b'\0')
            result = list(rich.file_strings(path))
            self.assertIn(1048590, [len(r['text']) for r in result])
            self.assertTrue(any(r['text'] == 'ransomware' for r in result))

    def test_streaming_batch_accept_groups_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compressed = gzip.compress(serialized(records()).encode())
            artifact = root / 'artifact.gz'
            artifact.write_bytes(compressed)
            state = dict(sha256=DIGEST, status='ok', graph_status='recovered', artifact_sha256=rich.hash_file(artifact))
            row = dict(sha256=DIGEST, group='mendeley_mal_test', group_folder='mendeley/ransomware/test/family',
                       corpus='mendeley', set='mal_test', original_split='mal_test', label='1', family='family',
                       kind='candidate', arch='x64', cohort_rows=[])
            payloads = {'inventory.json': json.dumps([row]).encode(), 'cohort_membership.json': b'[]',
                        'batch_index.json': json.dumps(dict(states=[state], final=True, metadata=True)).encode(),
                        'features/' + DIGEST + '.jsonl.gz': compressed}
            package = root / 'batch.tar'
            with tarfile.open(str(package), 'w') as archive:
                for name, raw in payloads.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(raw)
                    archive.addfile(info, io.BytesIO(raw))
            runner.accept(package, root / 'desktop')
            self.assertTrue((root / 'desktop/groups/mendeley/ransomware/test/family' / (DIGEST + '.jsonl.gz')).exists())
            self.assertIn('mal_test', (root / 'desktop/mendeley_samples.csv').read_text())

    def test_archive_paths_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.tar'
            with tarfile.open(str(path), 'w') as archive:
                archive.addfile(tarfile.TarInfo('../sample.exe'))
            with self.assertRaisesRegex(ValueError, 'unsafe_archive'):
                runner.accept(path, Path(directory) / 'desktop')

    def test_cleanup_requires_matching_ack_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('acks', 'features', 'status', 'exports'):
                (root / name).mkdir()
            source = root / 'source_fixture'
            source.write_bytes(b'never delete source')
            artifact = root / 'features' / (DIGEST + '.jsonl.gz')
            artifact.write_bytes(b'derived')
            (root / 'exports/batch_000000.tar').write_bytes(b'package')
            receipt = dict(name='batch_000000.tar', sha256='b' * 64)
            common.write_json(root / 'ready.json', receipt)
            common.write_json(root / 'acks/batch_000000.json', dict(sha256='c' * 64))
            states = [dict(sha256=DIGEST, status='ok')]
            with self.assertRaisesRegex(RuntimeError, 'invalid_transfer_ack'):
                runner.finish_transfer(root, receipt, states)
            self.assertTrue(artifact.exists())
            common.write_json(root / 'acks/batch_000000.json', dict(sha256='b' * 64))
            runner.finish_transfer(root, receipt, states)
            self.assertFalse(artifact.exists())
            self.assertEqual(source.read_bytes(), b'never delete source')
            self.assertEqual(json.loads((root / 'status' / (DIGEST + '.json')).read_text())['transferred_batch'], 'batch_000000')

    def test_memory_pressure_retries_without_publishing_graph(self):
        class Process:
            pid = 12345
            returncode = None
            def poll(self):
                return None
            def wait(self):
                self.returncode = -9
                return -9
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('features', 'status', 'scratch', 'logs'):
                (root / name).mkdir()
            source = root / 'fixture'
            source.write_bytes(b'fixture data, not a program')
            digest = rich.hash_file(source)
            job = dict(sha256=digest, source=str(source), size=source.stat().st_size, arch='x64', kind='candidate')
            with patch.object(runner.subprocess, 'Popen', side_effect=lambda *a, **k: Process()), \
                 patch.object(runner, 'process_memory', return_value=7 * 1024**3), \
                 patch.object(runner, 'available_memory', return_value=8 * 1024**3), \
                 patch.object(runner.signal, 'SIGKILL', 9, create=True), \
                 patch.object(runner.os, 'killpg', create=True) as kill:
                result = runner.run_job(job, root, 3600, 14400, 6144)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['reason'], 'resident_memory_pressure')
            self.assertEqual(len(result['attempts']), 2)
            self.assertEqual(kill.call_count, 2)
            self.assertEqual(list((root / 'features').iterdir()), [])
            self.assertTrue(source.exists())


if __name__ == '__main__':
    unittest.main()
