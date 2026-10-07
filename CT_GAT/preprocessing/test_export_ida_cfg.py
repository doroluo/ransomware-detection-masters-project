"""Run with python -m unittest discover -s CT_GAT/preprocessing -p test_export_ida_cfg.py."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('exporter', Path(__file__).with_name('export_ida_cfg.py'))
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class ExportTests(unittest.TestCase):
    def modules(self):
        first = NS(id=0, start_ea=4096, end_ea=4098, type=0)
        second = NS(id=1, start_ea=4098, end_ea=4100, type=2)
        first.succs = lambda: iter([second])
        second.succs = lambda: iter([])
        return {
            'idapro': NS(open_database=Mock(return_value=0), close_database=Mock()),
            'ida_auto': NS(auto_wait=lambda: True),
            'ida_bytes': NS(is_code=lambda f: True, get_full_flags=lambda ea: 1,
                            get_bytes=lambda ea, size: b'\x90\x90'),
            'ida_funcs': NS(get_func=lambda ea: object(), get_func_name=lambda ea: 'test'),
            'ida_gdl': NS(FC_NOEXT=1, FlowChart=lambda f, flags: [first, second]),
            'ida_idp': NS(is_call_insn=lambda insn: insn.ea == 4096),
            'ida_kernwin': NS(get_kernel_version=lambda: 'test'),
            'ida_lines': NS(tag_remove=lambda text: text),
            'ida_ua': NS(o_void=0, insn_t=lambda: NS(size=2, ops=[NS(type=0)]),
                         decode_insn=lambda insn, ea: setattr(insn, 'ea', ea) or 2),
            'idautils': NS(Functions=lambda: [4096], Heads=lambda start, end: [start],
                           CodeRefsFrom=lambda ea, flow: [8192]),
            'idc': NS(print_operand=lambda ea, i: '', print_insn_mnem=lambda ea: 'nop',
                      generate_disasm_line=lambda ea, flags: 'nop'),
        }

    def test_blocks_edges_calls_and_identity(self):
        modules = self.modules()
        with tempfile.TemporaryDirectory() as directory, patch.dict('sys.modules', modules):
            output = Path(directory) / 'graph.json'
            exporter.extract(Path('sample.exe'), output, 1, 'train', 'original.exe', 'abc')
            data = json.loads(output.read_text())
            self.assertEqual((data['label'], data['cohort'], data['sha256']), (1, 'train', 'abc'))
            function = data['functions'][0]
            self.assertEqual(function['edges'], [[0, 1]])
            self.assertEqual(function['calls'][0]['targets'], ['0x2000'])
            self.assertEqual(len(function['nodes']), 2)
            self.assertEqual(function['nodes'][0]['instructions'][0]['address'], '0x1000')
        modules['idapro'].close_database.assert_called_once_with(False)

    def test_empty_analysis_fails_without_publishing(self):
        modules = self.modules()
        modules['idautils'].Functions = lambda: []
        with tempfile.TemporaryDirectory() as directory, patch.dict('sys.modules', modules):
            output = Path(directory) / 'graph.json'
            with self.assertRaisesRegex(RuntimeError, 'no usable'):
                exporter.extract(Path('sample.exe'), output, 0, 'test', 'original.exe', 'abc')
            self.assertFalse(output.exists())
        modules['idapro'].close_database.assert_called_once_with(False)


if __name__ == '__main__':
    unittest.main()
