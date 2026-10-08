"""Portable Mendeley CFG reader. Standard library only; PyG conversion is optional."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

FEATURE_NAMES = ['log1p_instructions', 'log1p_calls', 'log1p_import_references',
                 'external_placeholder', 'log1p_in_degree', 'log1p_out_degree']


def verify_bundle(root):
    """Check every bundled file against the snapshot's SHA256SUMS.txt."""
    root = Path(root).resolve()
    count = 0
    for line in (root / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        expected, relative = line.split('  ', 1)
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError('unsafe checksum path')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1048576), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError('checksum mismatch: ' + relative)
        count += 1
    return count


def function_graph(record, imports):
    """Keep directed CFG topology and isolated/external blocks. No truncation."""
    blocks = record['blocks']
    ids = [b['id'] for b in blocks]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate block ID')
    remap = {bid: i for i, bid in enumerate(ids)}
    sources, targets = [], []
    incoming, outgoing = [0] * len(blocks), [0] * len(blocks)
    for source, target in record['edges']:
        if source not in remap or target not in remap:
            raise ValueError('edge references missing block')
        a, b = remap[source], remap[target]
        sources.append(a); targets.append(b)
        outgoing[a] += 1; incoming[b] += 1
    graph = dict(function_id=record['id'], function_name=record['name'], function_rva=record['rva'],
                 num_nodes=len(blocks), edge_index=[sources, targets], x=[],
                 node_tokens=[], node_operand_categories=[], node_api_references=[],
                 node_start_rva=[], node_is_external=[])
    for i, block in enumerate(blocks):
        insns = block['instructions']
        api_names = []
        for insn in insns:
            names = []
            for index in insn.get('import_ids', []):
                if not 0 <= index < len(imports):
                    raise ValueError('invalid import index')
                symbol = imports[index]
                names.append(symbol['dll'] + '!' + (symbol.get('name') or '#' + str(symbol.get('ordinal'))))
            api_names.append(names)
        graph['x'].append([math.log1p(len(insns)), math.log1p(sum(bool(x.get('is_call')) for x in insns)),
                           math.log1p(sum(len(n) for n in api_names)), float(block['external']),
                           math.log1p(incoming[i]), math.log1p(outgoing[i])])
        graph['node_tokens'].append([x['mnemonic'] for x in insns])
        graph['node_operand_categories'].append([[o['category'] for o in x['operands']] for x in insns])
        graph['node_api_references'].append(api_names)
        graph['node_start_rva'].append(block['start_rva'])
        graph['node_is_external'].append(block['external'])
    return graph


def iter_file_functions(path, expected_sample_id):
    """Stream one function at a time; caller need not unpack or understand JSONL."""
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        first = stream.readline()
        header = json.loads(first)
        if header.get('record') != 'header' or header.get('sha256') != expected_sample_id:
            raise ValueError('sample identity/header mismatch')
        if header.get('schema') != 'ida-cfg-research/2.1':
            raise ValueError('unsupported export schema')
        imports = header.get('pe', {}).get('imports', [])
        footer = None
        for line in stream:
            record = json.loads(line)
            if record['record'] == 'function':
                yield function_graph(record, imports)
            elif record['record'] == 'footer':
                footer = record
        if not footer or not footer.get('complete'):
            raise ValueError('incomplete export')


class MendeleyDataset:
    """Rows are binaries, not functions. Recommended splits exclude unusable CFGs."""
    def __init__(self, root, split='train', architecture=None, class_name=None):
        self.root = Path(root).resolve()
        if split not in ('train', 'test', 'all'):
            raise ValueError('split must be train, test or all')
        manifest = self.root / ('samples.csv' if split == 'all' else 'splits/' + split + '.csv')
        with manifest.open(encoding='utf-8', newline='') as stream:
            self.rows = list(csv.DictReader(stream))
        if architecture:
            self.rows = [r for r in self.rows if r['architecture'] == architecture]
        if class_name:
            if class_name not in ('goodware', 'ransomware'):
                raise ValueError('unknown class')
            self.rows = [r for r in self.rows if r['class_name'] == class_name]
        for row in self.rows:
            for field in ('label', 'num_functions', 'num_blocks', 'num_edges', 'num_instructions', 'eligible_for_gnn'):
                row[field] = int(row[field])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return dict(self.rows[index])

    def functions(self, sample):
        """Yield function graphs. Their shared label belongs to the parent binary."""
        if isinstance(sample, int):
            sample = self.rows[sample]
        path = (self.root / sample['cfg_path']).resolve()
        if self.root not in path.parents:
            raise ValueError('unsafe relative data path')
        return iter_file_functions(path, sample['sample_id'])

    def sample_graph(self, sample):
        """Disjoint union of function CFGs; NOT an interprocedural CFG."""
        if isinstance(sample, int):
            sample = self.rows[sample]
        graph = dict(num_nodes=0, edge_index=[[], []], x=[], node_tokens=[],
                     node_operand_categories=[], node_api_references=[], node_start_rva=[],
                     node_is_external=[], node_function_id=[], function_count=0)
        for function in self.functions(sample):
            offset = graph['num_nodes']
            for side in (0, 1):
                graph['edge_index'][side].extend(n + offset for n in function['edge_index'][side])
            for field in ('x', 'node_tokens', 'node_operand_categories', 'node_api_references',
                          'node_start_rva', 'node_is_external'):
                graph[field].extend(function[field])
            graph['node_function_id'].extend([function['function_id']] * function['num_nodes'])
            graph['num_nodes'] += function['num_nodes']
            graph['function_count'] += 1
        return graph

    @staticmethod
    def to_pyg(graph, sample):
        """Optional PyTorch Geometric adapter. Numeric x is a structural baseline."""
        import torch
        from torch_geometric.data import Data
        data = Data(x=torch.tensor(graph['x'], dtype=torch.float32).reshape(-1, len(FEATURE_NAMES)),
                    edge_index=torch.tensor(graph['edge_index'], dtype=torch.long).reshape(2, -1),
                    y=torch.tensor([int(sample['label'])], dtype=torch.long), num_nodes=graph['num_nodes'])
        data.sample_id = sample['sample_id']
        if 'function_id' in graph:
            data.function_id = graph['function_id']
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inspect', 'export-json', 'verify'])
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--split', default='train', choices=['train', 'test', 'all'])
    parser.add_argument('--architecture', choices=['x86', 'x64'])
    parser.add_argument('--class-name', choices=['goodware', 'ransomware'])
    parser.add_argument('--sample-id')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'verify':
        print(json.dumps({'verified_files': verify_bundle(args.root)}))
        return
    ds = MendeleyDataset(args.root, args.split, args.architecture, args.class_name)
    rows = [r for r in ds.rows if not args.sample_id or r['sample_id'] == args.sample_id]
    if not rows:
        parser.error('No matching sample in this split/filter')
    sample = next((r for r in rows if r['cfg_status'] == 'recovered'), rows[0])
    if args.command == 'export-json':
        if args.output is None:
            parser.error('--output is required for export-json')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write('{"sample":' + json.dumps(sample) + ',"functions":[\n')
            first = True
            for function in ds.functions(sample):
                if not first:
                    stream.write(',\n')
                json.dump(function, stream, indent=2)
                first = False
            stream.write('\n]}\n')
        print(str(args.output))
    else:
        function = next(ds.functions(sample), None)
        print(json.dumps(dict(samples_in_selection=len(ds), sample=sample,
                              first_function=None if function is None else dict(
                                  function_id=function['function_id'], num_nodes=function['num_nodes'],
                                  edge_index_shape=[2, len(function['edge_index'][0])],
                                  x_shape=[len(function['x']), len(FEATURE_NAMES)],
                                  first_block_tokens=function['node_tokens'][0][:15] if function['node_tokens'] else [])), indent=2))


if __name__ == '__main__':
    main()
