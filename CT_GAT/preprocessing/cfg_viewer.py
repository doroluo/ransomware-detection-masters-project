"""Build an offline, shareable CFG viewer from IDA JSONL(.gz) or readable JSON.

python cfg_viewer.py INPUT.jsonl.gz --output advisor_cfg.html
python cfg_viewer.py INPUT.jsonl.gz --function _wmain --output wmain.html

No binary execution, server, third-party Python package, or network connection.
The HTML can also open other exports using its local file picker.
"""
import argparse
import gzip
import json
from pathlib import Path

SCHEMA = 'ida-cfg-research/2.1'
MAX_EMBED_BYTES = 128 * 1024 * 1024


def records(path):
    """Gzip/JSONL streams one record at a time; readable JSON is an array."""
    path = Path(path)
    opener = gzip.open if path.suffix.lower() == '.gz' else open
    with opener(path, 'rt', encoding='utf-8-sig') as stream:
        first = stream.read(1)
        while first and first.isspace():
            first = stream.read(1)
        if first == '[':
            yield from json.loads(first + stream.read())
        elif first:
            yield json.loads(first + stream.readline())
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def validate_function(fn):
    if not isinstance(fn.get('name'), str) or not isinstance(fn.get('rva'), int):
        raise ValueError('Function needs a name and integer RVA')
    blocks = fn.get('blocks', [])
    ids = [b['id'] for b in blocks]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate block ID in {fn['name']}")
    known = set(ids)
    for edge in fn['edges']:
        if len(edge) != 2 or any(i not in known for i in edge):
            raise ValueError(f"Dangling CFG edge in {fn['name']}")
    for block in blocks:
        if not isinstance(block.get('instructions'), list):
            raise ValueError('Missing block instructions')


def load_export(path, functions=()):
    header, footer, chosen = None, None, []
    total, budget, matches, ids = 0, 0, set(), set()
    selectors = set(functions)
    for record in records(path):
        kind = record.get('record')
        if kind == 'header':
            if header is not None or record.get('schema') != SCHEMA:
                raise ValueError('Expected exactly one ida-cfg-research/2.1 header')
            header = {k: record.get(k) for k in ('schema', 'sha256', 'arch', 'input_kind', 'size', 'ida_version')}
            header['imports'] = record.get('pe', {}).get('imports', [])
        elif kind == 'function':
            if header is None:
                raise ValueError('Function before header')
            if record['id'] in ids:
                raise ValueError('Duplicate function ID')
            ids.add(record['id'])
            total += 1
            matched = selectors.intersection((record['name'], str(record['id'])))
            matches.update(matched)
            if selectors and not matched:
                continue
            validate_function(record)
            budget += len(json.dumps(record, ensure_ascii=True))
            if budget > MAX_EMBED_BYTES:
                raise ValueError('Selected functions exceed 128 MB. Select fewer with --function NAME_OR_ID.')
            chosen.append(record)
        elif kind == 'footer':
            if footer is not None:
                raise ValueError('Duplicate footer')
            footer = record
    if header is None or footer is None or footer.get('complete') is not True:
        raise ValueError('Incomplete IDA export: header and complete footer are required')
    if selectors - matches:
        raise ValueError('Functions not found: ' + ', '.join(sorted(selectors - matches)))
    if footer.get('counts', {}).get('functions', total) != total:
        raise ValueError('Footer function count disagrees with the export')
    return {'meta': header, 'functions': chosen, 'footer': footer,
            'source': Path(path).name, 'totalFunctions': total}


def write_viewer(data, output):
    output = Path(output)
    template = Path(__file__).with_name('cfg_viewer.html').read_text(encoding='utf-8')
    # A function name or import name is data, never executable markup/script.
    payload = json.dumps(data, ensure_ascii=True, separators=(',', ':')).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    token = '<script id="cfg-data" type="application/json">null</script>'
    if template.count(token) != 1:
        raise ValueError('Viewer template payload marker missing or duplicated')
    html = template.replace(token, '<script id="cfg-data" type="application/json">' + payload + '</script>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding='utf-8')
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input', type=Path)
    ap.add_argument('--output', type=Path, default=Path('advisor_cfg.html'))
    ap.add_argument('--function', action='append', default=[], metavar='NAME_OR_ID', help='Exact name or ID; repeat to include several')
    args = ap.parse_args()
    if args.input.resolve() == args.output.resolve() or args.output.resolve() == Path(__file__).with_name('cfg_viewer.html').resolve():
        ap.error('Output must differ from input and the viewer template')
    data = load_export(args.input, args.function)
    out = write_viewer(data, args.output)
    print(f"Viewer: {out.resolve()} ({len(data['functions'])}/{data['totalFunctions']} functions)")


if __name__ == '__main__':
    main()
