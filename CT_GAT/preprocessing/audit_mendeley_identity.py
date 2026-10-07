"""Read-only hash audit. Never repairs samples or treats names as identity."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--previous-manifest', type=Path)
    parser.add_argument('--vm-check', action='store_true')
    parser.add_argument('--verify-upx', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    review = json.loads(args.review.read_text(encoding='utf-8'))
    previous = {}
    if args.previous_manifest:
        with args.previous_manifest.open(newline='', encoding='utf-8-sig') as stream:
            previous = {r['sha256']: r for r in csv.DictReader(stream)}
    rows = []
    for original in review.get('unmatched', review.get('rows', [])):
        row = dict(original)
        expected = row['unverified_filename_candidates']
        row['previous_records'] = [previous[r['sha256']] for r in expected if r['sha256'] in previous] if previous else row.get('previous_records', [])
        if args.vm_check:
            path = Path(row['source']).resolve()
            roots = [Path.home() / 'goodware/Goodware_Mendeley', Path.home() / 'ransomware/Ransomware_Mendeley']
            if not any(root.resolve() in path.parents for root in roots):
                raise ValueError('unexpected sample location')
            # Read bytes only; output contains exclusively hashes/counts.
            raw = path.read_bytes()
            row['observed_sha256'] = hashlib.sha256(raw).hexdigest()
            row['observed_size'] = len(raw)
            row['matches_inventory'] = row['observed_sha256'] == row['sha256']
            row['size_deltas_from_previous'] = [len(raw) - int(r['size']) for r in row['previous_records']]
            row['newline_transform_matches'] = []
            for name, transformed in [('crlf_to_lf', raw.replace(b'\r\n', b'\n')),
                                      ('lf_to_crlf', raw.replace(b'\n', b'\r\n'))]:
                if hashlib.sha256(transformed).hexdigest() in {r['sha256'] for r in expected}:
                    row['newline_transform_matches'].append(name)
            row['zero_padding_matches'] = []
            for record in row['previous_records']:
                gap = int(record['size']) - len(raw)
                if 0 < gap <= 256 * 1024**2:
                    digest = hashlib.sha256(raw)
                    while gap:
                        count = min(gap, 1024 * 1024)
                        digest.update(b'\0' * count)
                        gap -= count
                    if digest.hexdigest() == record['sha256']:
                        row['zero_padding_matches'].append(record['sha256'])
            import pefile
            try:
                pe = pefile.PE(data=raw, fast_load=True)
                row['pe_integrity'] = dict(machine=pe.FILE_HEADER.Machine,
                    sections_beyond_eof=sum(s.PointerToRawData + s.SizeOfRawData > len(raw) for s in pe.sections if s.SizeOfRawData),
                    section_raw_end=max([s.PointerToRawData + s.SizeOfRawData for s in pe.sections if s.SizeOfRawData] or [0]),
                    declared_image_size=pe.OPTIONAL_HEADER.SizeOfImage,
                    overlay_offset=pe.get_overlay_data_start_offset(),
                    section_names=[s.Name.rstrip(b'\0').decode('ascii', 'replace') for s in pe.sections])
                pe.close()
            except Exception as error:
                row['pe_integrity'] = dict(error=type(error).__name__)
            if args.verify_upx:
                permitted = (Path.home() / 'work/out').resolve()
                if permitted not in args.out.resolve().parents:
                    raise ValueError('UPX audit output must be under ~/work/out/')
                row['upx_verification'] = dict(status='not_attempted')
                with tempfile.TemporaryDirectory(prefix=row['sha256'] + '_', dir=str(args.out.parent)) as directory:
                    copied = Path(directory) / (row['sha256'] + '.pe')
                    shutil.copyfile(str(path), str(copied))
                    copied.chmod(0o600)
                    log_path = args.out.parent / (row['sha256'] + '.upx_audit.log')
                    try:
                        with log_path.open('w') as log:
                            result = subprocess.run(['upx', '-d', str(copied)], stdout=log, stderr=subprocess.STDOUT,
                                                    timeout=120, check=False)
                        unpacked = copied.read_bytes()
                        unpacked_hash = hashlib.sha256(unpacked).hexdigest()
                        verified = unpacked_hash in {r['sha256'] for r in expected}
                        row['upx_verification'] = dict(status='verified_exact_cohort_hash' if result.returncode == 0 and verified else 'not_verified',
                                                       returncode=result.returncode, unpacked_sha256=unpacked_hash,
                                                       unpacked_size=len(unpacked))
                    except subprocess.TimeoutExpired:
                        row['upx_verification'] = dict(status='timeout')
        rows.append(row)
    result = dict(note='Read-only diagnostic. Filename matches are not verified sample identity; no relabeling or sample mutation.',
                  count=len(rows), groups=dict(collections.Counter(r['original_split'] for r in rows)),
                  same_size=sum(any(d == 0 for d in r.get('size_deltas_from_previous', [])) for r in rows),
                  newline_transform_matches=sum(bool(r.get('newline_transform_matches')) for r in rows), rows=rows)
    result['zero_padding_matches'] = sum(bool(r.get('zero_padding_matches')) for r in rows)
    result['files_with_sections_beyond_eof'] = sum(r.get('pe_integrity', {}).get('sections_beyond_eof', 0) > 0 for r in rows)
    result['upx_statuses'] = dict(collections.Counter(r.get('upx_verification', {}).get('status', 'not_attempted') for r in rows))
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
