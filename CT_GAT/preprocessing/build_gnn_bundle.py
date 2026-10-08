"""Build a portable Mendeley-only dataset without changing source exports."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

import gnn_dataset as reader


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def recommendation(row):
    reasons = []
    if row['eligible'] != '1':
        reasons.append(row['exclusion_reason'] or 'not_eligible')
    elif row['selected'] != '1':
        reasons.append(row['selection_reason'] or 'duplicate_identity')
    if row['kind'] != 'candidate':
        reasons.append(row['kind'])
    if row['arch'] not in ('x86', 'x64'):
        reasons.append('unsupported_architecture')
    if row['graph_status'] != 'recovered':
        reasons.append(row['graph_status'] or 'no_recovered_cfg')
    if row['feature_ready'] != '1':
        reasons.append('feature_missing')
    if row['effective_split'] not in ('train', 'test'):
        reasons.append('no_recommended_split')
    return ';'.join(dict.fromkeys(reasons))


def build(args):
    src, dest = args.source.resolve(), args.destination.resolve()
    if dest.exists():
        raise ValueError('Use a new destination to avoid overwriting a shared snapshot')
    rows = [r for r in read_csv(src / 'reconciliation/mendeley_samples.csv') if r['corpus'] == 'mendeley']
    by_hash = collections.defaultdict(list)
    for row in rows:
        by_hash[row['sha256']].append(row)
    dest.mkdir(parents=True)
    manifests, checksums = [], {}
    for i, (digest, originals) in enumerate(sorted(by_hash.items())):
        if not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('invalid sample ID')
        row = sorted(originals, key=lambda r: (r['selected'] != '1', r['original_split'], r['source']))[0]
        state = json.loads((src / 'status' / (digest + '.json')).read_text())
        source = src / 'features' / (digest + '.jsonl.gz')
        if state['status'] != 'ok' or sha(source) != state['artifact_sha256']:
            raise ValueError('unverified export ' + digest)
        # Retain the original class/split/family directory grouping.
        if not row['group_folder'].startswith('mendeley/'):
            raise ValueError('unexpected corpus directory')
        relative = 'cfg/' + row['group_folder'][len('mendeley/'):] + '/' + digest + '.jsonl.gz'
        target = dest / relative
        if dest not in target.resolve().parents:
            raise ValueError('unsafe group directory')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha(target) != state['artifact_sha256']:
            raise ValueError('copy checksum mismatch')
        checksums[relative] = state['artifact_sha256']
        reason = recommendation(row)
        counts = state['counts']
        manifests.append(dict(sample_id=digest, label=int(row['label']),
            class_name='goodware' if row['label'] == '0' else 'ransomware',
            original_split=row['original_split'].split('_')[-1],
            recommended_split=row['effective_split'] if not reason else '',
            architecture=row['arch'], family=row['family'], input_kind=row['kind'],
            cfg_status=row['graph_status'], eligible_for_gnn=int(not reason), exclusion_reason=reason,
            identity_group=row['identity_group'], representation=row['current_representation'],
            cohort_tag=row['matched_cohort_tag'], cfg_path=relative,
            num_functions=counts['functions'], num_blocks=counts['blocks'],
            num_edges=counts['edges'], num_instructions=counts['instructions'],
            original_file_occurrences=len(originals)))
        if (i + 1) % 250 == 0:
            print('verified_and_copied', i + 1, flush=True)
    fields = list(manifests[0])
    write_csv(dest / 'samples.csv', manifests)
    for split in ('train', 'test'):
        write_csv(dest / 'splits' / (split + '.csv'), [r for r in manifests if r['recommended_split'] == split], fields)
    train_ids = {r['identity_group'] for r in manifests if r['recommended_split'] == 'train'}
    test_ids = {r['identity_group'] for r in manifests if r['recommended_split'] == 'test'}
    if train_ids & test_ids:
        raise ValueError('identity leakage between recommended splits')
    for split, identities in [('train', train_ids), ('test', test_ids)]:
        if '' in identities or len(identities) != sum(r['recommended_split'] == split for r in manifests):
            raise ValueError('missing or duplicate identity within ' + split)
    write_csv(dest / 'splits/excluded.csv', [r for r in manifests if not r['eligible_for_gnn']], fields)
    write_csv(dest / 'provenance/original_occurrences.csv', rows)
    aliases = [r for r in read_csv(src / 'reconciliation/identity_aliases.csv') if r['corpus'] == 'mendeley']
    with (src / 'reconciliation/identity_aliases.csv').open(encoding='utf-8-sig', newline='') as stream:
        alias_fields = next(csv.reader(stream))
    write_csv(dest / 'provenance/identity_aliases.csv', aliases, alias_fields)
    shutil.copyfile(src / 'policy.json', dest / 'provenance/extraction_policy.json')
    shutil.copyfile(Path(reader.__file__), dest / 'dataset.py')
    shutil.copyfile(args.guide, dest / 'START_HERE.md')
    summary = dict(schema='mendeley-gnn-bundle/1', binary_samples=len(manifests),
        original_file_occurrences=len(rows), recommended_train=len(train_ids), recommended_test=len(test_ids),
        excluded_from_default_split=sum(not r['eligible_for_gnn'] for r in manifests),
        usable_cfgs=sum(r['cfg_status'] == 'recovered' for r in manifests),
        cfg_statuses=dict(collections.Counter(r['cfg_status'] for r in manifests)),
        recommended_groups=dict(collections.Counter(r['recommended_split'] + '/' + r['class_name'] + '/' + r['architecture']
                               for r in manifests if r['eligible_for_gnn'])),
        train_test_identity_overlap=0, baseline_feature_names=reader.FEATURE_NAMES)
    (dest / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    examples = dest / 'examples'; examples.mkdir()
    ds = reader.MendeleyDataset(dest)
    for label in ('goodware', 'ransomware'):
        row = min([r for r in ds.rows if r['class_name'] == label], key=lambda r: r['num_instructions'])
        graph = next(g for g in ds.functions(row) if g['num_nodes'] > 0)
        (examples / (row['sample_id'] + '.json')).write_text(json.dumps(dict(sample=row, function=graph), indent=2), encoding='utf-8')
    for file in dest.rglob('*'):
        if file.is_file():
            relative = file.relative_to(dest).as_posix()
            if relative not in checksums:
                checksums[relative] = sha(file)
    (dest / 'SHA256SUMS.txt').write_text(''.join(h + '  ' + name + '\n' for name,h in sorted(checksums.items())), encoding='utf-8')
    if args.zip:
        archive = dest.with_suffix('.zip')
        if archive.exists():
            raise ValueError('archive already exists')
        with zipfile.ZipFile(archive, 'w', allowZip64=True) as z:
            for file in dest.rglob('*'):
                if file.is_file():
                    z.write(file, dest.name + '/' + file.relative_to(dest).as_posix(),
                            compress_type=zipfile.ZIP_STORED if file.suffix == '.gz' else zipfile.ZIP_DEFLATED)
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:
                raise ValueError('archive verification failed')
        archive.with_suffix('.zip.sha256').write_text(sha(archive) + '  ' + archive.name + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--guide', type=Path, required=True)
    parser.add_argument('--zip', action='store_true')
    build(parser.parse_args())
