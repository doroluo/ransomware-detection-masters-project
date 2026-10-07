"""Build leakage-safe experiment manifests without modifying original provenance."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

from vm_ida_cfg import write_csv, write_json

POLICY = 'verified-upx-alias-test-reserved/1'


def reconcile(root):
    root = Path(root)
    inventory = json.loads((root / 'inventory.json').read_text())
    memberships = json.loads((root / 'cohort_membership.json').read_text())
    evidence_path = root / 'identity_evidence.json'
    evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else {'rows': []}
    parent = {}

    def find(value):
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    aliases = {}
    for row in evidence['rows']:
        proof = row.get('upx_verification', {})
        if proof.get('status') != 'verified_exact_cohort_hash':
            continue
        raw, unpacked = row['sha256'], proof['unpacked_sha256']
        if row.get('observed_sha256') != raw or not row.get('matches_inventory') or proof.get('returncode') != 0:
            raise ValueError('inconsistent identity proof')
        if unpacked not in {r['sha256'] for r in row['unverified_filename_candidates']}:
            raise ValueError('unpacked hash has no expected cohort identity')
        aliases[raw] = unpacked
        a, b = find(raw), find(unpacked)
        parent[max(a, b)] = min(a, b)
    members = collections.defaultdict(list)
    for row in memberships:
        members[(row['corpus'], row['set'], row['sha256'])].append(row)
    components = collections.defaultdict(list)
    for row in inventory:
        if row['sha256']:
            components[find(row['sha256'])].append(row)
    component_labels = {key: {r['label'] for r in rows if r['label'] != ''} for key, rows in components.items()}
    # Reserve test identities before exclusions: do not leak a quarantined test
    # identity into training through another collection or packed representation.
    test_ids = {key for key, rows in components.items() if any(r['original_split'].endswith('test') for r in rows)}
    statuses = {p.stem: json.loads(p.read_text()) for p in (root / 'status').glob('*.json')}
    rows, alias_rows = [], []
    for original in inventory:
        row = {key: original.get(key, '') for key in ('sha256', 'corpus', 'group', 'group_folder', 'original_split',
                                                       'family', 'label', 'label_semantics', 'arch', 'kind', 'source', 'relative_path')}
        digest = row['sha256']
        identity = find(digest) if digest else ''
        canonical = aliases.get(digest, digest)
        matched = members.get((row['corpus'], row['original_split'], digest), [])
        identity_status = 'exact_hash_match' if matched else 'no_cohort_match'
        if not matched and canonical != digest:
            matched = members.get((row['corpus'], row['original_split'], canonical), [])
            if matched:
                identity_status = 'verified_upx_transform'
        flags = {r['in_cohort'] for r in matched}
        labels = {r['label'] for r in matched}
        split = 'test' if row['original_split'].endswith('test') else 'train' if row['original_split'].endswith('train') else ''
        reason = ''
        if not digest or row['kind'] in ('dataset_support_file', 'symlink_not_followed', 'inventory_error'):
            reason = 'not_a_sample_candidate'
        elif len(component_labels[identity]) > 1 or (labels and labels != {row['label']}):
            reason = 'label_conflict'
        elif not split:
            reason = 'no_original_experiment_split'
        elif not matched:
            reason = 'unverified_cohort_membership'
        elif flags != {'1'}:
            reason = 'original_cohort_exclusion_or_conflict'
        elif split == 'train' and identity in test_ids:
            reason = 'identity_reserved_for_test'
        status = statuses.get(digest, {})
        feature = 'features/' + digest + '.jsonl.gz'
        ready = status.get('status') == 'ok' and (root / feature).is_file()
        row.update(identity_group=identity, cohort_sha256=canonical if matched else '',
                   identity_status=identity_status, current_representation='upx_packed' if digest in aliases else 'as_stored',
                   matched_cohort_tag='|'.join(sorted({r['tag'] for r in matched})),
                   cohort_inclusion='|'.join(sorted(flags)), effective_split=split if not reason else '',
                   eligible=int(not reason), exclusion_reason=reason, extraction_status=status.get('status', 'pending'),
                   graph_status=status.get('graph_status', ''), feature_ready=int(ready),
                   feature_file=feature if ready else '', selected=0, selection_reason='')
        rows.append(row)
        if digest in aliases:
            alias_rows.append({key: row[key] for key in ('sha256', 'cohort_sha256', 'identity_group', 'corpus', 'original_split',
                                                         'identity_status', 'current_representation', 'matched_cohort_tag')})
    selected = {}
    for row in sorted(rows, key=lambda r: (r['corpus'] != 'mendeley', r['sha256'], r['source'])):
        if not row['eligible']:
            continue
        key = row['identity_group']
        if key not in selected:
            selected[key] = row
            row['selected'] = 1
        else:
            row['selection_reason'] = 'duplicate_identity_in_same_split'
    train = [r for r in selected.values() if r['effective_split'] == 'train']
    test = [r for r in selected.values() if r['effective_split'] == 'test']
    if {r['identity_group'] for r in train} & {r['identity_group'] for r in test}:
        raise AssertionError('train/test identity leakage')
    out = root / 'reconciliation'
    out.mkdir(exist_ok=True)
    fields = list(rows[0]) if rows else []
    for name, subset in [('samples.csv', rows), ('mendeley_samples.csv', [r for r in rows if r['corpus'] == 'mendeley']),
                         ('train.csv', train), ('test.csv', test),
                         ('train_ready.csv', [r for r in train if r['feature_ready']]),
                         ('test_ready.csv', [r for r in test if r['feature_ready']]),
                         ('excluded.csv', [r for r in rows if not r['eligible']])]:
        write_csv(out / name, subset, fields)
    unique_aliases = {}
    for row in alias_rows:
        unique_aliases.setdefault(row['sha256'], row)
    write_csv(out / 'identity_aliases.csv', list(unique_aliases.values()), list(alias_rows[0]) if alias_rows else ['sha256', 'cohort_sha256'])
    report = dict(policy=POLICY, files=len(rows), verified_transform_hashes=len(aliases),
                  mendeley_identity_status=dict(collections.Counter(r['identity_status'] for r in rows if r['corpus'] == 'mendeley')),
                  exclusions=dict(collections.Counter(r['exclusion_reason'] for r in rows if not r['eligible'])),
                  selected_train=len(train), selected_test=len(test),
                  ready_train=sum(r['feature_ready'] for r in train), ready_test=sum(r['feature_ready'] for r in test),
                  train_test_identity_overlap=0,
                  evidence_sha256=hashlib.sha256(evidence_path.read_bytes()).hexdigest() if evidence_path.exists() else None,
                  note='Original inventory/cohorts/splits unchanged. Ready subsets are incomplete until extraction finishes. No near-duplicate guarantee.')
    write_json(out / 'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    print(json.dumps(reconcile(parser.parse_args().root), indent=2))
