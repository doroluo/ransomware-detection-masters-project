"""Export complete parameter records from the original campaign and attention study."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


METRICS = ('accuracy', 'auc', 'pr_auc', 'precision', 'detection_rate', 'specificity',
           'fpr', 'macro_f1', 'threshold', 'tp', 'tn', 'fp', 'fn', 'n')


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def effective_attention(model, spec):
    graph = model in (2, 3)
    transformer = model in (1, 3)
    heads = [spec['gat_heads']] * spec['gat_layers'] if graph else []
    if graph and spec.get('gat_head_policy', 'last_layer_one') == 'last_layer_one':
        heads[-1] = 1
    return dict(
        effective_transformer_heads=spec['transformer_heads'] if transformer else None,
        effective_gat_heads=compact(heads) if graph else None,
        transformer_attention_dropout=spec.get('attention_dropout', spec['dropout']) if transformer else None,
        gat_attention_dropout=spec.get('attention_dropout', spec['dropout']) if graph else None,
        feature_dropout=spec['dropout'],
        dropout_control='independent_attention' if 'attention_dropout' in spec else 'shared',
    )


def result_record(metrics_path, root, series, fold, stage, representation, ablation):
    result = json.loads(metrics_path.read_text())
    contract = json.loads((metrics_path.parent / 'config.json').read_text())
    expected = hashlib.sha256(compact(contract).encode()).hexdigest()
    if expected != result['fingerprint']:
        raise ValueError(f'Result contract mismatch at {metrics_path}')
    spec = contract['spec']
    split = 'test' if contract['evaluate_test'] else 'validation'
    score = dict(result['test'] if split == 'test' else result['validation_fixed'])
    score['precision'] = score['tp'] / (score['tp'] + score['fp']) if score['tp'] + score['fp'] else None
    score['specificity'] = score['tn'] / (score['tn'] + score['fp']) if score['tn'] + score['fp'] else None
    row = dict(
        series=series, model=contract['model'], fold=fold, stage=stage, split=split,
        representation=representation, ablation=ablation, seed=contract['seed'],
        pooling=spec['pooling'], **{name: score.get(name) for name in METRICS},
        validation_auc=result['validation_fixed']['auc'],
        validation_macro_f1=result['validation_fixed']['macro_f1'],
        **effective_attention(contract['model'], spec),
        **{key: compact(value) if isinstance(value, (dict, list)) else value for key, value in spec.items()
           if key not in ('pooling', 'ablation', 'paper_adapter_sources', 'attention_sources')},
        epochs_requested=contract['epochs'], epochs_run=result['epochs_run'],
        best_epoch=result['best_epoch'], patience=contract['patience'],
        optimizer='AdamW', optimizer_betas='[0.9,0.999]', optimizer_eps=1e-8,
        gradient_clip=1.0, class_weight_rule='inverse training class frequency',
        parameters=result['parameters'], epoch_seconds_mean=result['epoch_seconds_mean'],
        peak_gpu_mb=result['peak_gpu_mb'], gpu_name=result.get('gpu_name'),
        torch_version=contract['torch_version'], dataset_fingerprint=contract['dataset_fingerprint'],
        run_fingerprint=result['fingerprint'],
        scientific_source_digest=hashlib.sha256(compact(contract['source_hashes']).encode()).hexdigest(),
        adapter_source_digest=hashlib.sha256(compact(spec.get('paper_adapter_sources', {})).encode()).hexdigest(),
        attention_source_digest=hashlib.sha256(compact(spec.get('attention_sources', {})).encode()).hexdigest(),
        source_record=metrics_path.relative_to(root).as_posix(),
    )
    return row


def collect(campaign, study=None):
    campaign = Path(campaign)
    validation, held_out = [], []
    for folder in sorted((campaign / 'selection').glob('fold*')):
        fold = int(folder.name[4:])
        for path in sorted(folder.glob('model*/*_500/*/metrics.json')):
            stage = path.parents[1].name.removesuffix('_500')
            validation.append(result_record(path, campaign, 'primary_selection', fold, stage, 'signal500', 'none'))
    config = json.loads((campaign / 'CAMPAIGN.json').read_text())
    excluded = set(config.get('excluded_representations', []))
    for series in ('primary', 'comparisons', 'ablations'):
        for path in sorted((campaign / series).glob('**/trials/*/metrics.json')):
            job = path.parents[2]
            plan = json.loads((job / 'plan.json').read_text())
            rep = 'signal500_original' if series == 'primary' else (
                path.relative_to(campaign).parts[1] if series == 'comparisons' else 'signal500_common_api')
            if rep not in excluded:
                held_out.append(result_record(path, campaign, series, plan['fold'], 'confirmation', rep, plan['ablation']))
    if study and Path(study).exists():
        study = Path(study)
        for path in sorted(study.glob('fold*/model*/*/*/metrics.json')):
            if json.loads((path.parents[2] / 'plan.json').read_text()).get('smoke'):
                continue
            relative = path.relative_to(study).parts
            fold, stage = int(relative[0][4:]), relative[2]
            row = result_record(path, study, 'attention_followup', fold, stage, 'signal500_common_api', 'none')
            (held_out if row['split'] == 'test' else validation).append(row)
    return validation, held_out


def table(rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    return dict(columns=keys, rows=[[row.get(key) for key in keys] for row in rows])


def write_snapshot(campaign, study, output):
    validation, held_out = collect(campaign, study)
    rows = [
        ['Completed validation stage entries', len(validation)],
        ['Completed held-out runs', len(held_out)],
        ['Previously tested head counts', '2, 4, 8'],
        ['Previously tested shared dropout', '0.0, 0.1, 0.3, 0.5'],
        ['Requested new attention dropout', '0.0, 0.1, 0.2'],
        ['Heads for the follow-up', '2, 4, 8, 12'],
        ['Transformer width for the follow-up', '96 for every tested head count'],
        ['Existing GAT output layer', 'One head. A one-layer GAT therefore uses one head.'],
        ['Follow-up GAT layers', 'All layers use the requested head count.'],
        ['Model selection', 'Validation ROC-AUC, then macro-F1. Outer test scores do not select settings.'],
        ['Precision and specificity', 'Calculated from the recorded confusion counts.'],
        ['Existing result source', 'final_paper_20261103 campaign'],
        ['Snapshot UTC', datetime.now(timezone.utc).isoformat()],
        ['Performance goal', 'Evaluate the 90% target on held-out results. Improvements require measurement.'],
    ]
    snapshot = dict(
        summary=dict(columns=['Item', 'Value'], rows=rows),
        validation=table(validation), held_out=table(held_out),
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(snapshot, allow_nan=False), encoding='utf-8')
    temporary.replace(output)
    for name, records in (('validation', validation), ('held_out', held_out)):
        sheet = snapshot[name]
        with output.with_name(name + '_parameters.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(sheet['columns'])
            writer.writerows(sheet['rows'])
    return snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--study', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--xlsx-node', type=Path)
    parser.add_argument('--xlsx-builder', type=Path)
    parser.add_argument('--xlsx-out', type=Path)
    args = parser.parse_args(argv)
    if any((args.xlsx_node, args.xlsx_builder, args.xlsx_out)) and not all(
            (args.xlsx_node, args.xlsx_builder, args.xlsx_out)):
        parser.error('The Excel export needs --xlsx-node, --xlsx-builder and --xlsx-out together')
    snapshot = write_snapshot(args.campaign, args.study, args.out)
    if args.xlsx_node:
        subprocess.run([str(args.xlsx_node), str(args.xlsx_builder), str(args.out), str(args.xlsx_out)], check=True)
    print(compact({name: len(snapshot[name]['rows']) for name in ('validation', 'held_out')}))


if __name__ == '__main__':
    main()
