"""Paper tables with explicit split/seed scope and fold-specific validation thresholds."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from experiments.model_grid.data import digest
from experiments.model_grid.runner import atomic_json


METRICS = ('auc', 'pr_auc', 'accuracy', 'detection_rate', 'fpr', 'macro_f1')
GROUP = ('series', 'representation', 'ablation', 'model', 'pooling')


def scores(y, probability, decision):
    y, probability, decision = np.asarray(y), np.asarray(probability), np.asarray(decision, dtype=bool)
    tp = int(((y == 1) & decision).sum()); tn = int(((y == 0) & ~decision).sum())
    fp = int(((y == 0) & decision).sum()); fn = int(((y == 1) & ~decision).sum())
    return dict(n=len(y), auc=float(roc_auc_score(y, probability)),
        pr_auc=float(average_precision_score(y, probability)), accuracy=float((y == decision).mean()),
        detection_rate=tp / (tp + fn), fpr=fp / (fp + tn),
        macro_f1=float(f1_score(y, decision, average='macro', zero_division=0)), tp=tp, tn=tn, fp=fp, fn=fn)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temp = Path(path).with_suffix('.csv.tmp')
    with temp.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temp.replace(path)


def collect(root):
    rows = []
    for series in ('primary', 'comparisons', 'ablations'):
        for path in (root / series).glob('**/trials/*/metrics.json'):
            result = json.loads(path.read_text())
            if 'test' not in result:
                continue
            config = json.loads((path.parent / 'config.json').read_text())
            if result['fingerprint'] != digest(config) or not config['evaluate_test']:
                raise ValueError(f'Invalid completed result contract: {path}')
            job = path.parent.parent.parent
            plan = json.loads((job / 'plan.json').read_text())
            relative = path.relative_to(root).parts
            row = dict(series=series, representation='signal500_original' if series == 'primary' else
                relative[1] if series == 'comparisons' else 'signal500_common_api',
                ablation=plan['ablation'], model=config['model'], pooling=result['spec']['pooling'],
                fold=plan['fold'], seed=result['seed'], **result['test'],
                best_epoch=result['best_epoch'], epochs_run=result['epochs_run'], parameters=result['parameters'],
                seconds_per_epoch=result['epoch_seconds_mean'], peak_gpu_mb=result['peak_gpu_mb'],
                case_path=str(path.parent.resolve()), job_path=str(job.resolve()))
            for key in ('width', 'transformer_layers', 'transformer_heads', 'gat_hidden', 'gat_layers', 'gat_heads',
                        'block_length', 'sequence_length', 'batch_size', 'micro_batch_size', 'lr', 'dropout', 'weight_decay'):
                row[key] = result['spec'].get(key)
            rows.append(row)
    return sorted(rows, key=lambda r: tuple(str(r[k]) for k in (*GROUP, 'fold', 'seed')))


def oof_for(rows, expected_ids):
    if {r['fold'] for r in rows} != set(range(5)) or len(rows) != 5:
        return None
    by_id = {}
    for row in rows:
        path = Path(row['case_path']) / 'test_predictions.csv'
        provenance = json.loads((Path(row['job_path']) / 'dataset.json').read_text())
        expected_fold = set(provenance['splits']['test'])
        actual_fold = set()
        with path.open(newline='', encoding='utf-8') as handle:
            for prediction in csv.DictReader(handle):
                sid = prediction['sample_id']
                if sid in by_id or prediction['split'] != 'test':
                    raise ValueError('Duplicate/non-test prediction in OOF aggregation')
                actual_fold.add(sid)
                label, probability = int(prediction['label']), float(prediction['prob'])
                if not np.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError('Invalid prediction probability')
                if isinstance(expected_ids, dict) and label != expected_ids[sid]:
                    raise ValueError('OOF label differs from the authoritative cohort index')
                by_id[sid] = dict(sample_id=sid, label=label, probability=probability,
                    decision=probability >= row['threshold'], threshold=row['threshold'],
                    family=prediction['family'], arch=prediction['arch'], fold=row['fold'], seed=row['seed'])
        if actual_fold != expected_fold:
            raise ValueError('Prediction IDs differ from the declared outer test split')
    if set(by_id) != set(expected_ids):
        raise ValueError('OOF prediction cohort differs from the declared combined cohort')
    return [by_id[sid] for sid in sorted(by_id)]


def cluster_intervals(seed_predictions, index_rows, repeats=2000):
    """Class-stratified family/identity bootstrap; average metrics over fixed trained seeds."""
    first = seed_predictions[0]
    ids = [r['sample_id'] for r in first]
    for seed in seed_predictions:
        if [r['sample_id'] for r in seed] != ids:
            raise ValueError('Unpaired seed OOF cohorts')
    y = np.array([r['label'] for r in first])
    groups = {0: {}, 1: {}}
    for i, sid in enumerate(ids):
        label = int(y[i]); source = index_rows[sid]
        key = source['family'] if label else source['identity_group']
        groups[label].setdefault(key, []).append(i)
    rng = np.random.default_rng(20261103)
    samples = {k: [] for k in METRICS}
    arrays = [(np.array([r['probability'] for r in p]), np.array([r['decision'] for r in p])) for p in seed_predictions]
    for _ in range(repeats):
        positions = []
        for label in (0, 1):
            units = list(groups[label].values())
            for pick in rng.integers(0, len(units), len(units)):
                positions.extend(units[pick])
        idx = np.asarray(positions)
        values = [scores(y[idx], probability[idx], decision[idx]) for probability, decision in arrays]
        for key in METRICS:
            samples[key].append(statistics.mean(v[key] for v in values))
    return dict(repeats=repeats, grouping='ransomware family / benign identity, stratified by class',
        scope='Conditional on the trained models; does not refit models, thresholds, or hyperparameters',
        group_counts={str(k): len(v) for k, v in groups.items()},
        intervals={k: [float(x) for x in np.quantile(v, [.025, .975])] for k, v in samples.items()})


def publish(config, state, bootstrap=0):
    root = Path(config['root']); reports = root / 'reports'; reports.mkdir(parents=True, exist_ok=True)
    with Path(config['index']).open(newline='', encoding='utf-8') as handle:
        index = {r['sample_id']: r for r in csv.DictReader(handle)}
    excluded = json.loads((Path(config['builds']['signal500']) / 'settings.json').read_text())['excluded']
    expected = {sid: int(row['label']) for sid, row in index.items()
                if row.get('fold_v3') not in ('', None) and sid not in excluded}
    all_rows = collect(root)
    excluded_representations = set(config.get('excluded_representations', []))
    rows = [row for row in all_rows if row['representation'] not in excluded_representations]
    write_csv(reports / 'RESULTS.csv', rows)
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in GROUP)
        groups.setdefault(key, []).append(row)
    per_fold, oof_rows = [], []
    lines = ['# Final-paper results', '',
        'Only completed, contract-verified held-out results appear below. Partial queues and resource failures are reported separately.',
        'Primary tuning, common-vocabulary representation comparisons, and ablations are separate series.',
        'These existing outer folds have historical results; they are not a new external holdout.', '',
        '| Series | Representation | Ablation | Model | Pooling | Fold | Seeds | AUC | Detection | FPR |',
        '|---|---|---|---|---|---|---|---|---|---|']
    if excluded_representations:
        lines[6:6] = ['Paper comparisons exclude ' + ', '.join(sorted(excluded_representations)) +
            '. Completed raw results remain in the campaign archive.', '']
    for key, records in groups.items():
        for fold in config['folds']:
            rr = [r for r in records if r['fold'] == fold]
            if not rr:
                continue
            entry = dict(zip(GROUP, key)); entry.update(fold=fold, completed_seeds=len(rr), expected_seeds=len(config['seeds']))
            for metric in METRICS:
                v = [r[metric] for r in rr]
                entry[metric + '_mean'] = statistics.mean(v)
                entry[metric + '_seed_sd'] = statistics.stdev(v) if len(v) > 1 else None
            per_fold.append(entry)
            cell = lambda m: f'{entry[m + "_mean"]:.4f}' + (f' ± {entry[m + "_seed_sd"]:.4f}' if entry[m + '_seed_sd'] is not None else '')
            lines.append('| ' + ' | '.join(map(str, (*key, fold, f'{len(rr)}/{len(config["seeds"])}', cell('auc'), cell('detection_rate'), cell('fpr')))) + ' |')
        completed = []
        for seed in config['seeds']:
            predictions = oof_for([r for r in records if r['seed'] == seed], expected)
            if predictions is None:
                continue
            label = digest(dict(group=key, seed=seed))[:16]
            write_csv(reports / f'OOF_{label}.csv', predictions)
            value = scores([r['label'] for r in predictions], [r['probability'] for r in predictions],
                           np.array([r['decision'] for r in predictions]))
            oof_rows.append(dict(zip(GROUP, key), seed=seed, **value, threshold_policy='Per-fold validation-fitted thresholds'))
            completed.append(predictions)
        if bootstrap and len(completed) == len(config['seeds']):
            label = digest(dict(group=key, predictions=completed, bootstrap=bootstrap))[:16]
            target = reports / f'CI_{label}.json'
            if not target.exists():
                atomic_json(target, dict(group=dict(zip(GROUP, key)), **cluster_intervals(completed, index, bootstrap)))
    write_csv(reports / 'PER_FOLD_SUMMARY.csv', per_fold)
    write_csv(reports / 'OOF_SUMMARY.csv', oof_rows)
    ci_rows = []
    for path in reports.glob('CI_*.json'):
        doc = json.loads(path.read_text())
        if doc['group']['representation'] in excluded_representations:
            continue
        entry = dict(doc['group'])
        entry.update(bootstrap_repeats=doc['repeats'], grouping=doc['grouping'], scope=doc['scope'])
        for metric, interval in doc['intervals'].items():
            entry[metric + '_ci_lower'], entry[metric + '_ci_upper'] = interval
        ci_rows.append(entry)
    write_csv(reports / 'CI_SUMMARY.csv', ci_rows)
    if not rows:
        lines += ['', 'No new held-out results yet. Validation-only searches are running; smoke scores are not paper results.']
    lines += ['', 'Uncertainty shown in the per-fold table is sample SD across training seeds, not a confidence interval.',
        'OOF files combine exactly one prediction per sample per seed from five separately selected outer folds.',
        'Their hard decisions use the threshold fitted on the corresponding fold validation set.',
        'Optional confidence intervals resample family/identity groups conditional on the trained models.',
        'Runtime measurements reflect campaign load; they are not isolated deployment latency benchmarks.', '']
    (reports / 'SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')
    jobs = state.get('jobs', {})
    complete = sum(v.get('state') in ('complete', 'complete_with_resource_limits', 'complete_with_trial_failures') for v in jobs.values())
    active = [name for name, value in jobs.items() if value.get('state') == 'running']
    errors = [name for name, value in jobs.items() if value.get('state') in ('failed', 'resource_infeasible', 'blocked')]
    skipped = [name for name, value in jobs.items() if value.get('state') == 'skipped']
    progress = f'Completed jobs: {complete}/{len(jobs)}.'
    if skipped:
        progress = f'Completed jobs: {complete}/{len(jobs) - len(skipped)} scheduled; {len(skipped)} skipped ({len(jobs)} original jobs).'
    text = ['# Final-paper campaign status', '', f'Campaign state: {state.get("state", "running")}',
        f'First results checkpoint: {config["meeting_date"]}. Extended campaign target: {config["extended_date"]}. Paper due: {config["paper_due"]}.',
        f'{progress} Completed held-out seed/fold runs: {len(all_rows)}.',
        f'Paper-included runs: {len(rows)}; archived excluded runs: {len(all_rows) - len(rows)}.',
        f'Active jobs: {", ".join(active) or "none"}.', '',
        'The first priorities are complete validation grids, three-seed primary results across all five grouped folds, and the first controlled 250/500 signal/callwalk comparisons.',
        'Further block counts and ablations follow within the declared total budget.', '',
        'Selection is validation-only. Each outer fold has its own search; test scores never choose configurations.',
        'The new comparison series uses a shared semantic API vocabulary fitted only from training representations.',
        'Microbatch accumulation keeps the effective batch size fixed across graph sizes. Memory failures receive no invented scores.', '',
        'Files: SUMMARY.md, RESULTS.csv, PER_FOLD_SUMMARY.csv, OOF_SUMMARY.csv and STATUS.json.', '',
        'Create STOP in the campaign folder to stop dispatch and request an epoch-boundary checkpoint; rerun RESUME.ps1 to continue.',
        f'Campaign folder: {root}', '']
    if errors:
        text += ['Jobs requiring attention or reporting a limit:', *[f'- {j}' for j in errors], '']
    if skipped:
        text += ['Jobs excluded from the remaining schedule:',
            *[f'- {name}: {jobs[name]["reason"]}' for name in sorted(skipped)], '']
    if excluded_representations:
        text += ['Representations excluded from paper tables and charts: ' +
            ', '.join(sorted(excluded_representations)) + '.', '']
    text += ['Validation search progress:', '']
    for fold in config['folds']:
        folder = root / 'selection' / f'fold{fold}'
        if not folder.exists():
            continue
        completed = len(list(folder.glob('model*/*_500/*/metrics.json')))
        reused = len(list(folder.glob('model*/*_500/*/reused_from.json')))
        failures = len(list(folder.glob('model*/*_500/*/failure.json')))
        current_path = folder / 'status.json'
        current = json.loads(current_path.read_text()) if current_path.exists() else {}
        phase = f'Model {current["model"]}, {current.get("stage", "preparation")}' if current.get('model') else current.get('state', 'preparation')
        text.append(f'- Fold {fold}: {completed}/509 stage entries complete ({reused} reused); {failures} failed cases. Current: {phase}.')
    text.append('')
    (reports / 'STATUS.md').write_text('\n'.join(text), encoding='utf-8')
    atomic_json(reports / 'STATUS.json', state)
    try:
        from .figures import optimizer_figures, result_figures
        optimizer_figures(root)
        result_figures(root, rows)
    except ImportError:
        pass  # Numeric results remain available without plotting dependencies.
    return dict(completed_test_cases=len(all_rows), included_test_cases=len(rows),
        excluded_test_cases=len(all_rows) - len(rows), jobs_completed=complete,
        jobs_scheduled=len(jobs) - len(skipped), jobs_skipped=len(skipped), active=active, errors=errors)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--bootstrap', type=int, default=0)
    a = p.parse_args(argv)
    config = json.loads((a.root / 'CAMPAIGN.json').read_text())
    state = json.loads((a.root / 'STATUS.json').read_text())
    print(json.dumps(publish(config, state, a.bootstrap)), flush=True)


if __name__ == '__main__':
    main()
