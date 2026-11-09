"""Static research figures, with validation and held-out series kept separate."""
import json
from pathlib import Path


def optimizer_figures(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root); out = root / 'reports' / 'figures'; out.mkdir(parents=True, exist_ok=True)
    for model in (1, 2, 3):
        paths = sorted((root / 'selection' / 'fold0' / f'model{model}' / 'optimizer_500').glob('*/metrics.json'))
        if not paths:
            continue
        signature = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
        marker = out / f'model{model}_optimizer.sources.json'
        if marker.exists() and json.loads(marker.read_text()) == [list(v) for v in signature]:
            continue
        values = [json.loads(p.read_text()) for p in paths]
        pools = sorted({r['spec']['pooling'] for r in values})
        decays = sorted({r['spec']['weight_decay'] for r in values})
        fig, axes = plt.subplots(len(pools), len(decays), figsize=(12, 3.4 * len(pools)), squeeze=False, sharey=True)
        for row, pool in enumerate(pools):
            for col, decay in enumerate(decays):
                ax = axes[row, col]
                for drop in sorted({r['spec']['dropout'] for r in values}):
                    rr = sorted([r for r in values if r['spec']['pooling'] == pool and r['spec']['weight_decay'] == decay
                                 and r['spec']['dropout'] == drop], key=lambda r: r['spec']['lr'])
                    ax.plot([r['spec']['lr'] for r in rr], [r['validation_fixed']['auc'] for r in rr],
                            marker='o', label=f'Dropout {drop:g}')
                ax.set(xscale='log', xlabel='Learning rate', ylabel='Validation ROC-AUC',
                       title=f'{pool}; weight decay {decay:g}')
                ax.grid(alpha=.25)
        axes[0, -1].legend(fontsize=8)
        fig.suptitle(f'Model {model}: optimizer grid, outer fold 0 validation, seed 0 ({len(values)} completed settings)')
        fig.tight_layout(); fig.savefig(out / f'model{model}_optimizer_validation.png', dpi=180); plt.close(fig)
        marker.write_text(json.dumps(signature), encoding='utf-8')


def result_figures(root, rows):
    if not rows:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    out = Path(root) / 'reports' / 'figures'; out.mkdir(parents=True, exist_ok=True)
    primary = [r for r in rows if r['series'] == 'primary' and r['fold'] == 0]
    groups = {}
    for row in primary:
        groups.setdefault((row['model'], row['pooling']), []).append(row)
    groups = {k: v for k, v in groups.items() if {r['seed'] for r in v} == {0, 1, 2}}
    if not groups:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    labels = [f'Model {m}\n{p}' for m, p in sorted(groups)]
    for ax, metric, label in zip(axes, ('auc', 'detection_rate', 'fpr'), ('ROC-AUC', 'Detection rate', 'False-positive rate')):
        values = [[r[metric] for r in groups[k]] for k in sorted(groups)]
        ax.bar(range(len(values)), [np.mean(v) for v in values], yerr=[np.std(v, ddof=1) for v in values],
               capsize=4, color=['#386cb0', '#7fc97f', '#fdc086', '#beaed4'][:len(values)])
        ax.set(xticks=range(len(labels)), xticklabels=labels, ylabel=label, ylim=(0, 1))
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Primary models: outer test fold 0; mean and sample SD across seeds 0, 1, 2')
    fig.tight_layout(); fig.savefig(out / 'primary_fold0_test.png', dpi=180); plt.close(fig)
