"""Export draft-ready validation plots from an in-progress results.csv (python -m experiments.grid_study.plot --out ...)."""
import argparse
import csv
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (args.out/'results.csv').open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    out=args.out/'figures'; out.mkdir(exist_ok=True)
    for model in sorted({r['model'] for r in rows}):
        data=[r for r in rows if r['model']==model and r['stage']=='optimizer']
        pools=sorted({r['pooling'] for r in data})
        for pool in pools:
            rr=[r for r in data if r['pooling']==pool]
            decays=sorted({float(r['weight_decay']) for r in rr})
            fig,axes=plt.subplots(len(decays),3,squeeze=False,figsize=(12,3.3*len(decays)))
            for j,decay in enumerate(decays):
                for ax,metric in zip(axes[j],('auc','detection_rate','fpr')):
                    for drop in sorted({float(r['dropout']) for r in rr}):
                        line=sorted([r for r in rr if float(r['dropout'])==drop and float(r['weight_decay'])==decay],
                                    key=lambda r:float(r['lr']))
                        if line:
                            ax.plot([float(r['lr']) for r in line],[float(r[f'metric_{metric}']) for r in line],
                                    'o-',label=f'dropout {drop:g}')
                    ax.set(xscale='log',xlabel='Learning rate',ylabel=f'Validation {metric}',title=f'Weight decay {decay:g}')
                    ax.grid(alpha=.25)
                axes[j,0].legend(fontsize=8)
            fig.suptitle(f'Model {model}, {pool}: completed optimizer grid only; one seed, no seed error bars')
            fig.tight_layout(); fig.savefig(out/f'model{model}_{pool}_optimizer.png',dpi=180); plt.close(fig)
    print(out.resolve())


if __name__=='__main__': main()
