"""Build descriptive scientific figures from the saved final Excel export; no training."""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import statistics
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Rectangle
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
repo=Path(__file__).resolve().parents[4]
parser.add_argument('--workbook',type=Path,default=repo/'reports/final_paper/Final_Paper_Results_With_Hyperparameters_2026-11-24.xlsx')
parser.add_argument('--layout',type=Path,default=repo/'reports/final_paper/layout.json')
parser.add_argument('--verifier',type=Path,default=Path(__file__).with_name('verify.py'))
parser.add_argument('--out',type=Path,default=repo/'CT_GAT/pipeline/demo_results/hyperparameter_graphs')
args=parser.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
for name in ('png','svg','data'): (args.out/name).mkdir(exist_ok=True)
loader=importlib.util.spec_from_file_location('excel_verify',args.verifier)
module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
sheets,_,_=module.parse(args.workbook)
layout=json.loads(args.layout.read_text())

def read_rows(name,keys,count):
    cells=sheets[name]['cells']
    rows=[{k:cells.get(module.col(i)+str(row)) for i,k in enumerate(keys)} for row in range(2,count+2)]
    assert all(r[keys[0]] is not None for r in rows)
    return rows

validation=read_rows('Validation grid',layout['layouts']['Validation grid']['keys'],3139)
groups=read_rows('Five-fold summary',layout['skeys'],32)
original=[r for r in validation if r['series']!='attention_followup']
attention=[r for r in validation if r['series']=='attention_followup']
assert len(original)==2545 and len(attention)==594
assert all(r['seed']==0 for r in validation)
ARMS=[(1,'mean'),(2,'graph_mean_max'),(3,'mean'),(3,'cls')]
LABEL={(1,'mean'):'M1',(2,'graph_mean_max'):'M2',(3,'mean'):'M3 mean',(3,'cls'):'M3 CLS'}
COLOR={(1,'mean'):'#2878B5',(2,'graph_mean_max'):'#168A76',(3,'mean'):'#D98523',(3,'cls'):'#8057A6'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
    'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
    'axes.titleweight':'bold','figure.facecolor':'white','axes.facecolor':'white',
    'grid.color':'#E2E7ED','grid.linewidth':0.7,'svg.fonttype':'none','savefig.facecolor':'white'})
curves=[];figure_files=[];descriptive=[]

def save(fig,name,title,footer):
    fig.suptitle(title,fontsize=18,fontweight='bold',x=.045,ha='left',y=.975,color='#20364B')
    fig.text(.045,.018,footer,fontsize=9,color='#46596B',ha='left',va='bottom',linespacing=1.5)
    fig.savefig(args.out/'png'/(name+'.png'),dpi=210)
    svg=args.out/'svg'/(name+'.svg')
    fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    figure_files.append(name)
    plt.close(fig)

def arm_rows(rows,arm):return [r for r in rows if (r['model'],r['pooling'])==arm]

def marginal(rows,param,metric,tag,complete=False):
    """Within-fold marginal means, then equal fold weights. No inferential errors."""
    levels=sorted({r[param] for r in rows})
    folds=sorted({r['fold'] for r in rows})
    counts={(f,x):sum(r['fold']==f and r[param]==x for r in rows) for f in folds for x in levels}
    assert all(counts.values()),(tag,param,counts)
    assert all(len({counts[f,x] for x in levels})==1 for f in folds),(tag,param,'unbalanced grid',counts)
    fold_means={f:{x:statistics.mean(r[metric] for r in rows if r['fold']==f and r[param]==x) for x in levels} for f in folds}
    means=[statistics.mean(fold_means[f][x] for f in folds) for x in levels]
    for x,y in zip(levels,means):curves.append(dict(analysis=tag,parameter=param,level=x,metric=metric,mean=y,
        folds=len(folds),candidates=sum(counts[f,x] for f in folds),aggregation='within-fold marginal mean, equal fold weight'))
    descriptive.append(dict(analysis=tag,parameter=param,metric=metric,highest_mean_level=levels[int(np.argmax(means))],
                            highest_mean=max(means),lowest_mean=min(means),range=max(means)-min(means)))
    return levels,means,fold_means

def draw_curve(ax,x,y,arm,labels=None,style='-',marker='o',axis_levels=None):
    ticks=axis_levels if axis_levels is not None else x
    positions=[list(ticks).index(n) for n in x]
    assert all(ticks[p]==n for p,n in zip(positions,x))
    ax.plot(positions,y,label=LABEL[arm],color=COLOR[arm],marker=marker,lw=2,ms=5,ls=style)
    ax.set_xticks(range(len(ticks)),labels if labels is not None else [f'{n:g}' for n in ticks])
    ax.grid(axis='y');ax.set_axisbelow(True)

def legend(fig,axes,ncol=4,y=.905):
    handles={}
    for ax in np.array(axes).flatten():
        for h,label in zip(*ax.get_legend_handles_labels()):handles[label]=h
    order=sorted(handles,key=lambda k:list(LABEL.values()).index(k) if k in LABEL.values() else 10+list(handles).index(k))
    fig.legend([handles[k] for k in order],order,loc='upper center',bbox_to_anchor=(.5,y),ncol=ncol,frameon=False)

# 1. Full held-out evaluation, mean +/- sample SD across three pooled seed scores.
fig,axes=plt.subplots(2,3,figsize=(15,10))
fig.subplots_adjust(left=.07,right=.98,top=.84,bottom=.16,wspace=.28,hspace=.40)
metric_info=[('accuracy','Accuracy',100),('detection_rate','Ransomware detection',100),('macro_f1','Macro-F1',100),
             ('fpr','False-positive rate (lower is better)',100),('auc','ROC-AUC',1),('pr_auc','PR-AUC / average precision',1)]
test_data=[]
for ax,(metric,title,scale) in zip(axes.flat,metric_info):
    for j,(series,color,label) in enumerate([('primary','#365C87','Original primary signal500'),('attention_followup','#D47F25','Attention follow-up signal500')]):
        selected=[next(r for r in groups if r['series']==series and (r['model'],r['pooling'])==arm) for arm in ARMS]
        means=[r[metric+'_mean']*scale for r in selected];sd=[r[metric+'_seed_sd']*scale for r in selected]
        x=np.arange(4)+(j-.5)*.36
        bars=ax.bar(x,means,.34,color=color,label=label,yerr=sd,capsize=3,error_kw={'lw':1})
        for b,value,error in zip(bars,means,sd):ax.annotate(f'{value:.1f}' if scale==100 else f'{value:.3f}',
            (b.get_x()+b.get_width()/2,b.get_height()+error),xytext=(0,5),textcoords='offset points',ha='center',fontsize=8)
        for arm,r in zip(ARMS,selected):test_data.append(dict(series=series,model=LABEL[arm],metric=metric,mean=r[metric+'_mean'],seed_sd=r[metric+'_seed_sd']))
    ax.set_xticks(range(4),[LABEL[a] for a in ARMS]);ax.set_title(title)
    ax.set_ylabel('%' if scale==100 else 'Area under curve');ax.grid(axis='y');ax.set_axisbelow(True)
    if metric=='fpr':ax.set_ylim(0,26)
    elif scale==100:ax.set_ylim(0,106);ax.axhline(90,color='#8894A0',lw=.8,ls='--')
    else:ax.set_ylim(0,1.07)
legend(fig,axes,ncol=2,y=.91)
save(fig,'01_overall_held_out_performance','Overall test performance: original primary vs attention study',
     'Each seed pools all five held-out splits (5,618 samples); bars show mean +/- sample SD across three seeds.\n'
     'Cross-study comparison: the follow-up also fixes Transformer width at 96 and changes the GAT output-head policy.\n'
     'This is not an isolated causal estimate of changing heads/dropout. Original signal750 and ablations are separate experiments.')

# 2. Original optimizer factorial grids: all other parameters averaged within the same stage.
fig,axes=plt.subplots(2,3,figsize=(15,9))
fig.subplots_adjust(left=.07,right=.98,top=.83,bottom=.15,wspace=.25,hspace=.38)
parameters=[('lr','Learning rate'),('dropout','Shared dropout'),('weight_decay','Weight decay')]
for row,metric in enumerate(('accuracy','auc')):
    for c,(param,title) in enumerate(parameters):
        ax=axes[row,c]
        for arm in ARMS:
            rr=arm_rows([r for r in original if r['stage']=='optimizer'],arm)
            levels,means,_=marginal(rr,param,metric,'optimizer / '+LABEL[arm])
            labels=[f'{v:g}' for v in levels]
            draw_curve(ax,levels,np.array(means)*100,arm,labels)
        ax.set_title(title);ax.set_ylabel('Validation '+('accuracy (%) at threshold 0.5' if metric=='accuracy' else 'ROC-AUC (%)'))
legend(fig,axes,y=.905)
save(fig,'02_learning_rate_dropout_weight_decay','Original optimizer grid: learning rate, dropout and weight decay',
     'Validation only: 12-epoch, seed-0 screens. Each point averages the other optimizer settings within each split, then equally weights five splits.\n'
     'Levels are equally spaced categories (including learning rate); curves are descriptive marginal means, not held-out scores or causal estimates.\n'
     'The shared dropout setting in this original grid differs from the independent attention dropout in Figure 4.')

# 3. Original architecture, instruction caps and batch stages.
panels=[('width','Transformer width','architecture',(1,3)),('transformer_layers','Transformer layers','architecture',(1,3)),
        ('gat_hidden','GAT hidden size','architecture',(2,3)),('gat_layers','GAT layers','architecture',(2,3)),
        ('transformer_heads','Transformer heads','heads',(1,3)),('gat_heads','Requested GAT heads','heads',(2,3)),
        ('sequence_length','M1 instruction-sequence limit','architecture',(1,)),('block_length','M3 instructions per block','instruction',(3,)),
        ('batch_size','Logical training batch size','batch',(1,2,3))]
fig,axes=plt.subplots(3,3,figsize=(15,12))
fig.subplots_adjust(left=.07,right=.98,top=.87,bottom=.12,wspace=.27,hspace=.42)
for ax,(param,title,stage,models) in zip(axes.flat,panels):
    axis_levels=sorted({r[param] for r in original if r['stage']==stage and r['model'] in models})
    for arm in ARMS:
        if arm[0] not in models:continue
        rr=arm_rows([r for r in original if r['stage']==stage],arm)
        levels,means,_=marginal(rr,param,'auc',stage+' / '+LABEL[arm])
        draw_curve(ax,levels,np.array(means)*100,arm,axis_levels=axis_levels)
    ax.set_title(title);ax.set_ylabel('Validation ROC-AUC (%)')
legend(fig,axes,y=.93)
save(fig,'03_architecture_instruction_limits_batch','Original staged grid: architecture, input limits and batch size',
     'Within-stage validation marginal means; each split receives equal weight. Optimizer/architecture settings frozen by earlier stages can differ across splits.\n'
     'Original requested GAT heads apply before the final layer; the final layer always has one head. A one-layer GAT therefore has one actual head.\n'
     'Panels come from different stages and must not be combined as evidence of one globally optimal configuration. No statistical significance is claimed.')

# 4. Attention-only within-study effects, with explicit complete-grid restriction for M1.
complete_m1=[]
for f in range(5):
    rr=[r for r in attention if r['model']==1 and r['fold']==f]
    if len(rr)==12:complete_m1.append(f)
assert complete_m1==[0,1,4]
fig,axes=plt.subplots(2,3,figsize=(15,10.8))
fig.subplots_adjust(left=.07,right=.91,top=.84,bottom=.16,wspace=.31,hspace=.55)
for ax,param,title,reference,models in [(axes[0,0],'transformer_heads','Transformer heads: change from 4',4,(1,3)),
    (axes[0,1],'gat_heads','GAT heads: change from 4',4,(2,3)),
    (axes[0,2],'attention_dropout','Attention dropout: change from 0',0,(1,2,3))]:
    for arm in ARMS:
        if arm[0] not in models:continue
        rr=arm_rows(attention,arm)
        if arm[0]==1:rr=[r for r in rr if r['fold'] in complete_m1]
        levels,_,folds=marginal(rr,param,'auc','attention / '+LABEL[arm])
        changes=[statistics.mean(values[x]-values[reference] for values in folds.values())*100 for x in levels]
        draw_curve(ax,levels,changes,arm)
        for x,change in zip(levels,changes):curves.append(dict(analysis='attention matched marginal change / '+LABEL[arm],parameter=param,
            level=x,metric='auc_change_percentage_points',mean=change,folds=len(folds),candidates=len(rr),aggregation=f'within-fold change from {reference}; equal fold weight'))
    ax.set_title(title);ax.set_ylabel('Validation ROC-AUC change (points)');ax.axhline(0,color='#7B8793',lw=.8)
matrices=[];interaction_data=[]
heads=[2,4,8,12]
for drop in (0,.1,.2):
    grid=[]
    for th in heads:
        row=[]
        for gh in heads:
            differences=[]
            for f in range(5):
                def paired(t,g,d):
                    a=[r['auc'] for r in attention if r['model']==3 and r['fold']==f and r['transformer_heads']==t and r['gat_heads']==g and r['attention_dropout']==d]
                    assert len(a)==2
                    return statistics.mean(a)
                differences.append(paired(th,gh,drop)-paired(4,4,0))
            change=statistics.mean(differences)*100
            row.append(change)
            interaction_data.append(dict(transformer_heads=th,gat_heads=gh,attention_dropout=drop,
                validation_auc_change_points=change,reference_transformer_heads=4,reference_gat_heads=4,
                reference_attention_dropout=0,folds=5,pooling='paired mean and CLS'))
        grid.append(row)
    matrices.append(np.array(grid))
limit=max(abs(np.array(matrices).min()),abs(np.array(matrices).max()))
for ax,drop,grid in zip(axes[1],(0,.1,.2),matrices):
    image=ax.imshow(grid,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
    ax.set_xticks(range(4),heads);ax.set_yticks(range(4),heads)
    ax.set_xlabel('GAT heads');ax.set_ylabel('Transformer heads');ax.set_title(f'M3 interaction: attention dropout {drop:g}')
    for y in range(4):
        for x in range(4):ax.text(x,y,f'{grid[y,x]:+.2f}',ha='center',va='center',fontsize=10,color='white' if abs(grid[y,x])>limit*.6 else '#182B3C')
    if drop==0:ax.add_patch(Rectangle((.5,.5),1,1,fill=False,edgecolor='black',linewidth=1.8))
fig.colorbar(image,ax=list(axes[1]),fraction=.027,pad=.035,label='Validation ROC-AUC change (points)')
legend(fig,axes[0],y=.915)
save(fig,'04_attention_heads_dropout_interactions','Attention study: head counts, dropout and their interaction',
     'Top: marginal validation changes, with other attention settings averaged within split. M1 uses the three complete grids (splits 0, 1, 4); M2/M3 use all five.\n'
     'Six M1 screens failed with CUDA OOM in the other two splits. Bottom: M3 averages mean/CLS, then compares with T=4, G=4, dropout=0 within each split.\n'
     'Width 96 for M1/M3; all GAT layers use the chosen head count. Seed-0, 12-epoch screening results only; no held-out inference from individual cells.')

# 5. Complete local block-budget test comparisons; no incomplete M3 curves.
fig,axes=plt.subplots(1,2,figsize=(14,6.5))
fig.subplots_adjust(left=.07,right=.98,top=.80,bottom=.23,wspace=.23)
block_data=[]
for ax,metric,title in zip(axes,('accuracy','auc'),('Overall held-out accuracy','Overall held-out ROC-AUC')):
    for model in (1,2):
        arm=(model,'mean' if model==1 else 'graph_mean_max')
        for sampler,ls in [('signal','-'),('callwalk','--')]:
            rr=[next(r for r in groups if r['series']=='comparisons' and r['representation']==sampler+str(n) and r['model']==model) for n in (250,500,750,1000)]
            means=[r[metric+'_mean']*100 for r in rr];sd=[r[metric+'_seed_sd']*100 for r in rr]
            ax.errorbar([250,500,750,1000],means,yerr=sd,color=COLOR[arm],ls=ls,marker='o' if sampler=='signal' else 's',capsize=3,lw=2,label=LABEL[arm]+' / '+sampler)
            for n,r in zip((250,500,750,1000),rr):block_data.append(dict(model=LABEL[arm],sampler=sampler,blocks=n,metric=metric,mean=r[metric+'_mean'],seed_sd=r[metric+'_seed_sd']))
    ax.set_xticks([250,500,750,1000]);ax.set_xlabel('Maximum retained CFG blocks');ax.set_ylabel('Accuracy (%)' if metric=='accuracy' else 'ROC-AUC (%)');ax.set_title(title);ax.grid(axis='y')
legend(fig,axes,y=.905)
save(fig,'05_block_budget_sampling_test_performance','Block budget and sampling: complete five-fold test comparisons',
     'Controlled comparison series, shared training-only API vocabulary within each split. Points are pooled five-split test metrics, mean +/- sample SD across three seeds.\n'
     'Only M1/M2 have complete local curves at every cap. M3 is omitted because its returned comparisons do not cover all caps/samplers and five splits.\n'
     'The highest standard configuration here is M2 / signal750: 85.80% accuracy. This is separate from the original-primary signal500 series in Figure 1.')

def write_csv(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (args.out/'data'/name).open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
write_csv('marginal_validation_curves.csv',curves)
write_csv('overall_test_metrics.csv',test_data)
write_csv('block_budget_test_metrics.csv',block_data)
write_csv('attention_interaction_heatmap.csv',interaction_data)
(args.out/'data/descriptive_peaks.json').write_text(json.dumps(descriptive,indent=2)+'\n',encoding='utf-8')
summary=dict(source_workbook=args.workbook.name,source_sha256=hashlib.sha256(args.workbook.read_bytes()).hexdigest(),
             original_validation_stage_entries=2545,attention_successful_screens=594,attention_failed_screens=6,
             m1_complete_attention_grids=complete_m1,figures=figure_files,models_trained=0,
             limitations=['Validation curves are descriptive short single-seed screens, not held-out accuracy.',
                          'Marginal averaging can conceal interactions; stages are analyzed separately.',
                          'Test error bars are seed sample SD, not confidence intervals or family-sampling uncertainty.',
                          'Original versus attention differs in Transformer width and GAT head policy.',
                          'No statistical significance, calibrated infection probability or ensemble performance is claimed.'])
(args.out/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
shutil.copyfile(__file__,args.out/'data/build_figures.py')
print(json.dumps({'output':str(args.out),'figures':figure_files,'curve_rows':len(curves),'peaks':[x for x in descriptive if x['parameter'] in ('lr','dropout','weight_decay') and x['metric']=='auc']},indent=2))
