import argparse,csv,hashlib,json,statistics
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,f1_score,roc_auc_score

parser=argparse.ArgumentParser(description='Verify final attention records and reconstruct pooled test metrics without loading checkpoints.')
parser.add_argument('--study-root',type=Path,required=True)
parser.add_argument('--campaign-root',type=Path,required=True)
parser.add_argument('--index',type=Path,help='Override the index path recorded by the campaign.')
parser.add_argument('--signal500-build',type=Path,help='Override the signal500 build path recorded by the campaign.')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
ROOT=args.study_root.resolve()
CAMPAIGN=args.campaign_root.resolve()
data=json.loads((ROOT/'reports/records.json').read_text())

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def description(r):
    s=r['spec'];m=r['model'];lines=[]
    if m in (1,3):lines.append(f"Transformer: width {s['width']}, {s['transformer_layers']} layers, {s['transformer_heads']} heads")
    if m in (2,3):lines.append(f"GAT: hidden {s['gat_hidden']}, {s['gat_layers']} layers, actual heads {[s['gat_heads']]*s['gat_layers']}")
    if m==1:lines.append(f"First {s['sequence_length']} instructions; batch {s['batch_size']} (micro {s['micro_batch_size']})")
    elif m==3:lines.append(f"First {s['block_length']} instructions/block; batch {s['batch_size']} (micro {s['micro_batch_size']})")
    else:lines.append(f"77 handcrafted node features; batch {s['batch_size']} (micro {s['micro_batch_size']})")
    lines.append(f"Attention dropout {s['attention_dropout']:g}; feature dropout {s['dropout']:g}")
    lines.append(f"LR {s['lr']:g}; weight decay {s['weight_decay']:g}")
    return '\n'.join(lines)


def recipe(model,fold,pool):return f'A-M{model}-F{fold}-'+pool.replace('graph_mean_max','GAT').upper()

frozen=[];selection={};failures=[]
for folder in sorted(ROOT.glob('fold*/model*')):
    plan=json.loads((folder/'plan.json').read_text());status=json.loads((folder/'status.json').read_text())
    assert status['state']=='complete'
    selected=json.loads((folder/'selection.json').read_text())['specs']
    for spec in selected:
        key=(plan['model'],plan['fold'],spec['pooling']);selection[key]=spec
        obj=dict(model=plan['model'],fold=plan['fold'],pooling=spec['pooling'],spec=spec)
        frozen.append(dict(**spec,model=plan['model'],fold=plan['fold'],recipe_id=recipe(*key),run_configuration=description(obj),
            selection_rule='Attention validation ROC-AUC then macro-F1; Model 3 paired mean/CLS',
            source_record='attention_followup_20261112/'+str((folder/'selection.json').relative_to(ROOT)).replace('\\','/')))
    for path in folder.glob('*/*/failure.json'):
        d=json.loads(path.read_text());failures.append(dict(source_record='attention_followup_20261112/'+path.relative_to(ROOT).as_posix(),
           kind=d['state'],paper_included='Failure only (no score)',model=plan['model'],seed=d['seed'],pooling=d['spec']['pooling'],
           error='CUDA out of memory' if 'out of memory' in d['message'].lower() else d['state'],message=d['message'],
           spec_json=json.dumps(d['spec'],sort_keys=True,separators=(',',':'))))

tables={}
for section in ('validation','held_out'):
    rows=[]
    for values in data[section]['rows']:
        r=dict(zip(data[section]['columns'],values))
        if r['series']!='attention_followup':continue
        path=ROOT/r['source_record'];result=json.loads(path.read_text());config=json.loads((path.parent/'config.json').read_text());spec=config['spec']
        assert digest(config)==result['fingerprint']
        assert r['gpu_name']=='NVIDIA GeForce RTX 5080'
        r.update(spec=spec,paper_included='Yes',result_scope='Held-out test' if section=='held_out' else 'Validation only',
                 execution_profile='attention study controlled',device=result['device'],invocation_seconds=result.get('invocation_seconds'),
                 spec_json=json.dumps(spec,sort_keys=True,separators=(',',':')),source_hashes_json=json.dumps(config['source_hashes'],sort_keys=True,separators=(',',':')),
                 validation_accuracy=result['validation']['accuracy'],validation_detection_rate=result['validation']['detection_rate'],
                 validation_fpr=result['validation']['fpr'],validation_threshold=result['validation']['threshold'],
                 hardware_scope='RTX 5080, attention study',run_configuration=description(dict(model=r['model'],spec=spec)))
        key=(r['model'],r['fold'],r['pooling']);r['recipe_id']=recipe(*key)
        r['final_recipe']=r['recipe_id'];r['selected_final']='Yes' if spec==selection[key] else 'No'
        r['candidate_id']=f"A{r['model']}-F{r['fold']}-"+digest(dict(model=r['model'],fold=r['fold'],spec=spec))[:8].upper()
        if section=='held_out':
            assert spec==selection[key]
            provenance=json.loads((path.parents[2]/'dataset.json').read_text())
            r.update({k+'_samples':len(v) for k,v in provenance['splits'].items()})
        r['source_record']='attention_followup_20261112/'+r['source_record']
        rows.append(r)
    tables[section]=sorted(rows,key=lambda r:(r['model'],r['fold'],r['pooling'],r['seed'],r['source_record']))
assert len(tables['held_out'])==60 and len(tables['validation'])==594 and len(frozen)==20 and len(failures)==6

def score(predictions):
    y=np.array([r[0] for r in predictions]);p=np.array([r[1] for r in predictions]);q=np.array([r[2] for r in predictions],dtype=bool)
    tp=int(((y==1)&q).sum());tn=int(((y==0)&~q).sum());fp=int(((y==0)&q).sum());fn=int(((y==1)&~q).sum())
    return dict(n=len(y),accuracy=float((y==q).mean()),auc=float(roc_auc_score(y,p)),pr_auc=float(average_precision_score(y,p)),
                macro_f1=float(f1_score(y,q,average='macro')),detection_rate=tp/(tp+fn),fpr=fp/(fp+tn),
                precision=tp/(tp+fp),specificity=tn/(tn+fp),tp=tp,tn=tn,fp=fp,fn=fn)

config=json.loads((CAMPAIGN/'CAMPAIGN.json').read_text());excluded=json.loads(((args.signal500_build or Path(config['builds']['signal500']))/'settings.json').read_text())['excluded']
with (args.index or Path(config['index'])).open(newline='') as f:index={r['sample_id']:r for r in csv.DictReader(f)}
expected={sid:int(r['label']) for sid,r in index.items() if r.get('fold_v3') not in ('',None) and sid not in excluded}
assert len(expected)==5618
oof=[];per_fold=[];summaries=[]
metric_keys=('auc','pr_auc','accuracy','detection_rate','fpr','macro_f1')
for model,pool in ((1,'mean'),(2,'graph_mean_max'),(3,'cls'),(3,'mean')):
    entries=[r for r in tables['held_out'] if r['model']==model and r['pooling']==pool]
    for fold in range(5):
        rr=[r for r in entries if r['fold']==fold];assert len(rr)==3
        row=dict(series='attention_followup',representation='signal500_common_api',ablation='none',model=model,pooling=pool,fold=fold,completed_seeds=3,expected_seeds=3,
                 hardware_scope='RTX 5080, attention study',selected_recipe=recipe(model,fold,pool))
        for k in metric_keys:row[k+'_mean']=statistics.mean(r[k] for r in rr);row[k+'_seed_sd']=statistics.stdev(r[k] for r in rr)
        per_fold.append(row)
    pooled=[]
    for seed in (0,1,2):
        predictions={}
        for r in entries:
            if r['seed']!=seed:continue
            path=ROOT/r['source_record'].removeprefix('attention_followup_20261112/')
            provenance=json.loads((path.parents[2]/'dataset.json').read_text());expected_fold=set(provenance['splits']['test'])
            with (path.parent/'test_predictions.csv').open(newline='') as f:rr=list(csv.DictReader(f))
            assert len(rr)==len({v['sample_id'] for v in rr}) and {v['sample_id'] for v in rr}==expected_fold
            for v in rr:
                sid=v['sample_id'];label=int(v['label']);prob=float(v['prob'])
                assert sid not in predictions and label==expected[sid] and v['split']=='test' and np.isfinite(prob) and 0<=prob<=1
                predictions[sid]=(label,prob,prob>=r['threshold'])
        assert set(predictions)==set(expected)
        record=dict(series='attention_followup',representation='signal500_common_api',ablation='none',model=model,pooling=pool,seed=seed,
                    **score([predictions[k] for k in sorted(predictions)]),threshold_policy='Per-fold validation-fitted thresholds',hardware_scope='RTX 5080, attention study')
        oof.append(record);pooled.append(record)
    summaries.append(dict(model=model,pooling=pool,metrics={k:dict(mean=statistics.mean(r[k] for r in pooled),seed_sd=statistics.stdev(r[k] for r in pooled)) for k in (*metric_keys,'precision','specificity')}))

payload=dict(**tables,frozen=frozen,failures=failures,oof=oof,per_fold=per_fold,summaries=summaries,
             completed_utc=json.loads((ROOT/'status.json').read_text())['completed_utc'],
             source='attention_followup_20261112/reports/records.json',source_sha256=hashlib.sha256((ROOT/'reports/records.json').read_bytes()).hexdigest())
args.output.write_text(json.dumps(payload,allow_nan=False),encoding='utf-8')
print(json.dumps({key:len(payload[key]) for key in ('held_out','validation','frozen','failures','oof')}))
