"""Thirty-hour coarse search. Frozen-Transformer Model 3 is deliberately absent."""
import argparse
from contextlib import nullcontext
from torch.nn.attention import SDPBackend, sdpa_kernel
import csv
import gc
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import statistics
import time

import torch

from experiments.model_grid import runner
from experiments.model_grid.data import DATASETS, Prepared, digest
from experiments.model_grid.locking import exclusive
from .budget import Budget, BudgetExhausted
from .coverage import DiskPrepared
from . import grids


def compatible(saved, wanted):
    """Only the CLI dispatcher may differ. Model, packing, training and metric code must match exactly."""
    a, b = dict(saved), dict(wanted)
    for c in (a,b):
        c['source_hashes'] = {k:v for k,v in c['source_hashes'].items() if k != 'grid_search.py'}
    return a == b


def reuse_candidates(roots):
    paths = []
    for root in roots:
        if Path(root).exists():
            paths += sorted(Path(root).rglob('metrics.json'))
    return paths


def reuse_trial(paths, contract, out):
    for path in paths:
        config_path = path.parent/'config.json'
        if not config_path.is_file():
            continue
        saved = json.loads(config_path.read_text())
        if not compatible(saved, contract):
            continue
        result = json.loads(path.read_text())
        if result.get('fingerprint') != digest(saved):
            continue
        out.mkdir(parents=True, exist_ok=True)
        for name in ('config.json','metrics.json','history.json','validation_predictions.csv','test_predictions.csv','best.pt','last.pt'):
            if (path.parent/name).exists() and (path.parent/name).resolve() != (out/name).resolve():
                shutil.copy2(path.parent/name, out/name)
        runner.atomic_json(out/'reused_from.json', dict(path=str(path.parent.resolve()),
                          requested_contract=contract, original_fingerprint=result['fingerprint'],
                          rule='Exact input, model, seed, epoch, training and metric contract; CLI dispatcher may differ.'))
        return result
    return None


def publish_report(temp, target):
    """A viewer may deny rename on Windows; retain the completed snapshot without aborting training."""
    try:
        temp.replace(target)
    except PermissionError:
        print(f'Report is locked: {target}; latest snapshot retained at {temp}; training continues.', flush=True)


def report(root, records):
    """Rewrite a tidy, atomic analysis table after every completed configuration."""
    rows = []
    for r in records:
        result = r['result']
        metric = result['test'] if r['stage'] == 'confirm' else result['validation']
        rows.append(dict(model=r['model'], stage=r['stage'], blocks=r['blocks'], seed=result['seed'],
                         epochs_run=result['epochs_run'], best_epoch=result['best_epoch'],
                         **result['spec'], **{f'metric_{k}':v for k,v in metric.items()},
                         validation_auc=result['validation_fixed']['auc'],
                         seconds_per_epoch=result['epoch_seconds_mean'], path=r['path'], reused=r['reused']))
    if rows:
        temp = root/'results.csv.tmp'
        with temp.open('w',newline='',encoding='utf-8') as f:
            w = csv.DictWriter(f,fieldnames=list(dict.fromkeys(key for row in rows for key in row))); w.writeheader(); w.writerows(rows)
        publish_report(temp,root/'results.csv')
    lines = ['# Budgeted model search', '',
             'Models 1 and 3 train their position-aware instruction Transformers end to end. Model 2 uses handcrafted features.',
             'Screening is validation-only. Coverage pilots have a separate equal epoch budget and do not select the primary model.',
             'Confirmation is one grouped outer fold, mean ± sample SD across seeds; it is not five-fold CV.',
             'The preset is staged Cartesian search, not the Cartesian product of every stage. Budget-limited stages are partial grids.', '',
             '| Model | Stage | Block cap | Completed | Best validation AUC |', '|---|---|---|---|---|']
    for model,stage,blocks in sorted({(r['model'],r['stage'],r['blocks']) for r in records}):
        rr = [r for r in records if (r['model'],r['stage'],r['blocks']) == (model,stage,blocks)]
        lines.append(f"| {model} | {stage} | {blocks} | {len(rr)} | {max(r['result']['validation_fixed']['auc'] for r in rr):.4f} |")
    confirmed = [r for r in records if r['stage']=='confirm']
    if confirmed:
        lines += ['', '| Model | Pool | Seeds | Test AUC | Test detection | Test FPR |', '|---|---|---|---|---|---|']
        for model,pool in sorted({(r['model'],r['result']['spec']['pooling']) for r in confirmed}):
            rr = [r['result'] for r in confirmed if r['model']==model and r['result']['spec']['pooling']==pool]
            cells = []
            for metric in ('auc','detection_rate','fpr'):
                values = [r['test'][metric] for r in rr]
                cells.append(f'{statistics.mean(values):.4f}' + (f' ± {statistics.stdev(values):.4f}' if len(values)>1 else ''))
            lines.append(f"| {model} | {pool} | {','.join(str(r['seed']) for r in rr)} | " + ' | '.join(cells) + ' |')
    temp = root/'REPORT.md.tmp'; temp.write_text('\n'.join(lines)+'\n',encoding='utf-8'); publish_report(temp,root/'REPORT.md')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',type=int,choices=[1,2,3],default=[1,3])
    p.add_argument('--build',type=Path,required=True,help='500-block signal .pt build')
    p.add_argument('--build-250',type=Path)
    p.add_argument('--build-uncapped',type=Path)
    p.add_argument('--index',type=Path,required=True)
    p.add_argument('--dataset',choices=DATASETS,default='combined')
    p.add_argument('--out',type=Path,default=Path('runs/grid_wide'))
    p.add_argument('--reuse-from',nargs='*',type=Path,default=[])
    p.add_argument('--gpu-hours',type=float,default=30)
    p.add_argument('--learning-rates',nargs='+',type=float,default=[1e-4,3e-4,1e-3,3e-5,3e-3])
    p.add_argument('--dropouts',nargs='+',type=float,default=[.1,.3,0.,.5])
    p.add_argument('--weight-decays',nargs='+',type=float,default=[.0001,.01,0.])
    p.add_argument('--fold',type=int,choices=range(5),default=0)
    p.add_argument('--val-seed',type=int,default=0)
    p.add_argument('--screen-seed',type=int,default=0)
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--coverage-epochs',type=int,default=2)
    p.add_argument('--confirm-epochs',type=int,default=40)
    p.add_argument('--patience',type=int,default=8)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--smoke',type=int,default=0)
    p.add_argument('--wait-build-hours',type=float,default=12,help='Maximum CPU-only wait for a requested coverage build')
    p.add_argument('--wait-for',type=Path,help='Old queue root; wait for requested model statuses to finish before using the GPU')
    return p


class Study:
    def __init__(self,args):
        self.a=args; self.root=args.out; self.root.mkdir(parents=True,exist_ok=True)
        self.sources=runner.source_hashes()
        self.budget=Budget(self.root/'budget.json',args.gpu_hours)
        self.reuse=reuse_candidates(args.reuse_from)
        self.records=[]; self.data=None; self.data_key=None; self.reference_splits=None
        self.device=torch.device(args.device)
        self.choices={}; self.optimizer_choices={}; self.build_deadlines={}

    def prepare(self,model,spec,build,blocks):
        key=(model,str(build.resolve()),spec['sequence_length'],spec['block_length'])
        if key != self.data_key:
            self.data=None; gc.collect()
            tick=time.perf_counter()
            if blocks == 'uncapped':
                self.data=DiskPrepared(build,self.a.index,self.a.dataset,self.a.fold,self.a.val_seed,
                                       model,spec,self.root/'packed'/f'model{model}',self.a.smoke)
            else:
                self.data=Prepared(build,self.a.index,self.a.dataset,self.a.fold,self.a.val_seed,model,spec,self.a.smoke)
            self.data_key=key
            if self.reference_splits is None:
                self.reference_splits=self.data.provenance['splits']
            if self.data.provenance['splits'] != self.reference_splits:
                raise ValueError('coverage builds have different cohorts: refusing unmatched comparison')
            folder=self.root/f'model{model}'/f'input_{blocks}_seq{spec["sequence_length"]}_block{spec["block_length"]}'
            runner.atomic_json(folder/'dataset.json',dict(self.data.provenance,build_path=str(build.resolve()),
                                preparation_seconds=time.perf_counter()-tick))
            runner.atomic_json(folder/'coverage.json',self.data.coverage)
        return self.data

    def trial(self,model,spec,phase,stage,build,blocks,epochs,seed,test=False):
        folder=self.root/f'model{model}'/f'{stage}_{blocks}'/grids.trial_id(spec,epochs,seed,test)
        if (folder/'failure.json').exists():
            return None
        data=self.prepare(model,spec,build,blocks)
        contract=dict(model=model,spec=spec,seed=seed,epochs=epochs,patience=self.a.patience if test else epochs+1,
                      dataset_fingerprint=data.fingerprint,source_hashes=self.sources,evaluate_test=test,
                      torch_version=torch.__version__)
        reused=False
        if (folder/'metrics.json').exists():
            saved=json.loads((folder/'config.json').read_text())
            if not compatible(saved,contract):
                raise ValueError(f'completed trial contract changed: {folder}')
            result=json.loads((folder/'metrics.json').read_text()); reused=(folder/'reused_from.json').exists()
            if result.get('fingerprint') != digest(saved):
                raise ValueError(f'metrics/config fingerprint mismatch: {folder}')
        else:
            result=reuse_trial(self.reuse,contract,folder)
            reused=result is not None
        if result is None:
            if self.budget.remaining(phase)<=0:
                return None
            runner.atomic_json(self.root/'status.json',dict(state='training',model=model,stage=stage,blocks=blocks,
                               case=spec,seed=seed,completed=len(self.records),trial=str(folder)))
            try:
                # CPU activation offload preserves the model/gradients, while making full-graph pilots more feasible.
                # Apply to ALL coverage caps so their memory policy matches.
                offload=torch.autograd.graph.save_on_cpu(pin_memory=True) if stage=='coverage' and self.device.type=='cuda' else nullcontext()
                # CPU offload can compact a padded attention bias, violating fused-backend strides.
                # Record the backend in the spec so these are new trials, never resumed fused ones.
                attention=sdpa_kernel(SDPBackend.MATH) if spec.get('attention_backend')=='math' else nullcontext()
                with self.budget.charge(phase,folder.parent.parent/'STOP',folder), offload, attention:
                    result=runner.run_trial(model,spec,seed,epochs,contract['patience'],data,folder,
                                            self.device,self.sources,evaluate_test=test)
            except BudgetExhausted:
                return None
            except (torch.OutOfMemoryError,FloatingPointError,RuntimeError) as exc:
                # PyTorch gradient clipping reports numerical failure as RuntimeError.
                # Preserve its fail-fast behavior without aborting unrelated grid cases.
                gradient_failure=(str(exc).startswith('The total norm of order ')
                                  and 'for gradients from `parameters` is non-finite' in str(exc))
                if not isinstance(exc,(torch.OutOfMemoryError,FloatingPointError)) and not gradient_failure:
                    raise
                runner.atomic_json(folder/'failure.json',dict(error=type(exc).__name__,message=str(exc),
                                   spec=spec,model=model,stage=stage,blocks=blocks,not_a_valid_result=True))
                gc.collect()
                if self.device.type=='cuda': torch.cuda.empty_cache()
                return None
        self.reuse.append(folder/'metrics.json')
        self.records.append(dict(model=model,stage=stage,blocks=blocks,path=str(folder.resolve()),result=result,reused=reused))
        report(self.root,self.records)
        return result

    def screen(self,model,phase,stage,specs):
        if self.a.smoke:
            specs=specs[:2] if model==3 else specs[:1]
        phase_dir=self.root/f'model{model}'/stage
        runner.atomic_json(phase_dir/'planned_cases.json',specs)
        results=[]
        for spec in specs:
            result=self.trial(model,spec,phase,stage,self.a.build,'500',self.a.epochs,self.a.screen_seed)
            if result is not None: results.append(result)
        chosen=grids.choose(model,results)
        status=dict(planned=len(specs),completed=len(results),partial=len(results)<len(specs),specs=chosen,
                    selection='validation AUC then macro-F1 at 0.5; Model 3 averages only complete mean/CLS pairs',
                    test_used_for_selection=False,budget_remaining_seconds=self.budget.remaining(phase))
        runner.atomic_json(phase_dir/'selection.json',status)
        return chosen

    def coverage_case(self, model, blocks, build, wait=False):
        """Run ready coverage work first; only the final pass may wait for inputs."""
        statuspath=self.root/f'model{model}'/f'coverage_{blocks}'/'status.json'
        if wait and build is not None:
            deadline=self.build_deadlines.setdefault(str(build),time.monotonic()+self.a.wait_build_hours*3600)
            while (not (build/'build_summary.json').exists()
                   and self.budget.remaining('coverage')>0 and time.monotonic()<deadline):
                if (self.root/f'model{model}'/'STOP').exists():
                    raise InterruptedError('STOP file while waiting for coverage build')
                runner.atomic_json(self.root/'status.json',dict(state='waiting_for_coverage_build',model=model,
                                   blocks=blocks,path=str(build),gpu_budget_charged=False))
                time.sleep(20)
        if build is None or not (build/'build_summary.json').exists():
            runner.atomic_json(statuspath,dict(state='pending_build',path=str(build)))
            return False
        settings=json.loads((build/'settings.json').read_text())
        expected=0 if blocks=='uncapped' else int(blocks)
        if settings.get('num_blocks') != expected or settings.get('sample') != 'signal':
            raise ValueError(f'{blocks} build has the wrong cap/selector: {build}')
        rr=[]
        # Hold optimizer choice, reference architecture, batch size and epochs fixed across caps.
        for base in self.optimizer_choices[model]:
            spec=dict(base,batch_size=1)
            if model in (1,3):
                spec['attention_backend']='math'
            result=self.trial(model,spec,'coverage','coverage',build,blocks,self.a.coverage_epochs,self.a.screen_seed)
            if result is not None: rr.append(result)
        runner.atomic_json(statuspath,dict(state='complete' if len(rr)==len(self.optimizer_choices[model]) else 'partial',
                           completed=len(rr),epochs=self.a.coverage_epochs,validation_only=True))
        return True

    def run(self):
        for model in self.a.models:
            chosen=self.screen(model,'optimizer','optimizer',grids.optimizer(model,self.a.learning_rates,self.a.dropouts,self.a.weight_decays))
            if chosen: self.choices[model]=chosen; self.optimizer_choices[model]=chosen
        for model in self.a.models:
            if model not in self.choices: continue
            chosen=self.screen(model,'architecture','architecture',grids.architecture(model,self.choices[model][0]))
            if chosen: self.choices[model]=chosen
        # Instruction-limit selection uses validation and must finish before any outer test read.
        for model in self.a.models:
            if model==3 and model in self.choices:
                chosen=self.screen(model,'coverage','instruction',grids.instruction(model,self.choices[model][0]))
                if chosen: self.choices[model]=chosen
        selection=dict(specs=self.choices,block_cap=500,
                       rule='Staged validation selection; frozen before any new outer test evaluation',
                       coverage_pilots_used_for_selection=False)
        selection=json.loads(json.dumps(selection))
        selection_path=self.root/'primary_selection.json'
        if selection_path.exists() and json.loads(selection_path.read_text())!=selection:
            raise ValueError('primary selection changed after freezing; refusing to reuse held-out evaluations')
        runner.atomic_json(selection_path,selection)
        # Primary results need only the ready 500-block build. Do not idle behind uncapped preprocessing.
        if not self.a.smoke:
            for model in self.a.models:
                for seed,spec in itertools.product(self.a.seeds,self.choices.get(model,[])):
                    self.trial(model,spec,'confirm','confirm',self.a.build,'500',self.a.confirm_epochs,seed,test=True)
        # Avoid carrying a confirmation dataset with loaded test samples into validation-only pilots.
        self.data=None; self.data_key=None; gc.collect()
        deferred=[]
        for model in self.a.models:
            if model not in self.choices: continue
            for blocks,build in [('250',self.a.build_250),('500',self.a.build),('uncapped',self.a.build_uncapped)]:
                if not self.coverage_case(model,blocks,build) and build is not None:
                    deferred.append((model,blocks,build))
        for model,blocks,build in deferred:
            self.coverage_case(model,blocks,build,wait=True)
        pending=list(self.root.glob('model*/coverage_*/status.json'))
        pending=[str(p) for p in pending if json.loads(p.read_text())['state']!='complete']
        runner.atomic_json(self.root/'status.json',dict(state='queue_finished',completed=len(self.records),
                           pending_coverage=pending,used_hours=sum(self.budget.doc['used_seconds'].values())/3600,
                           note='Inspect stage selection.json files for partial grids; queue_finished does not imply every planned case completed.'))


def main(argv=None):
    args=parser().parse_args(argv)
    for name in ('models','learning_rates','dropouts','weight_decays','seeds'):
        vals=getattr(args,name)
        if len(vals)!=len(set(vals)): raise ValueError(f'duplicate {name}')
    if args.gpu_hours<=0 or args.wait_build_hours<0 or any(v<=0 for v in args.learning_rates) or any(not 0<=v<1 for v in args.dropouts):
        raise ValueError('invalid budget, learning rate or dropout')
    if any(v<0 for v in args.weight_decays+args.seeds+[args.screen_seed,args.val_seed]):
        raise ValueError('negative decay or seed')
    if min(args.epochs,args.coverage_epochs,args.confirm_epochs,args.patience,args.threads)<1:
        raise ValueError('epochs, patience and threads must be positive')
    if args.smoke: args.epochs=args.coverage_epochs=2
    summary={m:dict(optimizer=len(grids.optimizer(m,args.learning_rates,args.dropouts,args.weight_decays)),
                   architecture=len(grids.architecture(m,grids.base_spec(m))),
                   instruction=len(grids.instruction(m,grids.base_spec(m))),coverage=6 if m==3 else 3,
                   confirmation=len(args.seeds)*(2 if m==3 else 1)) for m in args.models}
    if args.plan_only:
        print(json.dumps(dict(models=summary,hours=args.gpu_hours,phase_hours={k:args.gpu_hours*v/30 for k,v in
              dict(optimizer=8,architecture=8,coverage=6,confirm=8).items()},frozen_model3=False),indent=2)); return
    args.out.mkdir(parents=True,exist_ok=True)
    with exclusive(args.out/'study.lock'):
        plan=dict(arguments={k:str(v) if isinstance(v,Path) else [str(x) if isinstance(x,Path) else x for x in v]
                             if isinstance(v,list) else v for k,v in vars(args).items() if k!='wait_for'},
                  cases=summary,sources=runner.source_hashes(),study_sources={p.name:hashlib.sha256(p.read_text(encoding='utf-8').encode()).hexdigest()
                      for p in Path(__file__).parent.glob('*.py')})
        plan=json.loads(json.dumps(plan))  # integer model keys round-trip through JSON as strings
        if (args.out/'plan.json').exists() and json.loads((args.out/'plan.json').read_text())!=plan:
            raise ValueError('study plan/source changed; choose a new output root')
        runner.atomic_json(args.out/'plan.json',plan)
        if args.wait_for:
            while True:
                states=[]
                for m in args.models:
                    p=args.wait_for/f'model{m}'/f'fold{args.fold}'/'status.json'
                    states.append(json.loads(p.read_text()).get('state') if p.exists() else 'missing')
                if all(s=='complete' for s in states): break
                if any(s in ('failed','stopped') for s in states):
                    raise RuntimeError(f'previous queue needs attention: {states}')
                runner.atomic_json(args.out/'status.json',dict(state='waiting_for_previous_queue',states=states,path=str(args.wait_for)))
                time.sleep(20)
        if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable')
        torch.set_num_threads(args.threads)
        try:
            Study(args).run()
        except BaseException as exc:
            runner.atomic_json(args.out/'status.json',dict(state='stopped' if isinstance(exc,InterruptedError) else 'failed',error=str(exc)))
            raise


if __name__=='__main__': main()
