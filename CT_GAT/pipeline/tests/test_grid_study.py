import copy
import json
import time

import pytest
import torch

from experiments.grid_study.budget import Budget, BudgetExhausted
from experiments.grid_study import grids
from experiments.grid_study.coverage import derive_payload, DiskPrepared
from experiments.grid_study.study import compatible, reuse_trial
from experiments.model_grid.data import digest
from experiments.model_grid import data as dm, runner


def test_grid_counts_ids_and_pair_selection():
    for m,n in ((1,60),(2,60),(3,120)):
        ss=grids.optimizer(m,[1e-4,3e-4,1e-3,3e-5,3e-3],[.1,.3,0,.5],[.0001,.01,0])
        assert len(ss)==n
        assert len({grids.trial_id(s,12,0,False) for s in ss})==n
    assert [len(grids.architecture(m,grids.base_spec(m))) for m in (1,2,3)]==[27,36,162]
    base=grids.base_spec(3)
    def r(width,pool,auc):
        return dict(spec=dict(base,width=width,pooling=pool),validation_fixed=dict(auc=auc,macro_f1=auc))
    # A strong unpaired run cannot win; different widths cannot be accidentally paired.
    chosen=grids.choose(3,[r(64,'mean',.7),r(64,'cls',.9),r(128,'mean',.99)])
    assert {s['width'] for s in chosen}=={64}


def test_budget_checkpoint_stop_and_resume(tmp_path):
    b=Budget(tmp_path/'budget.json',.000015,shares={'screen':1.})
    stop=tmp_path/'STOP'
    with pytest.raises(BudgetExhausted):
        with b.charge('screen',stop,'trial'):
            deadline=time.monotonic()+2
            while not stop.exists() and time.monotonic()<deadline: time.sleep(.005)
            assert stop.exists()
            raise InterruptedError('checkpoint boundary')
    assert not stop.exists()
    again=Budget(tmp_path/'budget.json',.000015,shares={'screen':1.})
    assert again.remaining('screen')<=0
    with pytest.raises(BudgetExhausted):
        with again.charge('screen',stop,'trial'): pass


def test_reuse_requires_identical_science_contract(tmp_path):
    a=dict(model=3,spec={'lr':.001},seed=0,epochs=12,patience=13,dataset_fingerprint='abc',
           source_hashes={'grid_search.py':'old','models.py':'same','runner.py':'same'},evaluate_test=False,torch_version='v')
    b=copy.deepcopy(a); b['source_hashes']['grid_search.py']='new'
    assert compatible(a,b)
    for key,value in [('epochs',40),('dataset_fingerprint','other'),('evaluate_test',True)]:
        c=copy.deepcopy(b); c[key]=value
        assert not compatible(a,c)
    c=copy.deepcopy(b); c['source_hashes']['models.py']='different'
    assert not compatible(a,c)
    old=tmp_path/'old'; old.mkdir()
    runner.atomic_json(old/'config.json',a)
    runner.atomic_json(old/'metrics.json',dict(fingerprint=digest(a),seed=0))
    out=tmp_path/'new'
    assert reuse_trial([old/'metrics.json'],b,out)['seed']==0
    assert json.loads((out/'config.json').read_text())==a
    assert (out/'reused_from.json').exists()


def test_signal_prefix_keeps_canonical_instruction_order_and_induced_edges():
    keys=[dict(function_id=1,block_id=i) for i in range(3)]
    p=dict(block_keys=keys,instruction_ids=torch.arange(20).reshape(4,5),insn_to_block=torch.tensor([0,1,1,2]),
           instruction_matched=torch.tensor([True,True,False,True]),instruction_rvas=torch.tensor([10,20,21,30]),
           edge_index=torch.tensor([[0,1,2],[1,2,1]]))
    for k in ('block_features','block_embeddings','block_types','block_external'): p[k]=torch.arange(3)
    for k in ('model1_block_keys','model2_block_keys','model3_block_keys'): p[k]=keys
    for k in ('edge_type','edge_attr','edge_back','edge_back_addr'): p[k]=torch.arange(3)
    q=derive_payload(p,dict(selected_block_ids=[keys[2],keys[1],keys[0]]),2)
    assert q['block_keys']==keys[1:]
    assert q['instruction_rvas'].tolist()==[20,21,30]
    assert q['insn_to_block'].tolist()==[0,0,1]
    assert q['edge_index'].tolist()==[[0,1],[1,0]]
    assert q['instruction_matched'].tolist()==[True,False,True]
    p['instruction_rvas']=p['instruction_rvas'].tolist()
    assert derive_payload(p,dict(selected_block_ids=[keys[2],keys[1],keys[0]]),2)['instruction_rvas']==[20,21,30]


@pytest.mark.parametrize('model',[1,2,3])
def test_disk_preparation_matches_resident_inputs_and_does_not_read_test(tmp_path,monkeypatch,model):
    splits={}
    for i,s in enumerate(('train','val','test')):
        splits[s]=[]
        for y in (0,1):
            sid=f'{i*2+y:064x}'
            splits[s].append(dict(sample_id=sid,label=y,family=f'{s}{y}',identity_group=sid,arch='x86'))
            ids=torch.ones(4,5,dtype=torch.long); ids[:,3]=dm.tok.FIRST_API_ID+(50 if s!='train' else y)
            torch.save(dict(instruction_ids=ids,insn_to_block=torch.tensor([0,0,1,1]),
                       instruction_matched=torch.ones(4,dtype=torch.bool),block_features=torch.zeros(2,77),
                       edge_index=torch.tensor([[0],[1]]),edge_attr=torch.eye(7)[:1],
                       edge_back=torch.tensor([False]),label=y),tmp_path/f'{sid}.pt')
    (tmp_path/'index.csv').write_text('synthetic')
    (tmp_path/'settings.json').write_text('{}')
    from experiments.grid_study import coverage
    monkeypatch.setattr(dm,'manifest_for',lambda *a:copy.deepcopy(splits))
    monkeypatch.setattr(coverage,'manifest_for',lambda *a:copy.deepcopy(splits))
    spec=grids.base_spec(model)
    args=(tmp_path,tmp_path/'index.csv','mendeley-x86',0,0,model,spec)
    a=dm.Prepared(*args); b=DiskPrepared(*args,cache=tmp_path/'packed')
    assert a.fingerprint==b.fingerprint
    assert a.coverage==b.coverage
    assert not {r['sample_id'] for r in splits['test']} & set(b.hashes)
    for sid in a.samples:
        aa,bb=a.samples[sid],b.samples[sid]
        field='ids' if model==1 else 'x' if model==2 else 'insn_ids'
        assert torch.equal(aa[field],bb[field])
    b.load_test()
    assert len(b.samples)==6
    if model==1:
        from experiments.grid_study import study
        out=tmp_path/'study'
        argv=['--models','1','--build',str(tmp_path),'--index',str(tmp_path/'index.csv'),
              '--out',str(out),'--device','cpu','--smoke','4','--wait-build-hours','0']
        study.main(argv)
        budget=json.loads((out/'budget.json').read_text())['used_seconds']
        study.main(argv)
        assert json.loads((out/'budget.json').read_text())['used_seconds']==budget
        assert json.loads((out/'status.json').read_text())['state']=='queue_finished'


def test_locked_rolling_reports_keep_snapshots_without_stopping(tmp_path,monkeypatch):
    from pathlib import Path
    from experiments.grid_study.study import report
    (tmp_path/'REPORT.md').write_text('previous report')
    (tmp_path/'results.csv').write_text('previous csv')
    def locked(*args, **kwargs):
        raise PermissionError('simulated Windows viewer lock')
    monkeypatch.setattr(Path,'replace',locked)
    metrics=runner.metrics([0,1],[.1,.9])
    result=dict(seed=0,epochs_run=2,best_epoch=2,spec=grids.base_spec(2),validation=metrics,
                validation_fixed=metrics,epoch_seconds_mean=1.)
    report(tmp_path,[dict(model=2,stage='optimizer',blocks='500',result=result,path='trial',reused=False)])
    assert (tmp_path/'REPORT.md').read_text()=='previous report'
    assert 'Budgeted model search' in (tmp_path/'REPORT.md.tmp').read_text()
    assert 'metric_auc' in (tmp_path/'results.csv.tmp').read_text()
    assert (tmp_path/'results.csv').read_text()=='previous csv'


def test_budget_locked_display_preserves_time_and_deadline(tmp_path, monkeypatch):
    from pathlib import Path
    path = tmp_path/'budget.json'
    b = Budget(path, .000015, shares={'screen': 1.})
    original = Path.replace
    def locked(self, target):
        if Path(target) == path:
            raise PermissionError('simulated Windows lock')
        return original(self, target)
    monkeypatch.setattr(Path, 'replace', locked)
    stop = tmp_path/'STOP'
    with pytest.raises(BudgetExhausted):
        with b.charge('screen', stop, 'trial'):
            deadline = time.monotonic()+2
            while not stop.exists() and time.monotonic()<deadline: time.sleep(.005)
            assert stop.exists()
            raise InterruptedError('checkpoint boundary')
    assert json.loads(path.read_text())['used_seconds']['screen'] == 0
    # Incomplete newest record cannot erase the last durable accounting record.
    (tmp_path/'budget.journal'/'99999999999999999999_partial.json').write_text('{')
    again = Budget(path, .000015, shares={'screen': 1.})
    assert again.doc['used_seconds'] == b.doc['used_seconds']
    assert again.remaining('screen') <= 0


def test_budget_recovers_newer_legacy_temp_and_refuses_corruption(tmp_path):
    path = tmp_path/'budget.json'
    b = Budget(path, 1, shares={'screen': 1.})
    record = copy.deepcopy(b.doc)
    record['used_seconds']['screen'] = 123.5
    path.with_suffix('.json.tmp').write_text(json.dumps(record))
    again = Budget(path, 1, shares={'screen': 1.})
    assert again.doc['used_seconds']['screen'] == 123.5
    bad = tmp_path/'bad.json'
    bad.write_text('{')
    with pytest.raises(RuntimeError, match='refusing to reset'):
        Budget(bad, 1, shares={'screen': 1.})


def test_budget_ledger_failure_requests_checkpoint_stop(tmp_path, monkeypatch):
    b = Budget(tmp_path/'budget.json', .000015, shares={'screen': 1.})
    original = b._persist
    calls = []
    def fail_once():
        calls.append(1)
        if len(calls) == 2:
            raise OSError('simulated disk full')
        original()
    monkeypatch.setattr(b, '_persist', fail_once)
    stop = tmp_path/'STOP'
    with pytest.raises(OSError, match='ledger write failed'):
        with b.charge('screen', stop, 'trial'):
            deadline = time.monotonic()+2
            while not stop.exists() and time.monotonic()<deadline: time.sleep(.005)
            assert stop.exists()
    assert b.doc['used_seconds']['screen'] > 0


@pytest.mark.parametrize('wait_hours', [0, 1])
def test_primary_results_and_all_ready_caps_precede_missing_build_wait(tmp_path, monkeypatch, wait_hours):
    from types import SimpleNamespace
    from experiments.grid_study import study
    args=study.parser().parse_args(['--models','1','2','3','--build',str(tmp_path/'b500'),
        '--build-250',str(tmp_path/'b250'),'--build-uncapped',str(tmp_path/'full'),
        '--index',str(tmp_path/'index.csv'),'--out',str(tmp_path/'out'),
        '--wait-build-hours',str(wait_hours)])
    for cap,build in [(250,args.build_250),(500,args.build)]:
        build.mkdir();(build/'build_summary.json').write_text('{}')
        (build/'settings.json').write_text(json.dumps(dict(num_blocks=cap,sample='signal')))
    obj=study.Study(args)
    events=[]
    bases={m:[grids.base_spec(m)] for m in args.models}
    bases[3]=[dict(bases[3][0],pooling=p) for p in ['mean','cls']]
    def screen(model,phase,stage,specs):
        assert not any(e[0]=='confirm' for e in events)
        events.append(('screen',model,stage))
        return copy.deepcopy(bases[model])
    def trial(model,spec,phase,stage,build,blocks,epochs,seed,test=False):
        if test:
            frozen=json.loads((args.out/'primary_selection.json').read_text())
            assert frozen['coverage_pilots_used_for_selection'] is False
            assert frozen['specs']['3']==bases[3]
            assert ('screen',3,'instruction') in events
            assert not any(e[0]=='wait' for e in events)
            obj.data=SimpleNamespace(test_loaded=True);obj.data_key='confirmation'
        if stage=='coverage':
            assert obj.data is None and obj.data_key is None
            assert sum(e[0]=='confirm' for e in events)==12
            assert spec==dict(next(b for b in bases[model] if b['pooling']==spec['pooling']),batch_size=1)
        events.append((stage,model,blocks))
        return dict(completed=True)
    def wait(seconds):
        assert seconds==20
        # All eight ready cap cases across all three models precede the first wait.
        assert sum(e[0]=='coverage' for e in events)==8
        assert sum(e[0]=='confirm' for e in events)==12
        events.append(('wait',))
        args.build_uncapped.mkdir()
        (args.build_uncapped/'build_summary.json').write_text('{}')
        (args.build_uncapped/'settings.json').write_text(json.dumps(dict(num_blocks=0,sample='signal')))
    monkeypatch.setattr(obj,'screen',screen);monkeypatch.setattr(obj,'trial',trial)
    monkeypatch.setattr(study.time,'sleep',wait)
    obj.run()
    state=json.loads((args.out/'status.json').read_text())
    assert state['state']=='queue_finished'
    assert len(state['pending_coverage'])==(3 if wait_hours==0 else 0)
    assert sum(e[0]=='wait' for e in events)==wait_hours
    # A resume cannot silently select new parameters after outer-test evaluation.
    frozen=json.loads((args.out/'primary_selection.json').read_text())
    frozen['specs']['1'][0]['width']=999
    (args.out/'primary_selection.json').write_text(json.dumps(frozen))
    events.clear()
    with pytest.raises(ValueError,match='primary selection changed'):
        obj.run()
    assert not any(e[0]=='confirm' for e in events)
