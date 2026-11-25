"""Checkpoint loading, thresholds and input contracts for the M1/M2/M3 demos."""
import gzip
import hashlib
import io
import json
import math
import zipfile

import pytest
import torch

from experiments.demo import predict
from experiments.demo.download import extract_verified
from experiments.model_grid.data import tok,pp


def recipe(model):
    return dict(model=model,name=f'M{model} test',block_cap=500,factory='attention' if model==3 else 'paper',
        fold=0,seed=0,threshold=.2,checkpoint='best.pt',run_fingerprint='synthetic-test',family_overall_accuracy=.8,
        training_api_names=['CreateFileW','CryptEncrypt'],spec=dict(width=8,transformer_heads=2,transformer_layers=1,
        sequence_length=16,block_length=4,block_chunk=64,dropout=.1,attention_dropout=0.,gat_head_policy='all_layers',
        gat_hidden=8,gat_heads=2,gat_layers=2,pooling='mean' if model!=2 else 'graph_mean_max',activation_checkpoint=False))


def payload():
    ids=torch.tensor([[1,1,1,tok.FIRST_API_ID,0],[1,1,1,tok.UNK_API,0],[1,1,1,tok.FIRST_API_ID+1,0],[1,1,1,0,0]])
    features=torch.zeros(2,77);features[:,0]=2
    return dict(file_id='a'*64,instruction_ids=ids,insn_to_block=torch.tensor([0,0,1,1]),
        instruction_matched=torch.ones(4,dtype=torch.bool),block_features=features,
        edge_index=torch.tensor([[0,1],[1,0]]),edge_attr=torch.zeros(2,7),edge_back=torch.tensor([False,True]))


@pytest.mark.parametrize('model',[1,2,3])
def test_saved_checkpoint_and_threshold_round_trip(tmp_path,model):
    from experiments.model_grid.models import make_model,collate
    if model==3:
        from experiments.attention_study.models import make_model
    r=recipe(model)
    graph,coverage=predict.prepare_tensor(payload(),r['training_api_names'],r,'a'*64)
    network=make_model(model,tok.FIRST_API_ID+2,r['spec'],0).eval()
    with torch.no_grad():
        for parameter in network.parameters():parameter.zero_()
        classifier=network.classifier if model!=3 else network.gat.classifier
        classifier[-1].bias[1]=math.log(.3/.7)
    torch.save(dict(model=network.state_dict(),epoch=7,fingerprint=r['run_fingerprint']),tmp_path/'best.pt')
    result=predict.score(graph,r,tmp_path,precision='fp32')
    assert result['ransomware_score']==pytest.approx(.3,abs=1e-6)
    assert result['prediction']=='ransomware'  # Saved threshold .2; a hard-coded .5 would be wrong.
    assert result['checkpoint_epoch']==7 and result['reference_seed']==0
    assert coverage['blocks_in_build']==2


def test_external_api_id_remaps_by_name_and_preserves_behavior():
    r=recipe(1);p=payload();p['instruction_ids'][:,4]=1
    sample,_=predict.prepare_tensor(p,['NewUnseenAPI','CreateFileW'],r,'a'*64)
    assert sample['ids'][0,3]==tok.UNK_API
    assert sample['ids'][2,3]==tok.FIRST_API_ID
    assert (sample['ids'][:,4]==1).all()


def test_budget_and_corpus_guards():
    r=recipe(2)
    with pytest.raises(ValueError,match='cap 500'):predict.check_settings(dict(sample='signal',num_blocks=750,seed=0),r)
    with pytest.raises(ValueError,match='archived corpus'):predict.check_sample_id('a'*64,{'a'*64})
    predict.check_sample_id('a'*64,{'a'*64},allow_known=True)
    with pytest.raises(ValueError,match='SHA-256'):predict.check_sample_id('filename.exe',set())


def write_exports(tmp_path):
    records=[dict(record='header',schema=pp.SCHEMA,sha256='a'*64,imagebase='0x400000'),
        dict(record='function',id=0,name='sub_401000',rva='0x1000',edges=[],blocks=[dict(id=0,
            start_rva='0x1000',end_rva='0x1002',instructions=[dict(rva='0x1000',mnemonic='mov',size=1),
                                                            dict(rva='0x1001',mnemonic='retn',size=1)])]),
        dict(record='footer',complete=True)]
    cfg=tmp_path/'example.jsonl.gz'
    with gzip.open(cfg,'wt',encoding='utf-8') as h:h.write('\n'.join(json.dumps(r) for r in records)+'\n')
    asm=tmp_path/'example.asm.gz'
    with gzip.open(asm,'wt',encoding='utf-8') as h:
        h.write('sub_401000 proc near\n    mov eax, ebx\n    retn\nsub_401000 endp\n')
    return cfg,asm


@pytest.mark.parametrize('model',[1,2,3])
def test_compatible_ida_exports_prepare_for_every_model(tmp_path,model):
    cfg,asm=write_exports(tmp_path)
    assert predict.cfg_sample_id(cfg)=='a'*64
    sample,coverage=predict.read_exports(cfg,asm,recipe(model),'a'*64)
    assert coverage['instructions_in_build']==2 and coverage['unmatched_in_build']==0
    assert len(sample['ids'])==2 if model==1 else sample.num_nodes==1
    with pytest.raises(ValueError,match='header SHA-256'):predict.read_exports(cfg,asm,recipe(model),'b'*64)


def test_checkpoint_archive_rejects_bad_hash_and_path(tmp_path):
    raw=io.BytesIO()
    with zipfile.ZipFile(raw,'w') as z:z.writestr('../escape.json','{}')
    blob=raw.getvalue()
    with pytest.raises(ValueError,match='SHA-256'):extract_verified(blob,tmp_path/'bad','0'*64)
    with pytest.raises(ValueError,match='Unsafe'):extract_verified(blob,tmp_path/'bad',hashlib.sha256(blob).hexdigest())
    assert not (tmp_path/'escape.json').exists()


def test_cli_refuses_existing_output_before_loading_weights(tmp_path):
    target=tmp_path/'existing.json';target.write_text('keep')
    with pytest.raises(SystemExit):predict.main(['--out',str(target),'--model','1'])
    assert target.read_text()=='keep'
