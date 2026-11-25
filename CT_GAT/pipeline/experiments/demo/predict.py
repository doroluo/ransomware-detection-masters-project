"""Run M1/M2/M3 reference checkpoints on compatible IDA exports or prepared samples."""
import argparse
from contextlib import contextmanager
import gzip
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
PIPELINE=HERE.parents[1]
DEFAULT_BUNDLE=PIPELINE/'demo_models'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1<<20),b''):digest.update(chunk)
    return digest.hexdigest()


def recipes():
    return json.loads((HERE/'recipes.json').read_text(encoding='utf-8'))


def verify_bundle(bundle,selection=None,check_sources=True):
    bundle=Path(bundle);selection=selection or recipes()
    manifest=json.loads((bundle/'bundle_manifest.json').read_text(encoding='utf-8'))
    if manifest['bundle_id']!=selection['bundle_id']:raise ValueError('Checkpoint bundle version differs from this code.')
    if json.loads((bundle/'recipes.json').read_text(encoding='utf-8'))!=selection:
        raise ValueError('Checkpoint recipes differ from the committed recipes.')
    for name,digest in manifest['files_sha256'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name:
            raise ValueError('Unsafe checkpoint manifest path.')
        if sha(bundle/relative)!=digest:raise ValueError('Checkpoint bundle hash mismatch: '+name)
    if check_sources:
        for name,digest in selection['model_source_sha256'].items():
            text=(PIPELINE/name).read_text(encoding='utf-8').encode('utf-8')
            if hashlib.sha256(text).hexdigest()!=digest:raise ValueError('Model/preprocessing source changed: '+name)
    return manifest


def check_sample_id(sample_id,known,allow_known=False):
    if not re.fullmatch('[0-9a-f]{64}',sample_id):raise ValueError('Sample ID must be the lowercase executable SHA-256.')
    if sample_id in known and not allow_known:
        raise ValueError('This ID is in the archived corpus. Use --allow-known-sample only for replay verification.')


def check_settings(settings,recipe):
    expected=('signal',recipe['block_cap'],0)
    if (settings.get('sample'),settings.get('num_blocks'),settings.get('seed'))!=expected:
        raise ValueError(f"M{recipe['model']} requires signal sampling, cap {recipe['block_cap']}, sampler seed 0.")


def cfg_sample_id(path):
    path=Path(path)
    if not path.name.endswith(('.jsonl.gz','.jsonl','.json')):
        raise ValueError('--cfg requires a compatible IDA CFG export, not an executable.')
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():continue
            header=json.loads(line)
            if header.get('record')!='header':raise ValueError('CFG export must begin with its header record.')
            return str(header.get('sha256','')).lower()
    raise ValueError('CFG export is empty.')


def prepare_tensor(payload,source_names,recipe,sample_id):
    import torch
    from experiments.final_paper.canonical import vocabulary_lookup
    from experiments.model_grid.data import Prepared,tok
    if not isinstance(payload,dict):raise ValueError('Expected a raw preprocessing dictionary, not a model or PyG graph.')
    if payload.get('file_id',sample_id)!=sample_id:raise ValueError('Payload file_id does not match its requested executable SHA-256.')
    required=('instruction_ids','insn_to_block','instruction_matched','block_features','edge_index','edge_attr')
    if any(k not in payload for k in required):raise ValueError('Missing project preprocessing tensors.')
    ids=payload['instruction_ids']
    if ids.ndim!=2 or ids.shape[1]!=5 or ids.numel()==0:raise ValueError('Expected nonempty five-channel instruction tokens.')
    features=payload['block_features']
    if features.ndim!=2 or features.shape[1]!=77 or not 0<len(features)<=recipe['block_cap']:
        raise ValueError('Expected 77 raw block features within the selected model block cap.')
    if not torch.isfinite(features).all():raise ValueError('Nonfinite block features.')
    if not isinstance(source_names,list) or len(source_names)!=len(set(source_names)):
        raise ValueError('Expected an ordered, unique API-name list.')
    lookup=vocabulary_lookup(source_names,recipe['training_api_names'])
    ids=ids.long().clone()
    if ids.min()<0 or ids[:,3].max()>=len(lookup):raise ValueError('Instruction IDs do not match the source API vocabulary.')
    ids[:,3]=lookup[ids[:,3]]
    p=dict(payload);p['instruction_ids']=ids
    # Reuse the paper loader without fitting any vocabulary or opening corpus splits.
    adapter=object.__new__(Prepared)
    adapter.model=recipe['model'];adapter.spec=recipe['spec'];adapter.coverage={}
    adapter.seen=torch.arange(tok.FIRST_API_ID,tok.FIRST_API_ID+len(recipe['training_api_names']))
    sample=adapter.pack(p,{'sample_id':sample_id,'label':0})
    adapter.mask(sample)
    return sample,adapter.coverage[sample_id]


def read_prepared(build,sample_id,recipe):
    import torch
    build=Path(build)
    check_settings(json.loads((build/'settings.json').read_text()),recipe)
    names=json.loads((build/'api_vocab.json').read_text())
    payload=torch.load(build/(sample_id+'.pt'),map_location='cpu',weights_only=True)
    return prepare_tensor(payload,names,recipe,sample_id)


def read_exports(cfg_path,asm_path,recipe,sample_id):
    import torch
    # Import through the model loader so the same flat tokenizer module is selected.
    from experiments.model_grid.data import pp,tok
    cfg_path,asm_path=Path(cfg_path),Path(asm_path)
    if cfg_sample_id(cfg_path)!=sample_id:raise ValueError('Requested sample ID differs from the CFG header SHA-256.')
    if not asm_path.name.endswith(('.asm.gz','.asm')):
        raise ValueError('--asm requires the corresponding IDA assembly listing.')
    vocab=tok.ApiVocab()
    for name in recipe['training_api_names']:vocab.id_for(name)
    torch.manual_seed(0)
    result=pp.process_sample(asm_path,cfg_path,label=0,seed=0,api_vocab=vocab,
        num_blocks=recipe['block_cap'],sample_mode='signal',freeze_vocab=True,file_id=sample_id,
        dataset_metadata={'sample_id':sample_id,'inference_only':True})
    models=result['models']
    p=dict(file_id=sample_id,instruction_ids=models['model1']['instruction_ids'],
        insn_to_block=models['model1']['insn_to_block'],instruction_matched=models['model1']['instruction_matched'],
        block_features=models['model2']['block_features'],edge_index=models['model2']['edge_index'],
        edge_attr=models['model2']['edge_attr'],edge_back=models['model2']['cfg_edge_back'])
    if not p['instruction_matched'].any():raise ValueError('No CFG instructions matched the assembly listing; check the export pair.')
    return prepare_tensor(p,vocab.names,recipe,sample_id)


@contextmanager
def inference_numerics(model_id,network,device,precision):
    import torch
    from torch.nn.attention import SDPBackend,sdpa_kernel
    previous_fastpath=torch.backends.mha.get_fastpath_enabled()
    original_encoder=None
    use_bf16=device.type=='cpu' and precision in ('auto','bf16') and model_id in (1,3)
    class EncoderAutocast(torch.nn.Module):
        def __init__(self,inner):super().__init__();self.inner=inner
        def forward(self,*values):
            with torch.autocast('cpu',dtype=torch.bfloat16):return self.inner(*values)
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        if use_bf16 and model_id==3:
            original_encoder=network.encoder;network.encoder=EncoderAutocast(original_encoder)
        with sdpa_kernel(SDPBackend.MATH):
            if use_bf16 and model_id==1:
                with torch.autocast('cpu',dtype=torch.bfloat16):yield
            else:yield
    finally:
        if original_encoder is not None:network.encoder=original_encoder
        torch.backends.mha.set_fastpath_enabled(previous_fastpath)


def score(sample,recipe,bundle,device='cpu',precision='auto'):
    import torch
    from experiments.model_grid.data import tok
    from experiments.model_grid.models import collate,forward,make_model
    if recipe['factory']=='attention':
        from experiments.attention_study.models import make_model
    device=torch.device(device)
    if device.type=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA/HIP device is unavailable; use --device cpu.')
    spec=dict(recipe['spec']);spec['activation_checkpoint']=False  # Training memory option; no weights or computation change.
    model_id=recipe['model']
    network=make_model(model_id,tok.FIRST_API_ID+len(recipe['training_api_names']),spec,recipe['seed'])
    weights=torch.load(Path(bundle)/recipe['checkpoint'],map_location='cpu',weights_only=True)
    if weights['fingerprint']!=recipe['run_fingerprint']:raise ValueError('Checkpoint fingerprint differs from the recorded run.')
    network.load_state_dict(weights['model'],strict=True)
    network.to(device).eval()
    batch,_=collate(model_id,[sample],device)
    with torch.inference_mode(),inference_numerics(model_id,network,device,precision):
        if model_id==1 and device.type=='cpu' and precision in ('auto','bf16'):
            ids,mask=batch
            logits=network(ids,padding_mask=mask).float()
        else:logits=forward(model_id,network,batch)
        probability=float(logits.softmax(1)[0,1].cpu())
    threshold=recipe['threshold']
    return dict(model=model_id,name=recipe['name'],reference_fold=recipe['fold'],reference_seed=recipe['seed'],
        ransomware_score=probability,threshold=threshold,prediction='ransomware' if probability>=threshold else 'benign',
        checkpoint_epoch=weights['epoch'],device=str(device),precision=precision,
        family_overall_accuracy=recipe['family_overall_accuracy'],
        accuracy_scope='Complete five-fold model-family evaluation; not the accuracy of this reference checkpoint or an external-file estimate.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['1','2','3','all'],default='all')
    parser.add_argument('--bundle',type=Path,default=DEFAULT_BUNDLE)
    inputs=parser.add_mutually_exclusive_group()
    inputs.add_argument('--build',type=Path,help='Prepared folder containing <SHA>.pt, settings.json, api_vocab.json')
    inputs.add_argument('--cfg',type=Path,help='Compatible IDA .jsonl.gz CFG export')
    parser.add_argument('--asm',type=Path,help='Corresponding IDA .asm.gz listing; required with --cfg')
    parser.add_argument('--sample-id',help='Executable SHA-256; defaults to the CFG filename stem')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--cpu-precision',dest='precision',choices=['auto','bf16','fp32'],default='auto',
                        help='CPU Transformer numerics; CUDA retains the original training autocast policy')
    parser.add_argument('--allow-known-sample',action='store_true',help='Corpus replay only; never external validation evidence')
    parser.add_argument('--out',type=Path,help='New output JSON file; existing files are refused')
    parser.add_argument('--list',action='store_true',help='Print the frozen recipes without loading weights')
    args=parser.parse_args(argv)
    selection=recipes()
    if args.list:
        print(json.dumps(selection['models'],indent=2));return 0
    if args.out is None or args.out.suffix.lower()!='.json' or args.out.exists():
        parser.error('--out must name a new .json file; existing results are never overwritten')
    if not (args.build or args.cfg):parser.error('Provide --cfg and --asm, or --build with --sample-id.')
    if args.cfg and not args.asm:parser.error('--cfg requires --asm from the same executable.')
    if args.build and args.model=='all':parser.error('Prepared caps differ: choose one --model per --build; --model all supports IDA exports.')
    try:
        sid=args.sample_id or (cfg_sample_id(args.cfg) if args.cfg else None)
        manifest=verify_bundle(args.bundle,selection)
        known=set(json.loads((args.bundle/'corpus_ids.json').read_text()))
        check_sample_id(sid or '',known,args.allow_known_sample)
        import torch
        torch.set_num_threads(1)
        choices=['1','2','3'] if args.model=='all' else [args.model]
        output=[]
        for choice in choices:
            recipe=dict(selection['models'][choice])
            recipe['training_api_names']=json.loads((args.bundle/recipe['vocabulary']).read_text())
            print(f"Preparing {recipe['name']} ({recipe['block_cap']} blocks)",flush=True)
            sample,coverage=(read_prepared(args.build,sid,recipe) if args.build else read_exports(args.cfg,args.asm,recipe,sid))
            result=score(sample,recipe,args.bundle,args.device,args.precision)
            result['coverage']=coverage;output.append(result)
        report=dict(sample_id=sid,scope='existing-corpus replay' if sid in known else 'candidate ID absent from archived corpus',
                    bundle_id=manifest['bundle_id'],results=output,
                    note='No retraining or executable execution. Scores are not calibrated infection probabilities. A single file cannot establish accuracy.')
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(report,indent=2));return 0
    except (ValueError,FileNotFoundError) as error:
        parser.error(str(error))


if __name__=='__main__':raise SystemExit(main())
