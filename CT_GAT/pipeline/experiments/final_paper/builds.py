"""Derive 750/1000-block signal and callwalk inputs from the retained uncapped tensors."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import gc
import json
from pathlib import Path
import shutil
import time

import torch

from experiments.grid_study.coverage import derive_payload
from experiments.model_grid import runner
from experiments.team_pipelines.common import paths_for, sample_rows
from experiments.team_pipelines.build_b import load_vocab_embedder

from .canonical import file_sha256
from .sampling import compact_cfg, pp


def select(cfg, sampler, cap):
    fn = pp.sample_signal_subgraph if sampler == 'signal' else pp.sample_callwalk_subgraph
    return fn(cfg, seed=0, num_blocks=cap)


def keys(payload):
    return {(k['function_id'], k['block_id']) for k in payload['block_keys']}


def derive_one(job):
    sid, row, uncapped, parents, outputs = job
    torch.set_num_threads(1)
    source, _ = paths_for(sid, row['corpus'])
    source_hash = file_sha256(source)
    cfg = compact_cfg(source, sid)
    parent_path = Path(uncapped) / f'{sid}.pt'
    parent_hash = file_sha256(parent_path)
    payload = torch.load(parent_path, mmap=True, map_location='cpu', weights_only=False)
    parent_checks = {}
    records = []
    for sampler in ('signal', 'callwalk'):
        parent = torch.load(Path(parents[sampler]) / f'{sid}.pt', mmap=True, map_location='cpu', weights_only=False)
        anchor = select(cfg, sampler, 500)
        if set(anchor.selected_keys) != keys(parent):
            raise ValueError(f'{sid}: {sampler} replay differs from the existing 500-block parent')
        parent_checks[sampler] = True
        largest = select(cfg, sampler, 1000)
        smaller = select(cfg, sampler, 750)
        if smaller.selected_keys != largest.selected_keys[:750]:
            raise ValueError('Sampler prefix is not stable')
        for cap, selected in ((750, smaller), (1000, largest)):
            destination = Path(outputs[f'{sampler}{cap}'])
            tiny = dict(selected_block_ids=[dict(function_id=a, block_id=b) for a, b in selected.selected_keys])
            out = derive_payload(payload, tiny, cap)
            target = destination / f'{sid}.pt'
            runner.save_checkpoint(target, out)
            records.append(dict(sample_id=sid, sampler=sampler, cap=cap, status='ok',
                n_blocks_total=len(cfg.blocks), n_blocks_kept=len(out['block_keys']),
                source_cfg_sha256=source_hash, parent_tensor_sha256=parent_hash,
                output_tensor_sha256=file_sha256(target)))
            del out
    return records


def reference_checks(ids, rows, uncapped):
    vocab, embedder = load_vocab_embedder(uncapped, pp)
    results = []
    for sid in ids:
        source, asm = paths_for(sid, rows[sid]['corpus'])
        cfg = compact_cfg(source, sid)
        full = pp.load_cfg(source, file_id=sid)
        payload = torch.load(Path(uncapped) / f'{sid}.pt', mmap=True, map_location='cpu', weights_only=False)
        for sampler in ('signal', 'callwalk'):
            for cap in (750, 1000):
                compact = select(cfg, sampler, cap)
                if compact.selected_keys != select(full, sampler, cap).selected_keys:
                    raise ValueError('Compact reader differs from the full CFG reader')
                tiny = dict(selected_block_ids=[dict(function_id=a, block_id=b) for a, b in compact.selected_keys])
                derived = derive_payload(payload, tiny, cap)
                fresh = pp.process_sample(asm, source, label=rows[sid]['label'], seed=0, api_vocab=vocab,
                    num_blocks=cap, sample_mode=sampler, file_id=sid, channel_embedder=embedder,
                    freeze_vocab=True, dataset_metadata=payload['dataset_metadata'])
                m = fresh['models']
                expected = {**m['model1'], **m['model2'], **m['model3'],
                    'instruction_rvas': m['instruction_rvas'], 'edge_back': m['model2']['cfg_edge_back'],
                    'edge_back_addr': m['model2']['cfg_edge_back_addr']}
                for name in ('instruction_ids', 'instruction_matched', 'instruction_rvas', 'insn_to_block',
                    'block_keys', 'block_features', 'block_embeddings', 'block_types', 'block_external',
                    'edge_index', 'edge_type', 'edge_attr', 'edge_back', 'edge_back_addr'):
                    same = torch.equal(derived[name], expected[name]) if isinstance(derived[name], torch.Tensor) else derived[name] == expected[name]
                    if not same:
                        raise ValueError(f'{sid}: fresh {sampler}{cap} differs in {name}')
                results.append(dict(sample_id=sid, sampler=sampler, cap=cap, exact_match=True))
                del fresh, m, expected, derived
                gc.collect()
        del full, cfg, payload
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--uncapped', type=Path, required=True)
    p.add_argument('--signal-500', type=Path, required=True)
    p.add_argument('--callwalk-500', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--limit', type=int, default=0, help='Validation-only subset; cannot create a complete production summary')
    a = p.parse_args(argv)
    torch.set_num_threads(1)
    source_settings = json.loads((a.uncapped / 'settings.json').read_text())
    summary = json.loads((a.uncapped / 'build_summary.json').read_text())
    rows = sample_rows('kfold_v3', 'all')
    ids = sorted(set(rows) - set(summary['excluded']))
    if len(ids) != 5618:
        raise ValueError('Unexpected combined cohort')
    outputs = {f'{sampler}{cap}': a.out_root / f'b_{"sig" if sampler == "signal" else "callwalk"}{cap}_combined_v3'
               for sampler in ('signal', 'callwalk') for cap in (750, 1000)}
    for sampler in ('signal', 'callwalk'):
        for cap in (750, 1000):
            folder = outputs[f'{sampler}{cap}']; folder.mkdir(parents=True, exist_ok=True)
            contract = dict(source_settings, sample=sampler, num_blocks=cap, derived_from=str(a.uncapped.resolve()),
                derivation='original-sampler-replay-and-uncapped-tensor-slice-v1', diagnostic_manifests_written=False,
                sampler_reader_sha256=file_sha256(Path(__file__).with_name('sampling.py')))
            settings = folder / 'settings.json'
            if settings.exists() and json.loads(settings.read_text()) != contract:
                raise ValueError('Derived build settings changed')
            runner.atomic_json(settings, contract)
            for name in ('api_vocab.json', 'channel_embedder.pt'):
                shutil.copy2(a.uncapped / name, folder / name)
    groups = {}
    for sid in ids:
        row = rows[sid]
        if 750 < int(row['num_blocks']) and 500 < int(row['num_instructions']) < 30000:
            groups.setdefault((row['corpus'], row['arch'], row['label']), []).append(sid)
    refs = [min(v, key=lambda sid: int(rows[sid]['num_instructions'])) for _, v in sorted(groups.items())]
    print(f'Validating {len(refs)} samples x two samplers x two caps against fresh preprocessing', flush=True)
    validation = reference_checks(refs, rows, a.uncapped)
    records = {name: {} for name in outputs}
    fields = ['sample_id', 'sampler', 'cap', 'status', 'n_blocks_total', 'n_blocks_kept', 'source_cfg_sha256', 'parent_tensor_sha256', 'output_tensor_sha256']
    for name, folder in outputs.items():
        path = folder / 'build_log.csv'
        if path.exists():
            with path.open(newline='') as handle:
                records[name] = {r['sample_id']: r for r in csv.DictReader(handle)}
    completed = set.intersection(*(set(v) for v in records.values()))
    for sid in list(completed):
        if any(not (folder / f'{sid}.pt').is_file() or file_sha256(folder / f'{sid}.pt') != records[name][sid]['output_tensor_sha256']
               for name, folder in outputs.items()):
            completed.remove(sid)
    todo = [sid for sid in ids if sid not in completed]
    if a.limit:
        todo = todo[:a.limit]
    handles, writers = {}, {}
    try:
        for name, folder in outputs.items():
            handle = (folder / 'build_log.csv').open('a', newline='', encoding='utf-8')
            handles[name] = handle; writers[name] = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
            if handle.tell() == 0: writers[name].writeheader()
        parents = {'signal': str(a.signal_500), 'callwalk': str(a.callwalk_500)}
        destinations = {k: str(v) for k, v in outputs.items()}
        jobs = [(sid, rows[sid], str(a.uncapped), parents, destinations) for sid in todo]
        last = time.monotonic()
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for batch in pool.map(derive_one, jobs, chunksize=1):
                for record in batch:
                    name = f'{record["sampler"]}{record["cap"]}'
                    writers[name].writerow(record); handles[name].flush()
                    records[name][record['sample_id']] = record
                completed.add(batch[0]['sample_id'])
                if time.monotonic() - last > 30:
                    print(f'750/1000 builds: {len(completed)}/{len(ids)} source samples, four outputs each', flush=True)
                    last = time.monotonic()
    finally:
        for handle in handles.values(): handle.close()
    for name, folder in outputs.items():
        done = len(records[name]) == len(ids)
        runner.atomic_json(folder / 'derivation_validation.json', dict(reference_checks=validation,
            parent_500_selection_checked=len(records[name]), diagnostic_manifests_written=0,
            production_complete=done, subset_validation_only=bool(a.limit)))
        if done:
            runner.atomic_json(folder / 'build_summary.json', dict(complete=True, processed=len(ids), total=5652,
                excluded=summary['excluded'], derived_from=str(a.uncapped.resolve())))
    print('Four additional block-count builds complete' if len(completed) == len(ids) else 'Subset validation complete', flush=True)


if __name__ == '__main__':
    main()
