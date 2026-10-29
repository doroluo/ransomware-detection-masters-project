"""Exact nested signal selection and disk-backed inputs for large coverage comparisons."""
import argparse
from collections.abc import MutableMapping
import hashlib
import json
from pathlib import Path
import shutil

import torch

from experiments.model_grid.data import Prepared, assert_disjoint, digest, manifest_for, subset, tok
from experiments.model_grid.runner import atomic_json, save_checkpoint


def derive_payload(payload, manifest, cap):
    """Signal's selected order is prefix-stable; tensor order is canonical, not rank order."""
    key = lambda x: (x['function_id'], x['block_id'])
    selected = set(map(key, manifest['selected_block_ids'][:cap]))
    nodes = torch.tensor([i for i,k in enumerate(payload['block_keys']) if key(k) in selected])
    if not len(nodes) or len(nodes) != len(selected):
        raise ValueError('manifest and tensor block keys disagree')
    mapping = torch.full((len(payload['block_keys']),), -1, dtype=torch.long)
    mapping[nodes] = torch.arange(len(nodes))
    imask = mapping[payload['insn_to_block']] >= 0
    emask = (mapping[payload['edge_index']] >= 0).all(0)
    out = dict(payload)
    for k in ('instruction_ids','instruction_matched','instruction_rvas'):
        out[k] = (payload[k][imask].clone() if isinstance(payload[k],torch.Tensor)
                  else [v for v,keep in zip(payload[k],imask.tolist()) if keep])
    out['insn_to_block'] = mapping[payload['insn_to_block'][imask]]
    for k in ('block_features','block_embeddings','block_types','block_external'):
        out[k] = payload[k][nodes].clone()
    for k in ('block_keys','model1_block_keys','model2_block_keys','model3_block_keys'):
        out[k] = [payload[k][i] for i in nodes.tolist()]
    out['edge_index'] = mapping[payload['edge_index'][:, emask]]
    for k in ('edge_type','edge_attr','edge_back','edge_back_addr'):
        out[k] = payload[k][emask].clone()
    return out


def derive_build(source, out, cap=250):
    source, out = Path(source), Path(out)
    settings = json.loads((source/'settings.json').read_text())
    if settings['sample'] != 'signal' or not 0 < cap < settings['num_blocks']:
        raise ValueError('derivation requires a smaller positive cap and a signal build')
    out.mkdir(parents=True, exist_ok=True)
    contract = dict(settings, num_blocks=cap, derived_from=str(source.resolve()),
                    derivation='signal-selection-prefix-v1',
                    source_settings_sha256=hashlib.sha256((source/'settings.json').read_bytes()).hexdigest())
    if (out/'settings.json').exists() and json.loads((out/'settings.json').read_text()) != contract:
        raise ValueError('coverage build settings changed')
    atomic_json(out/'settings.json', contract)
    shutil.copy2(source/'api_vocab.json', out/'api_vocab.json')
    paths = sorted(p for p in source.glob('*.pt') if len(p.stem) == 64)
    for i, p in enumerate(paths):
        if (out/p.name).exists():
            continue
        manifest = json.loads((source/f'{p.stem}.manifest.json').read_text())
        payload = torch.load(p, map_location='cpu', weights_only=False)
        save_checkpoint(out/p.name, derive_payload(payload, manifest, cap))
        if i % 100 == 0:
            print(f'250-block build: {i+1}/{len(paths)}', flush=True)
    atomic_json(out/'build_summary.json', dict(processed=len(paths), total=len(paths)+len(settings['excluded']),
                excluded=settings['excluded'], settings=contract, complete=True))


class DiskSamples(MutableMapping):
    """No resident full-dataset graph tensors. Each training batch loads its own samples."""
    def __init__(self, root):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.ids = set()
    def __getitem__(self, sid):
        if sid not in self.ids:
            raise KeyError(sid)
        return torch.load(self.root/f'{sid}.pt', map_location='cpu', weights_only=False)
    def __setitem__(self, sid, sample):
        save_checkpoint(self.root/f'{sid}.pt', sample); self.ids.add(sid)
    def __delitem__(self, sid):
        self.ids.remove(sid)
    def __iter__(self):
        return iter(self.ids)
    def __len__(self):
        return len(self.ids)
    def __contains__(self, sid):
        return sid in self.ids


class DiskPrepared(Prepared):
    """Use the same packing, API masking and provenance as Prepared, without retaining all graphs in RAM."""
    def __init__(self, build, index, dataset, fold, val_seed, model, spec, cache, smoke=0):
        self.build, self.index, self.model, self.spec = Path(build), Path(index), model, spec
        settings = json.loads((self.build/'settings.json').read_text())
        self.excluded = settings.get('excluded', {})
        self.splits = manifest_for(index, dataset, fold, val_seed)
        self.excluded_by_split = {s:{r['sample_id']:self.excluded[r['sample_id']] for r in rr
                                   if r['sample_id'] in self.excluded} for s,rr in self.splits.items()}
        self.splits = {s:[r for r in rr if r['sample_id'] not in self.excluded] for s,rr in self.splits.items()}
        assert_disjoint(self.splits)
        if smoke:
            for s in ('train','val'):
                self.splits[s] = subset(self.splits[s], smoke)
        self.rows = {r['sample_id']:r for rr in self.splits.values() for r in rr}
        if any(not (self.build/f'{sid}.pt').is_file() for sid in self.rows):
            raise FileNotFoundError('incomplete coverage build: refusing silent sample removal')
        self.samples, self.hashes, self.coverage = DiskSamples(cache), {}, {}
        self.seen = torch.empty(0, dtype=torch.long)
        for split in ('train','val'):
            for row in self.splits[split]:
                payload = self.read(row['sample_id'])
                ids = payload['instruction_ids'].long()
                if split == 'train':
                    self.seen = torch.unique(torch.cat([self.seen, ids[:,3][ids[:,3] >= tok.FIRST_API_ID]]))
                self.samples[row['sample_id']] = self.pack(payload, row)
        self.api_size = max(tok.FIRST_API_ID, int(self.seen.max())+1 if len(self.seen) else 0)
        for sid in self.samples:
            sample = self.samples[sid]; self.mask(sample); self.samples[sid] = sample
        self.provenance = dict(dataset=dataset, fold=fold, val_seed=val_seed, smoke=smoke,
            index_sha256=hashlib.sha256(self.index.read_bytes()).hexdigest(),
            splits={k:[r['sample_id'] for r in v] for k,v in self.splits.items()},
            build_exclusions_by_split=self.excluded_by_split, train_val_payload_sha256=dict(self.hashes),
            build_metadata={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [self.build/'settings.json', self.build/'api_vocab.json'] if p.is_file()})
        self.fingerprint = digest(self.provenance)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Derive exactly the 250-block signal subset, preserving canonical ordering.')
    parser.add_argument('--from-build', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    derive_build(args.from_build, args.out)
