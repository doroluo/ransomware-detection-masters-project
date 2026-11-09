"""Disk-backed controlled inputs with shared train-only APIs and explicit ablations."""
import hashlib
import json
from pathlib import Path

import torch

from experiments.grid_study.coverage import DiskPrepared, DiskSamples
from experiments.model_grid.data import assert_disjoint, digest, manifest_for, subset, tok
from experiments.team_pipelines.train_b import kind_slice
import preprocess_full as pp

from .canonical import file_sha256, vocabulary_lookup


ABLATIONS = {
    1: ('no_api_identity', 'no_behavior', 'shuffle_instructions'),
    2: ('no_api_kind_counts', 'no_behavior', 'edge_free'),
    3: ('no_api_identity', 'no_behavior', 'shuffle_instructions', 'edge_free'),
}


def protocol_hashes():
    folder = Path(__file__).parent
    return {name: file_sha256(folder / name) for name in ('canonical.py', 'data.py', 'trial.py', 'controlled.py')}


def ablate_sample(model, sample, name, sid):
    if name == 'none':
        return
    if name not in ABLATIONS[model]:
        raise ValueError(f'Unsupported Model {model} ablation: {name}')
    if name == 'no_api_kind_counts':
        sample.x[:, kind_slice(pp)] = 0
    elif name == 'no_behavior' and model == 2:
        columns = [i for i, key in enumerate(pp.HANDCRAFTED_FEATURE_NAMES) if key.startswith('behavior_')]
        sample.x[:, columns] = 0
    elif name == 'edge_free':
        sample.edge_index = torch.empty((2, 0), dtype=torch.long)
        sample.edge_attr = sample.edge_attr.new_empty((0, sample.edge_attr.shape[1]))
    else:
        ids = sample['ids'] if model == 1 else sample.insn_ids
        if name == 'no_api_identity':
            ids[:, 3][ids[:, 3] >= tok.FIRST_API_ID] = tok.UNK_API
        elif name == 'no_behavior':
            ids[:, 4] = tok.BEHAVIOR_MAP['NONE']
        elif name == 'shuffle_instructions':
            seed = int(hashlib.sha256((sid + ':paper-order-control-v1').encode()).hexdigest()[:15], 16)
            rng = torch.Generator().manual_seed(seed)
            if model == 1:
                sample['ids'] = ids[torch.randperm(len(ids), generator=rng)].clone()
            else:
                counts = torch.bincount(sample.insn_block, minlength=sample.num_nodes)
                start = 0
                shuffled = ids.clone()
                for count in counts.tolist():
                    shuffled[start:start + count] = ids[start:start + count][torch.randperm(count, generator=rng)]
                    start += count
                sample.insn_ids = shuffled


class ControlledPrepared(DiskPrepared):
    def __init__(self, build, index, dataset, fold, val_seed, model, spec, cache, vocabulary, smoke=0):
        self.fixed_names = vocabulary['names']
        self.fixed_seen = torch.arange(tok.FIRST_API_ID, tok.FIRST_API_ID + len(self.fixed_names))
        self.api_lookup = vocabulary_lookup(json.loads((Path(build) / 'api_vocab.json').read_text()), self.fixed_names)
        self.common_vocabulary = vocabulary
        self.ablation = spec.get('ablation', 'none')
        # The vocabulary is already fitted on training data. Mask each sample
        # immediately instead of writing the legacy unmasked corpus to disk and
        # rereading it for a second masking pass (especially costly uncapped).
        self.build, self.index, self.model, self.spec = Path(build), Path(index), model, spec
        settings = json.loads((self.build / 'settings.json').read_text())
        self.excluded = settings.get('excluded', {})
        splits = manifest_for(index, dataset, fold, val_seed)
        self.excluded_by_split = {s: {r['sample_id']: self.excluded[r['sample_id']]
            for r in rows if r['sample_id'] in self.excluded} for s, rows in splits.items()}
        self.splits = {s: [r for r in rows if r['sample_id'] not in self.excluded] for s, rows in splits.items()}
        assert_disjoint(self.splits)
        fit_smoke = vocabulary['contract'].get('smoke', 0)
        if fit_smoke and not smoke:
            raise ValueError('A smoke-only vocabulary cannot be used for production results')
        fit_rows = subset(self.splits['train'], fit_smoke) if fit_smoke else self.splits['train']
        if [r['sample_id'] for r in fit_rows] != vocabulary['contract']['train_ids']:
            raise ValueError('Common vocabulary was fitted on different training IDs')
        self.fit_payload_hashes = {}
        for name, entry in vocabulary['contract']['builds'].items():
            if Path(entry['path']).resolve() == self.build.resolve():
                self.fit_payload_hashes = vocabulary['training_payload_sha256'][name]
        if smoke:
            for s in ('train', 'val'):
                self.splits[s] = subset(self.splits[s], smoke)
        self.rows = {r['sample_id']: r for rows in self.splits.values() for r in rows}
        if any(not (self.build / f'{sid}.pt').is_file() for sid in self.rows):
            raise FileNotFoundError('Incomplete controlled build; no silent sample removal')
        self.samples, self.hashes, self.coverage = DiskSamples(cache), {}, {}
        self.api_size = tok.FIRST_API_ID + len(self.fixed_names)
        self.seen = self.fixed_seen
        for s in ('train', 'val'):
            for row in self.splits[s]:
                payload = self.read(row['sample_id'])
                sample = self.pack(payload, row)
                self.mask(sample)
                self.samples[row['sample_id']] = sample
                del sample, payload
        self.provenance = dict(dataset=dataset, fold=fold, val_seed=val_seed, smoke=smoke,
            index_sha256=file_sha256(self.index), splits={k: [r['sample_id'] for r in v] for k, v in self.splits.items()},
            build_exclusions_by_split=self.excluded_by_split, train_val_payload_sha256=dict(self.hashes),
            build_metadata={p.name: file_sha256(p) for p in [self.build / 'settings.json', self.build / 'api_vocab.json']})
        self.provenance['controlled_protocol'] = dict(api_vocabulary_fingerprint=vocabulary['fingerprint'],
            vocabulary_fit_scope='Union of training representations, excluding validation/test payloads',
            ablation=self.ablation, scientific_adapter_sources=protocol_hashes(),
            instruction_order_control='Fixed per sample; identical across seeds and paired pooling arms')
        self.fingerprint = digest(self.provenance)

    def read(self, sid):
        path = self.build / f'{sid}.pt'
        before = path.stat()
        fingerprint = file_sha256(path)
        if sid in self.fit_payload_hashes and self.fit_payload_hashes[sid] != fingerprint:
            raise ValueError('Training payload differs from the input used to fit the common vocabulary')
        if sid in self.hashes and self.hashes[sid] != fingerprint:
            raise ValueError(f'Input changed during controlled run: {sid}')
        self.hashes[sid] = fingerprint
        payload = torch.load(path, mmap=True, map_location='cpu', weights_only=False)
        if int(payload['label']) != self.rows[sid]['label']:
            raise ValueError(f'Label mismatch: {sid}')
        ids = payload['instruction_ids'].long().clone()
        if ids.numel() and (ids[:, 3].min() < 0 or ids[:, 3].max() >= len(self.api_lookup)):
            raise ValueError(f'Invalid source API ID: {sid}')
        ids[:, 3] = self.api_lookup[ids[:, 3]]
        payload['instruction_ids'] = ids
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Source payload changed while reading')
        return payload

    def mask(self, sample):
        seen = self.seen
        self.seen = self.fixed_seen
        super().mask(sample)
        self.seen = seen
        sid = getattr(sample, 'paper_sample_id', None)
        if self.model == 1:
            sid = sample.pop('_paper_sample_id')
        else:
            del sample.paper_sample_id
        ablate_sample(self.model, sample, self.ablation, sid)

    def pack(self, payload, row):
        sample = super().pack(payload, row)
        if self.model == 1:
            sample['_paper_sample_id'] = row['sample_id']
        else:
            sample.paper_sample_id = row['sample_id']
        return sample
