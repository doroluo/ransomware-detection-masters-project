"""Common fixed grouped splits; fold-specific API masking; no test loading during search."""
import hashlib
import io
import json
from pathlib import Path

import torch

from common.config import RunConfig
from common.manifests import load_index, make_manifest
from experiments.team_pipelines.train_b import block_counts, kind_slice, remap_api
from s03a_transformer.five_channel.model import tok
from s03b_cfg_feature.gat.graph_full_loader import _edge_attr, scale_block_features
import preprocess_full as pp  # loaded by the team's five-channel model

from .models import BlockData


DATASETS = {'mendeley-x86': ('kfold', 'x86'), 'mendeley-all': ('kfold', 'all'),
            'combined': ('kfold_v3', 'all'), 'combined-matched': ('kfold_v3', 'matched')}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def manifest_for(index, dataset, fold, val_seed):
    if Path(index).name != 'index.csv':
        raise ValueError('--index must point to the cache index.csv')
    protocol, arch = DATASETS[dataset]
    cfg = RunConfig(protocol=protocol, arch=arch, fold=fold, seed=0, val_seed=val_seed)
    manifest = make_manifest(cfg, load_index(Path(index).parent, protocol))
    splits = {s: getattr(manifest, s) for s in ('train', 'val', 'test')}
    assert_disjoint(splits)
    return splits


def assert_disjoint(splits):
    seen_ids, seen_groups = set(), set()
    for name, rows in splits.items():
        ids = {r['sample_id'] for r in rows}
        groups = {(r['label'], r['family'] if r['label'] else r['identity_group']) for r in rows}
        if len(ids) != len(rows) or ids & seen_ids or groups & seen_groups:
            raise ValueError(f'duplicate sample or family/identity leakage in {name}')
        if {r['label'] for r in rows} != {0, 1}:
            raise ValueError(f'{name} must contain both classes')
        seen_ids.update(ids)
        seen_groups.update(groups)


def subset(rows, n):
    """Balanced tiny subset inside a split, for a smoke test only."""
    if n < 4:
        raise ValueError('smoke size must be at least 4')
    return sum(([r for r in rows if r['label'] == label][:n // 2] for label in (0, 1)), [])


class Prepared:
    def __init__(self, build, index, dataset, fold, val_seed, model, spec, smoke=0):
        self.build, self.index = Path(build), Path(index)
        self.model, self.spec = model, spec
        self.splits = manifest_for(index, dataset, fold, val_seed)
        settings_path = self.build/'settings.json'
        if not settings_path.exists():
            settings_path = self.build/'build_info'/'build_settings.json'
        settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
        self.excluded = settings.get('excluded', {})
        if not isinstance(self.excluded, dict):
            raise ValueError('build exclusions must map sample IDs to recorded reasons')
        self.excluded_by_split = {s: {r['sample_id']: self.excluded[r['sample_id']] for r in rows
                                      if r['sample_id'] in self.excluded} for s, rows in self.splits.items()}
        self.splits = {s: [r for r in rows if r['sample_id'] not in self.excluded] for s, rows in self.splits.items()}
        assert_disjoint(self.splits)
        if smoke:
            for split in ('train', 'val'):
                self.splits[split] = subset(self.splits[split], smoke)
        self.rows = {r['sample_id']: r for rows in self.splits.values() for r in rows}
        self.samples, self.hashes, self.coverage = {}, {}, {}
        for sid in self.rows:
            if not (self.build / f'{sid}.pt').is_file():
                raise FileNotFoundError(f'{sid}.pt missing from build; no silent split exclusions are allowed')
        self.seen = torch.empty(0, dtype=torch.long)
        self.api_size = tok.FIRST_API_ID
        for split in ('train', 'val'):
            for row in self.splits[split]:
                sid = row['sample_id']
                payload = self.read(sid)
                ids = payload['instruction_ids'].long()
                if split == 'train':
                    self.seen = torch.unique(torch.cat([self.seen, ids[:, 3][ids[:, 3] >= tok.FIRST_API_ID]]))
                self.samples[sid] = self.pack(payload, row)
            if split == 'train':
                self.api_size = max(tok.FIRST_API_ID, int(self.seen.max()) + 1 if len(self.seen) else 0)
        # Mask after collecting ALL training API IDs; validation names never extend the vocabulary.
        for sample in self.samples.values():
            self.mask(sample)
        self.provenance = {
            'dataset': dataset, 'fold': fold, 'val_seed': val_seed, 'smoke': smoke,
            'index_sha256': hashlib.sha256(self.index.read_bytes()).hexdigest(),
            'splits': {k: [r['sample_id'] for r in v] for k, v in self.splits.items()},
            'build_exclusions_by_split': self.excluded_by_split,
            'train_val_payload_sha256': dict(self.hashes),
            'build_metadata': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [self.build/'settings.json', self.build/'api_vocab.json'] if p.is_file()},
        }
        self.fingerprint = digest(self.provenance)

    def read(self, sid):
        blob = (self.build / f'{sid}.pt').read_bytes()
        sha = hashlib.sha256(blob).hexdigest()
        if sid in self.hashes and self.hashes[sid] != sha:
            raise ValueError(f'input changed during run: {sid}')
        self.hashes[sid] = sha
        # These are the project's trusted tensor builds, never executable samples.
        payload = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=False)
        if int(payload['label']) != self.rows[sid]['label']:
            raise ValueError(f'label mismatch for {sid}')
        return payload

    def pack(self, p, row):
        sid, ids = row['sample_id'], p['instruction_ids'].long()
        owners = p['insn_to_block'].long()
        n = len(p['block_features'])
        if n < 1 or ids.ndim != 2 or ids.shape[1] != 5 or len(ids) != len(owners):
            raise ValueError(f'invalid instruction/block tensors: {sid}')
        if len(owners) and (owners.min() < 0 or owners.max() >= n):
            raise ValueError(f'invalid instruction owners: {sid}')
        if self.model == 1:
            kept = min(len(ids), self.spec['sequence_length'])
            sample = {'ids': ids[:kept].clone(), 'label': row['label']}
        elif self.model == 2:
            # The baseline retains the team's 77 handcrafted features, never embeddings.
            sample = BlockData(x=p['block_features'].float().clone(), edge_index=p['edge_index'].long(),
                               edge_attr=_edge_attr(p), y=torch.tensor(row['label']), num_nodes=n)
            sample._ids_for_counts = ids
            sample._owners_for_counts = owners
            sample._matched_for_counts = p['instruction_matched']
            kept = len(ids)
        else:
            order = torch.argsort(owners, stable=True)
            counts = torch.bincount(owners, minlength=n)
            sorted_owners = owners[order]
            starts = counts.cumsum(0) - counts
            position = torch.arange(len(ids)) - starts[sorted_owners]
            selected = order[position < self.spec['block_length']]
            sample = BlockData(insn_ids=ids[selected].clone(), insn_block=owners[selected].clone(),
                               edge_index=p['edge_index'].long(), edge_attr=_edge_attr(p),
                               y=torch.tensor(row['label']), num_nodes=n)
            kept = len(selected)
        selected_ids = sample['ids'] if self.model == 1 else (sample.insn_ids if self.model == 3 else ids)
        self.coverage[sid] = {'instructions_in_build': len(ids), 'instructions_used': kept, 'blocks_in_build': n,
                             'instructions_in_raw_cfg': int(row['num_instructions']) if row.get('num_instructions') else None,
                             'blocks_in_raw_cfg': int(row['num_blocks']) if row.get('num_blocks') else None,
                             'api_instructions_in_build': int((ids[:, 3] >= tok.FIRST_API_ID).sum()),
                             'api_instructions_used': int((selected_ids[:, 3] >= tok.FIRST_API_ID).sum()),
                             'unmatched_in_build': int((~p['instruction_matched'].bool()).sum())}
        return sample

    def mask(self, sample):
        if self.model == 1:
            sample['ids'] = remap_api(tok, sample['ids'], self.seen)
        elif self.model == 3:
            sample.insn_ids = remap_api(tok, sample.insn_ids, self.seen)
        else:
            ids = remap_api(tok, sample._ids_for_counts, self.seen)
            sample.x[:, kind_slice(pp)] = block_counts(tok, ids, sample._matched_for_counts,
                                                       sample._owners_for_counts, sample.num_nodes)
            sample.x = scale_block_features(sample.x)
            for key in ('_ids_for_counts', '_owners_for_counts', '_matched_for_counts'):
                delattr(sample, key)

    def load_test(self):
        """Called only after validation has selected the configuration; no test-time fitting."""
        for row in self.splits['test']:
            sid = row['sample_id']
            if sid not in self.samples:
                sample = self.pack(self.read(sid), row)
                self.mask(sample)
                self.samples[sid] = sample
