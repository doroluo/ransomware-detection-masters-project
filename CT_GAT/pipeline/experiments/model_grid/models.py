"""Position-aware Models 1/3 and the unchanged handcrafted-feature GAT (Model 2).

Uses the team's existing five-channel embedding/Transformer and FullBlockGAT.
The new Model 3 adapter changes block readout only for the CLS experiment.
"""
import torch
from torch import nn
from torch_geometric.data import Batch, Data

from s03a_transformer.five_channel.model import Model1TokenSequence
from s03b_cfg_feature.gat.gat_full_model import FullBlockGAT


class BlockData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        return self.num_nodes if key == 'insn_block' else super().__inc__(key, value, *args, **kwargs)


class BlockTransformer(nn.Module):
    """A real-instruction limit, plus a separate optional CLS slot; padding is ignored."""
    def __init__(self, api_size, block_length=64, dropout=.1, pooling='mean', width=64, heads=4, layers=2):
        super().__init__()
        if pooling not in ('mean', 'cls'):
            raise ValueError('pooling must be mean or cls')
        self.pooling, self.block_length = pooling, block_length
        # Allocate the extra position in BOTH arms: real instruction positions match.
        self.base = Model1TokenSequence(api_size, width, heads, layers,
                                       max_seq_len=block_length + 1, dropout=dropout)
        del self.base.classifier
        nn.init.normal_(self.base.pos_encoder, std=.02)
        if pooling == 'cls':
            self.cls = nn.Parameter(torch.empty(1, 1, width))
            nn.init.normal_(self.cls, std=.02)

    def forward(self, ids, pad):
        if ids.shape[1] > self.block_length:
            raise ValueError('more real instructions than configured block_length')
        empty = pad.all(1)
        x = self.base.embedder(ids) + self.base.pos_encoder[:, 1:ids.shape[1] + 1]
        if self.pooling == 'cls':
            cls = self.cls.expand(len(ids), -1, -1) + self.base.pos_encoder[:, :1]
            x = torch.cat([cls, x], 1)
            mask = torch.cat([pad.new_zeros((len(ids), 1)), pad], 1)
        else:
            mask = pad.clone()
            mask[empty, 0] = False  # avoid all-masked attention; zero the output below
        h = self.base.transformer_encoder(x, src_key_padding_mask=mask)
        if self.pooling == 'cls':
            pooled = h[:, 0]
        else:
            keep = (~pad).unsqueeze(-1)
            pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1)
        return pooled.masked_fill(empty[:, None], 0)


class Model3(nn.Module):
    def __init__(self, api_size, spec, seed):
        super().__init__()
        self.chunk = spec['block_chunk']
        self.encoder = BlockTransformer(api_size, spec['block_length'], spec['dropout'], spec['pooling'],
                                        spec['width'], spec['transformer_heads'], spec['transformer_layers'])
        self.norm = nn.LayerNorm(spec['width'])
        # Paired pooling arms start with the same GAT, regardless of CLS initialization.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 17011)
            self.gat = FullBlockGAT(spec['width'], 8, hidden=spec['gat_hidden'], heads=spec['gat_heads'],
                                   layers=spec['gat_layers'], dropout=spec['dropout'])

    def block_vectors(self, batch):
        ids, owners = batch.insn_ids, batch.insn_block
        counts = torch.bincount(owners, minlength=batch.num_nodes)
        starts = torch.cumsum(counts, 0) - counts
        outputs = []
        # Inputs are grouped by block and retain the original instruction order.
        for first in range(0, batch.num_nodes, self.chunk):
            last = min(first + self.chunk, batch.num_nodes)
            length = counts[first:last]
            width = max(1, int(length.max()))
            positions = torch.arange(width, device=ids.device)[None, :]
            keep = positions < length[:, None]
            indices = starts[first:last, None] + positions
            padded = ids.new_zeros((last - first, width, 5))
            padded[keep] = ids[indices[keep]]
            with torch.autocast(ids.device.type, dtype=torch.bfloat16, enabled=ids.device.type == 'cuda'):
                z = self.encoder(padded, ~keep)
            outputs.append(self.norm(z.float()).masked_fill((length == 0)[:, None], 0))
        return torch.cat(outputs)

    def forward(self, batch):
        # A fresh Batch is supplied for each step; cached CPU samples are never changed.
        batch.x = self.block_vectors(batch)
        return self.gat(batch)


def make_model(model, api_size, spec, seed):
    if model == 1:
        model1 = Model1TokenSequence(api_size, spec['width'], spec['transformer_heads'],
                                    spec['transformer_layers'], max_seq_len=spec['sequence_length'],
                                    dropout=spec['dropout'])
        nn.init.normal_(model1.pos_encoder, std=.02)
        return model1
    if model == 2:
        return FullBlockGAT(77, 8, hidden=spec['gat_hidden'], heads=spec['gat_heads'],
                            layers=spec['gat_layers'], dropout=spec['dropout'])
    return Model3(api_size, spec, seed)


def collate(model, samples, device):
    if model != 1:
        batch = Batch.from_data_list(samples).to(device)
        return batch, batch.y
    width = max(1, max(len(x['ids']) for x in samples))
    ids = torch.zeros(len(samples), width, 5, dtype=torch.long)
    pad = torch.ones(len(samples), width, dtype=torch.bool)
    for i, sample in enumerate(samples):
        n = len(sample['ids'])
        ids[i, :n] = sample['ids']
        pad[i, :max(n, 1)] = False
    return (ids.to(device), pad.to(device)), torch.tensor([x['label'] for x in samples], device=device)


def forward(model_id, model, batch):
    if model_id != 1:
        return model(batch).float()
    ids, mask = batch
    with torch.autocast(ids.device.type, dtype=torch.bfloat16, enabled=ids.device.type == 'cuda'):
        return model(ids, padding_mask=mask).float()
