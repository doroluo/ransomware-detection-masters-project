"""Model 3 trained end to end: Janice's transformer encodes the instructions of every basic block, Dorothy's GAT combines the block
vectors over the CFG, and one optimiser trains both (`experiments/team_pipelines/train_b.py --arm sig500v3-e2e`).

The frozen Model 3 of `s03b_cfg_feature` (arm `sig500v3-emb`) feeds the GAT the block embeddings that the build pooled with an
untrained `FiveChannelEmbedder`. Here the block vector is learned: for each block, the first `block_len` instructions (five token
ids each, the build's `instruction_ids`) go through `Model1TokenSequence` (`s03a_transformer/five_channel/model.py`, Janice Park's
file, unchanged: `embedder`, `pos_encoder`, `transformer_encoder`; her `classifier` is built with the module and never called),
the masked mean over the block's instructions is her pooling (`forward`, the last lines before `classifier`), a `LayerNorm` follows,
and `FullBlockGAT` (`s03b_cfg_feature/gat`, Dorothy Luo's file, unchanged, `in_dim=64`, `edge_dim=8`) takes the vectors as `x`.

  * `BlockSeqData`: one binary, with `insn_ids [M, 5]` (the first `block_len` instructions of each block, blocks in order, API ids
    already mapped to the fold's training vocabulary), `insn_block [M]` (the block each instruction belongs to), the CFG and `y`.
    `Batch.from_data_list` offsets `insn_block` by the nodes of the graphs before it (`__inc__`), so a batch is one flat list of
    instructions plus the graph.
  * `pad_blocks`: the flat list as `[n_blocks, L, 5]` ids and a `[n_blocks, L]` mask (True = padding), `L` the longest block of the
    batch up to `block_len`; no loop over blocks. A block with no instruction keeps one unmasked PAD row, so no row of the mask is
    entirely True (an all-masked row gives NaN in `TransformerEncoder`; the `fivechannel` trainer does the same for a binary).
  * `encode_blocks(m1, ids, pad_mask, lengths)`: `Model1TokenSequence.forward` up to the pooling, one vector per block; the vector
    of an empty block is zero.
  * `Model3EndToEnd`: the encoder under bf16 autocast on CUDA (as `fivechannel`), then `LayerNorm(64)`, the vector of an empty
    block set to zero again, and the GAT in fp32 on the batch whose `x` is now the block vectors.
  * `make_e2e_graph`: one build payload to a `BlockSeqData` (the API ids outside `seen` become UNK_API, as in `train_b.remap_api`;
    the 77 block features and the stored block embeddings are not read).

Register names and operand widths are never features: the ids are the build's five channels, as for `fivechannel`.
"""
import torch
import torch.nn as nn
from torch_geometric.data import Data

from experiments.team_pipelines.train_b import remap_api
from s03a_transformer.five_channel.model import Model1TokenSequence
from s03b_cfg_feature.gat.gat_full_model import FullBlockGAT

BLOCK_LEN = 64            # instructions per block the encoder reads (a longer block is cut: 0.97% of the blocks, 11% of the instructions)
D_MODEL = 64
AUTOCAST_ON_CPU = False      # `experiments.predict` sets it to rescore a run that was trained under bf16 autocast on cuda with bf16 on cpu
ENCODER = 'Model1TokenSequence d64 h4 L2'
GAT = 'FullBlockGAT hidden128 heads4 layers2'


class BlockSeqData(Data):
    """One binary for the end-to-end model: `insn_ids [M, 5]`, `insn_block [M]` (local block index), `edge_index`, `edge_attr`, `y`
    and `num_nodes` (the blocks). `insn_block` is offset by the blocks of the earlier graphs when a batch is collated."""

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'insn_block':
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)


def rank_in_block(insn_block, n_blocks):
    """(order, block, position, counts) of a flat instruction list: `order` sorts the instructions by block, keeping the order within
    a block (stable); `block` and `position` are the block and the index within the block of each sorted instruction; `counts` the
    instructions per block. The list need not be sorted already."""
    insn_block = insn_block.long()
    order = torch.argsort(insn_block, stable=True)
    block = insn_block[order]
    counts = torch.bincount(insn_block, minlength=n_blocks)
    starts = torch.cumsum(counts, 0) - counts
    position = torch.arange(block.numel(), device=block.device) - starts[block]
    return order, block, position, counts


def pad_blocks(insn_ids, insn_block, n_blocks, block_len=BLOCK_LEN):
    """(ids [n_blocks, L, 5], pad_mask [n_blocks, L] bool, lengths [n_blocks]) of a flat instruction list: the first `block_len`
    instructions of each block, in order, zero rows after them. `L` is the longest block (at most `block_len`, at least 1).
    `pad_mask` is True on padding; a block without instructions has one unmasked PAD row and length 0."""
    order, block, position, counts = rank_in_block(insn_block, n_blocks)
    lengths = counts.clamp(max=block_len)
    width = max(1, int(lengths.max())) if n_blocks else 1
    ids = torch.zeros(n_blocks, width, insn_ids.shape[1], dtype=insn_ids.dtype, device=insn_ids.device)
    keep = position < block_len
    ids[block[keep], position[keep]] = insn_ids[order][keep]
    pad_mask = torch.arange(width, device=ids.device)[None, :] >= lengths[:, None]
    pad_mask[lengths == 0, 0] = False
    return ids, pad_mask, lengths


def encode_blocks(m1, ids, pad_mask, lengths=None):
    """One vector per block, [n_blocks, d_model]: `Model1TokenSequence.forward` of `m1` on `ids [n_blocks, L, 5]` and `pad_mask`
    (True = padding) up to and without its `classifier`. With `lengths` (from `pad_blocks`) the vector of a block of length 0 is
    zero."""
    x = m1.embedder(ids)
    x = x + m1.pos_encoder[:, :ids.shape[1], :]
    x = m1.transformer_encoder(x, src_key_padding_mask=pad_mask)
    mask = (~pad_mask).unsqueeze(-1).float()
    pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    if lengths is not None:
        pooled = pooled.masked_fill((lengths == 0)[:, None], 0.0)
    return pooled


class Model3EndToEnd(nn.Module):
    """Janice's block encoder, a LayerNorm and Dorothy's GAT in one module. `forward(batch)` takes a collated `BlockSeqData` batch
    and returns the GAT's logits [n_graphs, 2]; it sets `batch.x` to the block vectors."""

    def __init__(self, api_vocab_size, block_len=BLOCK_LEN, edge_dim=8):
        super().__init__()
        self.block_len = int(block_len)
        self.encoder = Model1TokenSequence(api_vocab_size=api_vocab_size, d_model=D_MODEL, nhead=4, num_layers=2,
                                           max_seq_len=self.block_len)
        self.norm = nn.LayerNorm(D_MODEL)
        self.gat = FullBlockGAT(in_dim=D_MODEL, edge_dim=edge_dim)

    def block_vectors(self, batch):
        n_blocks = int(batch.num_nodes)
        ids, pad_mask, lengths = pad_blocks(batch.insn_ids, batch.insn_block, n_blocks, self.block_len)
        device = ids.device
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda' or AUTOCAST_ON_CPU):
            pooled = encode_blocks(self.encoder, ids, pad_mask, lengths)
        return self.norm(pooled.float()).masked_fill((lengths == 0)[:, None], 0.0)

    def forward(self, batch):
        batch.x = self.block_vectors(batch)
        return self.gat(batch)

    def n_parameters(self):
        """{'encoder', 'gat', 'total', 'unused_classifier'}: the trained parameters of the block encoder (embedder, positions,
        transformer layers) and its LayerNorm, of the GAT, their sum, and the parameters of Janice's `classifier`, which this
        model builds with her class and never calls (no gradient, never updated)."""
        count = lambda module: sum(p.numel() for p in module.parameters())              # noqa: E731
        unused = count(self.encoder.classifier)
        encoder = count(self.encoder) - unused + count(self.norm)
        gat = count(self.gat)
        return {'encoder': encoder, 'gat': gat, 'total': encoder + gat, 'unused_classifier': unused}


def make_e2e_graph(pp, gl, payload, seen, block_len=BLOCK_LEN):
    """The `BlockSeqData` of one build payload. The API ids that are not in `seen` become UNK_API (`seen=None`: the stored ids);
    each block keeps its first `block_len` instructions (blocks in order, the order within a block kept); the edges are
    `graph_full_loader`'s (one-hot relation + back-edge bit, edge_dim 8). `n_instructions` (before the cut), `n_tokens` (after) and
    `n_truncated` (blocks longer than `block_len`) are recorded on the graph."""
    ids = payload['instruction_ids'] if seen is None else remap_api(pp.tok, payload['instruction_ids'], seen)
    n_blocks = int(payload['block_features'].shape[0])
    order, block, position, counts = rank_in_block(payload['insn_to_block'], n_blocks)
    keep = position < block_len
    data = BlockSeqData(insn_ids=ids[order][keep].long(), insn_block=block[keep], edge_index=payload['edge_index'].long(),
                        edge_attr=gl._edge_attr(payload), y=torch.tensor(int(payload['label']), dtype=torch.long),
                        num_nodes=n_blocks)
    data.edge_type = payload['edge_type'].long()
    data.sample_id = str(payload['file_id'])
    data.n_instructions = int(ids.shape[0])
    data.n_tokens = int(keep.sum())
    data.n_truncated = int((counts > block_len).sum())
    return data
