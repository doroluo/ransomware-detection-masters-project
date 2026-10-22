"""Model 1, the token-sequence classifier, on the fold vocabulary of a block export (s03c_cfg_transformer/EXPORT_SCHEMA.md).

Adapted from `src/models/token_sequence.py` of the branch `origin/TokenSeqModel` (Janice Park). Kept from her model: the
learned positional embedding, `nn.TransformerEncoder`, the masked mean pool and the two-layer MLP head, and the
convention that `pad_mask` is True on padding. Changed:
  * her `FiveChannelEmbedder` (opcode, two operands, api, behaviour, each with its own vocabulary) is replaced by
    `InstructionEmbedder`: one embedding table over the fold vocabulary, an instruction position embeds
    E[mn] + E[sig] + E[beh], an API position embeds E[api], plus a 2-way kind embedding;
  * `max_seq_len` defaults to 512 and a window longer than it raises instead of being cut by the slice;
  * `encode(ids, kind, pad_mask) -> [B, d]` returns the window vector; windows with no token at all are encoded as zeros
    (a softmax over nothing would give NaN);
  * `BinaryTokenSequence` encodes every window of a binary and masked-means the window vectors per binary before the head.
"""
import torch
import torch.nn as nn

INSTRUCTION, API = 0, 1


class InstructionEmbedder(nn.Module):
    """ids [B, L, 3] (mn, sig, beh; unused slots 0; an API position has its id in slot 0) and kind [B, L] (0 instruction,
    1 api) -> [B, L, d_model]. Id 0 (PAD) embeds to zero, so a sum over the three slots serves both kinds."""

    def __init__(self, n_ids: int, d_model: int = 64) -> None:
        super().__init__()
        self.n_ids, self.d_model = n_ids, d_model
        self.embedding = nn.Embedding(n_ids, d_model, padding_idx=0)
        self.kind_embedding = nn.Embedding(2, d_model)

    def forward(self, ids: torch.Tensor, kind: torch.Tensor) -> torch.Tensor:
        return self.embedding(ids).sum(dim=-2) + self.kind_embedding(kind)


class Model1TokenSequence(nn.Module):
    """Window encoder and head. `forward` gives one logit pair per window; `encode` gives the pooled window vector."""

    def __init__(self, n_ids: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, num_classes: int = 2,
                 max_seq_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model, self.max_seq_len = d_model, max_seq_len
        self.embedder = InstructionEmbedder(n_ids, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout,
                                           batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.classifier = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(d_model, num_classes))

    def encode(self, ids: torch.Tensor, kind: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """ids [B, L, 3], kind [B, L], pad_mask [B, L] bool (True on padding) -> [B, d_model]: the masked mean of the
        encoder output over the real positions; zeros for a window that is all padding."""
        seq_len = ids.shape[1]
        if seq_len > self.max_seq_len:
            raise ValueError(f'window of {seq_len} positions is longer than max_seq_len {self.max_seq_len}')
        out = self.pos_encoder.new_zeros(ids.shape[0], self.d_model)
        live = ~pad_mask.all(dim=1)
        if not bool(live.any()):
            return out
        x = self.embedder(ids[live], kind[live]) + self.pos_encoder[:, :seq_len, :]
        pad = pad_mask[live]
        x = self.transformer_encoder(x, src_key_padding_mask=pad)
        keep = (~pad).unsqueeze(-1).to(x.dtype)
        pooled = (x * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        return out.index_copy(0, live.nonzero(as_tuple=True)[0], pooled)

    def forward(self, ids: torch.Tensor, kind: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """logits [B, num_classes], one row per window."""
        return self.classifier(self.encode(ids, kind, pad_mask))


class BinaryTokenSequence(nn.Module):
    """All windows of one or more binaries -> logits [n_binaries, num_classes]. Each window is encoded by `model`, the
    window vectors of a binary are averaged over its non-empty windows, and `model.classifier` reads the mean."""

    def __init__(self, model: Model1TokenSequence, window_chunk: int | None = None) -> None:
        super().__init__()
        self.model, self.window_chunk = model, window_chunk

    def encode(self, ids, kind, pad_mask, win2bin, n_binaries: int | None = None) -> torch.Tensor:
        """[n_binaries, d_model]: masked mean of the window vectors per binary (zeros for a binary with no token)."""
        n = n_binaries if n_binaries is not None else (int(win2bin.max()) + 1 if len(win2bin) else 0)
        step = self.window_chunk or max(len(ids), 1)
        parts = [self.model.encode(ids[i:i + step], kind[i:i + step], pad_mask[i:i + step])
                 for i in range(0, len(ids), step)]
        vecs = torch.cat(parts) if parts else self.model.pos_encoder.new_zeros(0, self.model.d_model)
        live = (~pad_mask.all(dim=1)).to(vecs.dtype).unsqueeze(-1)
        total = vecs.new_zeros(n, self.model.d_model).index_add(0, win2bin, vecs * live)
        count = vecs.new_zeros(n, 1).index_add(0, win2bin, live)
        return total / count.clamp(min=1.0)

    def forward(self, ids, kind, pad_mask, win2bin, n_binaries: int | None = None) -> torch.Tensor:
        return self.model.classifier(self.encode(ids, kind, pad_mask, win2bin, n_binaries))
