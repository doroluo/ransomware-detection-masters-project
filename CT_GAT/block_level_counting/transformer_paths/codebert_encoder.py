"""CodeBERT block encoder built on the strings of our block export (TRANSFORMER_CONTRACT.md, team CodeBERT arm).

`CodeBertBlockEncoder` turns the fold-vocabulary ids of one block back into the vocabulary strings of the CFG cache and
embeds that text with the frozen `microsoft/codebert-base`. It never sees an operand: the strings are the signature
(mnemonic and operand categories, e.g. `xor r,r`) and API reference (e.g. `kernel32.dll!CreateFileW`) strings of the
cache vocabulary, so no register name and no address can reach the model.

Text rule. The text of a block is its signature and API strings in sequence order, joined with ' ; ':
  * a `sig` token is the string of its signature (`xor r,r`), an `api` token the string of its API reference, verbatim
    from `<cache>/vocab.json` (the export layout puts every signature before every API reference);
  * a token mapped to its kind's UNK id, and the specials CLS, SEP and MASK, give the placeholder `<unk>`;
  * PAD is dropped, and so are the `mn` and `beh` tokens (the signature string already carries the mnemonic, and a
    behaviour id is not text);
  * an empty block gives the empty text, which CodeBERT reads as `<s></s>`.
Example: `xor r,r ; mov r,m ; kernel32.dll!CreateFileW`.

Encoding. The text goes through the CodeBERT tokenizer (truncated to `max_subword_len` subwords, `<s>` and `</s>`
included), the frozen model runs in eval mode (fp16 on CUDA, fp32 on CPU, autocast switched off so that the caller's
bf16 autocast does not change it), the hidden state of `layer` (default -1, the last) is averaged over the non-pad
subwords in fp32, and the result is a float32 vector of 768 numbers. Long inputs are cut at the end, so for a block with
more than `max_subword_len` subwords the last strings are not seen.

The CodeBERT weights are not parameters of this module: they are held outside the registered parameters and loaded from
the local Hugging Face cache (`HF_HOME`, `local_files_only`) when the module is built, so `state_dict` is empty and a
checkpoint is a few hundred kilobytes. `init_kwargs` records the revision of the model and the sha256 of its
model.safetensors, and loading refuses weights with another sha.

    HF_HOME=C:/Users/chaoa/Downloads/hf_home python -m transformer_paths.codebert_encoder --write-checkpoints \
        --export ours_cap0 --name codebert
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common.config import RUNS
from transformer_paths.encoder_contract import save_checkpoint
from transformer_paths.reader import BlockStore

MODEL_NAME = 'microsoft/codebert-base'
SEPARATOR = ' ; '
PLACEHOLDER = '<unk>'
SPECIAL_IDS = (1, 2, 3)       # CLS, SEP, MASK; PAD (0) is dropped
TEXT_KINDS = ('sig', 'api')   # the kinds that become text
BATCH_SUBWORDS = 1 << 16      # subwords per CodeBERT forward: bounds the GPU memory, does not change the vectors


def hub_dir(model_name):
    home = os.environ.get('HF_HOME')
    if not home:
        raise RuntimeError('HF_HOME is not set; it must point at the Hugging Face home that holds the CodeBERT files')
    return Path(home) / 'hub' / ('models--' + model_name.replace('/', '--'))


def weights_file(model_name):
    """The model.safetensors of the local Hugging Face cache (any snapshot that has one)."""
    found = sorted(hub_dir(model_name).glob('snapshots/*/model.safetensors'))
    if not found:
        raise FileNotFoundError(f'no model.safetensors for {model_name} under {hub_dir(model_name)}')
    return found[0]


def model_revision(model_name):
    """Commit sha of the main snapshot of the local cache (the config and tokenizer files)."""
    return (hub_dir(model_name) / 'refs' / 'main').read_text().strip()


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def fold_pieces(store: BlockStore, fold):
    """Object array over the fold ids: the text piece of each id ('' = not text). A kept signature or API token gives
    its cache vocabulary string, the kind's UNK id and CLS / SEP / MASK give PLACEHOLDER, everything else ('' for PAD,
    mn and beh tokens) is dropped."""
    cache = Path(store.meta['cache'])
    text = (cache / 'vocab.json').read_text(encoding='utf-8')
    if hashlib.sha256(text.encode('utf-8')).hexdigest() != store.meta['vocab_sha']:
        raise ValueError(f'{cache / "vocab.json"} is not the vocabulary this export was made with (vocab_sha differs)')
    strings = json.loads(text)
    data = store.fold(fold)
    vocab, lut = data['vocab'], data['lut']
    pieces = np.full(vocab['n_ids'], '', dtype=object)
    pieces[list(SPECIAL_IDS)] = PLACEHOLDER
    for kind in TEXT_KINDS:
        offset, size = vocab['global_offsets'][kind], len(strings[kind])
        if store.meta['global_vocab']['sizes'][kind] != size:
            raise ValueError(f'{kind}: {size} strings but the export has {store.meta["global_vocab"]["sizes"][kind]}')
        unk = vocab['kinds'][kind]['unk']
        pieces[unk] = PLACEHOLDER
        for g, s in enumerate(strings[kind]):
            if lut[offset + g] != unk:
                pieces[lut[offset + g]] = s
    return pieces


def block_text(pieces, ids):
    """The text of one block: `ids` are its fold-vocabulary ids (padding already removed), `pieces` the table of
    `fold_pieces`."""
    return SEPARATOR.join(p for p in pieces[np.asarray(ids, dtype=np.int64)] if p)


class CodeBertBlockEncoder(nn.Module):
    """Frozen CodeBERT on the text of a block (module docstring has the text rule); satisfies BlockEncoder."""

    def __init__(self, export, fold, model_name=MODEL_NAME, max_subword_len=256, layer=-1, pooling='mean',
                 revision=None, weights_sha256=None, batch_subwords=BATCH_SUBWORDS, max_len=192):
        super().__init__()
        if pooling != 'mean':
            raise ValueError(f"pooling {pooling!r}: only 'mean' (over the non-pad subwords) is implemented")
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        self.d_out, self.max_len = 768, max_len
        self.max_subword_len, self.layer, self.batch_subwords = int(max_subword_len), int(layer), int(batch_subwords)
        store = BlockStore(export)
        self._pieces = fold_pieces(store, int(fold))
        self._lut_size = len(self._pieces)
        weights = weights_file(model_name)
        digest = sha256_of(weights)
        if weights_sha256 is not None and digest != weights_sha256:
            raise ValueError(f'{weights}: sha256 {digest} but the checkpoint was made with {weights_sha256}')
        if revision is not None and model_revision(model_name) != revision:
            raise ValueError(f'{model_name}: revision {model_revision(model_name)} but the checkpoint has {revision}')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        config = AutoConfig.from_pretrained(model_name, local_files_only=True)
        from safetensors.torch import load_file
        hf = AutoModel.from_config(config)
        hf.load_state_dict(load_file(str(weights)))
        hf.eval().requires_grad_(False)
        object.__setattr__(self, '_hf', hf)     # not a registered submodule: stays out of state_dict
        self.hf_device = torch.device('cpu')

    def texts(self, tokens, mask):
        """The text of each block of a batch (module docstring)."""
        tokens, mask = np.asarray(tokens.cpu()), np.asarray(mask.cpu())
        out = []
        for row, keep in zip(tokens, mask):
            ids = row[keep]
            if len(ids) and (ids.min() < 0 or ids.max() >= self._lut_size):
                raise ValueError('a token id is outside the fold vocabulary')
            out.append(block_text(self._pieces, ids))
        return out

    def _on(self, device):
        if self.hf_device != device:
            dtype = torch.float16 if device.type == 'cuda' else torch.float32
            self._hf.to(device=device, dtype=dtype)
            self.hf_device = device
        return self._hf

    @torch.no_grad()
    def forward(self, tokens, mask):
        device = tokens.device
        hf = self._on(device)
        enc = self.tokenizer(self.texts(tokens, mask), truncation=True, max_length=self.max_subword_len,
                             padding=False)
        ids = enc['input_ids']
        lengths = np.array([len(x) for x in ids])
        out = torch.empty(len(ids), self.d_out, dtype=torch.float32, device=device)
        order = np.argsort(lengths, kind='stable')
        pad_id = self.tokenizer.pad_token_id
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and (end - start + 1) * lengths[order[end]] <= self.batch_subwords:
                end += 1
            sel = order[start:end]
            width = int(lengths[sel].max())
            batch = torch.full((len(sel), width), pad_id, dtype=torch.long)
            attn = torch.zeros(len(sel), width, dtype=torch.long)
            for r, i in enumerate(sel):
                batch[r, :lengths[i]], attn[r, :lengths[i]] = torch.tensor(ids[i]), 1
            batch, attn = batch.to(device), attn.to(device)
            with torch.autocast(device.type, enabled=False):
                res = hf(input_ids=batch, attention_mask=attn, output_hidden_states=self.layer != -1)
                hidden = res.last_hidden_state if self.layer == -1 else res.hidden_states[self.layer]
                w = attn.unsqueeze(-1).float()
                out[torch.as_tensor(sel, device=device)] = (hidden.float() * w).sum(1) / w.sum(1)
            start = end
        return out


def init_kwargs_for(export_root, fold, **overrides):
    """The init_kwargs of a checkpoint: the export, the fold, the model, and the revision and sha of its weights."""
    model_name = overrides.get('model_name', MODEL_NAME)
    return {'export': str(export_root), 'fold': int(fold), 'model_name': model_name, 'max_subword_len': 256,
            'layer': -1, 'pooling': 'mean', 'revision': model_revision(model_name),
            'weights_sha256': sha256_of(weights_file(model_name)), 'batch_subwords': BATCH_SUBWORDS, 'max_len': 192,
            **overrides}


def write_checkpoints(export_root, name, folds=range(5), out_root=None, **overrides):
    """RUNS/encoders/<name>/fold{k}/best.pt per fold (encoder_contract.build_checkpoint, the fold manifest's train and
    val ids); returns the paths. Nothing is trained: the encoder is the frozen CodeBERT of the init_kwargs."""
    store = BlockStore(export_root)
    paths = []
    for k in folds:
        manifest = store.fold(k)['manifest']
        ids = lambda split: [store.binaries[i]['sample_id'] for i in manifest[split]]
        init = init_kwargs_for(export_root, k, **overrides)
        path = Path(out_root or RUNS / 'encoders') / name / f'fold{k}' / 'best.pt'
        save_checkpoint(path, CodeBertBlockEncoder(**init), init, export_root, k, ids('train'), ids('val'))
        paths.append(path)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--write-checkpoints', action='store_true', required=True)
    ap.add_argument('--export', required=True, help='export root, or a name under RUNS/export')
    ap.add_argument('--name', default='codebert')
    ap.add_argument('--folds', type=int, nargs='*', default=list(range(5)))
    ap.add_argument('--max-subword-len', type=int, default=256)
    ap.add_argument('--layer', type=int, default=-1)
    args = ap.parse_args()
    root = Path(args.export) if Path(args.export).exists() else RUNS / 'export' / args.export
    for path in write_checkpoints(root, args.name, args.folds, max_subword_len=args.max_subword_len, layer=args.layer):
        print(path, f'{path.stat().st_size / 1e6:.2f} MB')


if __name__ == '__main__':
    main()
