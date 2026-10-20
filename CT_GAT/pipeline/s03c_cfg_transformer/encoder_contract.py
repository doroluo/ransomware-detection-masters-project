"""The contract between a teammate's per-block encoder and this repository (TRANSFORMER_CONTRACT.md).

BlockEncoder: the interface. save_checkpoint / load_checkpoint / load_encoder: the checkpoint dict. check_checkpoint:
python -m s03c_cfg_transformer.encoder_contract --ckpt RUNS/encoders/<name>/fold0/best.pt --export RUNS/export/<export>
--fold 0, to run before handing a checkpoint over; precompute_blockvec.py runs it again and refuses a failing one.
"""
import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from common.config import EXPORT
from s03c_cfg_transformer.block_export import sha_of_ids
from common.reader import BlockStore, FoldView

CHECKPOINT_KEYS = ('module', 'class', 'init_kwargs', 'state_dict', 'export_sha', 'fold', 'manifest_sha',
                   'train_ids', 'val_ids', 'train_ids_sha', 'val_ids_sha', 'vocab_sha', 'd_out', 'max_len')
# module names that checkpoints written before the stage-3c layout record -> where the class lives now
LEGACY_MODULES = {'transformer_paths.reference_encoder': 's03c_cfg_transformer.encoders.reference',
                  'transformer_paths.codebert_encoder': 's03c_cfg_transformer.encoders.codebert'}
CHECK_BLOCKS = 64
REL_TOL, ABS_TOL = 1e-3, 1e-4


class ContractError(ValueError):
    """A checkpoint breaks the contract; the message lists every failed check."""


@runtime_checkable
class BlockEncoder(Protocol):
    """A pure function of one block's token sequence. forward(tokens [B, L] long fold-vocabulary ids, mask [B, L]
    bool, True on tokens) -> [B, d_out] float. `d_out` is the vector size and `max_len` the longest sequence accepted
    (the export cuts blocks at 192 tokens, so max_len >= 192 loses nothing)."""
    d_out: int
    max_len: int

    def forward(self, tokens, mask):
        ...


def vocab_sha_of(store: BlockStore, fold):
    """sha256 of the fold's global -> fold id table, as recorded in vocab.json."""
    return store.fold(fold)['vocab']['lut_sha']


def importable_module(model):
    """Name under which type(model) can be imported again; a class defined in a script run as __main__ is refused unless
    the script ran as `python -m package.module`."""
    module = type(model).__module__
    if module == '__main__':
        spec = getattr(sys.modules['__main__'], '__spec__', None)
        if spec is None:
            raise ContractError('the encoder class is defined in a script run as __main__; put it in an importable '
                                'module so that load_encoder can import it')
        module = spec.name
    return module


def build_checkpoint(model, init_kwargs, export_root, fold, train_ids, val_ids):
    """The checkpoint dict of a trained encoder: `model` is an instance of a class importable as
    type(model).__module__.type(model).__qualname__, built by `type(model)(**init_kwargs)`. train_ids and val_ids are
    the sample ids the encoder was trained and selected on."""
    store = BlockStore(export_root)
    return {'module': importable_module(model), 'class': type(model).__qualname__, 'init_kwargs': dict(init_kwargs),
            'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
            'export_sha': store.export_sha, 'fold': int(fold), 'manifest_sha': store.fold(fold)['manifest']['manifest_sha'],
            'train_ids': sorted(train_ids), 'val_ids': sorted(val_ids), 'train_ids_sha': sha_of_ids(train_ids),
            'val_ids_sha': sha_of_ids(val_ids), 'vocab_sha': vocab_sha_of(store, fold), 'd_out': int(model.d_out),
            'max_len': int(model.max_len)}


def save_checkpoint(path, model, init_kwargs, export_root, fold, train_ids, val_ids):
    """Write the checkpoint dict of `build_checkpoint` to `path` (parent directories are made)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_checkpoint(model, init_kwargs, export_root, fold, train_ids, val_ids), path)


def load_checkpoint(path):
    """The checkpoint dict (plain types and tensors only: torch.load with weights_only)."""
    return torch.load(path, map_location='cpu', weights_only=True)


def load_encoder(path, device='cpu'):
    """The encoder of a checkpoint on `device`, in eval mode: module.class(**init_kwargs) with the state_dict."""
    ckpt = load_checkpoint(path)
    cls = getattr(importlib.import_module(LEGACY_MODULES.get(ckpt['module'], ckpt['module'])), ckpt['class'])
    model = cls(**ckpt['init_kwargs'])
    model.load_state_dict(ckpt['state_dict'])
    return model.to(device).eval()


def sample_blocks(store: BlockStore, fold, n=CHECK_BLOCKS):
    """(tokens, mask) of n distinct blocks of the fold's training split, spread over the split, and the shortest block
    of the split (about 2% of the blocks are empty: no token at all)."""
    view = FoldView(store, fold, 'train')
    ids, _ = view.distinct_blocks()
    pick = ids[torch.linspace(0, len(ids) - 1, min(n, len(ids))).long().numpy()]
    lengths = np.diff(np.asarray(store.blk_tok_ptr))[ids]
    return view.padded(np.unique(np.append(pick, ids[np.argmin(lengths)])))


@torch.no_grad()
def check_checkpoint(path, export_root, fold, device='cpu'):
    """Load a checkpoint and test it against the export. Checks: the dict has every key; export_sha, fold,
    manifest_sha and vocab_sha match the export; the id lists match their shas; train ids are a subset of the fold's
    train, val ids a subset of its val, and no test id is in either; the encoder loads and reports d_out and max_len;
    its output on real blocks, an empty block included, has shape [B, d_out], is finite and is the same on a second eval pass; it does not
    depend on the other blocks of the batch or on padding. Returns {check: detail} and raises ContractError listing
    every failed check."""
    store = BlockStore(export_root)
    problems, passed = [], {}

    def check(name, ok, detail=''):
        if ok:
            passed[name] = 'ok'
        else:
            problems.append(f'{name}: {detail}')

    ckpt = load_checkpoint(path)
    missing = [k for k in CHECKPOINT_KEYS if k not in ckpt]
    if missing:
        raise ContractError(f'{path}: missing keys {missing}')
    manifest = store.fold(fold)['manifest']
    by_index = lambda split: {store.binaries[i]['sample_id'] for i in manifest[split]}
    train, val, test = by_index('train'), by_index('val'), by_index('test')
    check('export_sha', ckpt['export_sha'] == store.export_sha, f"{ckpt['export_sha'][:12]} vs {store.export_sha[:12]}")
    check('fold', ckpt['fold'] == fold, f"{ckpt['fold']} vs {fold}")
    check('manifest_sha', ckpt['manifest_sha'] == manifest['manifest_sha'], 'the fold manifest differs')
    check('vocab_sha', ckpt['vocab_sha'] == vocab_sha_of(store, fold), 'the fold vocabulary differs')
    check('train_ids_sha', ckpt['train_ids_sha'] == sha_of_ids(ckpt['train_ids']), 'sha does not match train_ids')
    check('val_ids_sha', ckpt['val_ids_sha'] == sha_of_ids(ckpt['val_ids']), 'sha does not match val_ids')
    check('train_ids_in_train', bool(ckpt['train_ids']) and set(ckpt['train_ids']) <= train,
          f"{len(set(ckpt['train_ids']) - train)} ids outside the fold's train")
    check('val_ids_in_val', bool(ckpt['val_ids']) and set(ckpt['val_ids']) <= val,
          f"{len(set(ckpt['val_ids']) - val)} ids outside the fold's val")
    leaked = (set(ckpt['train_ids']) | set(ckpt['val_ids'])) & test
    check('no_test_ids', not leaked, f'{len(leaked)} test ids in train_ids / val_ids')
    try:
        encoder = load_encoder(path, device)
    except Exception as e:
        raise ContractError(f'{path}: cannot load the encoder: {type(e).__name__}: {e}') from e
    check('d_out_max_len', (encoder.d_out, encoder.max_len) == (ckpt['d_out'], ckpt['max_len']),
          f"module has d_out {encoder.d_out}, max_len {encoder.max_len}")
    tokens, mask = (t.to(device) for t in sample_blocks(store, fold))
    tokens, mask = tokens[:, :encoder.max_len], mask[:, :encoder.max_len]
    out = encoder(tokens, mask)
    check('output_shape', tuple(out.shape) == (len(tokens), ckpt['d_out']), f'{tuple(out.shape)}, expected '
          f"({len(tokens)}, {ckpt['d_out']})")
    check('finite', bool(torch.isfinite(out).all()), 'NaN or inf in the output')
    check('eval_deterministic', torch.equal(out, encoder(tokens, mask)), 'two eval passes differ')
    close = lambda a, b: torch.allclose(a.float(), b.float(), rtol=REL_TOL, atol=ABS_TOL)
    one = torch.cat([encoder(tokens[i:i + 1], mask[i:i + 1]) for i in range(min(8, len(tokens)))])
    check('batch_independent', close(one, out[:len(one)]), 'a block alone and in a batch give different vectors')
    room = min(5, encoder.max_len - tokens.shape[1])
    pad = torch.zeros(len(tokens), room, dtype=tokens.dtype, device=device)
    padded = encoder(torch.cat([tokens, pad], 1), torch.cat([mask, pad.bool()], 1)) if room > 0 else out
    check('padding_independent', close(padded, out), 'extra padding changes the vectors')
    if problems:
        raise ContractError(f'{path}: ' + '; '.join(problems))
    return passed


def main():
    ap = argparse.ArgumentParser(description=check_checkpoint.__doc__.split('\n')[0])
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--export', required=True, help='export root, or a name under RUNS/export')
    ap.add_argument('--fold', type=int, required=True)
    args = ap.parse_args()
    export_root = Path(args.export) if Path(args.export).exists() else EXPORT / args.export
    print(json.dumps(check_checkpoint(args.ckpt, export_root, args.fold), indent=1))
    print('contract ok')


if __name__ == '__main__':
    main()
