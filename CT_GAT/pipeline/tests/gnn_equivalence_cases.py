"""The configurations and the computation of the batch / forward / backward equivalence fixture (fixtures/gnn_equivalence.npz).

`compute(api, cache)` runs the data path and one training step of the GNN on a cache and returns {key: array}. `api` names
the classes of one code base: `RunConfig, load_index, make_manifest, TokenVocab, ImportVocab, loader, HierGNN`. The same
function ran on the 3-block-level-counting branch (3764cc7; `fixtures/make_gnn_equivalence.py`) to record the fixture, and
`test_gnn_equivalence.py` runs it on this code base and compares.
"""
import numpy as np
import torch
import torch.nn.functional as F

CASES = {
    'default': {},
    'augmented': dict(mask_blocks=0.15, drop_edges=0.2, flip_edges=0.2),
    'tokens_functions': dict(mask_tokens=0.15, drop_functions=0.15, drop_library=0.3),
    'shuffle': dict(control='shuffle'),
    'chain_mean': dict(control='chain', encoder='mean'),
    'no_edges': dict(control='no_edges'),
    'noapi': dict(vocab='noapi', mask_blocks=0.15),
    'imports': dict(imports='header', readout='gated'),
    'stack_forward': dict(stack_tfidf=True, direction='forward', gnn_fn_layers=2, gnn_bin_layers=1),
    'budget_untyped': dict(budget=100, untyped=True, no_call_mp=True),
}
N_TRAIN_BATCHES, N_EVAL_BATCHES = 3, 2


def config(api, kw):
    return api.RunConfig(model='gnn', protocol='kfold', fold=0, num_workers=0, d_model=16, batch_size=4, **kw)


def flat(prefix, batch, out):
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[f'{prefix}/{key}'] = value.numpy()
        elif key == 'sample_id':
            out[f'{prefix}/{key}'] = np.array(value)
        else:
            out[f'{prefix}/{key}'] = np.asarray(value)


def compute(api, cache):
    out = {}
    index = api.load_index(cache)
    for name, kw in CASES.items():
        cfg = config(api, kw)
        manifest = api.make_manifest(cfg, index)
        vocab = api.TokenVocab(manifest.train, cfg)
        imports = api.ImportVocab(manifest.train, cfg.min_df, cache) if cfg.imports else None
        stack = {r['sample_id']: np.float32(0.1 * i - 1) for i, r in enumerate(manifest.train + manifest.val)} \
            if cfg.stack_tfidf else None
        out[f'{name}/vocab_total'] = np.array(vocab.total)
        out[f'{name}/ids'] = np.array([r['sample_id'] for r in manifest.train + manifest.val + manifest.test])

        def batches(rows, train, n):
            data = api.loader(rows, vocab, cfg, train=train, stack=stack, imports=imports)
            return [b for _, b in zip(range(n), data)]

        train = batches(manifest.train, True, N_TRAIN_BATCHES)
        for i, batch in enumerate(train):
            flat(f'{name}/train{i}', batch, out)
        for i, batch in enumerate(batches(manifest.val, False, N_EVAL_BATCHES)):
            flat(f'{name}/eval{i}', batch, out)

        torch.manual_seed(0)
        model = api.HierGNN(cfg, vocab.total, None, len(imports) if imports is not None else 0)
        model.train()
        torch.manual_seed(1)
        logits = model(train[0])
        loss = F.binary_cross_entropy_with_logits(logits, train[0]['label'])
        loss.backward()
        out[f'{name}/logits'], out[f'{name}/loss'] = logits.detach().numpy(), loss.detach().numpy()
        for pname, p in model.named_parameters():
            out[f'{name}/grad/{pname}'] = np.zeros(p.shape, np.float32) if p.grad is None else p.grad.numpy()
    return out
