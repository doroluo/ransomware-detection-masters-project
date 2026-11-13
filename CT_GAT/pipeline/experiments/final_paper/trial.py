"""Paper-only microbatch accumulation; original model-grid training source is unchanged.

The epoch/checkpoint/metric protocol follows experiments.model_grid.runner.run_trial.
Only sample-weighted microbatch accumulation and optional activation checkpointing differ.
"""
import json
from pathlib import Path
import random
import statistics
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from s04_evaluation.val_threshold import grid_threshold
from experiments.model_grid.runner import (source_hashes, atomic_json, save_checkpoint, batches,
    metrics, predict, write_predictions, rng_state, restore_rng, cpu_state)
from experiments.model_grid.data import digest
from experiments.model_grid.models import Model3, collate, forward, make_model as original_make_model


class CheckpointModel3(Model3):
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
                    z = checkpoint(self.encoder, padded, ~keep, use_reentrant=False)
                outputs.append(self.norm(z.float()).masked_fill((length == 0)[:, None], 0))
            return torch.cat(outputs)


def make_model(model, api_size, spec, seed):
    if model == 3 and spec.get('activation_checkpoint'):
        return CheckpointModel3(api_size, spec, seed)
    return original_make_model(model, api_size, spec, seed)


def run_trial(model_id, spec, seed, epochs, patience, data, out, device, sources, evaluate_test=False, progress_seconds=0):
    out.mkdir(parents=True, exist_ok=True)
    contract = dict(model=model_id, spec=spec, seed=seed, epochs=epochs, patience=patience,
                    dataset_fingerprint=data.fingerprint, source_hashes=sources, evaluate_test=evaluate_test,
                    torch_version=torch.__version__)
    fingerprint = digest(contract)
    if (out/'config.json').exists():
        if json.loads((out/'config.json').read_text()) != contract:
            raise ValueError(f'run settings/input/source changed: choose a new output directory: {out}')
    else:
        atomic_json(out/'config.json', contract)
    if (out/'metrics.json').exists():
        return json.loads((out/'metrics.json').read_text())
    if source_hashes() != sources:
        raise ValueError('source changed during queue; refusing mixed implementations')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    net = make_model(model_id, data.api_size, spec, seed).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=spec['lr'], weight_decay=spec['weight_decay'])
    train_ids = [r['sample_id'] for r in data.splits['train']]
    counts = np.bincount([data.rows[s]['label'] for s in train_ids], minlength=2)
    class_weight = torch.tensor(len(train_ids) / (2 * counts), dtype=torch.float32, device=device)
    history, start_epoch, best_key, stale = [], 1, (-1., -1.), 0
    if (out/'last.pt').exists():
        last = torch.load(out/'last.pt', map_location='cpu', weights_only=False)
        if last['fingerprint'] != fingerprint:
            raise ValueError('checkpoint contract mismatch')
        net.load_state_dict(last['model'])
        opt.load_state_dict(last['optimizer'])
        history, start_epoch, best_key, stale = last['history'], last['epoch'] + 1, tuple(last['best_key']), last['stale']
        restore_rng(last['rng'])
    start_time = time.perf_counter()
    last_progress = 0.

    def progress(phase, force=False, **details):
        nonlocal last_progress
        now = time.perf_counter()
        if progress_seconds and (force or now - last_progress >= progress_seconds):
            row = dict(phase=phase, seed=seed, elapsed_seconds=now - start_time, **details)
            atomic_json(out/'progress.json', row)
            print(json.dumps(dict(run=out.name, **row)), flush=True)
            last_progress = now

    for epoch in range(start_epoch, epochs + 1):
        if stale >= patience:
            break
        if (out.parent.parent/'STOP').exists():
            raise InterruptedError('STOP file: saved checkpoints can be resumed')
        net.train()
        order = np.random.default_rng([seed, epoch, 7919]).permutation(train_ids).tolist()
        tick = time.perf_counter()
        total = 0.
        done = 0
        progress('training', force=True, epoch=epoch, samples_done=done, samples_total=len(order))
        for batch_index, part in enumerate(batches(order, spec['batch_size'])):
            # A logical batch has exactly one optimizer step. Microbatches contribute
            # sample-weighted gradients, including a short final logical batch.
            opt.zero_grad(set_to_none=True)
            for micro_index, micro in enumerate(batches(part, spec.get('micro_batch_size', len(part)))):
                if batch_index == 0 and micro_index == 0:
                    progress('load_and_transfer', force=True, epoch=epoch)
                batch, y = collate(model_id, [data.samples[s] for s in micro], device)
                if batch_index == 0 and micro_index == 0:
                    progress('forward', force=True, epoch=epoch)
                logits = forward(model_id, net, batch)
                weighted = F.cross_entropy(logits, y, reduction='none') * class_weight[y]
                loss = weighted.sum() / len(part)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'nonfinite loss in {out.name}, epoch {epoch}')
                if batch_index == 0 and micro_index == 0:
                    progress('backward', force=True, epoch=epoch)
                loss.backward()
                total += float(weighted.detach().sum())
                del batch, y, logits, loss, weighted
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            done += len(part)
            progress('training', epoch=epoch, samples_done=done, samples_total=len(order),
                     training_seconds=time.perf_counter() - tick)
        progress('validation', force=True, epoch=epoch, samples_done=done, samples_total=len(order))
        val = predict(model_id, net, data, 'val', spec.get('micro_batch_size', spec['batch_size']), device)
        score = metrics(val[1], val[2])
        key = (score['auc'], score['macro_f1'])
        row = dict(epoch=epoch, train_loss=total / len(order), seconds=time.perf_counter() - tick,
                   val_auc=score['auc'], val_macro_f1=score['macro_f1'])
        history.append(row)
        progress('saving_checkpoints', force=True, epoch=epoch)
        if key > best_key:
            best_key, stale = key, 0
            save_checkpoint(out/'best.pt', {'model': cpu_state(net), 'epoch': epoch, 'fingerprint': fingerprint})
        else:
            stale += 1
        save_checkpoint(out/'last.pt', dict(model=cpu_state(net), optimizer=opt.state_dict(), epoch=epoch,
                         history=history, best_key=best_key, stale=stale, rng=rng_state(), fingerprint=fingerprint))
        atomic_json(out/'history.json', history)
        print(json.dumps(dict(run=out.name, seed=seed, **row)), flush=True)
        progress('epoch_complete', force=True, **row)
    best = torch.load(out/'best.pt', map_location='cpu', weights_only=False)
    net.load_state_dict(best['model'])
    val = predict(model_id, net, data, 'val', spec.get('micro_batch_size', spec['batch_size']), device)
    threshold = grid_threshold(np.array(val[1]), np.array(val[2]))
    result = dict(spec=spec, seed=seed, best_epoch=best['epoch'], epochs_run=len(history),
                  validation=metrics(val[1], val[2], threshold), validation_fixed=metrics(val[1], val[2]),
                  epoch_seconds_mean=statistics.mean(r['seconds'] for r in history),
                  invocation_seconds=time.perf_counter() - start_time,
                  device=str(device), gpu_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                  peak_gpu_mb=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0,
                  parameters=sum(p.numel() for p in net.parameters()), fingerprint=fingerprint)
    write_predictions(out/'validation_predictions.csv', data, val, 'val')
    if evaluate_test:
        data.load_test()
        test = predict(model_id, net, data, 'test', spec.get('micro_batch_size', spec['batch_size']), device)
        result['test'] = metrics(test[1], test[2], threshold)
        write_predictions(out/'test_predictions.csv', data, test, 'test')
    atomic_json(out/'metrics.json', result)
    del net, opt
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result
