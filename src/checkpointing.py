"""Atomic, weights-only-loadable checkpoints including all sampling/RNG state."""
from pathlib import Path
import json
import os
import random
import numpy as np
import torch


def rng_state(train_source, val_source):
    state = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': [state[0], state[1].tolist(), state[2], state[3], state[4]],
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        'train_sampler': json.dumps(train_source.rng.bit_generator.state),
        'val_sampler': json.dumps(val_source.rng.bit_generator.state),
    }


def restore_rng(state, train_source, val_source):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
    train_source.rng.bit_generator.state = json.loads(state['train_sampler'])
    val_source.rng.bit_generator.state = json.loads(state['val_sampler'])


def save_checkpoint(path, model, optimizer, scaler, train_source, val_source, progress, signature):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'signature': signature, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'rng': rng_state(train_source, val_source), 'progress': progress,
    }
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer, scaler, train_source, val_source, signature):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['signature'] != signature:
        raise RuntimeError('Checkpoint settings, source, data or hardware differ. Use a new output directory for a new experiment.')
    model.load_state_dict(payload['model'])
    optimizer.load_state_dict(payload['optimizer'])
    if scaler is not None:
        scaler.load_state_dict(payload['scaler'])
    restore_rng(payload['rng'], train_source, val_source)
    return payload['progress']
