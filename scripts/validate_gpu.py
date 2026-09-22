#!/usr/bin/env python3
"""Actual full-size CUDA/AMP gates; CPU CI must never stand in for this artifact."""
from pathlib import Path
import json
import sys
import gc
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from src.model import TinyGPT, ModelConfig
from src.data import TokenMemmap
from src.train import _train_step
from src.diagnostics import gradient_check
from src.utils import seed_everything, choose_precision, device_info
from src.evidence import load_config, atomic_json, source_digest, utcnow


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('Full-depth GPU validation requires CUDA; CPU results are not substituted.')
    cfg = load_config(); device = torch.device('cuda')
    dtype, scaling, precision = choose_precision(device, cfg.get('precision', 'auto'))
    existing_path = ROOT / 'results/gpu_correctness.json'
    if existing_path.exists():
        existing = json.loads(existing_path.read_text())
        comparable = ['gpu_name', 'gpu_total_memory_gib', 'torch_version', 'cuda_version']
        current = device_info()
        if (existing.get('source_sha256') == source_digest() and existing.get('precision') == precision
                and all(existing.get('hardware', {}).get(k) == current.get(k) for k in comparable)
                and any(r.get('passed') for r in existing.get('variants', {}).values())):
            print('Reusing matching full-depth GPU validation:', existing_path)
            return
        raise RuntimeError('GPU validation belongs to a different protocol/hardware. Use fresh output directories.')
    rows = {}
    for variant in ['midpoint', 'leapfrog']:
        seed_everything(cfg['seed'])
        model = TinyGPT(ModelConfig(**json.loads((ROOT / f'configs/model_{variant}.json').read_text()))).to(device)
        source = TokenMemmap(cfg['train_path'], model.cfg.seq_len, device, cfg['seed'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['max_lr'], betas=tuple(cfg['betas']), weight_decay=cfg['weight_decay'])
        scaler = torch.amp.GradScaler('cuda') if scaling else None
        try:
            for _ in range(3):
                x, y = source.batch(1)
                _train_step(model, optimizer, scaler, x, y, dtype, cfg['grad_clip'])
            x, y = source.batch(1)
            rows[variant] = gradient_check(model, x, y, dtype)
            rows[variant]['successful_precheck_updates'] = 3
        except (FloatingPointError, RuntimeError) as e:
            rows[variant] = {'passed': False, 'error': f'{type(e).__name__}: {e}'}
        del model, optimizer, scaler, source
        gc.collect(); torch.cuda.empty_cache()
    report = {'schema_version': 2, 'created_utc': utcnow(), 'source_sha256': source_digest(),
              'hardware': device_info(), 'precision': precision, 'variants': rows}
    atomic_json(ROOT / 'results/gpu_correctness.json', report)
    print(json.dumps(report, indent=2))
    if not any(r['passed'] for r in rows.values()):
        raise RuntimeError('No reversible variant passed the full-depth precision gate. Use fp32 for all runs in a fresh directory.')


if __name__ == '__main__': main()
