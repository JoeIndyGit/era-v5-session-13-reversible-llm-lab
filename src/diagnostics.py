"""Numerical gates in the same precision and context as the measured run."""
import copy
from contextlib import nullcontext
import math
import torch
from torch import nn


def amp_context(device, dtype):
    return torch.autocast(device.type, dtype=dtype) if dtype != torch.float32 else nullcontext()


def thresholds(dtype):
    # Predeclared tolerances, not thresholds fitted to a selected result.
    if dtype == torch.float32:
        return {'gradient_relative_l2': 1e-4, 'reconstruction_relative_l2': 1e-4, 'reconstruction_max_abs': 1e-3}
    return {'gradient_relative_l2': 0.05, 'reconstruction_relative_l2': 0.02, 'reconstruction_max_abs': 0.05}


@torch.no_grad()
def reconstruction_check(model, idx, dtype):
    pos = torch.arange(idx.size(1), device=idx.device)
    with amp_context(idx.device, dtype):
        x = model.tok_emb(idx) + model.pos_emb(pos)[None, :, :]
        report = model.stack.roundtrip_error(x)
    limits = thresholds(dtype)
    report.update({'precision': str(dtype).split('.')[-1], 'context_length': idx.size(1),
                   'batch_size': idx.size(0), 'layers': model.cfg.n_layers, 'thresholds': limits})
    report['passed'] = bool(
        math.isfinite(report['max_abs_error']) and math.isfinite(report['max_relative_l2_error'])
        and report['max_abs_error'] <= limits['reconstruction_max_abs']
        and report['max_relative_l2_error'] <= limits['reconstruction_relative_l2'])
    return report


class AutogradReferenceStack(nn.Module):
    def __init__(self, blocks, variant, h):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.variant, self.h = variant, h

    def forward(self, x):
        previous, current = x, x + self.h * self.blocks[0](x)
        for block in self.blocks[1:]:
            following = (previous + 2 * self.h * block(current) if self.variant == 'midpoint'
                         else 2 * current - previous + self.h ** 2 * block(current))
            previous, current = current, following
        return current


def gradient_check(model, x, y, dtype):
    """Compare every LM parameter gradient, including the tied embedding/head."""
    reference = copy.deepcopy(model)
    reference.stack = AutogradReferenceStack(reference.stack.blocks, model.cfg.reversible_variant, model.cfg.reversible_h)
    model.zero_grad(set_to_none=True); reference.zero_grad(set_to_none=True)
    with amp_context(x.device, dtype):
        actual, loss = model(x, y)
    loss.backward()
    with amp_context(x.device, dtype):
        expected, reference_loss = reference(x, y)
    reference_loss.backward()
    squared_error = squared_reference = max_absolute = 0.0
    all_finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    for (name, p), (other_name, q) in zip(model.named_parameters(), reference.named_parameters()):
        if name != other_name or p.grad is None or q.grad is None:
            raise AssertionError(f'Missing/misaligned gradient: {name} / {other_name}')
        a, b = p.grad.float(), q.grad.float()
        all_finite = all_finite and bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        squared_error += float((a - b).square().sum())
        squared_reference += float(b.square().sum())
        max_absolute = max(max_absolute, float((a - b).abs().max()))
    relative = math.sqrt(squared_error / max(squared_reference, 1e-24))
    recon = reconstruction_check(model, x, dtype)
    report = {'gradient_relative_l2_error': relative, 'gradient_max_abs_error': max_absolute,
              'output_max_abs_error': float((actual.detach().float() - expected.detach().float()).abs().max()),
              'loss': float(loss.detach()), 'reference_loss': float(reference_loss.detach()),
              'all_finite': all_finite, 'reconstruction': recon, 'thresholds': thresholds(dtype),
              'layers': model.cfg.n_layers, 'context_length': x.size(1),
              'parameters': model.num_parameters(), 'precision': str(dtype).split('.')[-1]}
    report['passed'] = bool(all_finite and relative <= thresholds(dtype)['gradient_relative_l2'] and recon['passed'])
    model.zero_grad(set_to_none=True)
    return report
