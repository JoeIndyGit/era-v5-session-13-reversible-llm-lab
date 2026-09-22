from typing import Sequence
import torch
import torch.nn as nn


def _param_slices(blocks, params):
    slices = []
    offset = 0
    for block in blocks:
        n = len(list(block.parameters()))
        slices.append((offset, offset + n))
        offset += n
    if offset != len(params):
        raise RuntimeError("Parameter flattening mismatch in reversible stack.")
    return slices


def _autocast_state(x):
    device_type = x.device.type
    enabled = torch.is_autocast_enabled(device_type)
    dtype = torch.get_autocast_dtype(device_type) if device_type in ("cuda", "cpu") else None
    return device_type, enabled, dtype


class _ReversibleMidpointFn(torch.autograd.Function):
    """Memory-efficient explicit-midpoint recurrence with Euler bootstrap."""
    @staticmethod
    def forward(ctx, x, *args):
        blocks, h = args[-2], float(args[-1])
        params = args[:-2]
        slices = _param_slices(blocks, params)
        device_type, autocast_enabled, autocast_dtype = _autocast_state(x)

        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=autocast_dtype,
                                enabled=autocast_enabled and device_type in ("cuda", "cpu")):
                p0 = x
                p1 = p0 + h * blocks[0](p0)
                p_prev, p_cur = p0, p1
                for layer in range(1, len(blocks)):
                    p_next = p_prev + (2.0 * h) * blocks[layer](p_cur)
                    p_prev, p_cur = p_cur, p_next

        ctx.blocks, ctx.h, ctx.params, ctx.slices = blocks, h, params, slices
        ctx.device_type, ctx.autocast_enabled, ctx.autocast_dtype = device_type, autocast_enabled, autocast_dtype
        ctx.save_for_backward(p_prev, p_cur)
        return p_cur

    @staticmethod
    def backward(ctx, grad_output):
        blocks, h, params, slices = ctx.blocks, ctx.h, ctx.params, ctx.slices
        p_penultimate, p_final = ctx.saved_tensors
        p_next, p_cur = p_final.detach(), p_penultimate.detach()
        g_next, g_cur = grad_output, torch.zeros_like(p_cur)
        param_grads = [None] * len(params)

        for layer in range(len(blocks) - 1, 0, -1):
            start, end = slices[layer]
            layer_params = list(params[start:end])
            p_req = p_cur.detach().requires_grad_(True)
            with torch.enable_grad():
                with torch.autocast(device_type=ctx.device_type, dtype=ctx.autocast_dtype,
                                    enabled=ctx.autocast_enabled and ctx.device_type in ("cuda", "cpu")):
                    delta = blocks[layer](p_req)
            grads = torch.autograd.grad(
                delta, [p_req] + layer_params,
                grad_outputs=(2.0 * h) * g_next,
                allow_unused=True,
            )
            for j, g in enumerate(grads[1:], start=start):
                param_grads[j] = g
            with torch.no_grad():
                p_older = p_next - (2.0 * h) * delta.detach()
            p_next, p_cur = p_cur, p_older
            g_next, g_cur = g_cur + grads[0], g_next

        start, end = slices[0]
        layer_params = list(params[start:end])
        p0_req = p_cur.detach().requires_grad_(True)
        with torch.enable_grad():
            with torch.autocast(device_type=ctx.device_type, dtype=ctx.autocast_dtype,
                                enabled=ctx.autocast_enabled and ctx.device_type in ("cuda", "cpu")):
                delta0 = blocks[0](p0_req)
        grads = torch.autograd.grad(delta0, [p0_req] + layer_params,
                                    grad_outputs=h * g_next, allow_unused=True)
        for j, g in enumerate(grads[1:], start=start):
            param_grads[j] = g
        grad_x = g_cur + g_next + grads[0]
        return (grad_x, *param_grads, None, None)


class _ReversibleLeapfrogFn(torch.autograd.Function):
    """Memory-efficient leapfrog recurrence with Euler bootstrap."""
    @staticmethod
    def forward(ctx, x, *args):
        blocks, h = args[-2], float(args[-1])
        params = args[:-2]
        slices = _param_slices(blocks, params)
        device_type, autocast_enabled, autocast_dtype = _autocast_state(x)
        h2 = h * h

        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=autocast_dtype,
                                enabled=autocast_enabled and device_type in ("cuda", "cpu")):
                p0 = x
                p1 = p0 + h * blocks[0](p0)
                p_prev, p_cur = p0, p1
                for layer in range(1, len(blocks)):
                    p_next = 2.0 * p_cur - p_prev + h2 * blocks[layer](p_cur)
                    p_prev, p_cur = p_cur, p_next

        ctx.blocks, ctx.h, ctx.params, ctx.slices = blocks, h, params, slices
        ctx.device_type, ctx.autocast_enabled, ctx.autocast_dtype = device_type, autocast_enabled, autocast_dtype
        ctx.save_for_backward(p_prev, p_cur)
        return p_cur

    @staticmethod
    def backward(ctx, grad_output):
        blocks, h, params, slices = ctx.blocks, ctx.h, ctx.params, ctx.slices
        h2 = h * h
        p_penultimate, p_final = ctx.saved_tensors
        p_next, p_cur = p_final.detach(), p_penultimate.detach()
        g_next, g_cur = grad_output, torch.zeros_like(p_cur)
        param_grads = [None] * len(params)

        for layer in range(len(blocks) - 1, 0, -1):
            start, end = slices[layer]
            layer_params = list(params[start:end])
            p_req = p_cur.detach().requires_grad_(True)
            with torch.enable_grad():
                with torch.autocast(device_type=ctx.device_type, dtype=ctx.autocast_dtype,
                                    enabled=ctx.autocast_enabled and ctx.device_type in ("cuda", "cpu")):
                    delta = blocks[layer](p_req)
            grads = torch.autograd.grad(
                delta, [p_req] + layer_params,
                grad_outputs=h2 * g_next,
                allow_unused=True,
            )
            for j, g in enumerate(grads[1:], start=start):
                param_grads[j] = g
            with torch.no_grad():
                p_older = 2.0 * p_cur - p_next + h2 * delta.detach()
            new_g_cur = g_cur + 2.0 * g_next + grads[0]
            new_g_older = -g_next
            p_next, p_cur = p_cur, p_older
            g_next, g_cur = new_g_cur, new_g_older

        start, end = slices[0]
        layer_params = list(params[start:end])
        p0_req = p_cur.detach().requires_grad_(True)
        with torch.enable_grad():
            with torch.autocast(device_type=ctx.device_type, dtype=ctx.autocast_dtype,
                                enabled=ctx.autocast_enabled and ctx.device_type in ("cuda", "cpu")):
                delta0 = blocks[0](p0_req)
        grads = torch.autograd.grad(delta0, [p0_req] + layer_params,
                                    grad_outputs=h * g_next, allow_unused=True)
        for j, g in enumerate(grads[1:], start=start):
            param_grads[j] = g
        grad_x = g_cur + g_next + grads[0]
        return (grad_x, *param_grads, None, None)


class _ReversibleStackBase(nn.Module):
    fn = None
    variant = None
    def __init__(self, blocks: Sequence[nn.Module], h: float = 0.25):
        super().__init__()
        if len(blocks) < 2:
            raise ValueError("Reversible stack requires at least two blocks.")
        self.blocks = nn.ModuleList(blocks)
        self.h = float(h)

    def forward(self, x):
        params = tuple(p for block in self.blocks for p in block.parameters())
        return self.fn.apply(x, *params, tuple(self.blocks), self.h)


class ReversibleMidpointStack(_ReversibleStackBase):
    fn = _ReversibleMidpointFn
    variant = "midpoint"

    @torch.no_grad()
    def roundtrip_error(self, x: torch.Tensor):
        states = [x.detach().clone()]
        p1 = x + self.h * self.blocks[0](x)
        states.append(p1.detach().clone())
        p_prev, p_cur = x, p1
        for layer in range(1, len(self.blocks)):
            p_next = p_prev + (2.0 * self.h) * self.blocks[layer](p_cur)
            states.append(p_next.detach().clone())
            p_prev, p_cur = p_cur, p_next
        rec_next, rec_cur = states[-1], states[-2]
        reconstructed = [None] * len(states)
        reconstructed[-1], reconstructed[-2] = rec_next, rec_cur
        for layer in range(len(self.blocks) - 1, 0, -1):
            rec_older = rec_next - (2.0 * self.h) * self.blocks[layer](rec_cur)
            reconstructed[layer - 1] = rec_older
            rec_next, rec_cur = rec_cur, rec_older
        return _error_report(states, reconstructed)


class ReversibleLeapfrogStack(_ReversibleStackBase):
    fn = _ReversibleLeapfrogFn
    variant = "leapfrog"

    @torch.no_grad()
    def roundtrip_error(self, x: torch.Tensor):
        h2 = self.h * self.h
        states = [x.detach().clone()]
        p1 = x + self.h * self.blocks[0](x)
        states.append(p1.detach().clone())
        p_prev, p_cur = x, p1
        for layer in range(1, len(self.blocks)):
            p_next = 2.0 * p_cur - p_prev + h2 * self.blocks[layer](p_cur)
            states.append(p_next.detach().clone())
            p_prev, p_cur = p_cur, p_next
        rec_next, rec_cur = states[-1], states[-2]
        reconstructed = [None] * len(states)
        reconstructed[-1], reconstructed[-2] = rec_next, rec_cur
        for layer in range(len(self.blocks) - 1, 0, -1):
            rec_older = 2.0 * rec_cur - rec_next + h2 * self.blocks[layer](rec_cur)
            reconstructed[layer - 1] = rec_older
            rec_next, rec_cur = rec_cur, rec_older
        return _error_report(states, reconstructed)


def _error_report(states, reconstructed):
    errors = [(a.float() - b.float()).abs().max().item() for a, b in zip(states, reconstructed)]
    return {
        "max_abs_error": max(errors),
        "mean_layer_max_abs_error": sum(errors) / len(errors),
        "per_state_max_abs_error": errors,
    }
