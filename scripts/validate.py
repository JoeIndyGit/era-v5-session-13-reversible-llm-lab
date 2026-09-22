import copy
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.data import TokenMemmap, count_valid_targets
from src.model import ModelConfig, TinyGPT, TransformerDelta
from src.reversible import ReversibleMidpointStack, ReversibleLeapfrogStack
from src.train import cosine_lr


def parameter_count_test():
    cfg = ModelConfig()
    model = TinyGPT(cfg)
    expected = 20_000_768
    actual = model.num_parameters()
    assert actual == expected, (actual, expected)
    breakdown = model.parameter_breakdown()
    block_share = breakdown["transformer_blocks"] / actual
    assert block_share > 0.85, breakdown
    print(f"PASS parameter count: {actual:,} ({actual/1e6:.6f}M); {block_share:.1%} is transformer blocks")


def architecture_parity_test():
    torch.manual_seed(123); b = TinyGPT(ModelConfig(reversible=False))
    for variant in ("midpoint", "leapfrog"):
        torch.manual_seed(123); r = TinyGPT(ModelConfig(reversible=True, reversible_variant=variant))
        assert b.num_parameters() == r.num_parameters()
        for (bn, bp), (rn, rp) in zip(b.named_parameters(), r.named_parameters()):
            assert bp.shape == rp.shape, (bn, rn, bp.shape, rp.shape)
            assert torch.equal(bp, rp), f"Initialization mismatch: {bn} vs {rn}"
    print("PASS baseline/midpoint/leapfrog initial parameter tensors are identical")


def _gradient_equivalence(variant):
    torch.manual_seed(7)
    cfg = ModelConfig(vocab_size=128, seq_len=16, d_model=32, n_heads=4,
                      n_layers=4, d_ff=64, reversible=False)
    blocks_custom = torch.nn.ModuleList([TransformerDelta(cfg) for _ in range(cfg.n_layers)])
    blocks_naive = copy.deepcopy(blocks_custom)
    x1 = torch.randn(2, 12, cfg.d_model, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    h = 0.25
    Stack = ReversibleMidpointStack if variant == "midpoint" else ReversibleLeapfrogStack
    y1 = Stack(blocks_custom, h=h)(x1); y1.square().mean().backward()

    p0=x2; p1=p0+h*blocks_naive[0](p0); p_prev,p_cur=p0,p1
    for layer in range(1, len(blocks_naive)):
        if variant == "midpoint":
            p_next = p_prev + 2*h*blocks_naive[layer](p_cur)
        else:
            p_next = 2*p_cur - p_prev + (h*h)*blocks_naive[layer](p_cur)
        p_prev,p_cur=p_cur,p_next
    y2=p_cur; y2.square().mean().backward()
    out_err=(y1-y2).abs().max().item()
    xgrad_err=(x1.grad-x2.grad).abs().max().item()
    pgrad_err=max((a.grad-b.grad).abs().max().item() for a,b in zip(blocks_custom.parameters(), blocks_naive.parameters()))
    assert out_err < 1e-6 and xgrad_err < 1e-5 and pgrad_err < 1e-5
    print(f"PASS {variant} custom backward equivalence:", {"output":out_err,"input_grad":xgrad_err,"parameter_grad":pgrad_err})


def roundtrip_test():
    torch.manual_seed(11)
    for variant in ("midpoint", "leapfrog"):
        cfg=ModelConfig(vocab_size=128,seq_len=16,d_model=32,n_heads=4,n_layers=8,d_ff=64,
                        reversible=True,reversible_variant=variant,reversible_h=0.25)
        model=TinyGPT(cfg); x=torch.randn(2,12,cfg.d_model)
        diag=model.stack.roundtrip_error(x)
        assert diag["max_abs_error"] < 1e-4, (variant,diag)
        print(f"PASS {variant} reversible round-trip:", diag["max_abs_error"])


def exact_token_budget_test():
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/"tiny.bin"; arr=np.memmap(p,dtype=np.uint16,mode="w+",shape=(10_000,))
        arr[:]=np.arange(10_000,dtype=np.uint16)%100; arr.flush()
        src=TokenMemmap(p,seq_len=32,device=torch.device("cpu"),seed=1)
        _,y=src.batch(batch_size=4,valid_target_tokens=101)
        assert y.numel()==128 and count_valid_targets(y)==101 and int((y==-100).sum())==27
    print("PASS exact final-partial-batch token accounting")


def lr_schedule_test():
    max_lr,min_lr=3e-4,3e-5
    assert cosine_lr(1_000_000,50_000_000,max_lr,min_lr,1_000_000)==max_lr
    assert abs(cosine_lr(50_000_000,50_000_000,max_lr,min_lr,1_000_000)-min_lr)<1e-12
    print("PASS token-based LR schedule endpoints")


if __name__ == "__main__":
    parameter_count_test(); architecture_parity_test()
    _gradient_equivalence("midpoint"); _gradient_equivalence("leapfrog")
    roundtrip_test(); exact_token_budget_test(); lr_schedule_test()
