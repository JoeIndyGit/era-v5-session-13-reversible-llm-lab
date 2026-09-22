from pathlib import Path
from collections import deque
import csv
import gc
import json
import math
import time
from contextlib import nullcontext

import numpy as np
import torch

from .data import TokenMemmap, count_valid_targets
from .model import TinyGPT, ModelConfig
from .utils import (
    seed_everything, device_info, choose_precision,
    cuda_sync, reset_peak_memory, peak_memory,
)


def cosine_lr(tokens_seen, target_tokens, max_lr, min_lr, warmup_tokens):
    if tokens_seen < warmup_tokens:
        return max_lr * max(tokens_seen, 1) / warmup_tokens
    progress = min(1.0, (tokens_seen - warmup_tokens) / max(1, target_tokens - warmup_tokens))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def evaluate(model, source, batch_size, batches, autocast_dtype):
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = source.batch(batch_size)
        ctx = (
            torch.autocast("cuda", dtype=autocast_dtype)
            if source.device.type == "cuda"
            else nullcontext()
        )
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def _train_step(model, optimizer, scaler, x, y, autocast_dtype, grad_clip):
    optimizer.zero_grad(set_to_none=True)
    ctx = (
        torch.autocast("cuda", dtype=autocast_dtype)
        if x.device.type == "cuda"
        else nullcontext()
    )
    with ctx:
        _, loss = model(x, y)

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    return loss, float(grad_norm)


def _qualitative_samples(model, tokenizer_path, device, seed=2026):
    """Deterministic top-1 generations stored as qualitative sanity checks."""
    if not tokenizer_path or not Path(tokenizer_path).exists():
        return None
    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        prompts = [
            "Once upon a time",
            "The little dragon",
            "Lily went into the forest",
        ]
        out = []
        for i, prompt in enumerate(prompts):
            torch.manual_seed(seed + i)
            ids = tokenizer.encode(prompt).ids
            x = torch.tensor([ids], dtype=torch.long, device=device)
            y = model.generate(x, max_new_tokens=64, temperature=1.0, top_k=1)
            out.append({"prompt": prompt, "completion": tokenizer.decode(y[0].tolist())})
        return out
    except Exception as e:
        return [{"error": f"qualitative generation unavailable: {type(e).__name__}: {e}"}]


def run_experiment(cfg, model_cfg, experiment_name):
    """Run one complete token-budget experiment and emit auditable JSON + CSV."""
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype, needs_scaler, precision_name = choose_precision(device)

    model = TinyGPT(model_cfg).to(device)
    params = model.num_parameters()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["max_lr"],
        betas=tuple(cfg["betas"]),
        weight_decay=cfg["weight_decay"],
    )
    scaler = (
        torch.amp.GradScaler("cuda", enabled=True)
        if device.type == "cuda" and needs_scaler else None
    )

    train_source = TokenMemmap(
        cfg["train_path"], model_cfg.seq_len, device, seed=cfg["seed"]
    )
    val_source = TokenMemmap(
        cfg["val_path"], model_cfg.seq_len, device, seed=cfg["seed"] + 1
    )

    out_dir = Path(cfg.get("results_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{experiment_name}_steps.csv"
    result_path = out_dir / f"{experiment_name}.json"

    log_fields = [
        "step", "tokens_seen", "train_loss", "train_loss_100step_mean",
        "val_loss", "lr", "grad_norm", "step_time_s", "tokens_per_s",
        "peak_allocated_gib", "peak_reserved_gib",
    ]

    tokens_seen = 0
    step = 0
    next_eval = cfg["eval_every_tokens"]
    speed_samples = []
    recent_losses = deque(maxlen=100)
    final_val = None

    reset_peak_memory()
    cuda_sync()
    run_start = time.perf_counter()

    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()

        while tokens_seen < cfg["target_tokens"]:
            step += 1
            remaining = cfg["target_tokens"] - tokens_seen
            full_tokens = cfg["batch_size"] * model_cfg.seq_len
            valid_tokens = min(full_tokens, remaining)

            x, y = train_source.batch(
                cfg["batch_size"], valid_target_tokens=valid_tokens
            )
            lr = cosine_lr(
                tokens_seen, cfg.get("lr_schedule_target_tokens", cfg["target_tokens"]),
                cfg["max_lr"], cfg["min_lr"], cfg["warmup_tokens"]
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            cuda_sync()
            t0 = time.perf_counter()
            loss, grad_norm = _train_step(
                model, optimizer, scaler, x, y, autocast_dtype, cfg["grad_clip"]
            )
            cuda_sync()
            step_time = time.perf_counter() - t0

            actual_tokens = count_valid_targets(y)
            tokens_seen += actual_tokens
            recent_losses.append(float(loss.item()))
            tok_s = actual_tokens / step_time
            if step > cfg["speed_warmup_steps"] and actual_tokens == full_tokens:
                speed_samples.append(tok_s)

            if tokens_seen >= next_eval or tokens_seen >= cfg["target_tokens"]:
                final_val = evaluate(
                    model, val_source,
                    cfg["eval_batch_size"],
                    cfg["eval_batches"],
                    autocast_dtype,
                )
                while next_eval <= tokens_seen:
                    next_eval += cfg["eval_every_tokens"]

            mem = peak_memory()
            writer.writerow({
                "step": step,
                "tokens_seen": tokens_seen,
                "train_loss": float(loss.item()),
                "train_loss_100step_mean": float(np.mean(recent_losses)),
                "val_loss": "" if final_val is None else final_val,
                "lr": lr,
                "grad_norm": grad_norm,
                "step_time_s": step_time,
                "tokens_per_s": tok_s,
                **mem,
            })
            f.flush()

    cuda_sync()
    wall = time.perf_counter() - run_start
    mem = peak_memory()

    reconstruction = None
    if model_cfg.reversible and hasattr(model.stack, "roundtrip_error"):
        probe = torch.randint(
            0, model_cfg.vocab_size, (1, min(32, model_cfg.seq_len)), device=device
        )
        pos = torch.arange(probe.size(1), device=device)
        with torch.no_grad():
            x = model.tok_emb(probe) + model.pos_emb(pos)[None, :, :]
            reconstruction = model.stack.roundtrip_error(x)

    dataset_meta = None
    meta_path = Path(cfg["train_path"]).parent / "dataset_meta.json"
    if meta_path.exists():
        dataset_meta = json.loads(meta_path.read_text())

    qualitative = _qualitative_samples(
        model, (dataset_meta or {}).get("tokenizer_path"), device, seed=2026
    )

    result = {
        "experiment": experiment_name,
        "architecture": "reversible" if model_cfg.reversible else "baseline",
        "reversible_variant": model_cfg.reversible_variant if model_cfg.reversible else None,
        "bootstrap": "euler" if model_cfg.reversible else None,
        "reversible_h": model_cfg.reversible_h if model_cfg.reversible else None,
        "model": model.architecture_summary(),
        "parameters": params,
        "target_tokens": cfg["target_tokens"],
        "tokens_seen": tokens_seen,
        "optimizer_steps": step,
        "batch_size": cfg["batch_size"],
        "seq_len": model_cfg.seq_len,
        "tokens_per_full_step": cfg["batch_size"] * model_cfg.seq_len,
        "precision": precision_name,
        "final_step_train_loss": float(loss.item()),
        "final_train_loss_100step_mean": float(np.mean(recent_losses)),
        "final_val_loss": float(final_val),
        "final_val_perplexity": float(math.exp(min(final_val, 20))),
        "mean_tokens_per_s": float(np.mean(speed_samples)) if speed_samples else None,
        "median_tokens_per_s": float(np.median(speed_samples)) if speed_samples else None,
        "p10_tokens_per_s": float(np.percentile(speed_samples, 10)) if speed_samples else None,
        "p90_tokens_per_s": float(np.percentile(speed_samples, 90)) if speed_samples else None,
        "throughput_samples_count": len(speed_samples),
        "wall_clock_s": wall,
        **mem,
        "reconstruction": reconstruction,
        "hardware": device_info(),
        "dataset": dataset_meta,
        "qualitative_samples": qualitative,
        "training_config": cfg,
    }
    result_path.write_text(json.dumps(result, indent=2))
    return result


def _probe_one_batch(base_cfg, model_cfg, batch_size, trial_steps, reserve_limit):
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    gc.collect()
    seed_everything(base_cfg["seed"])

    model = TinyGPT(ModelConfig(**model_cfg.to_dict())).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=base_cfg["max_lr"],
        betas=tuple(base_cfg["betas"]),
        weight_decay=base_cfg["weight_decay"],
    )
    dtype, needs_scaler, _ = choose_precision(device)
    scaler = torch.amp.GradScaler("cuda", enabled=True) if needs_scaler else None
    source = TokenMemmap(
        base_cfg["train_path"], model_cfg.seq_len, device,
        seed=base_cfg["seed"] + batch_size,
    )

    ok = True
    error = None
    reset_peak_memory()
    try:
        last_loss = None
        for _ in range(trial_steps):
            x, y = source.batch(batch_size)
            loss, _ = _train_step(model, opt, scaler, x, y, dtype, base_cfg["grad_clip"])
            last_loss = float(loss.item())
            cuda_sync()
            if not math.isfinite(last_loss):
                ok = False
                error = f"non-finite loss: {last_loss}"
                break
        mem = peak_memory()
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        reserve_fraction = mem["peak_reserved_gib"] / total
        if reserve_fraction > reserve_limit:
            ok = False
            error = f"reserved fraction {reserve_fraction:.3f} > {reserve_limit:.3f}"
    except torch.cuda.OutOfMemoryError:
        ok = False
        error = "CUDA OOM"
        torch.cuda.empty_cache()

    attempt = {
        "batch_size": int(batch_size),
        "stable": bool(ok),
        "error": error,
        "last_loss": last_loss,
        **peak_memory(),
    }

    del model, opt, source
    if scaler is not None:
        del scaler
    gc.collect()
    torch.cuda.empty_cache()
    return attempt


def find_max_stable_batch(
    base_cfg,
    model_cfg,
    start_batch=None,
    trial_steps=10,
    reserve_limit=0.96,
    max_batch_cap=8192,
):
    """Find maximum memory-feasible integer batch using failure bracketing + binary search."""
    if not torch.cuda.is_available():
        raise RuntimeError("Maximum GPU batch probing requires CUDA.")

    start = int(start_batch or base_cfg["batch_size"])
    attempts = []
    cache = {}

    def probe(b):
        b = int(b)
        if b not in cache:
            cache[b] = _probe_one_batch(
                base_cfg, model_cfg, b, trial_steps, reserve_limit
            )
            attempts.append(cache[b])
        return cache[b]

    first = probe(start)
    if not first["stable"]:
        high = start
        low = 0
        candidate = max(1, start // 2)
        while candidate >= 1:
            a = probe(candidate)
            if a["stable"]:
                low = candidate
                break
            high = candidate
            if candidate == 1:
                break
            candidate = max(1, candidate // 2)
        if low == 0:
            raise RuntimeError("Even batch size 1 is unstable on this GPU.")
    else:
        low = start
        high = None
        candidate = start * 2
        while candidate <= max_batch_cap:
            a = probe(candidate)
            if a["stable"]:
                low = candidate
                candidate *= 2
            else:
                high = candidate
                break
        if high is None:
            if low < max_batch_cap:
                a = probe(max_batch_cap)
                if a["stable"]:
                    low = max_batch_cap
                else:
                    high = max_batch_cap
            if high is None:
                return {
                    "largest_stable_batch": low,
                    "largest_feasible_batch": low,
                    "first_failed_batch": None,
                    "search_complete": False,
                    "note": f"No failure observed up to safety cap {max_batch_cap}; value is a lower bound.",
                    "attempts": sorted(attempts, key=lambda x: x["batch_size"]),
                    "trial_steps": trial_steps,
                    "reserve_limit": reserve_limit,
                    "max_batch_cap": max_batch_cap,
                }

    while high - low > 1:
        mid = (low + high) // 2
        a = probe(mid)
        if a["stable"]:
            low = mid
        else:
            high = mid

    return {
        "largest_stable_batch": low,
        "largest_feasible_batch": low,
        "first_failed_batch": high,
        "search_complete": True,
        "note": "Largest stable integer batch bracketed by an observed failure.",
        "attempts": sorted(attempts, key=lambda x: x["batch_size"]),
        "trial_steps": trial_steps,
        "reserve_limit": reserve_limit,
        "max_batch_cap": max_batch_cap,
    }


def run_variant_selection(base_cfg, candidate_cfgs, pilot_tokens=1_000_000, results_dir="results"):
    """Controlled Midpoint-vs-Leapfrog pilot with predeclared selection rule."""
    out = Path(results_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant, model_cfg in candidate_cfgs.items():
        cfg = dict(base_cfg)
        cfg.update({
            "target_tokens": int(pilot_tokens),
            "lr_schedule_target_tokens": int(base_cfg["target_tokens"]),
            "eval_every_tokens": max(500_000, int(pilot_tokens // 2)),
            "eval_batches": min(10, int(base_cfg.get("eval_batches", 10))),
            "speed_warmup_steps": min(5, int(base_cfg.get("speed_warmup_steps", 5))),
            "results_dir": str(out / "pilots"),
        })
        try:
            r = run_experiment(cfg, model_cfg, f"pilot_{variant}")
            recon = (r.get("reconstruction") or {}).get("max_abs_error", float("inf"))
            stable = bool(np.isfinite(r["final_val_loss"]) and np.isfinite(r["final_train_loss_100step_mean"]) and recon < 1e-3)
            rows.append({
                "variant": variant, "stable": stable,
                "val_loss": r["final_val_loss"],
                "train_loss": r["final_train_loss_100step_mean"],
                "median_tokens_per_s": r["median_tokens_per_s"],
                "peak_allocated_gib": r["peak_allocated_gib"],
                "reconstruction_max_abs_error": recon,
                "result_file": str(Path(cfg["results_dir"]) / f"pilot_{variant}.json"),
            })
        except Exception as e:
            rows.append({"variant": variant, "stable": False, "error": f"{type(e).__name__}: {e}"})

    stable = [r for r in rows if r.get("stable")]
    if not stable:
        raise RuntimeError(f"No reversible variant passed the pilot: {rows}")
    selected = min(stable, key=lambda r: (r["val_loss"], -(r.get("median_tokens_per_s") or 0)))
    report = {
        "pilot_tokens_per_candidate": int(pilot_tokens),
        "selection_rule": "lowest held-out validation loss among stable candidates; throughput is tie-breaker",
        "candidates": rows,
        "selected_variant": selected["variant"],
        "selected": selected,
    }
    (out / "variant_selection.json").write_text(json.dumps(report, indent=2))
    (out / "selected_variant.json").write_text(json.dumps({
        "selected_variant": selected["variant"],
        "config_file": f"configs/model_{selected['variant']}.json",
        "selection_source": "results/variant_selection.json",
    }, indent=2))
    return report


def run_accumulation_control(cfg, model_cfg, experiment_name, physical_batch, grad_accum_steps):
    """Optional baseline control matching a larger effective batch via exact token-weighted accumulation."""
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype, needs_scaler, precision_name = choose_precision(device)
    model = TinyGPT(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["max_lr"], betas=tuple(cfg["betas"]), weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=True) if device.type=="cuda" and needs_scaler else None
    train_source = TokenMemmap(cfg["train_path"], model_cfg.seq_len, device, seed=cfg["seed"])
    val_source = TokenMemmap(cfg["val_path"], model_cfg.seq_len, device, seed=cfg["seed"]+1)
    out_dir=Path(cfg.get("results_dir","results")); out_dir.mkdir(parents=True,exist_ok=True)
    log_path=out_dir/f"{experiment_name}_steps.csv"; result_path=out_dir/f"{experiment_name}.json"
    fields=["step","tokens_seen","train_loss","train_loss_100step_mean","val_loss","lr","grad_norm","step_time_s","tokens_per_s","peak_allocated_gib","peak_reserved_gib"]
    tokens_seen=step=0; next_eval=cfg["eval_every_tokens"]; speeds=[]; losses=deque(maxlen=100); final_val=None
    reset_peak_memory(); cuda_sync(); start=time.perf_counter()
    with log_path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        while tokens_seen < cfg["target_tokens"]:
            step += 1; remaining=cfg["target_tokens"]-tokens_seen
            effective_capacity=physical_batch*model_cfg.seq_len*grad_accum_steps
            update_tokens=min(effective_capacity,remaining)
            lr=cosine_lr(tokens_seen,cfg.get("lr_schedule_target_tokens",cfg["target_tokens"]),cfg["max_lr"],cfg["min_lr"],cfg["warmup_tokens"])
            for g in optimizer.param_groups: g["lr"]=lr
            optimizer.zero_grad(set_to_none=True); cuda_sync(); t0=time.perf_counter()
            consumed=0; micro_losses=[]
            for _ in range(grad_accum_steps):
                valid=min(physical_batch*model_cfg.seq_len, update_tokens-consumed)
                if valid<=0: break
                x,y=train_source.batch(physical_batch,valid_target_tokens=valid)
                ctx=torch.autocast("cuda",dtype=autocast_dtype) if device.type=="cuda" else nullcontext()
                with ctx: _,loss=model(x,y)
                weight=valid/update_tokens
                if scaler is not None: scaler.scale(loss*weight).backward()
                else: (loss*weight).backward()
                micro_losses.append((float(loss.item()),valid)); consumed += valid
            if scaler is not None:
                scaler.unscale_(optimizer); grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["grad_clip"]); scaler.step(optimizer); scaler.update()
            else:
                grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["grad_clip"]); optimizer.step()
            cuda_sync(); elapsed=time.perf_counter()-t0; tokens_seen += consumed
            avg_loss=sum(v*n for v,n in micro_losses)/consumed; losses.append(avg_loss); tok_s=consumed/elapsed
            if step>cfg["speed_warmup_steps"] and consumed==effective_capacity: speeds.append(tok_s)
            if tokens_seen>=next_eval or tokens_seen>=cfg["target_tokens"]:
                final_val=evaluate(model,val_source,cfg["eval_batch_size"],cfg["eval_batches"],autocast_dtype)
                while next_eval<=tokens_seen: next_eval+=cfg["eval_every_tokens"]
            mem=peak_memory(); w.writerow({"step":step,"tokens_seen":tokens_seen,"train_loss":avg_loss,"train_loss_100step_mean":float(np.mean(losses)),"val_loss":"" if final_val is None else final_val,"lr":lr,"grad_norm":float(grad_norm),"step_time_s":elapsed,"tokens_per_s":tok_s,**mem}); f.flush()
    cuda_sync(); wall=time.perf_counter()-start; mem=peak_memory()
    result={
        "experiment":experiment_name,"architecture":"baseline_gradient_accumulation","parameters":model.num_parameters(),
        "target_tokens":cfg["target_tokens"],"tokens_seen":tokens_seen,"optimizer_steps":step,
        "physical_batch_size":int(physical_batch),"grad_accum_steps":int(grad_accum_steps),
        "effective_batch_size":int(physical_batch*grad_accum_steps),"seq_len":model_cfg.seq_len,
        "precision":precision_name,"final_train_loss_100step_mean":float(np.mean(losses)),"final_val_loss":float(final_val),
        "final_val_perplexity":float(math.exp(min(final_val,20))),"mean_tokens_per_s":float(np.mean(speeds)) if speeds else None,
        "median_tokens_per_s":float(np.median(speeds)) if speeds else None,"wall_clock_s":wall,**mem,"hardware":device_info(),"training_config":cfg,
    }
    result_path.write_text(json.dumps(result,indent=2)); return result
