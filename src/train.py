from pathlib import Path
from collections import deque
import csv
import copy
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


def _amp(device, dtype):
    return torch.autocast(device.type, dtype=dtype) if device.type == 'cuda' and dtype != torch.float32 else nullcontext()


@torch.no_grad()
def evaluate(model, source, batch_size, batches, autocast_dtype):
    # A fixed held-out sample is shared across every checkpoint and architecture.
    state = copy.deepcopy(source.rng.bit_generator.state)
    training = model.training
    model.eval()
    losses = []
    try:
        for _ in range(batches):
            x, y = source.batch(batch_size)
            with _amp(source.device, autocast_dtype):
                _, loss = model(x, y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite validation loss')
            losses.append(loss.item())
    finally:
        source.rng.bit_generator.state = state
        model.train(training)
    return float(np.mean(losses))


def _train_step(model, optimizer, scaler, x, y, autocast_dtype, grad_clip):
    """Return only after a real update; replay an overflowing FP16 batch."""
    for retry in range(16):
        optimizer.zero_grad(set_to_none=True)
        with _amp(x.device, autocast_dtype):
            _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        if scaler is not None:
            old_scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < old_scale:
                continue
        else:
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
            optimizer.step()
        if not math.isfinite(float(norm)):
            raise FloatingPointError('Non-finite gradient norm after successful update')
        return loss.detach(), float(norm), retry
    raise FloatingPointError('FP16 gradients overflowed after 16 attempts; use a safer precision for all runs.')


def _qualitative_samples(model, tokenizer_path, device, seed=2026):
    if not tokenizer_path or not Path(tokenizer_path).exists():
        return None
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    rows = []
    for i, prompt in enumerate(['Once upon a time', 'The little dragon', 'Lily went into the forest']):
        torch.manual_seed(seed + i)
        x = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)
        y = model.generate(x, max_new_tokens=64, temperature=1.0, top_k=1)
        rows.append({'prompt': prompt, 'completion': tokenizer.decode(y[0].tolist())})
    return rows


def run_experiment(cfg, model_cfg, experiment_name, stop_after_steps=None):
    """Exactly budgeted training with atomic recovery and independently auditable logs.

    stop_after_steps is used only by the interruption regression test. It never
    writes a completed result and is not an alternative benchmark budget.
    """
    from .checkpointing import save_checkpoint, load_checkpoint
    from .evidence import (SCHEMA_VERSION, utcnow, verified_dataset, run_identity,
                           canonical_hash, atomic_json, sha256)
    from .diagnostics import reconstruction_check

    seed_everything(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda' and not cfg.get('allow_cpu', False):
        raise RuntimeError('The measured submission pipeline requires CUDA. CPU execution is reserved for isolated tests.')
    dtype, needs_scaler, precision = choose_precision(device, cfg.get('precision', 'auto'))
    dataset = verified_dataset(cfg)
    hardware = device_info()
    identity = run_identity(cfg, model_cfg, dataset, hardware, precision)
    signature = canonical_hash(identity)
    out = Path(cfg.get('results_dir', 'results')); out.mkdir(parents=True, exist_ok=True)
    result_path = out / f'{experiment_name}.json'
    log_path = out / f'{experiment_name}_steps.csv'
    checkpoint_path = Path(cfg.get('checkpoint_dir', 'checkpoints')) / f'{experiment_name}.pt'

    if result_path.exists():
        result = json.loads(result_path.read_text())
        if (result.get('status') == 'completed' and result.get('signature') == signature
                and result.get('tokens_seen') == cfg['target_tokens'] and log_path.exists()
                and result.get('steps_sha256') == sha256(log_path)):
            from scripts.audit_results import check_run
            problems = check_run(result, log_path, cfg['target_tokens'])
            if problems:
                raise RuntimeError('Completed result failed re-audit: ' + '; '.join(problems))
            print(f'Reusing verified complete run: {experiment_name}')
            return result
        raise RuntimeError(f'{result_path} is stale or incomplete. Preserve it and choose a new results/checkpoint directory.')

    model = TinyGPT(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['max_lr'], betas=tuple(cfg['betas']), weight_decay=cfg['weight_decay'])
    scaler = torch.amp.GradScaler('cuda') if needs_scaler else None
    train_source = TokenMemmap(cfg['train_path'], model_cfg.seq_len, device, seed=cfg['seed'])
    val_source = TokenMemmap(cfg['val_path'], model_cfg.seq_len, device, seed=cfg['seed'] + 1)
    progress = {'rows': [], 'tokens_seen': 0, 'step': 0, 'next_eval': cfg['eval_every_tokens'],
                'started_utc': utcnow(), 'active_seconds': 0.0, 'session': 0,
                'evaluation_peak_allocated_gib': None, 'resume_count': 0}
    if checkpoint_path.exists():
        progress = load_checkpoint(checkpoint_path, model, optimizer, scaler, train_source, val_source, signature)
        progress['resume_count'] += 1
        print(f'Resuming {experiment_name}: {progress["tokens_seen"]:,} tokens / {progress["step"]:,} updates')
    elif log_path.exists():
        raise RuntimeError(f'{log_path} exists without a compatible checkpoint; preserve it before starting a fresh run.')

    rows = progress['rows']
    recent = deque([r['train_loss'] for r in rows[-100:]], maxlen=100)
    final_val = next((r['val_loss'] for r in reversed(rows) if r['val_loss'] != ''), None)
    base_seconds = progress['active_seconds']
    progress['session'] += 1
    session_steps = 0
    fields = ['step', 'tokens_seen', 'valid_tokens', 'train_loss', 'train_loss_100step_mean',
              'val_loss', 'lr', 'grad_norm', 'step_time_s', 'tokens_per_s', 'steady_state',
              'overflow_retries', 'session', 'peak_allocated_gib', 'peak_reserved_gib']
    reset_peak_memory(); cuda_sync(); start = time.perf_counter()

    def checkpoint():
        progress['active_seconds'] = base_seconds + time.perf_counter() - start
        save_checkpoint(checkpoint_path, model, optimizer, scaler, train_source, val_source, progress, signature)

    # Rewriting committed rows discards any CSV tail written after the last atomic checkpoint.
    with log_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows); f.flush()
        while progress['tokens_seen'] < cfg['target_tokens']:
            valid = min(cfg['batch_size'] * model_cfg.seq_len, cfg['target_tokens'] - progress['tokens_seen'])
            x, y = train_source.batch(cfg['batch_size'], valid_target_tokens=valid)
            assert count_valid_targets(y) == valid
            lr = cosine_lr(progress['tokens_seen'], cfg.get('lr_schedule_target_tokens', cfg['target_tokens']),
                           cfg['max_lr'], cfg['min_lr'], cfg['warmup_tokens'])
            for group in optimizer.param_groups: group['lr'] = lr
            if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
            cuda_sync(); t0 = time.perf_counter()
            loss, norm, retries = _train_step(model, optimizer, scaler, x, y, dtype, cfg['grad_clip'])
            cuda_sync(); duration = time.perf_counter() - t0
            mem = peak_memory()
            progress['step'] += 1; session_steps += 1; progress['tokens_seen'] += valid
            recent.append(float(loss))
            val_at_step = ''
            if progress['tokens_seen'] >= progress['next_eval'] or progress['tokens_seen'] == cfg['target_tokens']:
                if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
                final_val = evaluate(model, val_source, cfg['eval_batch_size'], cfg['eval_batches'], dtype)
                val_at_step = final_val
                eval_peak = peak_memory()['peak_allocated_gib']
                if eval_peak is not None:
                    progress['evaluation_peak_allocated_gib'] = max(eval_peak, progress['evaluation_peak_allocated_gib'] or 0)
                while progress['next_eval'] <= progress['tokens_seen']: progress['next_eval'] += cfg['eval_every_tokens']
            row = {'step': progress['step'], 'tokens_seen': progress['tokens_seen'], 'valid_tokens': valid,
                   'train_loss': float(loss), 'train_loss_100step_mean': float(np.mean(recent)), 'val_loss': val_at_step,
                   'lr': lr, 'grad_norm': norm, 'step_time_s': duration, 'tokens_per_s': valid / duration,
                   'steady_state': session_steps > cfg['speed_warmup_steps'] and valid == cfg['batch_size'] * model_cfg.seq_len and retries == 0,
                   'overflow_retries': retries, 'session': progress['session'], **mem}
            rows.append(row); writer.writerow(row); f.flush()
            if progress['step'] % cfg.get('log_every_steps', 100) == 0 or progress['tokens_seen'] == cfg['target_tokens']:
                print(f'{experiment_name}: {progress["tokens_seen"]:,}/{cfg["target_tokens"]:,} tokens | loss {float(loss):.4f} | {valid/duration:,.0f} tok/s', flush=True)
            if (progress['step'] % cfg.get('checkpoint_every_steps', 500) == 0
                    or progress['tokens_seen'] == cfg['target_tokens'] or stop_after_steps == progress['step']):
                checkpoint()
            if stop_after_steps == progress['step'] and progress['tokens_seen'] < cfg['target_tokens']:
                return {'status': 'interrupted_for_test', 'tokens_seen': progress['tokens_seen']}

    wall = base_seconds + time.perf_counter() - start
    optimizer.zero_grad(set_to_none=True)
    if device.type == 'cuda': torch.cuda.empty_cache()
    reconstruction = None
    if model_cfg.reversible:
        # The actual context, tokenizer inputs and autocast regime used in training.
        probe_x, _ = val_source.batch(1)
        reconstruction = reconstruction_check(model, probe_x, dtype)
        if not reconstruction['passed']:
            raise RuntimeError(f'Trained-model reconstruction failed: {reconstruction}. The final checkpoint is preserved.')
    qualitative = _qualitative_samples(model, dataset.get('tokenizer_path'), device)
    steady = [r for r in rows if r['steady_state']]
    speeds = [r['tokens_per_s'] for r in steady]
    max_metric = lambda key: max((r[key] for r in rows if r[key] is not None), default=None)
    result = {
        'schema_version': SCHEMA_VERSION, 'status': 'completed', 'signature': signature, 'identity': identity,
        'started_utc': progress['started_utc'], 'completed_utc': utcnow(),
        'experiment': experiment_name, 'architecture': 'reversible' if model_cfg.reversible else 'baseline',
        'reversible_variant': model_cfg.reversible_variant if model_cfg.reversible else None,
        'bootstrap': 'euler' if model_cfg.reversible else None,
        'reversible_h': model_cfg.reversible_h if model_cfg.reversible else None,
        'model': model.architecture_summary(), 'parameters': model.num_parameters(),
        'target_tokens': cfg['target_tokens'], 'tokens_seen': progress['tokens_seen'], 'optimizer_steps': progress['step'],
        'batch_size': cfg['batch_size'], 'seq_len': model_cfg.seq_len, 'tokens_per_full_step': cfg['batch_size'] * model_cfg.seq_len,
        'precision': precision, 'final_step_train_loss': rows[-1]['train_loss'],
        'final_train_loss_100step_mean': float(np.mean(recent)), 'final_val_loss': float(final_val),
        'final_val_perplexity': float(math.exp(final_val)),
        'mean_tokens_per_s': float(np.mean(speeds)) if speeds else None,
        'median_tokens_per_s': float(np.median(speeds)) if speeds else None,
        'aggregate_tokens_per_s': sum(r['valid_tokens'] for r in steady) / sum(r['step_time_s'] for r in steady) if steady else None,
        'p10_tokens_per_s': float(np.percentile(speeds, 10)) if speeds else None,
        'p90_tokens_per_s': float(np.percentile(speeds, 90)) if speeds else None,
        'throughput_samples_count': len(steady), 'wall_clock_s': wall,
        'peak_allocated_gib': max_metric('peak_allocated_gib'), 'peak_reserved_gib': max_metric('peak_reserved_gib'),
        'evaluation_peak_allocated_gib': progress['evaluation_peak_allocated_gib'],
        'overflow_retries': sum(r['overflow_retries'] for r in rows), 'resume_count': progress['resume_count'],
        'reconstruction': reconstruction, 'hardware': hardware, 'dataset': dataset,
        'qualitative_samples': qualitative, 'training_config': cfg,
        'steps_file': log_path.name, 'steps_sha256': sha256(log_path),
        'checkpoint_file': str(checkpoint_path), 'checkpoint_sha256': sha256(checkpoint_path),
    }
    atomic_json(result_path, result)
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
    dtype, needs_scaler, precision = choose_precision(device, base_cfg.get("precision", "auto"))
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
        successful_updates = 0
        overflow_retries = 0
        for _ in range(trial_steps):
            x, y = source.batch(batch_size)
            loss, _, retries = _train_step(model, opt, scaler, x, y, dtype, base_cfg["grad_clip"])
            overflow_retries += retries
            successful_updates += 1
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
    except FloatingPointError as exc:
        ok = False
        error = f"numerical failure: {exc}"
    except torch.cuda.OutOfMemoryError:
        ok = False
        error = "CUDA OOM"
        torch.cuda.empty_cache()

    attempt = {
        "batch_size": int(batch_size),
        "stable": bool(ok),
        "error": error,
        "last_loss": last_loss,
        "successful_updates": successful_updates,
        "overflow_retries": overflow_retries,
        **peak_memory(),
    }

    del model, opt, source
    if scaler is not None:
        del scaler
    gc.collect()
    torch.cuda.empty_cache()
    return attempt


def _search_max_stable_batch(
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
            if (cache[b].get("error") or "").startswith("numerical failure"):
                raise RuntimeError("A numerical failure is not evidence of a memory boundary: " + cache[b]["error"])
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


def find_max_stable_batch(base_cfg, model_cfg, start_batch=None, trial_steps=10, reserve_limit=0.96, max_batch_cap=8192):
    from .evidence import verified_dataset, run_identity, canonical_hash, atomic_json, utcnow
    if not torch.cuda.is_available():
        raise RuntimeError("Maximum GPU batch probing requires CUDA.")
    if trial_steps < 10 or not 0 < reserve_limit < 1:
        raise ValueError("Capacity probes require at least 10 successful updates and a reserve limit between 0 and 1.")
    _, _, precision = choose_precision(torch.device("cuda"), base_cfg.get("precision", "auto"))
    identity = run_identity(base_cfg, model_cfg, verified_dataset(base_cfg), device_info(), precision)
    protocol = dict(start_batch=start_batch or base_cfg["batch_size"], trial_steps=trial_steps,
                    reserve_limit=reserve_limit, max_batch_cap=max_batch_cap)
    signature = canonical_hash({"identity": identity, "protocol": protocol})
    name = "reversible_batch_probe" if model_cfg.reversible else "baseline_batch_probe"
    path = Path(base_cfg.get("results_dir", "results")) / (name + ".json")
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("signature") == signature and existing.get("search_complete"):
            from scripts.audit_results import check_probe
            problems = check_probe(existing)
            if problems:
                raise RuntimeError("Saved capacity probe failed re-audit: " + "; ".join(problems))
            return existing
        raise RuntimeError(f"Stale or incomplete batch evidence at {path}; preserve it and use a new experiment directory.")
    started = utcnow()
    result = _search_max_stable_batch(base_cfg, model_cfg, **protocol)
    result.update(schema_version=2, signature=signature, identity=identity, protocol=protocol, started_utc=started, completed_utc=utcnow())
    atomic_json(path, result)
    return result


def run_variant_selection(base_cfg, candidate_cfgs, pilot_tokens=1_000_000, results_dir="results"):
    """Controlled Midpoint-vs-Leapfrog pilot with predeclared selection rule."""
    out = Path(results_dir); out.mkdir(parents=True, exist_ok=True)
    from .evidence import atomic_json, utcnow, source_digest
    gate_path = out / "gpu_correctness.json"
    gates = json.loads(gate_path.read_text()) if gate_path.exists() else None
    if not base_cfg.get("allow_cpu", False) and (not gates or gates.get("source_sha256") != source_digest()):
        raise RuntimeError("Run the full-depth GPU precision checks before selecting a variant.")
    rows = []
    for variant, model_cfg in candidate_cfgs.items():
        if gates and not gates["variants"].get(variant, {}).get("passed"):
            rows.append({"variant": variant, "stable": False, "error": "Failed the full-depth GPU precision gate"})
            continue
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
            stable = bool(np.isfinite(r["final_val_loss"]) and np.isfinite(r["final_train_loss_100step_mean"]) and r["reconstruction"]["passed"])
            rows.append({
                "variant": variant, "stable": stable,
                "val_loss": r["final_val_loss"],
                "train_loss": r["final_train_loss_100step_mean"],
                "median_tokens_per_s": r["median_tokens_per_s"],
                "peak_allocated_gib": r["peak_allocated_gib"],
                "reconstruction_max_abs_error": recon,
                "result_file": str(Path(cfg["results_dir"]) / f"pilot_{variant}.json"),
                "signature": r["signature"],
                "completed_utc": r["completed_utc"],
            })
        except Exception as e:
            rows.append({"variant": variant, "stable": False, "error": f"{type(e).__name__}: {e}"})

    stable = [r for r in rows if r.get("stable")]
    if not stable:
        raise RuntimeError(f"No reversible variant passed the pilot: {rows}")
    selected = min(stable, key=lambda r: (r["val_loss"], -(r.get("median_tokens_per_s") or 0)))
    report = {
        "schema_version": 2, "selected_utc": utcnow(), "source_sha256": source_digest(),
        "pilot_tokens_per_candidate": int(pilot_tokens),
        "selection_rule": "lowest held-out validation loss among stable candidates; throughput is tie-breaker",
        "candidates": rows,
        "selected_variant": selected["variant"],
        "selected": selected,
    }
    old_path = out / "variant_selection.json"
    if old_path.exists():
        old = json.loads(old_path.read_text())
        old_signatures = {r["variant"]: r.get("signature") for r in old.get("candidates", []) if r.get("stable")}
        signatures = {r["variant"]: r.get("signature") for r in rows if r.get("stable")}
        if old.get("source_sha256") != source_digest() or old.get("selected_variant") != report["selected_variant"] or old_signatures != signatures:
            raise RuntimeError("Existing variant selection belongs to a different experiment; preserve it and use fresh output directories.")
        report = old
    atomic_json(out / "variant_selection.json", report)
    atomic_json(out / "selected_variant.json", {
        "selected_variant": selected["variant"],
        "config_file": f"configs/model_{selected['variant']}.json",
        "selection_source": "results/variant_selection.json",
        "selected_utc": report["selected_utc"],
        "source_sha256": report["source_sha256"],
    })
    return report


def run_accumulation_control(cfg, model_cfg, experiment_name, physical_batch, grad_accum_steps):
    """Optional baseline control matching a larger effective batch via exact token-weighted accumulation."""
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype, needs_scaler, precision_name = choose_precision(device, cfg.get("precision", "auto"))
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
