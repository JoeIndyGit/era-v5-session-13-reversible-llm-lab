#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json
import subprocess
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import prepare_tinystories_cache
from src.model import ModelConfig
from src.train import run_experiment, run_variant_selection, find_max_stable_batch


def sh(*args: str) -> None:
    subprocess.run(list(args), cwd=ROOT, check=True)


def load_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the graded runs. This runner refuses to generate "
            "CPU-only speed/memory evidence because it would not satisfy the GPU experiment."
        )
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)} | {p.total_memory/1024**3:.2f} GiB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-tokens", type=int, default=2_000_000)
    ap.add_argument("--probe-steps", type=int, default=10)
    ap.add_argument("--reserve-limit", type=float, default=0.96)
    ap.add_argument("--force", action="store_true", help="rerun completed phases")
    args = ap.parse_args()

    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (ROOT / "assets").mkdir(exist_ok=True)

    require_cuda()

    print("\n[0/6] Correctness + environment")
    sh(sys.executable, "scripts/validate.py")
    sh(sys.executable, "scripts/capture_environment.py")

    print("\n[1/6] Dataset/tokenizer cache")
    prepare_tinystories_cache(
        out_dir=ROOT / "data",
        train_tokens=52_000_000,
        val_tokens=1_000_000,
        vocab_size=10_000,
        tokenizer_training_stories=100_000,
    )

    common = load_json(ROOT / "configs/common.json")
    common["train_path"] = str(ROOT / common["train_path"])
    common["val_path"] = str(ROOT / common["val_path"])
    common["results_dir"] = str(results)

    print("\n[2/6] Pre-registered reversible variant selection")
    selected_path = results / "selected_variant.json"
    if args.force or not selected_path.exists():
        candidates = {
            "midpoint": ModelConfig(**load_json(ROOT / "configs/model_midpoint.json")),
            "leapfrog": ModelConfig(**load_json(ROOT / "configs/model_leapfrog.json")),
        }
        report = run_variant_selection(
            common, candidates, pilot_tokens=args.pilot_tokens, results_dir=str(results)
        )
        print("Selected:", report["selected_variant"])
    selected = load_json(selected_path)
    selected_cfg = ModelConfig(**load_json(ROOT / selected["config_file"]))
    baseline_cfg = ModelConfig(**load_json(ROOT / "configs/model_baseline.json"))

    print("\n[3/6] Required fixed-batch 50M runs")
    if args.force or not (results / "baseline_fixed.json").exists():
        run_experiment(common, baseline_cfg, "baseline_fixed")
    else:
        print("SKIP baseline_fixed: authoritative JSON exists")

    if args.force or not (results / "reversible_fixed.json").exists():
        run_experiment(common, selected_cfg, "reversible_fixed")
    else:
        print("SKIP reversible_fixed: authoritative JSON exists")

    print("\n[4/6] 10-update batch frontiers + reversible maximum-batch 50M run")
    bp_path = results / "baseline_batch_probe.json"
    rp_path = results / "reversible_batch_probe.json"

    if args.force or not bp_path.exists():
        bp = find_max_stable_batch(
            common, baseline_cfg, start_batch=common["batch_size"],
            trial_steps=args.probe_steps, reserve_limit=args.reserve_limit,
        )
        write_json(bp_path, bp)
    else:
        bp = load_json(bp_path)

    if args.force or not rp_path.exists():
        rp = find_max_stable_batch(
            common, selected_cfg, start_batch=common["batch_size"],
            trial_steps=args.probe_steps, reserve_limit=args.reserve_limit,
        )
        write_json(rp_path, rp)
    else:
        rp = load_json(rp_path)

    if not bp.get("search_complete") or not rp.get("search_complete"):
        raise RuntimeError(
            "At least one batch search reached its safety cap without an observed failure. "
            "Increase max_batch_cap before claiming a maximum."
        )

    max_batch = int(rp.get("largest_feasible_batch", rp["largest_stable_batch"]))
    max_cfg = dict(common)
    max_cfg["batch_size"] = max_batch
    max_cfg["eval_batch_size"] = min(common["eval_batch_size"], max_batch)
    if args.force or not (results / "reversible_max_batch.json").exists():
        run_experiment(max_cfg, selected_cfg, "reversible_max_batch")
    else:
        print("SKIP reversible_max_batch: authoritative JSON exists")

    print("\n[5/6] Final evidence audit")
    sh(sys.executable, "scripts/audit_results.py")

    print("\n[6/6] Render report + figures from measured evidence")
    sh(sys.executable, "scripts/render_report.py")
    print("\nCOMPLETE: required evidence is in results/, figures in assets/, README is populated.")


if __name__ == "__main__":
    main()
