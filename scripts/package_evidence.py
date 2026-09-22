#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission_evidence"
ZIP = OUT / "era-v5-session-13-evidence.zip"

INCLUDE_ROOT_FILES = [
    "README.md",
    "EXPERIMENT_CARD.md",
    "SUBMISSION_CHECKLIST.md",
    "requirements.txt",
]
INCLUDE_DIRS = [
    "results",
    "assets",
    "configs",
    "src",
    "scripts",
    "notebooks",
    ".github/workflows",
]
EXCLUDE_NAMES = {".gitkeep", "__pycache__", ".ipynb_checkpoints"}
EXCLUDE_SUFFIXES = {".pyc"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def eligible_files():
    files = []
    for rel in INCLUDE_ROOT_FILES:
        p = ROOT / rel
        if p.exists():
            files.append(p)
    for rel in INCLUDE_DIRS:
        d = ROOT / rel
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if not p.is_file():
                continue
            if any(part in EXCLUDE_NAMES for part in p.parts):
                continue
            if p.suffix in EXCLUDE_SUFFIXES:
                continue
            files.append(p)
    return sorted(set(files))


def main():
    # A bundle must never bless an incomplete experiment.
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "audit_results.py")],
        cwd=ROOT, check=True
    )

    required = [
        ROOT / "results" / "baseline_fixed.json",
        ROOT / "results" / "reversible_fixed.json",
        ROOT / "results" / "reversible_max_batch.json",
        ROOT / "results" / "variant_selection.json",
        ROOT / "results" / "baseline_batch_probe.json",
        ROOT / "results" / "reversible_batch_probe.json",
        ROOT / "results" / "environment.txt",
        ROOT / "assets" / "executive_summary.png",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        raise SystemExit("Cannot package incomplete evidence. Missing: " + ", ".join(missing))

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    files = eligible_files()
    manifest_files = []
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        manifest_files.append({
            "path": rel,
            "bytes": p.stat().st_size,
            "sha256": sha256(p),
        })

    manifest = {
        "assignment": "ERA V5 Session 13 — Reversible LLM Training Lab",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "audit": "PASS",
        "required_training_tokens_per_main_run": 50_000_000,
        "parameter_count": 20_000_768,
        "excluded": [
            "data/ token caches",
            "checkpoints/",
            "Python caches",
        ],
        "files": manifest_files,
    }
    manifest_path = OUT / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
        z.write(manifest_path, "MANIFEST.json")

    print(f"Evidence bundle: {ZIP}")
    print(f"Files: {len(files)}")
    print(f"ZIP SHA-256: {sha256(ZIP)}")


if __name__ == "__main__":
    main()
