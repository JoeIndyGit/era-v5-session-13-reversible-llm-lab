#!/usr/bin/env python3
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md",
    "EXPERIMENT_CARD.md",
    "SUBMISSION_CHECKLIST.md",
    "COLAB_RUNBOOK.md",
    "requirements.txt",
    "configs/common.json",
    "configs/model_baseline.json",
    "configs/model_midpoint.json",
    "configs/model_leapfrog.json",
    "src/model.py",
    "src/reversible.py",
    "src/train.py",
    "src/data.py",
    "src/evidence.py",
    "src/checkpointing.py",
    "src/diagnostics.py",
    "scripts/validate_gpu.py",
    "tests/test_experiment.py",
    "scripts/validate.py",
    "scripts/audit_results.py",
    "scripts/render_report.py",
    "scripts/run_full_submission.py",
    "scripts/package_evidence.py",
    "notebooks/00_setup_and_validation.ipynb",
    "notebooks/00b_variant_selection.ipynb",
    "notebooks/01_baseline_50m.ipynb",
    "notebooks/02_reversible_fixed_batch_50m.ipynb",
    "notebooks/03_reversible_max_batch_50m.ipynb",
    "notebooks/04_analysis_and_report.ipynb",
    "notebooks/05_optional_matched_effective_batch_control.ipynb",
    "notebooks/06_one_click_colab_submission.ipynb",
]

errors = []

for rel in REQUIRED:
    if not (ROOT / rel).exists():
        errors.append(f"missing required file: {rel}")

for p in (ROOT / "configs").glob("*.json"):
    try:
        json.loads(p.read_text())
    except Exception as e:
        errors.append(f"invalid config JSON {p.relative_to(ROOT)}: {e}")

for p in (ROOT / "notebooks").glob("*.ipynb"):
    try:
        nb = json.loads(p.read_text())
        code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
        for cell in code_cells:
            source = "".join(cell.get("source", []))
            if any(line.lstrip().startswith("!") for line in source.splitlines()):
                errors.append(f"{p.name}: shell command must propagate failure through check=True")
        if nb.get("nbformat") != 4:
            errors.append(f"{p.name}: expected nbformat 4")
        if not isinstance(nb.get("cells"), list) or not nb["cells"]:
            errors.append(f"{p.name}: no notebook cells")
    except Exception as e:
        errors.append(f"invalid notebook JSON {p.relative_to(ROOT)}: {e}")

readme = (ROOT / "README.md").read_text()
for marker in [
    "<!-- RESULTS_TABLE_START -->",
    "<!-- RESULTS_TABLE_END -->",
    "<!-- FINDINGS_START -->",
    "<!-- FINDINGS_END -->",
    "<!-- BATCH_TABLE_START -->",
    "<!-- BATCH_TABLE_END -->",
    "<!-- VARIANT_TABLE_START -->",
    "<!-- VARIANT_TABLE_END -->",
]:
    if marker not in readme:
        errors.append(f"README missing report-render marker: {marker}")

runner = (ROOT / "scripts" / "run_full_submission.py").read_text()
if "require_cuda()" not in runner:
    errors.append("full submission runner no longer enforces CUDA")

if errors:
    print("REPOSITORY INTEGRITY: FAIL")
    for e in errors:
        print(" -", e)
    sys.exit(1)

print("REPOSITORY INTEGRITY: PASS")
print(f"Validated {len(REQUIRED)} required files, configs, notebooks, README markers, and CUDA gate.")
