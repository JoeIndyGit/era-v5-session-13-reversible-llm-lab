# Submission Checklist

## Pre-flight
- [ ] GPU runtime enabled.
- [ ] `python scripts/validate.py` passes Midpoint **and** Leapfrog gates.
- [ ] `results/environment.txt` captured.
- [ ] `dataset_meta.json` contains tokenizer + train + validation SHA-256 hashes.

## Variant selection
- [ ] Run `00b_variant_selection.ipynb`.
- [ ] `results/variant_selection.json` contains both candidates.
- [ ] `results/selected_variant.json` exists **before** required reversible 50M runs.

## Required 50M evidence
- [ ] `baseline_fixed.json` + CSV.
- [ ] `reversible_fixed.json` + CSV.
- [ ] `baseline_batch_probe.json`.
- [ ] `reversible_batch_probe.json`.
- [ ] Both probes use `trial_steps: 10`.
- [ ] Both searches have an observed failure bracket (`search_complete: true`).
- [ ] `reversible_max_batch.json` + CSV.
- [ ] Every required run reports exactly `tokens_seen = 50,000,000`.

## Final report
- [ ] `python scripts/audit_results.py` reports **PASS**.
- [ ] Run notebook 04.
- [ ] README no longer contains result placeholders.
- [ ] `executive_summary.png` exists.
- [ ] Loss, memory, throughput, quality-memory, and batch-frontier plots exist.
- [ ] Variant-selection and batch-frontier tables are populated.
- [ ] Allocated + reserved memory, validation loss/PPL, tok/s, reconstruction error are visible.
- [ ] Qualitative top-1 completions are present or a recorded reason explains their absence.

## Optional research bonus
- [ ] Run notebook 05 and preserve `baseline_matched_effective_batch.json` if compute budget permits.

## GitHub
- [ ] All notebooks committed.
- [ ] Source/config/scripts committed.
- [ ] Raw result JSON/CSV + environment + generated assets committed.
- [ ] GitHub Actions validate passes.
- [ ] README images render.
- [ ] Submit the repository/README link.

## Final evidence bundle
- [ ] Run `python scripts/package_evidence.py` only after the final audit passes.
- [ ] `submission_evidence/MANIFEST.json` exists.
- [ ] `submission_evidence/era-v5-session-13-evidence.zip` exists.
- [ ] Preserve the printed ZIP SHA-256 alongside the submitted artifact.
- [ ] Verify the bundle excludes `data/` caches and temporary checkpoints.
