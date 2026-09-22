# Submission checklist

- [ ] Full-depth CUDA/precision gates pass for the selected reversible variant.
- [ ] Midpoint/leapfrog pilot evidence explains the variant decision before required reversible training.
- [ ] Baseline completes exactly 50,000,000 successful-update target tokens.
- [ ] Reversible fixed-batch run completes exactly 50,000,000 tokens at the same batch.
- [ ] Capacity probes show an adjacent passing/failing memory boundary, with at least 10 successful updates at the passing batch.
- [ ] Reversible maximum-batch run uses the measured batch and completes exactly 50,000,000 tokens.
- [ ] Runs share model sizes, data, seed, optimizer, GPU type, software versions and precision.
- [ ] Final losses, perplexity, median/aggregate tokens/s and allocated/reserved peaks match the step logs.
- [ ] Reconstruction is checked at full depth and actual training precision after training.
- [ ] Executed notebooks contain real outputs, no errors, and match their source notebooks.
- [ ] README contains measured tables, six figures, qualitative examples and interpretation/limitations.
- [ ] `python scripts/audit_results.py` passes.
- [ ] `python scripts/package_evidence.py` produces the evidence ZIP and SHA-256 manifest.
- [ ] Measured results, generated figures, README and executed notebooks are committed to GitHub.

The optional matched-effective-batch control is useful additional analysis; it is not required for the three-run assignment.
