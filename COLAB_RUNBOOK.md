# Colab execution and recovery

## Start

1. Open [notebook 06](https://colab.research.google.com/github/JoeIndyGit/era-v5-session-13-reversible-llm-lab/blob/main/notebooks/06_one_click_colab_submission.ipynb).
2. Select **Runtime → Change runtime type → GPU**.
3. Keep **PERSIST_TO_DRIVE = True**, authorize your own Drive mount, and run every cell.
4. Keep the default fixed batch of 16 and precision `auto` unless the initial checks show that the hardware needs another choice. Select any alternative before starting the comparison.

The repository, checkpoints and measured output persist in `MyDrive/ERA_V5_S13/repository`. The approximately 106 MB of token arrays are read into host RAM for each training run, so training does not repeatedly sample random windows from Drive.

The runner executes the actual notebooks and writes outputs into `executed_notebooks/`. It performs full-depth GPU precision checks, two 2M-token variant pilots, three required 50M-token runs, both batch-capacity probes, the final audit, report generation and packaging.

## If the runtime disconnects

Reconnect to the same GPU type with the same PyTorch/CUDA versions. Reopen notebook 06 and run its cells again with identical settings. The persisted repository remains at its original source commit while results/checkpoints exist. Completed artifacts are checked before reuse; incomplete training resumes at the most recent atomic checkpoint.

A checkpoint contains parameters, Adam state, gradient scaler, token/step counters, sampling state and Python/NumPy/Torch/CUDA RNG state. Work after the last checkpoint is replayed; it is not counted twice. At most 499 successful updates normally need replaying. The final update also creates a checkpoint.

A notebook kernel restart on the same retained runtime and a full Colab runtime replacement are different events. Drive preserves files across both; an unmounted temporary `/content` checkout does not survive a full runtime replacement.

If source, data, precision or hardware changes, the pipeline rejects reuse. Preserve the existing experiment folder and create a fresh checkout/output folder for the new comparison. Do not combine its throughput/memory figures with previous hardware results.

## Numerical or memory failures

A failed mixed-precision correctness gate stops the run before the pilots. If neither candidate passes, use `PRECISION = 'fp32'` for all three runs in a **fresh experiment folder**. If the fixed batch cannot fit, similarly select a smaller fixed batch before running a new comparison.

The maximum-batch search needs an observed memory failure or the declared 96% allocator limit immediately above its passing batch. A numerical failure is not a capacity result. If a search hits its safety cap, increase the cap deliberately in a fresh capacity measurement; do not report the lower bound as a measured maximum.

## Completion

The final console must contain:

```text
FINAL EVIDENCE AUDIT: PASS
COMPLETE: measured results, figures, README, executed notebooks and evidence ZIP are ready.
```

The ZIP is `submission_evidence/era-v5-session-13-evidence.zip`; `MANIFEST.json` lists the SHA-256 hashes of its contents. The bundle includes the tokenizer and dataset metadata but excludes token arrays and training checkpoints.

Commit these generated paths to the GitHub repository:

- `README.md`
- `results/`, including pilot JSON/CSV, selected variant, capacity probes, GPU checks and environment
- `assets/`
- `executed_notebooks/`

Use the README to present the findings and the CSV logs to verify every headline number. The grader should not need to rerun training merely to see the submitted measurements.
