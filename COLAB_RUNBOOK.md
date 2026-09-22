# Colab Execution Runbook

This is the shortest reliable path from a blank Colab runtime to a submission-complete Session 13 evidence bundle.

## 1. Open the canonical one-click notebook

Use:

`https://colab.research.google.com/github/JoeIndyGit/era-v5-session-13-reversible-llm-lab/blob/main/notebooks/06_one_click_colab_submission.ipynb`

The notebook now **clones the repository itself**, so it does not assume that GitHub files already exist under `/content`.

## 2. Select one GPU and keep it for the entire graded run

In Colab:

`Runtime → Change runtime type → GPU`

The assignment compares throughput and CUDA peak memory. Do not intentionally move the three required runs across different GPU types.

The final evidence audit checks GPU name, total GPU memory, CUDA/PyTorch version, precision, dataset hashes, and the Git commit recorded by every required run.

## 3. Run notebook 06 top-to-bottom

The orchestrator executes:

```text
correctness validation
        ↓
environment capture
        ↓
TinyStories 10K BPE + 52M/1M token caches
        ↓
Midpoint vs Leapfrog 2M-token selection pilot
        ↓
baseline fixed-batch — exactly 50M tokens
        ↓
selected reversible fixed-batch — exactly 50M tokens
        ↓
baseline 10-update batch frontier
        ↓
reversible 10-update batch frontier
        ↓
selected reversible max-batch — exactly 50M tokens
        ↓
final evidence audit
        ↓
README tables + figures
        ↓
evidence ZIP + SHA-256 manifest
```

The main command is:

```bash
python scripts/run_full_submission.py
```

## 4. Do not call a capped search a maximum

Both baseline and reversible batch searches must end with an **observed failing batch** above the largest passing batch.

The final report therefore uses the precise wording:

> maximum memory-feasible batch under the 10-update probe rule

If a search reaches its safety cap without observing failure, the orchestrator stops rather than overclaiming a maximum.

## 5. Completion gate

The run is complete only when these messages appear:

```text
FINAL EVIDENCE AUDIT: PASS
COMPLETE: required evidence is in results/, figures in assets/, README is populated.
```

Then run:

```bash
python scripts/package_evidence.py
```

This must create:

```text
submission_evidence/
├── MANIFEST.json
└── era-v5-session-13-evidence.zip
```

The manifest includes a SHA-256 digest for every packaged artifact.

## 6. Preserve these files in GitHub

Commit the generated:

- `README.md`
- `results/*.json`
- `results/*_steps.csv`
- `results/environment.txt`
- `assets/*.png`
- executed notebooks, if desired for the class submission

Do **not** commit the large `data/` cache or temporary checkpoints.

## 7. Reviewer evidence hierarchy

Use the README for the narrative. If a number is challenged, trace it to:

```text
README / figure
      ↓
result JSON
      ↓
step CSV or batch-probe JSON
      ↓
environment + dataset hashes
      ↓
Git commit
```

That makes the assignment auditable rather than screenshot-dependent.
