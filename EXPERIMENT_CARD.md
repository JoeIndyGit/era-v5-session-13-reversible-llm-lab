# Experiment card

## Question
How do reversible dynamics change activation memory, throughput, physical batch capacity and held-out language-model quality?

## Model and required runs
All architectures have 20,000,768 parameters, 22 layers, hidden dimension 256, eight attention heads and context 256.

1. Baseline: fixed batch (default 16), exactly 50M successful-update target tokens.
2. Selected reversible variant: the same fixed batch and 50M tokens.
3. Selected reversible variant: measured maximum memory-feasible batch and 50M tokens.

## Predeclared variant selection
Full-depth GPU checks at the actual precision precede 2M-token midpoint and leapfrog pilots. Both use h=0.25, an Euler bootstrap, identical initialization/data/batch and the initial portion of the 50M-token LR schedule. Among candidates passing finite-loss, gradient and reconstruction gates, lowest validation loss wins; median throughput breaks a tie. Tolerances are recorded by `src/diagnostics.py` and in each numerical artifact.

## Capacity criterion
At least 10 successful optimizer updates; no more than 96% of total GPU memory reserved by the allocator. Binary search must end with adjacent observed passing/failing integer batches. Numerical failures do not establish a memory boundary; a search cap supplies only a lower bound.

## Measurements and controls
Final held-out loss/perplexity, final training loss, median and aggregate steady-state tokens/s, training allocated/reserved peaks, separate evaluation peak, reconstruction error, overflow retries and recovery count. The same data, initialization seed, optimizer settings, GPU type, software and precision are shared. A token-based LR schedule controls the budget; larger batches still change update count.

## Evidence
Raw JSON and CSV, pinned dataset revision and hashes, full-depth GPU checks, variant selection, batch probes, source fingerprint, environment, executed notebook outputs and generated README/figures. Audit metrics are recalculated from logs. Small CPU regression tests are kept separate from measured GPU evidence.

## Scope
One run per required condition; no multi-seed confidence intervals. The optional matched-effective-batch baseline is an additional research control, excluded from the required pipeline.
