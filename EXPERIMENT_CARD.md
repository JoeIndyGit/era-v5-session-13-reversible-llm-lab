# Experiment Card

## Primary research question
Can a reversible transformer trade recomputation for enough activation-memory savings to increase physical batch capacity while preserving comparable LM quality?

## Pre-registered reversible variant selection
A 2M-token pilot compares Midpoint and Leapfrog at h=0.25 under identical seeds/data/batch/optimizer. Stable = finite losses plus reconstruction max error <1e-3. Primary selection metric = held-out validation loss; throughput is tie-breaker. The selected variant is persisted before required 50M reversible runs.

## Required runs
1. `baseline_fixed`: standard residual, batch 16, exactly 50M valid targets.
2. `reversible_fixed`: selected reversible variant, batch 16, exactly 50M valid targets.
3. `reversible_max_batch`: selected reversible variant, measured maximum memory-feasible physical batch under the 10-update capacity rule, exactly 50M valid targets.

## Extra controls
- baseline maximum-batch probe using the same 10-update rule.
- optional baseline gradient-accumulation run matched to reversible effective batch.
- deterministic top-1 qualitative generations.

## Primary outcomes
Final 100-step mean train loss; final validation loss/perplexity; median/mean training tokens/s; peak allocated/reserved CUDA GiB; wall time; batch-capacity uplift; reconstruction error.

## Integrity rules
No fabricated metrics. No manual README result entry. One hardware type across required runs. Preserve raw evidence and environment snapshot.
