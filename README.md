# ERA V5 — Session 13: Reversible LLM Training Lab

**20,000,768 parameters · three 50M-token experiments · explicit midpoint and leapfrog · measured batch capacity**

This experiment asks how reversible transformer dynamics change training memory, throughput, batch capacity and language-model quality. Every headline result is generated from recorded training output. The tables remain unpopulated until the experiments and evidence audit complete.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JoeIndyGit/era-v5-session-13-reversible-llm-lab/blob/main/notebooks/06_one_click_colab_submission.ipynb)

Select a GPU, run notebook **06**, and keep its Google Drive persistence enabled. It executes the core notebooks, saves their actual outputs, resumes interrupted training and generates the report. See [COLAB_RUNBOOK.md](COLAB_RUNBOOK.md) for recovery and submission steps.

## Assignment and evidence

| Requirement | Experiment or evidence |
|---|---|
| Train a roughly 20M LLM | Exact parameter-count assertion: **20,000,768** |
| Baseline, 50M tokens, fixed batch | `01_baseline_50m.ipynb`; default batch **16** |
| Train with reversibility at the same batch | `02_reversible_fixed_batch_50m.ipynb` |
| Report which reversible variant worked | Full-depth GPU checks, then `00b_variant_selection.ipynb` |
| Reversible model at maximum batch | `03_reversible_max_batch_50m.ipynb` |
| Final loss, speed and memory | Result JSON, step CSV and generated tables below |
| Other findings | Quality differences, recomputation cost, batch uplift, reconstruction accuracy and limitations |
| Detailed README and notebooks | This report, notebook sources, and generated `executed_notebooks/` |

The required runs use **150,000,000 successful-update target tokens in total**. Two preliminary variant pilots add 2M tokens each. Capacity probes and correctness checks are recorded separately.

## Model and data

| Component | Setting |
|---|---:|
| Architecture | Decoder-only causal transformer |
| Vocabulary | 10,000-token ByteLevel BPE |
| Context length | 256 |
| Transformer layers | 22 |
| Hidden dimension / attention heads | 256 / 8 |
| MLP dimension | 1,024 |
| Dropout | 0.0 |
| Embedding and language-model head | Tied weights |
| Parameters | 20,000,768 |
| Share in transformer blocks | 86.87% |

The model is deliberately deep so activation storage contributes meaningfully to the experiment. A custom vocabulary avoids spending most of the parameter budget on embeddings.

TinyStories is downloaded at a recorded Hugging Face dataset revision. A ByteLevel BPE is trained on 100,000 training stories. Setup produces a 52M-token training cache and a separate 1M-token validation cache. The tokenizer, dataset revision and SHA-256 hashes are recorded; cached files are rehashed before reuse. Sampling uses reproducible random windows with replacement, so **50M processed target tokens does not mean 50M distinct corpus positions**.

Host RAM holds the approximately 106 MB of token arrays, avoiding random Google Drive reads during training. The validation windows are fixed across evaluations and all three architectures.

## Reversible dynamics

A transformer block produces an update $f_{\theta_\ell}(p)$ comprising causal attention and an MLP. The baseline uses the ordinary residual update. The reversible variants retain the final two hidden states and reconstruct earlier states during backward recomputation.

**Explicit midpoint**

$$p_{\ell+1}=p_{\ell-1}+2h f_{\theta_\ell}(p_\ell)$$

Its inverse is $p_{\ell-1}=p_{\ell+1}-2h f_{\theta_\ell}(p_\ell)$.

**Leapfrog**

$$p_{\ell+1}=2p_\ell-p_{\ell-1}+h^2 f_{\theta_\ell}(p_\ell)$$

Its inverse is $p_{\ell-1}=2p_\ell-p_{\ell+1}+h^2 f_{\theta_\ell}(p_\ell)$.

Both start with an **Euler bootstrap**, $p_1=p_0+h f_{\theta_0}(p_0)$, using $h=0.25$. Dropout is zero so recomputation uses the same deterministic update function.

These recurrences follow Gal et al., [Reversing Large Language Models for Efficient Training and Fine-Tuning](https://arxiv.org/abs/2512.02056). Identical parameter tensors do **not** make the baseline and reversible forward functions identical. The comparison therefore measures the complete architecture/training trade-off; its loss difference is not solely an effect of activation storage.

## Correctness before long runs

CPU checks cover exact parameter count, identical initialization, both custom backward passes against autograd, reconstruction, partial-batch accounting, and the token-based learning-rate schedule.

```bash
python scripts/validate.py
python -m unittest discover -s tests -v
```

The regression suite interrupts real small-model training and verifies **identical final model weights, Adam states, sampling states and losses after recovery**, for baseline, midpoint and leapfrog. It also tests overflow retries, stale-checkpoint rejection, tampered evidence, capacity boundaries and notebook error handling. These miniature tests are not benchmark results.

Before variant pilots, `scripts/validate_gpu.py` checks the **full 22-layer model at context 256**, after three optimizer updates, under the actual selected CUDA precision. It compares every parameter gradient with an ordinary-autograd reference and measures reconstruction error. CPU correctness cannot substitute for this GPU artifact.

Predeclared numerical tolerances are recorded in every diagnostic. FP32 reconstruction must have relative L2 error at most $10^{-4}$ and maximum absolute error at most $10^{-3}$; mixed precision uses 0.02 and 0.05 respectively. The 0.02 relative bound is applied to the full-depth BF16 GPU check and to the trained-model round-trip; the observed error is recorded for every run. Gradient relative L2 tolerance is $10^{-4}$ for FP32 and 0.05 for mixed precision. Both absolute and relative errors are reported, and the trained reversible models must pass reconstruction again at the end.

If neither variant passes, the pipeline stops. A new, controlled FP32 experiment is the documented fallback; precision is never changed halfway through a comparison.

## Variant selection

Eligible midpoint and leapfrog models receive identical initialization, sampling, batch, optimizer and a **2M-token pilot**. Their LR schedule follows the opening portion of the 50M-token schedule. Among candidates passing the numerical gates, lowest held-out validation loss wins; median throughput breaks an exact tie. The decision is timestamped before the required reversible runs.

<!-- VARIANT_TABLE_START -->
> Run notebook 00b to generate the measured variant decision.
<!-- VARIANT_TABLE_END -->

## Controlled training protocol

All three runs share data, seed, model parameter shapes, optimizer settings, GPU type, software versions and numerical precision. Each starts from fresh, identically seeded parameters; the pilots are not continued into the required runs.

| Setting | Value |
|---|---:|
| Seed | 1337 |
| Optimizer | AdamW |
| Betas / weight decay | (0.9, 0.95) / 0.1 |
| Maximum / minimum learning rate | 0.0003 / 0.00003 |
| Warmup | 1M tokens |
| Decay | Cosine by successful-update token count |
| Gradient clipping | Global norm 1.0 |
| Required budget | Exactly 50,000,000 valid targets per run |
| Fixed batch | 16 by default; configurable before starting the comparison |

The final partial batch masks surplus targets with `-100`. At batch 16 and context 256, a required run has **12,208 successful updates**, with 128 valid targets in the last update. FP16 overflow replays the same batch at a lower scale; skipped optimizer steps never advance the token budget. Retries are reported.

### Maximum batch

Both architectures are probed with full forward/backward/AdamW updates. Candidates must complete at least **10 successful updates** and remain within **96% of total GPU memory reserved by the allocator**. Search doubles until an observed memory boundary, then finds adjacent passing/failing integer batches by binary search. A numerical failure is not accepted as a memory boundary, and a safety-cap result is only a lower bound.

Run C trains for the complete 50M tokens at the measured reversible maximum. The claim is precisely **maximum memory-feasible batch under this declared probe and allocator-margin rule**, not an unrestricted hardware maximum or a guarantee of convergence.

<!-- BATCH_TABLE_START -->
> Run notebook 03 to generate both capacity measurements.
<!-- BATCH_TABLE_END -->

## Measurement definitions

- **Final loss:** final-step training loss and the final 100-update mean are preserved; held-out cross-entropy and perplexity are the primary quality comparison. A 100-update window covers different token counts at different batches.
- **Speed:** synchronized training-step timing includes forward, backward, optimizer update and clipping. Sampling/transfer, validation, checkpoint writes and reporting are outside this timer. The first 20 updates after each process start, partial batches and overflow retries are excluded from steady-state summaries. Both median tokens/s and total measured tokens divided by total measured seconds are reported.
- **Memory:** training peak allocated and reserved CUDA GiB include the model, gradients, optimizer and activations. Evaluation peak allocated memory is recorded separately. Reserved memory may include allocator blocks retained from earlier operations. CUDA figures exclude host RAM, driver allocations outside PyTorch and other processes.
- **Recovery:** checkpoints include model, optimizer, scaler, token counters, Python/NumPy/Torch/CUDA RNGs and both data samplers. They are written atomically every 500 updates. After a crash, any CSV tail beyond the latest committed checkpoint is discarded and those updates are replayed.
- **Provenance:** completed runs and checkpoints are reused only when source, protocol, hardware, precision and data signatures match. Step CSV hashes are verified before completed results are reused.

<!-- HARDWARE_TABLE_START -->
> Hardware and precision will be populated from the measured runs.
<!-- HARDWARE_TABLE_END -->

## Measured results

<!-- RESULTS_TABLE_START -->
> No benchmark values are pre-filled. Execute the required runs and notebook 04.
<!-- RESULTS_TABLE_END -->

<!-- FINDINGS_START -->
> Measured findings will be generated after the evidence audit passes.
<!-- FINDINGS_END -->

![Experiment summary](assets/executive_summary.png)

Notebook 04 generates loss-versus-token, memory, throughput, quality-versus-memory and batch-capacity figures in `assets/`. The loss plot includes held-out validation points to avoid relying only on differently sized smoothing windows.

## Qualitative language-model check

<!-- QUALITATIVE_TABLE_START -->
> Greedy completions from the fixed-batch runs will appear here.
<!-- QUALITATIVE_TABLE_END -->

## Interpretation and limitations

The report should be read as one controlled run per condition, without confidence intervals across training seeds. Same-batch measurements quantify the observed memory/throughput trade-off. Increasing batch also changes optimizer-step count and optimization dynamics, even with a token-based LR schedule; a better or worse maximum-batch loss cannot be attributed to memory savings alone.

Reconstruction removes the need to retain every layer's activations, but embeddings, logits, gradients and Adam states still occupy memory. The $B\times T\times V$ logits can limit batch capacity after transformer activation storage is reduced. No speedup or memory reduction is assumed in advance.

`05_optional_matched_effective_batch_control.ipynb` provides an additional baseline gradient-accumulation comparison. It is a research extension, not a replacement for any required experiment and not included in the required evidence audit.

## Execution, recovery and submission

```bash
pip install -r requirements.txt
python scripts/run_full_submission.py --precision auto --batch-size 16
```

The runner executes **00 → 00b → 01 → 02 → 03 → 04** in separate notebook kernels. Live text is streamed, actual cell outputs are saved after execution, and a failed cell stops the pipeline. Re-run the same command to resume the same experiment.

```bash
python scripts/audit_results.py
python scripts/package_evidence.py
```

The final audit recomputes losses, perplexity, throughput and memory peaks from CSV logs; verifies all three token budgets; checks GPU precision, data and source provenance; validates the variant decision and measured batch handoff; and requires completed notebook outputs matching their sources.

Commit generated **`README.md`, `results/`, `assets/`, and `executed_notebooks/`**. The compact evidence ZIP also preserves the tokenizer, dataset metadata, code and a SHA-256 manifest. Keep large training checkpoints and token arrays outside GitHub.

## References

- Gal et al. [Reversing Large Language Models for Efficient Training and Fine-Tuning](https://arxiv.org/abs/2512.02056).
- Eldan and Li. [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759).
- PyTorch. [Automatic mixed precision examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).

Repository: [JoeIndyGit/era-v5-session-13-reversible-llm-lab](https://github.com/JoeIndyGit/era-v5-session-13-reversible-llm-lab).
