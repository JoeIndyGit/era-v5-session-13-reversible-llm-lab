#!/usr/bin/env python3
"""Fail-closed audit: recompute headline metrics from logs and verify provenance."""
from pathlib import Path
import argparse
import csv
import json
import math
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evidence import canonical_hash, sha256, source_digest

REQUIRED_RUNS = ['baseline_fixed', 'reversible_fixed', 'reversible_max_batch']
TRAIN_NOTEBOOKS = ['00_setup_and_validation', '00b_variant_selection', '01_baseline_50m',
                   '02_reversible_fixed_batch_50m', '03_reversible_max_batch_50m']


def finite_number(value, positive=False):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and (not positive or value > 0)


def close(a, b):
    return finite_number(a) and finite_number(b) and math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-9)


def read_json(path, errors):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as e:
        errors.append(f'{path.name}: missing or invalid JSON ({type(e).__name__})')
        return None


def check_run(result, log_path, expected_tokens=50_000_000):
    """Also used on tiny real CPU runs by regression tests; final audit adds CUDA gates."""
    errors = []
    if not result or result.get('schema_version') != 2 or result.get('status') != 'completed':
        return ['missing a completed schema-v2 result']
    if result.get('signature') != canonical_hash(result.get('identity')):
        errors.append('run signature does not match its recorded identity')
    if result.get('tokens_seen') != expected_tokens or result.get('target_tokens') != expected_tokens:
        errors.append('token budget is incomplete')
    if not log_path.exists(): return errors + ['step CSV is missing']
    if result.get('steps_sha256') != sha256(log_path): errors.append('step CSV hash differs from result')
    try:
        with log_path.open() as stream:
            rows = list(csv.DictReader(stream))
        if not rows: return errors + ['step CSV is empty']
        budget = int(result['batch_size']) * int(result['seq_len'])
        seen = 0; session_counts = {}; steady = []; losses = []; vals = []
        allocated = []; reserved = []; retries_total = 0
        for number, row in enumerate(rows, 1):
            valid = int(row['valid_tokens']); loss = float(row['train_loss']); duration = float(row['step_time_s'])
            if int(row['step']) != number: errors.append(f'nonconsecutive step at CSV row {number}')
            if valid != min(budget, expected_tokens - seen): errors.append(f'incorrect valid-token count at step {number}')
            seen += valid
            if int(row['tokens_seen']) != seen: errors.append(f'cumulative-token mismatch at step {number}')
            if not finite_number(loss) or not finite_number(duration, True) or not finite_number(float(row['grad_norm'])):
                errors.append(f'non-finite loss/gradient or invalid duration at step {number}')
            if not close(float(row['tokens_per_s']), valid / duration): errors.append(f'throughput mismatch at step {number}')
            losses.append(loss)
            if not close(float(row['train_loss_100step_mean']), statistics.mean(losses[-100:])):
                errors.append(f'rolling loss mismatch at step {number}')
            if row['val_loss'] != '': vals.append(float(row['val_loss']))
            session = int(row['session']); session_counts[session] = session_counts.get(session, 0) + 1
            retries = int(row['overflow_retries']); retries_total += retries
            eligible = session_counts[session] > result['training_config']['speed_warmup_steps'] and valid == budget and retries == 0
            if (row['steady_state'] == 'True') != eligible: errors.append(f'incorrect throughput eligibility at step {number}')
            if eligible: steady.append((valid, duration, float(row['tokens_per_s'])))
            if row['peak_allocated_gib']: allocated.append(float(row['peak_allocated_gib']))
            if row['peak_reserved_gib']: reserved.append(float(row['peak_reserved_gib']))
        if result.get('hardware', {}).get('device') == 'cuda':
            if len(allocated) != len(rows) or len(reserved) != len(rows): errors.append('GPU memory is missing from step rows')
            if any(not finite_number(a, True) or not finite_number(b, True) or a > b for a, b in zip(allocated, reserved)):
                errors.append('invalid allocated/reserved memory in step rows')
        if len(rows) != result.get('optimizer_steps'): errors.append('optimizer-step total differs from CSV')
        if seen != expected_tokens: errors.append('CSV does not reach the token budget')
        if retries_total != result.get('overflow_retries'): errors.append('overflow retry total differs from CSV')
        if not close(losses[-1], result.get('final_step_train_loss')): errors.append('final-step loss differs from CSV')
        if not close(statistics.mean(losses[-100:]), result.get('final_train_loss_100step_mean')): errors.append('final mean loss differs from CSV')
        if not vals or not close(vals[-1], result.get('final_val_loss')): errors.append('final validation loss differs from CSV')
        if rows[-1]['val_loss'] == '': errors.append('final validation was not performed at the token budget')
        if not close(math.exp(result['final_val_loss']), result.get('final_val_perplexity')): errors.append('perplexity differs from exp(validation loss)')
        if not steady: errors.append('no steady-state throughput samples')
        else:
            checks = {'median_tokens_per_s': statistics.median(x[2] for x in steady),
                      'mean_tokens_per_s': statistics.mean(x[2] for x in steady),
                      'aggregate_tokens_per_s': sum(x[0] for x in steady) / sum(x[1] for x in steady)}
            for key, value in checks.items():
                if not close(value, result.get(key)): errors.append(f'{key} differs from CSV')
        if result.get('throughput_samples_count') != len(steady): errors.append('throughput sample count differs from CSV')
        for key, numbers in [('peak_allocated_gib', allocated), ('peak_reserved_gib', reserved)]:
            if numbers and not close(max(numbers), result.get(key)): errors.append(f'{key} differs from CSV')
    except (ValueError, KeyError, TypeError, ZeroDivisionError, OverflowError) as e:
        errors.append(f'invalid result/CSV structure: {type(e).__name__}: {e}')
    return errors


def check_probe(probe):
    if not probe: return ['missing probe']
    errors = []
    if probe.get('schema_version') == 2 and probe.get('signature') != canonical_hash({'identity': probe.get('identity'), 'protocol': probe.get('protocol')}):
        errors.append('capacity signature does not match its protocol')
    low = probe.get('largest_stable_batch'); high = probe.get('first_failed_batch')
    if not probe.get('search_complete') or not isinstance(low, int) or high != low + 1:
        return ['maximum lacks an adjacent observed passing/failing bracket']
    attempts = {r.get('batch_size'): r for r in probe.get('attempts', [])}
    passing, failing = attempts.get(low, {}), attempts.get(high, {})
    if not passing.get('stable') or passing.get('successful_updates', 0) < 10:
        errors.append('claimed maximum did not complete 10 successful optimizer updates')
    if failing.get('stable') is not False or not (failing.get('error') == 'CUDA OOM' or (failing.get('error') or '').startswith('reserved fraction')):
        errors.append('upper boundary is not an observed memory/safety-limit failure')
    if probe.get('trial_steps', 0) < 10: errors.append('probe uses fewer than 10 updates')
    return errors


def audit(root=ROOT, require_notebooks=True):
    root = Path(root); out = root / 'results'; errors = []
    runs = {name: read_json(out / f'{name}.json', errors) for name in REQUIRED_RUNS}
    selection = read_json(out / 'variant_selection.json', errors)
    selected = read_json(out / 'selected_variant.json', errors)
    gpu = read_json(out / 'gpu_correctness.json', errors)
    probes = {name: read_json(out / f'{name}_batch_probe.json', errors) for name in ['baseline', 'reversible']}
    if not (out / 'environment.txt').is_file(): errors.append('environment.txt is missing')
    source = source_digest()
    for name, run in runs.items():
        if not run: continue
        errors.extend(f'{name}: {e}' for e in check_run(run, out / f'{name}_steps.csv'))
        if run.get('parameters') != 20_000_768: errors.append(f'{name}: incorrect parameter count')
        hardware = run.get('hardware', {})
        if hardware.get('device') != 'cuda': errors.append(f'{name}: GPU evidence is missing')
        for field in ['gpu_name', 'gpu_total_memory_gib', 'cuda_version', 'torch_version', 'git_commit']:
            if hardware.get(field) is None: errors.append(f'{name}: missing hardware/source field {field}')
        for key in ['peak_allocated_gib', 'peak_reserved_gib', 'median_tokens_per_s', 'aggregate_tokens_per_s']:
            if not finite_number(run.get(key), True): errors.append(f'{name}: invalid {key}')
        if run.get('identity', {}).get('source_sha256') != source: errors.append(f'{name}: results were produced by different source code')
        data = run.get('dataset') or {}
        for field in ['tokenizer_sha256', 'train_sha256', 'val_sha256', 'dataset_revision']:
            if not data.get(field): errors.append(f'{name}: missing dataset provenance {field}')
        recon = run.get('reconstruction')
        if name != 'baseline_fixed':
            if (not recon or not recon.get('passed') or recon.get('layers') != 22 or recon.get('context_length') != 256
                    or recon.get('precision') != {'bf16': 'bfloat16', 'fp16': 'float16', 'fp32': 'float32'}.get(run.get('precision'))):
                errors.append(f'{name}: reconstruction is not validated at actual depth/context/precision')
    if all(runs.values()):
        reference = runs['baseline_fixed']
        for name, run in runs.items():
            for field in ['gpu_name', 'gpu_total_memory_gib', 'cuda_version', 'torch_version', 'git_commit']:
                if run['hardware'].get(field) != reference['hardware'].get(field): errors.append(f'{name}: hardware/source mismatch {field}')
            if run.get('precision') != reference.get('precision'): errors.append(f'{name}: precision mismatch')
            for field in ['tokenizer_sha256', 'train_sha256', 'val_sha256', 'dataset_revision']:
                if (run.get('dataset') or {}).get(field) != (reference.get('dataset') or {}).get(field): errors.append(f'{name}: dataset mismatch {field}')
            for field in ['seed', 'target_tokens', 'max_lr', 'min_lr', 'warmup_tokens', 'betas', 'weight_decay', 'grad_clip', 'eval_every_tokens', 'eval_batches', 'eval_batch_size']:
                if run['training_config'].get(field) != reference['training_config'].get(field): errors.append(f'{name}: protocol mismatch {field}')
        if reference.get('batch_size') != runs['reversible_fixed'].get('batch_size'): errors.append('fixed batches differ')
    for name, probe in probes.items():
        errors.extend(f'{name} probe: {e}' for e in check_probe(probe))
        if probe and probe.get('identity', {}).get('source_sha256') != source: errors.append(f'{name} probe: different source')
        if probe and runs['baseline_fixed']:
            for field in ['hardware', 'precision', 'dataset']:
                if probe.get('identity', {}).get(field) != runs['baseline_fixed'].get('identity', {}).get(field): errors.append(f'{name} probe: provenance mismatch {field}')
    if probes['reversible'] and runs['reversible_max_batch']:
        if runs['reversible_max_batch']['batch_size'] != probes['reversible']['largest_stable_batch']: errors.append('maximum run does not use the measured maximum')
        if runs['reversible_max_batch']['started_utc'] < probes['reversible']['completed_utc']: errors.append('maximum run predates its capacity measurement')
    if selection and selected and gpu:
        if selection.get('pilot_tokens_per_candidate') != 2_000_000: errors.append('variant pilot budget differs from declared 2M tokens')
        chosen = selection.get('selected_variant')
        stable = [r for r in selection.get('candidates', []) if r.get('stable')]
        if not stable or min(stable, key=lambda r: (r['val_loss'], -(r.get('median_tokens_per_s') or 0)))['variant'] != chosen:
            errors.append('selected variant does not follow the declared selection rule')
        if selected.get('selected_variant') != chosen: errors.append('variant handoff disagrees with selection')
        gate = gpu.get('variants', {}).get(chosen, {})
        if not gate.get('passed') or gate.get('layers') != 22 or gate.get('context_length') != 256 or gate.get('parameters') != 20_000_768:
            errors.append('selected variant failed/misses the full-depth GPU gate')
        if gpu.get('source_sha256') != source or selection.get('source_sha256') != source: errors.append('GPU/selection evidence uses different source')
        if runs['baseline_fixed']:
            if gpu.get('precision') != runs['baseline_fixed'].get('precision'): errors.append('GPU validation precision differs from training')
            for field in ['gpu_name', 'gpu_total_memory_gib', 'torch_version', 'cuda_version']:
                if gpu.get('hardware', {}).get(field) != runs['baseline_fixed'].get('hardware', {}).get(field): errors.append(f'GPU correctness hardware mismatch: {field}')
        for name in ['reversible_fixed', 'reversible_max_batch']:
            run = runs[name]
            if run and (run.get('reversible_variant') != chosen or run['started_utc'] < selection['selected_utc']): errors.append(f'{name}: variant selection was not respected before training')
        for candidate in stable:
            path = out / 'pilots' / f'pilot_{candidate["variant"]}.json'
            pilot = read_json(path, errors)
            if pilot:
                errors.extend(f'{path.name}: {e}' for e in check_run(pilot, path.with_name(path.stem + '_steps.csv'), selection['pilot_tokens_per_candidate']))
                if pilot.get('signature') != candidate.get('signature') or not close(pilot.get('final_val_loss'), candidate.get('val_loss')):
                    errors.append(f'{path.name}: selection summary differs from pilot')
                if pilot['completed_utc'] > selection['selected_utc']: errors.append('selection predates pilot completion')
    if require_notebooks:
        for name in TRAIN_NOTEBOOKS:
            nb = read_json(root / 'executed_notebooks' / (name + '.ipynb'), errors)
            if not nb: continue
            original = root / 'notebooks' / (name + '.ipynb')
            if not original.exists() or nb.get('metadata', {}).get('execution_record', {}).get('source_notebook_sha256') != sha256(original):
                errors.append(f'{name}: executed notebook does not match its source')
            if nb.get('metadata', {}).get('execution_record', {}).get('status') != 'completed':
                errors.append(f'{name}: execution did not complete')
            cells = [c for c in nb.get('cells', []) if c.get('cell_type') == 'code']
            if not cells or any(c.get('execution_count') is None for c in cells): errors.append(f'{name}: notebook has unexecuted cells')
            if any(o.get('output_type') == 'error' for c in cells for o in c.get('outputs', [])): errors.append(f'{name}: notebook contains execution errors')
    return errors


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--without-notebooks', action='store_true')
    args = parser.parse_args()
    errors = audit(require_notebooks=not args.without_notebooks)
    print('FINAL EVIDENCE AUDIT: ' + ('FAIL' if errors else 'PASS'))
    for error in errors: print(' -', error)
    if errors: raise SystemExit(1)
    print('Three complete, comparable runs; metrics recomputed from CSV; GPU gates, pilots, batch boundary and executed notebooks verified.')


if __name__ == '__main__': main()
