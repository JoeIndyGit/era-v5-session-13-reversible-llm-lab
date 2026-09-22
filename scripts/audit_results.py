from pathlib import Path
import json, math, sys

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'
required=['baseline_fixed','reversible_fixed','reversible_max_batch']
errors=[]

def load(name):
    p=R/f'{name}.json'
    if not p.exists(): errors.append(f'missing {p.relative_to(ROOT)}'); return None
    return json.loads(p.read_text())

runs={n:load(n) for n in required}
sel_path=R/'selected_variant.json'; sel=json.loads(sel_path.read_text()) if sel_path.exists() else None
if sel is None: errors.append('missing results/selected_variant.json')
probes={}
for n in ['baseline_batch_probe','reversible_batch_probe']:
    p=R/f'{n}.json'
    if not p.exists(): errors.append(f'missing results/{n}.json')
    else: probes[n]=json.loads(p.read_text())
if not (R/'environment.txt').exists(): errors.append('missing results/environment.txt')

if all(runs.values()):
    for n,r in runs.items():
        if r.get('parameters')!=20_000_768: errors.append(f'{n}: parameter count {r.get("parameters")}')
        if r.get('tokens_seen')!=50_000_000: errors.append(f'{n}: tokens_seen {r.get("tokens_seen")}')
        for k in ['final_train_loss_100step_mean','final_val_loss','median_tokens_per_s','peak_allocated_gib','peak_reserved_gib']:
            v=r.get(k)
            if v is None or (isinstance(v,(int,float)) and not math.isfinite(v)): errors.append(f'{n}: invalid {k}={v}')
    if runs['baseline_fixed'].get('batch_size')!=runs['reversible_fixed'].get('batch_size'):
        errors.append('fixed-batch baseline and reversible batches differ')
    for field in ['gpu_name','gpu_total_memory_gib','cuda_version','torch_version','git_commit']:
        vals={str(r.get('hardware',{}).get(field)) for r in runs.values()}
        if len(vals)!=1: errors.append(f'hardware mismatch for {field}: {vals}')
    precisions={r.get('precision') for r in runs.values()}
    if len(precisions)!=1: errors.append(f'precision mismatch: {precisions}')
    for field in ['tokenizer_sha256','train_sha256','val_sha256']:
        vals={str((r.get('dataset') or {}).get(field)) for r in runs.values()}
        if len(vals)!=1 or 'None' in vals: errors.append(f'dataset provenance mismatch/missing for {field}: {vals}')
    if sel:
        chosen=sel.get('selected_variant')
        for n in ['reversible_fixed','reversible_max_batch']:
            if runs[n].get('reversible_variant')!=chosen: errors.append(f'{n}: variant {runs[n].get("reversible_variant")} != preselected {chosen}')

if probes:
    for n,p in probes.items():
        if int(p.get('trial_steps',0))<10: errors.append(f'{n}: trial_steps < 10')
        if not p.get('search_complete'): errors.append(f'{n}: batch search lacks observed failure bracket')
    if runs.get('reversible_max_batch') and 'reversible_batch_probe' in probes:
        expected=probes['reversible_batch_probe'].get('largest_feasible_batch',probes['reversible_batch_probe'].get('largest_stable_batch'))
        if runs['reversible_max_batch'].get('batch_size')!=expected: errors.append('reversible max run batch does not equal measured probe maximum')

if errors:
    print('FINAL EVIDENCE AUDIT: FAIL')
    for e in errors: print(' -',e)
    sys.exit(1)
print('FINAL EVIDENCE AUDIT: PASS')
print('All required runs are comparable, provenance-matched, complete, and tied to the pre-selected reversible variant.')
