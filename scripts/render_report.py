from pathlib import Path
import json, re
import pandas as pd
import matplotlib.pyplot as plt

ORDER=["baseline_fixed","reversible_fixed","reversible_max_batch"]

def load_results(results_dir="results"):
    out={}
    for n in ORDER:
        p=Path(results_dir)/f"{n}.json"
        if p.exists(): out[n]=json.loads(p.read_text())
    return out

def load_probes(results_dir="results"):
    out={}
    for n in ["baseline_batch_probe","reversible_batch_probe"]:
        p=Path(results_dir)/f"{n}.json"
        if p.exists(): out[n]=json.loads(p.read_text())
    return out

def load_variant(results_dir="results"):
    p=Path(results_dir)/"variant_selection.json"
    return json.loads(p.read_text()) if p.exists() else None

def _label(n): return n.replace("_"," ").title()

def plot_results(results, results_dir="results", assets_dir="assets"):
    assets=Path(assets_dir); assets.mkdir(parents=True,exist_ok=True)
    curves=[]
    for n in ORDER:
        p=Path(results_dir)/f"{n}_steps.csv"
        if p.exists(): curves.append((n,pd.read_csv(p)))
    if curves:
        fig,ax=plt.subplots(figsize=(9,5))
        for n,df in curves:
            line, = ax.plot(df.tokens_seen/1e6,df.train_loss_100step_mean,label=_label(n)+" train")
            val = df.dropna(subset=["val_loss"])
            if not val.empty:
                ax.plot(val.tokens_seen/1e6,val.val_loss,linestyle="--",marker=".",color=line.get_color(),label=_label(n)+" validation")
        ax.set(xlabel="Training tokens (millions)",ylabel="100-step mean cross-entropy",title="Training loss over the 50M-token budget"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(assets/'loss_vs_tokens.png',dpi=180); plt.close(fig)
    labels=[_label(n) for n in ORDER if n in results]
    for key,ylabel,title,file in [
        ("peak_allocated_gib","Peak allocated GPU memory (GiB)","Peak training memory","memory_comparison.png"),
        ("median_tokens_per_s","Median training tokens / second","Measured training throughput","throughput_comparison.png")]:
        vals=[results[n][key] for n in ORDER if n in results]
        if labels and all(v is not None for v in vals):
            fig,ax=plt.subplots(figsize=(8,5)); ax.bar(labels,vals); ax.set_ylabel(ylabel); ax.set_title(title); ax.tick_params(axis="x",rotation=15); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(assets/file,dpi=180); plt.close(fig)
    xs=[];ys=[];names=[]
    for n in ORDER:
        if n in results and results[n].get("peak_allocated_gib") is not None:
            xs.append(results[n]["peak_allocated_gib"]);ys.append(results[n]["final_val_loss"]);names.append(_label(n))
    if xs:
        fig,ax=plt.subplots(figsize=(7,5)); ax.scatter(xs,ys,s=70)
        for x,y,n in zip(xs,ys,names): ax.annotate(n,(x,y),xytext=(6,6),textcoords="offset points")
        ax.set(xlabel="Peak allocated memory (GiB)",ylabel="Final validation loss",title="Quality–memory trade-off");ax.grid(alpha=.25);fig.tight_layout();fig.savefig(assets/'quality_memory_frontier.png',dpi=180);plt.close(fig)
    probes=load_probes(results_dir)
    if probes:
        fig,ax=plt.subplots(figsize=(8,5))
        for key,probe in probes.items():
            rows=sorted(probe["attempts"],key=lambda x:x["batch_size"]); x=[r["batch_size"] for r in rows]; y=[r["peak_reserved_gib"] for r in rows]
            label="standard residual" if key.startswith("baseline") else "selected reversible"
            ax.plot(x,y,marker="o",label=label)
            for r in rows:
                if not r["stable"]: ax.annotate("fail",(r["batch_size"],r["peak_reserved_gib"]),xytext=(4,4),textcoords="offset points")
        ax.set(xlabel="Batch size (sequences)",ylabel="Peak reserved GPU memory (GiB)",title="Measured batch-capacity frontier");ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(assets/'batch_capacity_frontier.png',dpi=180);plt.close(fig)
    if len(results)==3:
        b,r,m=[results[n] for n in ORDER]; probes=load_probes(results_dir); bp=probes.get('baseline_batch_probe'); rp=probes.get('reversible_batch_probe')
        rows=[
            ["Fixed batch",b["batch_size"],r["batch_size"],m["batch_size"]],
            ["Peak allocated GiB",f'{b["peak_allocated_gib"]:.2f}',f'{r["peak_allocated_gib"]:.2f}',f'{m["peak_allocated_gib"]:.2f}'],
            ["Median tok/s",f'{b["median_tokens_per_s"]:,.0f}',f'{r["median_tokens_per_s"]:,.0f}',f'{m["median_tokens_per_s"]:,.0f}'],
            ["Validation loss",f'{b["final_val_loss"]:.4f}',f'{r["final_val_loss"]:.4f}',f'{m["final_val_loss"]:.4f}'],
        ]
        fig,ax=plt.subplots(figsize=(10,4.5));ax.axis('off');table=ax.table(cellText=rows,colLabels=["Metric","Baseline","Reversible fixed","Reversible max"],loc='center');table.auto_set_font_size(False);table.set_fontsize(11);table.scale(1,1.6)
        mem=100*(1-r['peak_allocated_gib']/b['peak_allocated_gib']); speed=100*(r['median_tokens_per_s']/b['median_tokens_per_s']-1)
        extra=f"Same-batch memory delta: {-mem:+.1f}%   |   Same-batch speed delta: {speed:+.1f}%"
        if bp and rp: extra += f"   |   Max-batch uplift: {rp['largest_stable_batch']/bp['largest_stable_batch']:.2f}×"
        ax.set_title("Experiment at a glance\n"+extra,pad=18);fig.tight_layout();fig.savefig(assets/'executive_summary.png',dpi=180,bbox_inches='tight');plt.close(fig)

def markdown_table(results):
    if len(results)<3: return "> Results are intentionally not pre-filled. Run the required notebooks, then notebook 04 to populate this table from measured JSON artifacts."
    rows=[]
    for n in ORDER:
        r=results[n];recon=r.get('reconstruction') or {}; variant=(f"{r.get('reversible_variant')} + {r.get('bootstrap')} bootstrap" if r.get('reversible_variant') else 'standard residual')
        rows.append("| {} | {} | {} | {:.4f} | {:.4f} | {:.2f} | {:,.0f} | {:.2f} | {:.2f} | {:,} | {} |".format(_label(n),variant,r['batch_size'],r['final_train_loss_100step_mean'],r['final_val_loss'],r['final_val_perplexity'],r['median_tokens_per_s'],r['peak_allocated_gib'],r['peak_reserved_gib'],r['optimizer_steps'],f"{recon.get('max_abs_error',0):.2e}" if recon else 'n/a'))
    return "| Experiment | Architecture | Batch | Final train loss¹ | Final val loss | Val PPL | Median tok/s | Peak alloc GiB | Peak reserved GiB | Steps | Reconstruction max error |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"+'\n'.join(rows)+"\n\n¹ Mean of the final 100 optimizer-step losses."

def variant_table(results_dir="results"):
    v=load_variant(results_dir)
    if not v: return "> Variant-selection evidence will appear after notebook 00b."
    lines=[]
    for r in v['candidates']:
        lines.append(f"| {r['variant']} | {'yes' if r.get('stable') else 'no'} | {r.get('val_loss','—')} | {r.get('median_tokens_per_s','—')} | {r.get('reconstruction_max_abs_error','—')} |")
    return "| Variant | Stable | Pilot val loss | Median tok/s | Reconstruction max error |\n|---|---|---:|---:|---:|\n"+'\n'.join(lines)+f"\n\n**Selected for the required 50M reversible runs:** `{v['selected_variant']}`. Selection rule: {v['selection_rule']}."

def qualitative_table(results):
    if 'baseline_fixed' not in results or 'reversible_fixed' not in results: return "> Deterministic generation samples will appear after the two fixed-batch runs."
    b=results['baseline_fixed'].get('qualitative_samples') or []; r=results['reversible_fixed'].get('qualitative_samples') or []
    if not b or not r or 'error' in b[0] or 'error' in r[0]: return "> Qualitative samples were unavailable; see the result JSON for the recorded reason."
    rows=[]
    for a,c in zip(b,r):
        esc=lambda x:str(x).replace('|','\\|').replace('\n',' ')
        rows.append(f"| {esc(a['prompt'])} | {esc(a['completion'])} | {esc(c['completion'])} |")
    return "| Prompt | Baseline greedy completion | Reversible greedy completion |\n|---|---|---|\n"+'\n'.join(rows)

def findings(results,results_dir="results"):
    if len(results)<3:return "Measured findings will be generated after all three 50M-token runs complete."
    b,r,m=[results[n] for n in ORDER];probes=load_probes(results_dir)
    speed=100*(r['median_tokens_per_s']/b['median_tokens_per_s']-1);maxspeed=100*(m['median_tokens_per_s']/b['median_tokens_per_s']-1);val=r['final_val_loss']-b['final_val_loss']
    mem_delta = 100 * (r['peak_allocated_gib'] / b['peak_allocated_gib'] - 1)
    lines=[f"- At the same batch size, peak allocated memory changed by **{mem_delta:+.1f}%** for the selected reversible variant relative to baseline.",f"- Its same-batch throughput delta was **{speed:+.1f}%**.",f"- At the maximum reversible batch, throughput changed by **{maxspeed:+.1f}%** relative to the fixed-batch baseline.",f"- Fixed-batch validation-loss delta was **{val:+.4f}** (reversible − baseline).",f"- Reversible round-trip max-absolute reconstruction error was **{r['reconstruction']['max_abs_error']:.2e}**."]
    bp=probes.get('baseline_batch_probe');rp=probes.get('reversible_batch_probe')
    if bp and rp:
        lines.insert(2,f"- Maximum memory-feasible batch changed from **{bp['largest_stable_batch']} → {rp['largest_stable_batch']} sequences ({rp['largest_stable_batch']/bp['largest_stable_batch']:.2f}×)** under the same 10-update probe rule.")
    lines.extend([
        f"- Aggregate steady-state throughput (total measured tokens / total measured seconds) was **{b['aggregate_tokens_per_s']:,.0f}**, **{r['aggregate_tokens_per_s']:,.0f}**, and **{m['aggregate_tokens_per_s']:,.0f} tok/s** for baseline, reversible fixed, and reversible maximum batch respectively.",
        f"- Reversible maximum-batch validation-loss delta against reversible fixed batch was **{m['final_val_loss']-r['final_val_loss']:+.4f}**; update counts were **{r['optimizer_steps']:,}** and **{m['optimizer_steps']:,}**.",
        f"- Trained reversible fixed-batch reconstruction relative L2 error was **{r['reconstruction']['max_relative_l2_error']:.2e}**, measured at context **{r['reconstruction']['context_length']}** in **{r['precision']}**.",
        f"- Overflow retries were **{b['overflow_retries']} / {r['overflow_retries']} / {m['overflow_retries']}**; recoveries were **{b['resume_count']} / {r['resume_count']} / {m['resume_count']}** (baseline / reversible fixed / reversible max)."
    ])
    return '\n'.join(lines)

def batch_probe_table(results_dir="results"):
    p=load_probes(results_dir)
    if len(p)<2:return "> Batch-frontier table will appear after notebook 03 measures both baseline and reversible capacity."
    rows=[]
    for k in ['baseline_batch_probe','reversible_batch_probe']:
        x=p[k]; label='Standard residual' if k.startswith('baseline') else 'Selected reversible'
        rows.append(f"| {label} | {x['largest_stable_batch']} | {x.get('first_failed_batch') or 'not observed'} | {x.get('trial_steps')} | {'yes' if x.get('search_complete') else 'no — lower bound'} |")
    return "| Architecture | Largest feasible batch | First observed failure | Trial updates | Search bracket complete |\n|---|---:|---:|---:|---|\n"+'\n'.join(rows)

def hardware_table(results):
    if len(results)<3:
        return "> Hardware provenance will appear after all three required runs."
    r=results['baseline_fixed']; h=r.get('hardware') or {}
    return ("| Field | Measured value |\n|---|---|\n"
            f"| GPU | {h.get('gpu_name')} |\n"
            f"| GPU memory | {h.get('gpu_total_memory_gib')} GiB |\n"
            f"| Precision | {r.get('precision')} |\n"
            f"| PyTorch | {h.get('torch_version')} |\n"
            f"| CUDA | {h.get('cuda_version')} |\n"
            f"| Git commit | {h.get('git_commit')} |")

def update_readme(readme_path='README.md',results_dir='results'):
    from scripts.audit_results import audit
    errors = audit(Path(readme_path).resolve().parent)
    if errors:
        raise RuntimeError("Report generation requires a passing evidence audit: " + "; ".join(errors))
    p=Path(readme_path);text=p.read_text();results=load_results(results_dir)
    reps={
        r'<!-- RESULTS_TABLE_START -->.*?<!-- RESULTS_TABLE_END -->':f"<!-- RESULTS_TABLE_START -->\n{markdown_table(results)}\n<!-- RESULTS_TABLE_END -->",
        r'<!-- FINDINGS_START -->.*?<!-- FINDINGS_END -->':f"<!-- FINDINGS_START -->\n{findings(results,results_dir)}\n<!-- FINDINGS_END -->",
        r'<!-- BATCH_TABLE_START -->.*?<!-- BATCH_TABLE_END -->':f"<!-- BATCH_TABLE_START -->\n{batch_probe_table(results_dir)}\n<!-- BATCH_TABLE_END -->",
        r'<!-- VARIANT_TABLE_START -->.*?<!-- VARIANT_TABLE_END -->':f"<!-- VARIANT_TABLE_START -->\n{variant_table(results_dir)}\n<!-- VARIANT_TABLE_END -->",
        r'<!-- QUALITATIVE_TABLE_START -->.*?<!-- QUALITATIVE_TABLE_END -->':f"<!-- QUALITATIVE_TABLE_START -->\n{qualitative_table(results)}\n<!-- QUALITATIVE_TABLE_END -->",
        r'<!-- HARDWARE_TABLE_START -->.*?<!-- HARDWARE_TABLE_END -->':f"<!-- HARDWARE_TABLE_START -->\n{hardware_table(results)}\n<!-- HARDWARE_TABLE_END -->",
    }
    for pat,repl in reps.items(): text=re.sub(pat,lambda match, value=repl: value,text,flags=re.S)
    p.write_text(text);return results

if __name__=='__main__':
    r=update_readme();plot_results(r);print(f"Rendered report from {len(r)} required experiment result files.")
