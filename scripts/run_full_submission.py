#!/usr/bin/env python3
"""Execute the actual notebooks, preserving their real outputs after every cell."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NOTEBOOKS = ['00_setup_and_validation', '00b_variant_selection', '01_baseline_50m',
             '02_reversible_fixed_batch_50m', '03_reversible_max_batch_50m', '04_analysis_and_report']


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the measured runs; CPU test outputs are never submission evidence.')
    print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)


def execute_notebook(source, destination, cwd=ROOT):
    import nbformat
    import tempfile
    from jupyter_client import AsyncKernelManager
    from nbclient import NotebookClient
    nb = nbformat.read(source, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == 'code': cell.outputs = []; cell.execution_count = None
    from src.evidence import sha256
    nb.metadata['execution_record'] = {'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
                                       'source_notebook_sha256': sha256(source)}
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    def persist(**kwargs):
        temp = destination.with_name(destination.name + '.tmp')
        nbformat.write(nb, temp)
        os.replace(temp, destination)
    class StreamingClient(NotebookClient):
        def process_message(self, msg, cell, cell_index):
            result = super().process_message(msg, cell, cell_index)
            if msg.get('msg_type') == 'stream':
                print(msg['content'].get('text', ''), end='', flush=True)
                persist()
            return result
    sockets = tempfile.TemporaryDirectory(prefix='era-s13-kernel-')
    manager = AsyncKernelManager(kernel_name='python3', transport='ipc', ip=str(Path(sockets.name) / 'kernel'))
    client = StreamingClient(nb, km=manager, timeout=None, kernel_name='python3', allow_errors=False,
                            resources={'metadata': {'path': str(cwd)}}, on_cell_executed=persist)
    try:
        client.execute(cleanup_kc=True)
        nb.metadata['execution_record']['status'] = 'completed'
    except BaseException:
        nb.metadata['execution_record']['status'] = 'failed'
        raise
    finally:
        nb.metadata['execution_record']['finished_utc'] = datetime.now(timezone.utc).isoformat()
        persist()
        sockets.cleanup()
    return destination


def main():
    from src.evidence import atomic_json
    parser = argparse.ArgumentParser()
    parser.add_argument('--precision', choices=['auto', 'fp32', 'fp16', 'bf16'], default='auto')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1: raise ValueError('Batch size must be positive')
    os.chdir(ROOT)
    require_cuda()
    config_path = ROOT / 'results/execution_config.json'
    overrides = {'precision': args.precision, 'batch_size': args.batch_size}
    if config_path.exists() and json.loads(config_path.read_text()) != overrides:
        raise RuntimeError('Execution settings changed. Preserve this experiment and use a fresh checkout for the new comparison.')
    atomic_json(config_path, overrides)
    os.environ['ERA_S13_CONFIG'] = str(config_path)
    for number, name in enumerate(NOTEBOOKS, 1):
        print(f'[{number}/{len(NOTEBOOKS)}] Executing {name}', flush=True)
        execute_notebook(ROOT / 'notebooks' / f'{name}.ipynb', ROOT / 'executed_notebooks' / f'{name}.ipynb')
    subprocess.run([sys.executable, 'scripts/audit_results.py'], cwd=ROOT, check=True)
    subprocess.run([sys.executable, 'scripts/package_evidence.py'], cwd=ROOT, check=True)
    print('COMPLETE: measured results, figures, README, executed notebooks and evidence ZIP are ready.', flush=True)


if __name__ == '__main__': main()
