"""Shared provenance and atomic output helpers; no benchmark values live here."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 2


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def source_digest():
    files = sorted(list((ROOT / 'src').glob('*.py')) + list((ROOT / 'scripts').glob('*.py')))
    return canonical_hash({str(p.relative_to(ROOT)): sha256(p) for p in files})


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def verified_dataset(cfg):
    path = Path(cfg['train_path']).parent / 'dataset_meta.json'
    if not path.exists():
        if not cfg.get('allow_cpu', False):
            raise RuntimeError('Dataset provenance is missing; run setup first.')
        # Small isolated CPU tests also hash the actual input bytes.
        return {'train_sha256': sha256(cfg['train_path']), 'val_sha256': sha256(cfg['val_path']), 'test_fixture': True}
    meta = json.loads(path.read_text())
    for name, actual in [('train', cfg['train_path']), ('val', cfg['val_path']), ('tokenizer', meta['tokenizer_path'])]:
        if sha256(actual) != meta.get(name + '_sha256'):
            raise RuntimeError(f'{name} contents differ from the recorded dataset hash')
    return meta


def run_identity(cfg, model_cfg, dataset, hardware, precision):
    ignored = {'train_path', 'val_path', 'results_dir', 'checkpoint_dir', 'checkpoint_every_steps',
               'log_every_steps', 'resume', 'allow_cpu'}
    return {
        'schema_version': SCHEMA_VERSION,
        'source_sha256': source_digest(),
        'model': model_cfg.to_dict(),
        'training': {k: v for k, v in cfg.items() if k not in ignored},
        'dataset': {k: dataset.get(k) for k in ['train_sha256', 'val_sha256', 'tokenizer_sha256', 'dataset_revision']},
        'hardware': {k: hardware.get(k) for k in ['device', 'gpu_name', 'gpu_total_memory_gib', 'torch_version', 'cuda_version']},
        'precision': precision,
    }


def load_config():
    """Notebook and command-line execution use the same effective settings."""
    cfg = json.loads((ROOT / 'configs/common.json').read_text())
    override = os.environ.get('ERA_S13_CONFIG')
    if override:
        cfg.update(json.loads(Path(override).read_text()))
    return cfg
