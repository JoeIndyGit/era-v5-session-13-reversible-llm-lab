import random
import subprocess
import numpy as np
import torch


def seed_everything(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_info():
    info = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_memory_gib": p.total_memory / (1024**3),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        })
    else:
        info.update({
            "gpu_name": None,
            "gpu_total_memory_gib": None,
            "bf16_supported": False,
        })
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        info["git_commit"] = None
    return info


def choose_precision(device):
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, False, "bf16"
        return torch.float16, True, "fp16"
    return torch.float32, False, "fp32-cpu"


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_memory():
    if not torch.cuda.is_available():
        return {"peak_allocated_gib": None, "peak_reserved_gib": None}
    return {
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
    }
