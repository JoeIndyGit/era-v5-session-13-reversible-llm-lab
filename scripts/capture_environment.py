from pathlib import Path
import platform, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "results" / "environment.txt"
out.parent.mkdir(parents=True, exist_ok=True)

def cmd(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e:
        return f"UNAVAILABLE: {e}"

sections = [
    ("timestamp_utc", cmd([sys.executable, "-c", "import datetime; print(datetime.datetime.now(datetime.timezone.utc).isoformat())"])),
    ("python", sys.version),
    ("platform", platform.platform()),
    ("git_commit", cmd(["git", "rev-parse", "HEAD"])),
    ("nvidia_smi", cmd(["nvidia-smi"])),
    ("pip_freeze", cmd([sys.executable, "-m", "pip", "freeze"])),
]
out.write_text("\n\n".join(f"## {k}\n{v}" for k,v in sections))
print(out)
