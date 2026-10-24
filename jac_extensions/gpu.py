"""Pick a GPU that carries no other user's process and has enough free memory.

Import and call ``claim(need_mb)`` *before* importing torch. Blocks (polling every
``interval`` s) until such a GPU exists, then sets CUDA_VISIBLE_DEVICES. Pass
need_mb=0 / set JACEXT_DEVICE=cpu to stay on the CPU.
"""
import os
import pwd
import subprocess
import time

ME = pwd.getpwuid(os.getuid()).pw_name
MARGIN_MB = 1500


def _query():
    gpus = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True).stdout
    owners = {}
    for line in apps.strip().splitlines():
        pid, uuid = [v.strip() for v in line.split(",")]
        try:
            user = subprocess.run(["ps", "-o", "user=", "-p", pid], capture_output=True,
                                  text=True).stdout.strip() or "?"
        except Exception:
            user = "?"
        owners.setdefault(uuid, set()).add(user)
    out = []
    for line in gpus.strip().splitlines():
        index, uuid, used, total = [v.strip() for v in line.split(",")]
        foreign = owners.get(uuid, set()) - {ME}
        out.append((int(index), int(total) - int(used), foreign))
    return out


def claim(need_mb, allowed=None, interval=60, verbose=True):
    if need_mb <= 0 or os.environ.get("JACEXT_DEVICE") == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return None
    while True:
        if allowed is None and os.environ.get("JACEXT_GPU"):
            allowed = [int(v) for v in os.environ["JACEXT_GPU"].split(",")]
        candidates = [(free, index) for index, free, foreign in _query()
                      if not foreign and free >= need_mb + MARGIN_MB
                      and (allowed is None or index in allowed)]
        if candidates:
            free, index = max(candidates)
            # nvidia-smi indices are PCI-bus order; CUDA's default order may differ
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
            if verbose:
                print(f"[gpu] using GPU {index} ({free} MB free, need {need_mb})", flush=True)
            return index
        if verbose:
            print(f"[gpu] no free GPU for {need_mb} MB; waiting {interval}s", flush=True)
        time.sleep(interval)
