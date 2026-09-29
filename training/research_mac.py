"""Apple Silicon preflight for v41; no model/data changes and no CPU fallback."""
import os
from pathlib import Path
import platform
import shutil
import subprocess

MPS_ENV = {"PYTORCH_ENABLE_MPS_FALLBACK":"0", "PYTORCH_MPS_FAST_MATH":"0"}


def configure_environment():
    # Call before importing torch. Explicit conflicting settings are rejected by preflight.
    for key,value in MPS_ENV.items(): os.environ.setdefault(key,value)


def preflight(root, memory_fraction=.85):
    import torch
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("v41 MPS requires native arm64 Python on Apple Silicon, not Rosetta")
    if not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable; use setup_mac.sh and native Python 3.11. No CPU fallback.")
    if any(os.environ.get(k) != v for k,v in MPS_ENV.items()):
        raise ValueError("Disable MPS fallback/fast math BEFORE importing torch; restart the kernel")
    if not 0 < memory_fraction <= 1:
        raise ValueError("MPS memory fraction must be in (0,1]; unlimited allocation is forbidden")
    torch.mps.set_per_process_memory_fraction(memory_fraction)
    chip = subprocess.check_output(["sysctl","-n","machdep.cpu.brand_string"],text=True).strip()
    ram = int(subprocess.check_output(["sysctl","-n","hw.memsize"],text=True))
    recommended = torch.mps.recommended_max_memory()
    free = shutil.disk_usage(Path(root)).free
    # Exercise autograd and optimizer, not only is_available(). No real training here.
    parameter = torch.nn.Parameter(torch.ones(8,8,device="mps"))
    optimizer = torch.optim.AdamW([parameter],lr=1e-4,foreach=False)
    parameter.square().mean().backward(); optimizer.step(); torch.mps.synchronize()
    if not torch.isfinite(parameter).all(): raise RuntimeError("MPS numerical preflight failed")
    del parameter,optimizer
    torch.mps.empty_cache()
    print(f"MPS: {chip} | RAM {ram/2**30:.0f} GiB | allocator limit "
          f"{recommended*memory_fraction/2**30:.1f} GiB | free disk {free/2**30:.1f} GiB",flush=True)
    if free < 50*2**30: print("WARNING: keep at least 50 GiB free for the long queue and result ZIPs.",flush=True)
    return {"chip":chip,"unified_memory_bytes":ram,"recommended_mps_bytes":recommended,
            "allocator_fraction":memory_fraction,"environment":{k:os.environ[k] for k in MPS_ENV}}


def memory_sample():
    import torch
    return {"mps_allocated_bytes":torch.mps.current_allocated_memory(),
            "mps_driver_bytes":torch.mps.driver_allocated_memory()}


if __name__ == "__main__":
    configure_environment()
    preflight(Path(__file__).resolve().parents[1])
