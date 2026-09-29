"""CUDA preflight and synchronization without a training/PyTorch dependency."""
import argparse
import ctypes
import json
import subprocess


class CudaSynchronizer:
    def __init__(self):
        self.library = ctypes.CDLL("libcudart.so.12")
        self.library.cudaDeviceSynchronize.argtypes = []
        self.library.cudaDeviceSynchronize.restype = ctypes.c_int
        self.library.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.library.cudaGetErrorString.restype = ctypes.c_char_p
        self()

    def __call__(self):
        status = self.library.cudaDeviceSynchronize()
        if status:
            error = self.library.cudaGetErrorString(status).decode()
            raise RuntimeError(f"CUDA synchronization failed: {error}")


def gpu_inventory():
    try:
        return {"nvidia_smi": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv"],
            text=True, stderr=subprocess.STDOUT, timeout=10).strip()}
    except (OSError, subprocess.SubprocessError) as error:
        return {"nvidia_smi": None, "warning": str(error)}


def validate_cuda_placement(events):
    """Allow CPU shape metadata only, never silent CPU neural-network computation."""
    kernels = [e["args"] for e in events if e.get("cat") == "Node" and e.get("args", {}).get("provider")]
    cuda = [e for e in kernels if e["provider"] == "CUDAExecutionProvider"]
    unexpected = [e for e in kernels if e["provider"] != "CUDAExecutionProvider"
                  and not (e["provider"] == "CPUExecutionProvider" and e.get("op_name") == "Shape")]
    if not cuda or unexpected:
        raise RuntimeError(f"CUDA placement check failed; unexpected CPU kernels: {unexpected}")
    return {"cuda_kernel_events": len(cuda),
            "cpu_shape_events": len(kernels) - len(cuda),
            "cpu_compute_fallback": False}


def main():
    from pathlib import Path
    import onnxruntime as ort
    from .benchmark import weight_inventory
    from .core import ROOT
    from .runtime import DEFAULT_PROFILE, PROFILE_NAMES, Runtime, write_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILE_NAMES, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Use a new report path")
    runtime = Runtime(args.profile, "CUDAExecutionProvider")
    CudaSynchronizer()()
    sessions = [e.session for e in getattr(runtime.encoder, "members", [runtime.encoder])]
    weights = weight_inventory(ROOT / "models")
    if not weights["passed"]:
        raise RuntimeError("Delivered weights exceed the organizer limit or are missing")
    result = {**runtime.metadata(), **gpu_inventory(), "onnxruntime": ort.__version__,
              "actual_providers": [s.get_providers() for s in sessions],
              "placement": [s.reid_cuda_placement for s in sessions],
              "weights": weights, "preflight_passed": True,
              "official_gpu_verified": False}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
