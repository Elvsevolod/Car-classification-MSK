import json
from types import SimpleNamespace

import numpy as np
import pytest

from backend import benchmark as timing
from backend.gpu import CudaSynchronizer, validate_cuda_placement


def event(provider, operator):
    return {"cat": "Node", "args": {"provider": provider, "op_name": operator}}


def test_cuda_shape_metadata_is_allowed_but_cpu_convolution_is_not():
    cuda = event("CUDAExecutionProvider", "Conv")
    shape = event("CPUExecutionProvider", "Shape")
    assert validate_cuda_placement([cuda, shape])["cpu_shape_events"] == 1
    with pytest.raises(RuntimeError, match="placement"):
        validate_cuda_placement([cuda, event("CPUExecutionProvider", "Conv")])
    with pytest.raises(RuntimeError, match="placement"):
        validate_cuda_placement([shape])


def test_cuda_synchronization_errors_are_fatal(monkeypatch):
    class Function:
        def __init__(self, result):
            self.result = result
        def __call__(self, *args):
            return self.result
    library = SimpleNamespace(cudaDeviceSynchronize=Function(35),
                              cudaGetErrorString=Function(b"insufficient driver"))
    monkeypatch.setattr("backend.gpu.ctypes.CDLL", lambda _: library)
    with pytest.raises(RuntimeError, match="insufficient driver"):
        CudaSynchronizer()


def test_benchmark_synchronizes_each_timed_extraction(monkeypatch, tmp_path):
    actions = []
    session = SimpleNamespace(get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    encoder = SimpleNamespace(session=session, preprocess=lambda *args: 0,
                              encode_batch=lambda batch: actions.append("extract") or np.ones((len(batch), 2)))
    runtime = SimpleNamespace(encoder=encoder, metadata=lambda: {})
    monkeypatch.setattr(timing, "Runtime", lambda *args: runtime)
    monkeypatch.setattr(timing, "CudaSynchronizer", lambda: lambda: actions.append("sync"))
    monkeypatch.setattr(timing, "read_rows", lambda _: [{"image_id": "one"}])
    monkeypatch.setattr(timing, "bbox", lambda _: (0, 0, 1, 1))
    monkeypatch.setattr(timing, "ImageIndex", lambda _: SimpleNamespace(resolve=lambda _: tmp_path / "one.png"))
    from PIL import Image
    Image.new("RGB", (1, 1)).save(tmp_path / "one.png")
    monkeypatch.setattr(timing, "encode_rows", lambda *args: actions.append("full") or np.ones((2, 2)))
    monkeypatch.setattr(timing, "gpu_inventory", lambda: {})
    monkeypatch.setattr(timing, "weight_inventory", lambda _: {"passed": True})
    class Memory:
        def __init__(self, *args): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def report(self): return {}
    monkeypatch.setattr(timing, "MemorySampler", Memory)
    result = timing.benchmark("test", tmp_path, "CUDAExecutionProvider", warmup=2, samples=3,
                              throughput_seconds=.001, progress=lambda _: None)
    assert actions[:15] == ["sync", "extract", "sync"] * 5
    assert actions[-3:] == ["sync", "full", "sync"]
    assert len(result["batch1_latency_ms"]) == 3
    assert set(result["throughput"]) == {"1", "8", "16", "32"}
    assert all(v["seconds"] >= .001 for v in result["throughput"].values())


def test_compare_exports_rejects_decision_and_vector_drift(monkeypatch, tmp_path):
    from backend.compare_exports import compare_exports
    monkeypatch.setattr("backend.compare_exports.validate_artifacts", lambda *args: None)
    paths = [tmp_path / "one", tmp_path / "two"]
    manifest = dict.fromkeys(("profile", "profile_fingerprint", "encoder_fingerprint", "embedding_ids", "query_csv_sha256",
                              "gallery_csv_sha256", "image_sha256", "provider"), "same")
    for path in paths:
        path.mkdir()
        (path / "export_manifest.json").write_text(json.dumps(manifest))
        (path / "submission.csv").write_text("q,g1,g2\n")
        (path / "candidates.csv").write_text("query_id,gallery_id,confidence\nq,g1,0.7\n")
        np.save(path / "embeddings.npy", np.ones((3, 2), dtype=np.float32))
    assert compare_exports(tmp_path, *paths)["exact_embeddings"]
    np.save(paths[1] / "embeddings.npy", np.ones((3, 2), dtype=np.float32) + 1e-5)
    with pytest.raises(ValueError, match="Embeddings"):
        compare_exports(tmp_path, *paths)
    assert compare_exports(tmp_path, *paths, atol=2e-4)["passed"]
    (paths[1] / "candidates.csv").write_text("query_id,gallery_id,confidence\nq,g2,0.7\n")
    with pytest.raises(ValueError, match="candidate IDs"):
        compare_exports(tmp_path, *paths, atol=2e-4)
