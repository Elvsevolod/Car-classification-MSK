"""No new tolerances: optimized preprocessing must preserve every float32 pixel."""
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageOps

from backend import dual_role, runtime
from backend.core import Encoder, ROOT
from backend.frozen_encoder import FrozenEncoder, PolicyEncoder
from tests.test_release_runtime import dataset


def preprocessing_encoder():
    member = FrozenEncoder.__new__(FrozenEncoder)
    member.size = 256
    member.bundle = {"preprocessing": {"resize_mode": "square"}}
    policy = PolicyEncoder.__new__(PolicyEncoder)
    policy.members = [member]
    encoder = dual_role.DualRoleEncoder.__new__(dual_role.DualRoleEncoder)
    encoder.mvp, encoder.r1 = Encoder.__new__(Encoder), policy
    return encoder


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA", "P"])
@pytest.mark.parametrize("orientation", range(1, 9))
@pytest.mark.parametrize("format", ["PNG", "JPEG"])
def test_shared_crop_is_bitwise_equal_to_original_formula(mode, orientation, format, monkeypatch):
    image = Image.fromarray(np.random.default_rng(7).integers(0, 256, (41, 63, 3), dtype=np.uint8)).convert(mode)
    if format == "JPEG" and mode in ("RGBA", "P"):
        image = image.convert("RGB")
    exif = Image.Exif()
    exif[274] = orientation
    buffer = BytesIO()
    image.save(buffer, format=format, exif=exif)
    buffer.seek(0)
    with Image.open(buffer) as image:
        oriented = ImageOps.exif_transpose(image).convert("RGB")
        encoder = preprocessing_encoder()
        original_crop = dual_role.crop_image
        calls = []
        def counted(*args):
            calls.append(1)
            return original_crop(*args)
        monkeypatch.setattr(dual_role, "crop_image", counted)
        for box in ((0, 0, oriented.width, oriented.height),
                    (oriented.width - 1, oriented.height - 1, 1, 1), (2, 3, 20, 17)):
            actual = encoder.preprocess(image, box)
            x, y, w, h = box
            for size, value in zip((208, 256), actual):
                # The pre-optimization formula, independent of production helpers.
                crop = oriented.crop((x, y, x + w, y + h)).resize((size, size), Image.Resampling.BILINEAR)
                pixels = np.asarray(crop, dtype=np.float32) / np.float32(255)
                pixels = (pixels - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
                np.testing.assert_array_equal(value, pixels.transpose(2, 0, 1))
                assert value.dtype == np.float32 and value.flags.c_contiguous
        assert len(calls) == 3


@pytest.mark.parametrize("box", [(-1, 0, 2, 2), (0, 0, 0, 1), (0, 0, 11, 10), (0.5, 0, 1, 1)])
def test_shared_crop_still_rejects_invalid_bbox(box):
    with pytest.raises(ValueError):
        preprocessing_encoder().preprocess(Image.new("RGB", (10, 10)), box)


@pytest.mark.parametrize("count", [1, 8, 16, 32])
def test_r1_sessions_receive_the_same_unchanged_batch(count):
    received = []
    def run(outputs, inputs):
        received.append(inputs["image"])
        return [np.ones((count, 512), dtype=np.float32)]
    policy = PolicyEncoder.__new__(PolicyEncoder)
    policy.members = []
    for _ in range(3):
        member = FrozenEncoder.__new__(FrozenEncoder)
        member.size, member.dimension = 256, 512
        member.input_name, member.output_name = "image", "embedding"
        member.session = SimpleNamespace(run=run)
        policy.members.append(member)
    source = np.random.default_rng(1).normal(size=(count, 3, 256, 256)).astype(np.float32)
    result = policy.encode_batch(list(source))
    assert result.shape == (count, 1536)
    assert received[0] is received[1] is received[2]
    np.testing.assert_array_equal(received[0], source)


def test_export_builds_only_one_image_index(dataset, tmp_path, monkeypatch):
    calls = []
    original = runtime.ImageIndex
    def counted(directory):
        calls.append(directory)
        return original(directory)
    monkeypatch.setattr(runtime, "ImageIndex", counted)
    runtime.export(runtime.Runtime(), dataset, tmp_path / "export")
    assert calls == [dataset / "images"]


def test_cuda_nvrtc_is_pinned_and_discoverable():
    assert "nvidia-cuda-nvrtc-cu12==12.2.140" in (ROOT / "requirements-gpu.txt").read_text()
    assert "/opt/venv/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:" in (ROOT / "Dockerfile").read_text()
