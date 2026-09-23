"""Technical ONNX parity check; does not install weights into the MVP."""
import copy
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from torch.nn import functional as F

from backend.core import sha256


class ImageOnly(nn.Module):
    def __init__(self, image_encoder):
        super().__init__()
        self.image_encoder = image_encoder

    def forward(self, images):
        _, feature, projected = self.image_encoder(images)
        return F.normalize(torch.cat((feature[:, 0], projected[:, 0]), dim=1), dim=1)


def export_smoke(model, samples, destination):
    """Export image weights only and verify dynamic batches 1 and 2 with real crops."""
    if len(samples) < 2:
        raise ValueError("Need at least two real crops for the export check")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoder = ImageOnly(copy.deepcopy(model.image_encoder).cpu()).eval()
    samples = samples[:2].detach().cpu().float()
    with torch.no_grad():
        torch.onnx.export(encoder, samples[:1], str(destination), dynamo=False,
                          input_names=["images"], output_names=["embeddings"],
                          dynamic_axes={"images": {0: "batch"}, "embeddings": {0: "batch"}},
                          opset_version=17)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(destination), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    errors, cosines = [], []
    for batch in (samples[:1], samples):
        with torch.no_grad():
            expected = encoder(batch).numpy()
        actual = session.run(["embeddings"], {"images": batch.numpy()})[0]
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-3)
        np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1, atol=1e-5)
        if actual.shape != (len(batch), 1280):
            raise ValueError("Incorrect export embedding shape")
        errors.append(float(np.abs(actual - expected).max()))
        cosines.extend((actual * expected).sum(1).tolist())
    return {"sha256": sha256(destination), "bytes": destination.stat().st_size,
            "batches_checked": [1, 2], "embedding_dimension": 1280,
            "max_absolute_error": max(errors), "minimum_cosine": min(cosines),
            "image_only": True, "mvp_changed": False,
            "exporter": "torch.onnx legacy dynamo=False, opset 17 (pinned torch environment)"}
