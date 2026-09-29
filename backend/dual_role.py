"""v24 inference port: lossless MVP512/R1_1536 blocks, separate ranking and acceptance."""
import numpy as np

from .core import Encoder, PREPROCESS, crop_image, normalize
from .frozen_encoder import PolicyEncoder, digest

LAYOUT = {"ranking": [0, 512], "candidate": [512, 2048], "normalization": "unit blocks, no global renormalization"}


def validate_blocks(values):
    if values.dtype != np.float32 or values.ndim != 2 or values.shape[1] != 2048 or not np.isfinite(values).all():
        raise ValueError("Dual-role requires finite float32 MVP512 + R1_1536 vectors")
    if any(not np.allclose(np.linalg.norm(block, axis=1), 1., atol=1e-5, rtol=0)
           for block in (values[:, :512], values[:, 512:])):
        raise ValueError("Dual-role requires two unit blocks, not a globally normalized vector")


class DualRoleEncoder:
    def __init__(self, bundle, provider):
        self.mvp = Encoder(provider=provider)
        self.r1 = PolicyEncoder(bundle, provider)
        if (self.r1.dimension != 1536 or self.r1.size != 256 or len(self.r1.members) != 3
                or self.r1.bundle["candidate_policy"] != "raw_top1"):
            raise ValueError("Dual-role requires the three-member frozen R1/256 candidate")
        self.members = [self.mvp, *self.r1.members]
        self.size, self.dimension, self.provider = [208, 256], 2048, provider
        self.preprocessing = {"ranking": PREPROCESS, "candidate": self.r1.members[0].bundle["preprocessing"],
                              "layout": LAYOUT}
        self.fingerprint = digest({"schema": "dual-role-v24", "mvp": self.mvp.fingerprint,
                                   "r1": self.r1.fingerprint, "preprocessing": self.preprocessing})

    def preprocess(self, image, box):
        crop = crop_image(image, box)
        return self.mvp.preprocess_crop(crop), self.r1.preprocess_crop(crop)

    def encode_batch(self, batch):
        mvp, r1 = zip(*batch)
        values = np.concatenate([self.mvp.encode_batch(mvp), self.r1.encode_batch(r1)], axis=1)
        validate_blocks(values)
        return values

    def encode(self, image, box):
        return self.encode_batch([self.preprocess(image, box)])[0]


def split_unit(values):
    validate_blocks(values)
    return normalize(np.ascontiguousarray(values[:, :512])), normalize(np.ascontiguousarray(values[:, 512:]))


def ranking_blocks(values, r1_weight=0.):
    """v25 weighted concatenation; preserve the original R1 candidate space."""
    mvp, r1 = split_unit(values)
    if r1_weight == 0:
        return mvp, r1
    if r1_weight != .5:
        raise ValueError("Only the frozen v25 equal mixture is deployed")
    mixed = normalize(np.concatenate([mvp * np.float32(np.sqrt(1-r1_weight)),
                                      r1 * np.float32(np.sqrt(r1_weight))], axis=1))
    # Match research: normalize the mixture, then the query/gallery matrices.
    return normalize(mixed), r1
