"""v20: a small learned visual pair head. No labels/IDs enter its inference API."""
from dataclasses import asdict, dataclass
import hashlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from backend.core import sha256
from training.audit import digest
from training.osnet_ablation_suite import save_checkpoint
from training.stage6 import write_json


@dataclass(frozen=True)
class HeadConfig:
    images_per_identity: int = 4
    pool: int = 50
    positives_per_query: int = 2
    negatives_per_query: int = 8
    hidden: int = 64
    epochs: int = 12
    batch_size: int = 256
    learning_rate: float = .001
    weight_decay: float = .0001

    def validate(self):
        integers = (self.images_per_identity, self.pool, self.positives_per_query,
                    self.negatives_per_query, self.hidden, self.epochs, self.batch_size)
        if any(type(x) is not int or x < 1 for x in integers) or self.images_per_identity < 2:
            raise ValueError("Invalid pair-head counts")
        if not np.isfinite([self.learning_rate, self.weight_decay]).all() or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid pair-head optimizer")


def training_rows(rows, allowed, *, seed, per_identity=4):
    """Deterministic camera-diverse subsample of TRAIN only; no annotation edits."""
    groups = {}
    for row in rows:
        if row["vehicle_id"] in allowed:
            groups.setdefault(row["vehicle_id"], {}).setdefault(row["camera_id"], []).append(row)
    result = []
    for identity, cameras in sorted(groups.items()):
        queues = [sorted(values, key=lambda r: digest([seed, r["image_id"]]))
                  for _, values in sorted(cameras.items())]
        selected = []
        for offset in range(per_identity):
            for values in queues:
                if offset < len(values) and len(selected) < per_identity:
                    selected.append(values[offset])
        result.extend(selected)
    if not result or not {r["vehicle_id"] for r in result} <= set(allowed):
        raise ValueError("No allowed pair training rows")
    return result


def mine_pairs(rows, vectors, config):
    """Cross-camera positives and real cosine-neighbor hard negatives.

    Same-ID/same-camera examples and self-pairs are ignored, never negatives.
    Hard positives can be outside the top-K pool. IDs/cameras are used only here.
    """
    config.validate()
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) != len(rows) or not np.isfinite(vectors).all():
        raise ValueError("Invalid mining vectors")
    if len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate mining image IDs")
    identities = np.array([r["vehicle_id"] for r in rows])
    cameras = np.array([r["camera_id"] for r in rows])
    pairs, targets = [], []
    for i in range(len(rows)):
        scores = vectors @ vectors[i]
        order = np.argsort(-scores, kind="stable")
        pool = order[order != i][:config.pool]
        positive = np.flatnonzero((identities == identities[i]) & (cameras != cameras[i]))
        positive = positive[np.argsort(scores[positive], kind="stable")][:config.positives_per_query]
        negative = pool[identities[pool] != identities[i]][:config.negatives_per_query]
        # Avoid anchors with only one class; keep training risk balanced across usable anchors.
        if not len(positive) or not len(negative):
            continue
        pairs.extend((i, int(j)) for j in [*positive, *negative])
        targets.extend([1.] * len(positive) + [0.] * len(negative))
    if not pairs:
        raise ValueError("No cross-camera positives plus hard negatives in train")
    return np.asarray(pairs, dtype=np.int64), np.asarray(targets, dtype=np.float32)


def pair_features(query_tokens, candidate_tokens, cosine):
    """One image vs a candidate batch; symmetric 4x4 visual correspondence features.

    The head sees the symmetric 16x16 local similarity matrix, directional
    best-match summaries and raw visual cosine. No gallery labels or filenames.
    """
    q, g = np.asarray(query_tokens, dtype=np.float32), np.asarray(candidate_tokens, dtype=np.float32)
    cosine = np.asarray(cosine, dtype=np.float32)
    if (q.shape != (16, 512) or g.ndim != 3 or g.shape[1:] != q.shape
            or cosine.shape != (len(g),) or not all(np.isfinite(x).all() for x in (q, g, cosine))):
        raise ValueError("Expected normalized 4x4 OSNet tokens and one cosine per candidate")
    norm_q, norm_g = np.linalg.norm(q, axis=-1), np.linalg.norm(g, axis=-1)
    if np.any((norm_q > 1e-6) & (abs(norm_q - 1) > 1e-4)) or np.any((norm_g > 1e-6) & (abs(norm_g - 1) > 1e-4)):
        raise ValueError("Local tokens must be unit length or zero no-evidence tokens")
    cross = np.clip(np.einsum("td,ksd->kts", q, g), -1, 1)
    left, right = np.sort(cross.max(axis=2), axis=1), np.sort(cross.max(axis=1), axis=1)
    matrix = (cross + cross.transpose(0, 2, 1)) * .5
    return np.concatenate([matrix.reshape(len(g), -1), (left + right) * .5,
                           np.abs(left - right), cosine[:, None]], axis=1).astype(np.float32)


class PairHead(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.register_buffer("center", torch.zeros(289))
        self.register_buffer("scale", torch.ones(289))
        self.network = nn.Sequential(nn.Linear(289, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, features):
        return self.network((features - self.center) / self.scale).squeeze(-1)


def fit_head(features, targets, directory, *, seed, signature, config=HeadConfig()):
    """Fixed epochs; train-only normalizer. Resume exact epoch boundaries on CPU."""
    config.validate()
    x, y = torch.as_tensor(features, dtype=torch.float32), torch.as_tensor(targets, dtype=torch.float32)
    if x.shape != (len(y), 289) or not torch.isfinite(x).all() or set(y.tolist()) != {0., 1.}:
        raise ValueError("Need finite training features and both binary classes")
    directory.mkdir(parents=True, exist_ok=True)
    signature = digest({"context": signature, "seed": seed, "config": asdict(config),
                        "features": hashlib.sha256(x.numpy().tobytes()).hexdigest(), "labels": digest(y.tolist())})
    # CPU head keeps training cheap and deterministic; the encoder uses the source device.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = PairHead(config.hidden)
    model.center.copy_(x.mean(0))
    model.scale.copy_(x.std(0, unbiased=False).clamp_min(.001))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    path, start, history = directory / "last.pt", 0, []
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["signature"] != signature:
            raise ValueError("Head resume recipe/features changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, history = state["epoch"], state["history"]
    positive_weight = (1 - y).sum() / y.sum()
    for epoch in range(start, config.epochs):
        order = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed + epoch))
        total = 0.
        for ids in order.split(config.batch_size):
            loss = F.binary_cross_entropy_with_logits(model(x[ids]), y[ids], pos_weight=positive_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite pair-head loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        history.append({"epoch": epoch + 1, "train_loss": total / len(y)})
        save_checkpoint(path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                               "signature": signature, "config": asdict(config), "epoch": epoch + 1, "history": history})
        write_json(directory / "history.json", history)
        print(f"  pair head seed={seed}: epoch {epoch+1}/{config.epochs}, train loss={total/len(y):.5f}", flush=True)
    return {"path": str(path), "sha256": sha256(path), "signature": signature,
            "pairs": len(y), "positives": int(y.sum()), "config": asdict(config), "history": history,
            "selection": "fixed last epoch; no evaluation labels in training or normalization"}


def load_head(summary):
    from pathlib import Path
    path = Path(summary["path"])
    if sha256(path) != summary["sha256"]:
        raise ValueError("Pair head checkpoint changed")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["signature"] != summary["signature"] or saved["epoch"] != saved["config"]["epochs"]:
        raise ValueError("Incomplete or mismatched pair head")
    model = PairHead(saved["config"]["hidden"])
    model.load_state_dict(saved["model"])
    if not all(torch.isfinite(x).all() for x in model.state_dict().values()):
        raise ValueError("Non-finite pair head")
    return model.eval()


def score_pairs(head, query_tokens, gallery_tokens, cosine):
    if head.training:
        raise ValueError("Pair head inference requires eval mode")
    with torch.inference_mode():
        result = torch.sigmoid(head(torch.from_numpy(pair_features(query_tokens, gallery_tokens, cosine)))).numpy()
    if not np.isfinite(result).all():
        raise ValueError("Invalid head scores")
    return result
