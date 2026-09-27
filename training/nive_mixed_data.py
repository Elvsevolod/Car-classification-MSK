"""v32 read-only data audit and deterministic, coverage-aware auxiliary schedule."""
from collections import Counter, defaultdict, deque
from pathlib import Path
import random

import numpy as np
from PIL import Image

from backend.core import bbox, crop_image
from training.audit import digest
from training.nive_transfer import NiVeDataset, audit_nive
from training.osnet_ablation_suite import freeze_json, load_json


def label_domains(organizer, external, allowed):
    labels = {identity: i for i, identity in enumerate(sorted(allowed))}
    target = [{**row, "label": labels[row["vehicle_id"]],
               "vehicle_id": f"organizer:{row['vehicle_id']}"}
              for row in organizer if row["vehicle_id"] in labels]
    if {row["label"] for row in target} != set(range(len(labels))):
        raise ValueError("Missing organizer train identities")
    ids = sorted({row["vehicle_id"] for row in external})
    source_labels = {identity: i for i, identity in enumerate(ids)}
    source = []
    for row in external:
        parts = Path(row["path"]).parts
        if (len(parts) != 3 or parts[0] != "train" or not parts[1].isdigit()
                or row["vehicle_id"] != f"nive:{parts[1]}" or Path(row["path"]).suffix != ".jpg"):
            raise ValueError("Only namespaced NiVe train photographs are allowed")
        source.append({**row, "label": source_labels[row["vehicle_id"]]})
    return target, source


def cyclic_schedule(rows, steps, p, k, seed, forbidden=None):
    """Cycle every image before reuse; prefer a different camera/view proxy for K=2.

    Same algorithm for N0 and N1. For N0, exclude the current main image IDs;
    the main schedule itself stays the historical camera-aware PK sampler.
    """
    if k != 2 or len({r["vehicle_id"] for r in rows}) < p:
        raise ValueError("Auxiliary schedule requires P distinct identities and K=2")
    rng = random.Random(seed)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["vehicle_id"]].append(i)
    if any(len(indices) < k for indices in groups.values()):
        raise ValueError("Each auxiliary identity needs at least two unique photographs")
    queues = {identity: [] for identity in groups}
    identity_queue, batches = deque(), []
    identities = sorted(groups)
    for step in range(steps):
        blocked = set(forbidden[step]) if forbidden is not None else set()
        chosen, batch = set(), []
        attempts = 0
        while len(chosen) < p:
            if not identity_queue:
                order = identities.copy(); rng.shuffle(order); identity_queue.extend(order)
            identity = identity_queue.popleft()
            attempts += 1
            if attempts > 3 * len(identities):
                raise ValueError("Cannot build an auxiliary batch disjoint from main images")
            available = [i for i in groups[identity] if rows[i]["image_id"] not in blocked]
            if identity in chosen or len(available) < k:
                identity_queue.append(identity)
                continue
            selected = []
            for _ in range(k):
                options = [i for i in queues[identity] if i in available and i not in selected]
                if not options:
                    refill = groups[identity].copy(); rng.shuffle(refill)
                    queues[identity].extend(i for i in refill if i not in queues[identity])
                    options = [i for i in queues[identity] if i in available and i not in selected]
                cross = [i for i in options if not selected or
                         rows[i]["camera_id"] != rows[selected[0]]["camera_id"]]
                index = (cross or options)[0]
                queues[identity].remove(index)
                selected.append(index)
            chosen.add(identity); batch.extend(selected)
        batches.append(batch)
    return batches


def coverage(rows, batches):
    counts = Counter(i for batch in batches for i in batch)
    identities = {rows[i]["vehicle_id"] for i in counts}
    return {"logical_images": sum(counts.values()), "unique_images": len(counts),
            "available_images": len(rows), "image_fraction": len(counts) / len(rows),
            "identities_seen": len(identities),
            "presentations_per_image": {rows[i]["image_id"]: counts[i] for i in range(len(rows))}}


def describe_image(image):
    gray = np.asarray(image.convert("L").resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32)
    small = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
    dhash = int.from_bytes(np.packbits(small[:, 1:] > small[:, :-1]).tobytes(), "big")
    laplace = (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] +
               gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1])
    return {"dhash": f"{dhash:016x}", "brightness_0_255": float(gray.mean()),
            "blur_laplacian_variance_at_64": float(laplace.var()), "crop_size": list(image.size)}


def near_pairs(left, right, same_set=False):
    """Exact Hamming<=3 lookup in 64-bit dHash via four 16-bit bands (no ANN)."""
    index = defaultdict(set)
    hashes = {key: int(value["dhash"], 16) for key, value in right.items()}
    for key, value in hashes.items():
        for band in range(4):
            index[band, (value >> (16 * band)) & 65535].add(key)
    pairs = []
    for key, item in left.items():
        value = int(item["dhash"], 16)
        candidates = set().union(*(index[band, (value >> (16 * band)) & 65535] for band in range(4)))
        for other in sorted(candidates):
            if same_set and key >= other:
                continue
            distance = (value ^ hashes[other]).bit_count()
            if distance <= 3:
                pairs.append({"external": key, "reference": other, "hamming": distance})
    return pairs


def distribution(rows, descriptions):
    identity_counts = Counter(r["vehicle_id"] for r in rows)
    groups = defaultdict(set)
    for row in rows:
        groups[row["vehicle_id"]].add(row["camera_id"])
    values = list(descriptions.values())
    quantiles = lambda xs: np.quantile(xs, [0, .1, .5, .9, 1]).tolist()
    return {"images": len(rows), "identities": len(identity_counts),
            "images_per_identity": dict(identity_counts),
            "groups_per_identity": {str(k): len(v) for k, v in groups.items()},
            "brightness_quantiles": quantiles([v["brightness_0_255"] for v in values]),
            "blur_quantiles": quantiles([v["blur_laplacian_variance_at_64"] for v in values]),
            "crop_width_quantiles": quantiles([v["crop_size"][0] for v in values]),
            "crop_height_quantiles": quantiles([v["crop_size"][1] for v in values]),
            "limitation": "Brightness is not a night label; blur at normalized 64px is diagnostic only"}


def audit_domains(directory, organizer, allowed, dataset, external, nive_root, fingerprint):
    """Cache only after byte fingerprints have been checked by the caller."""
    path = directory / "domain_audit.json"
    if path.exists():
        saved = load_json(path)
        if saved["input_fingerprint"] != fingerprint:
            raise ValueError("Audit inputs changed; use a new run")
        return saved
    target, protected, source = {}, {}, {}
    for number, row in enumerate(organizer):
        with Image.open(dataset / "images" / f"{row['image_id']}.jpg") as image:
            crop = crop_image(image, bbox(row))
            item = describe_image(crop)
            protected[row["image_id"] + ":crop"] = item
            # Also catch reuse of a whole frame, not only an organizer crop.
            protected[row["image_id"] + ":frame"] = describe_image(image.convert("RGB"))
            if row["vehicle_id"] in allowed:
                target[row["image_id"]] = item
        if (number + 1) % 500 == 0:
            print(f"AUDIT organizer: {number+1}/{len(organizer)}", flush=True)
    for number, row in enumerate(external):
        with Image.open(nive_root / row["path"]) as image:
            source[row["path"]] = describe_image(image)
        if (number + 1) % 1000 == 0:
            print(f"AUDIT NiVe train: {number+1}/{len(external)}", flush=True)
    cross = near_pairs(source, protected)
    internal = near_pairs(source, source, same_set=True)
    saved = {"input_fingerprint": fingerprint, "near_duplicate_method": "64-bit dHash Hamming<=3; suspect, not proof",
             "near_duplicate_limitation": "Does not rule out recrops, changed viewpoints or missed near-duplicates",
             "cross_domain_suspects": cross, "source_internal_suspects": internal,
             "organizer_train": distribution([r for r in organizer if r["vehicle_id"] in allowed], target),
             "nive_train": distribution(external, source), "organizer_descriptions": target,
             "nive_descriptions": source, "source_groups": "filename prefix is view proxy, not physical camera",
             "review_samples": {"organizer": random.Random(20260915).sample(sorted(target), min(24, len(target))),
                                "nive": random.Random(20260915).sample(sorted(source), min(24, len(source)))}}
    freeze_json(path, saved)
    return saved


def reviewed_external(rows, audit, review):
    """Only explicitly confirmed EXTERNAL duplicates are excluded; organizer is never edited."""
    suspects = audit["cross_domain_suspects"]
    signature = digest(suspects)
    if not suspects and not review:
        return rows, []
    if not review or review.get("suspects_sha256") != signature:
        raise ValueError("Near-duplicate suspects need manual review; see domain_audit.json before training")
    paths = {p["external"] for p in suspects}
    decisions = review.get("decisions", {})
    if set(decisions) != paths or any(v not in {"confirmed_duplicate", "not_duplicate"} for v in decisions.values()):
        raise ValueError("Review must resolve every suspicious external path explicitly")
    excluded = [{"path": p, "reason": "manually_confirmed_external_duplicate"}
                for p, decision in sorted(decisions.items()) if decision == "confirmed_duplicate"]
    forbidden = {p["path"] for p in excluded}
    return [r for r in rows if r["path"] not in forbidden], excluded
