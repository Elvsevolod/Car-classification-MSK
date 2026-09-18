"""Small, explicit training pipeline used by notebooks/train_osnet.ipynb."""
import io
import json
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from tqdm.auto import tqdm

from backend.core import DATASET, STOCK_MODEL, bbox, crop_image, read_rows, sha256
from backend.evaluate import SEED, make_protocol, make_splits, write_json
from backend.scoring import calibrate, metrics, ranked_queries
from training.osnet import ReIDTrainerModel, export_encoder_onnx, load_encoder_from_onnx
from training.preprocessing import ResizeCrop


@dataclass
class TrainConfig:
    seed: int = SEED
    epochs: int = 20
    identities_per_batch: int = 8
    images_per_identity: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 5e-4
    triplet_margin: float = 0.3
    triplet_weight: float = 1.0
    consistency_weight: float = 0.2
    label_smoothing: float = 0.1
    num_workers: int = 0
    mode: str = "development"  # development: 925 IDs; final: train+validation (1234 IDs)
    use_augmentation: bool = True

    @property
    def batch_size(self):
        return self.identities_per_batch * self.images_per_identity

    def validate(self):
        if self.mode not in {"development", "final"}:
            raise ValueError("mode must be 'development' or 'final'")
        if self.identities_per_batch < 2 or self.images_per_identity < 2:
            raise ValueError("Triplet loss requires at least 2 identities and 2 images per identity")
        if self.epochs < 1 or self.learning_rate <= 0 or self.consistency_weight < 0:
            raise ValueError("epochs and learning_rate must be positive; consistency_weight must be non-negative")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def format_duration(seconds):
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def progress_line(epoch, total_epochs, timing, best_map=None):
    parts = [f"Epoch {epoch}/{total_epochs}", f"осталось эпох: {total_epochs - epoch}",
             f"эпоха: {format_duration(timing['epoch_seconds'])}",
             f"train: {format_duration(timing['train_seconds'])}"]
    if "evaluation_seconds" in timing:
        parts.append(f"evaluation: {format_duration(timing['evaluation_seconds'])}")
    parts += [f"прошло: {format_duration(timing['elapsed_seconds'])}",
              f"ETA: {format_duration(timing['estimated_remaining_seconds'])}"]
    if best_map is not None:
        parts.append(f"best mAP: {best_map:.4f}")
    return " | ".join(parts)


def ensure_splits(dataset=DATASET, path=Path("artifacts/splits.json"), seed=SEED):
    rows = read_rows(dataset / "train.csv")
    csv_hash = sha256(dataset / "train.csv")
    if path.exists():
        saved = json.loads(path.read_text())
        identities = saved.get("identities", {})
        flattened = [value for values in identities.values() for value in values]
        expected = sorted({row["vehicle_id"] for row in rows})
        if saved.get("train_csv_sha256") == csv_hash and sorted(flattened) == expected:
            return rows, saved
    print("Creating identity/frame-disjoint split…")
    hashes = {row["image_id"]: sha256(dataset / "images" / f"{row['image_id']}.jpg") for row in tqdm(rows)}
    identities = make_splits(rows, hashes, seed)
    protocols = {name: make_protocol(rows, identities[name], seed) for name in ("calibration", "validation")}
    saved = {
        "seed": seed, "identities": identities, "train_csv_sha256": csv_hash, "frame_sha256": hashes,
        "protocols": {name: {"query_ids": [r["image_id"] for r in query],
                              "gallery_ids": [r["image_id"] for r in gallery]}
                      for name, (query, gallery) in protocols.items()},
    }
    write_json(path, saved)
    return rows, saved


def split_summary(rows, split):
    owner = {identity: name for name, identities in split["identities"].items() for identity in identities}
    images = defaultdict(int)
    for row in rows:
        images[owner[row["vehicle_id"]]] += 1
    return {name: {"identities": len(ids), "images": images[name]} for name, ids in split["identities"].items()}


class RandomDownscale:
    """Simulate a low-resolution camera crop without changing its final size."""
    def __init__(self, probability=.25, scale=(.45, .8)):
        self.probability = probability
        self.scale = scale

    def __call__(self, image):
        if random.random() >= self.probability:
            return image
        width, height = image.size
        factor = random.uniform(*self.scale)
        small = image.resize((max(1, round(width * factor)), max(1, round(height * factor))),
                             Image.Resampling.BILINEAR)
        return small.resize((width, height), Image.Resampling.BILINEAR)


class RandomJPEGCompression:
    """Round-trip a crop through JPEG to vary compression artefacts."""
    def __init__(self, probability=.25, quality=(35, 85)):
        self.probability = probability
        self.quality = quality

    def __call__(self, image):
        if random.random() >= self.probability:
            return image
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=random.randint(*self.quality))
        output.seek(0)
        with Image.open(output) as compressed:
            return compressed.convert("RGB")


class RandomLowerCenterOcclusion:
    """Hide a random lower-centre patch so it cannot become an identity shortcut."""
    def __init__(self, probability=.35, width=(.25, .55), height=(.06, .16), center_y=(.62, .82)):
        self.probability = probability
        self.width = width
        self.height = height
        self.center_y = center_y

    def __call__(self, tensor):
        if random.random() >= self.probability:
            return tensor
        _, image_height, image_width = tensor.shape
        patch_width = max(1, round(image_width * random.uniform(*self.width)))
        patch_height = max(1, round(image_height * random.uniform(*self.height)))
        center_x = round(image_width * random.uniform(.42, .58))
        center_y = round(image_height * random.uniform(*self.center_y))
        left = min(max(0, center_x - patch_width // 2), image_width - patch_width)
        top = min(max(0, center_y - patch_height // 2), image_height - patch_height)
        result = tensor.clone()
        fill = tensor.mean(dim=(1, 2), keepdim=True)
        result[:, top:top + patch_height, left:left + patch_width] = fill
        return result


def image_transforms(robust=False, resize_mode="square"):
    operations = [ResizeCrop(resize_mode), transforms.RandomHorizontalFlip(),
                  transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.1, hue=.02)]
    if robust:
        operations += [transforms.RandomApply([transforms.GaussianBlur(5, sigma=(.1, 1.8))], p=.25),
                       RandomDownscale(), RandomJPEGCompression()]
    operations.append(transforms.ToTensor())
    if robust:
        operations += [transforms.RandomErasing(p=.3, scale=(.02, .12), ratio=(.3, 3.3), value="random"),
                       RandomLowerCenterOcclusion()]
    operations.append(transforms.Normalize([.485, .456, .406], [.229, .224, .225]))
    return transforms.Compose(operations)


class VehicleDataset(Dataset):
    def __init__(self, rows, dataset=DATASET, augment=False, resize_mode="square"):
        self.rows = rows
        self.dataset = dataset
        self.augment = augment
        self.clean_transform = image_transforms(resize_mode=resize_mode) if augment else transforms.Compose([
            ResizeCrop(resize_mode), transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        self.robust_transform = image_transforms(robust=True, resize_mode=resize_mode) if augment else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.dataset / "images" / f"{row['image_id']}.jpg") as image:
            cropped = crop_image(image, bbox(row)).convert("RGB")
            clean = self.clean_transform(cropped)
            if self.augment:
                return clean, self.robust_transform(cropped), row["label"], row["image_id"]
        return clean, row["label"], row["image_id"]


class PKBatchSampler(Sampler):
    """Each batch contains P vehicle identities and K images of every identity."""
    def __init__(self, rows, identities_per_batch, images_per_identity, seed=SEED):
        self.groups = defaultdict(list)
        for index, row in enumerate(rows):
            self.groups[row["label"]].append(index)
        self.identities = sorted(self.groups)
        self.p = identities_per_batch
        self.k = images_per_identity
        self.seed = seed
        self.epoch = 0
        if len(self.identities) < self.p:
            raise ValueError("Not enough identities for one PK batch")

    def __len__(self):
        return len(self.identities) // self.p

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        identities = self.identities.copy()
        rng.shuffle(identities)
        usable = len(identities) - len(identities) % self.p
        for start in range(0, usable, self.p):
            batch = []
            for identity in identities[start:start + self.p]:
                choices = self.groups[identity]
                batch.extend(rng.sample(choices, self.k) if len(choices) >= self.k
                             else rng.choices(choices, k=self.k))
            yield batch


def prepare_training(rows, split, config, dataset=DATASET):
    config.validate()
    train_ids = list(split["identities"]["train"])
    if config.mode == "final":
        train_ids += split["identities"]["validation"]
    train_ids = sorted(train_ids)
    labels = {identity: index for index, identity in enumerate(train_ids)}
    selected = [{**row, "label": labels[row["vehicle_id"]]} for row in rows if row["vehicle_id"] in labels]
    dataset_object = VehicleDataset(selected, dataset, config.use_augmentation)
    sampler = PKBatchSampler(selected, config.identities_per_batch, config.images_per_identity, config.seed)
    loader = DataLoader(dataset_object, batch_sampler=sampler, num_workers=config.num_workers,
                        pin_memory=torch.cuda.is_available(), persistent_workers=config.num_workers > 0)
    return selected, labels, sampler, loader


def batch_hard_triplet_loss(embeddings, labels, margin=0.3):
    embeddings = F.normalize(embeddings, dim=1)
    distances = torch.cdist(embeddings, embeddings)
    same = labels[:, None].eq(labels[None, :])
    same.fill_diagonal_(False)
    different = ~labels[:, None].eq(labels[None, :])
    hardest_positive = distances.masked_fill(~same, -torch.inf).max(dim=1).values
    hardest_negative = distances.masked_fill(~different, torch.inf).min(dim=1).values
    if not torch.isfinite(hardest_positive).all() or not torch.isfinite(hardest_negative).all():
        raise ValueError("Every batch must contain >=2 identities and >=2 images per identity")
    return F.relu(hardest_positive - hardest_negative + margin).mean()


def consistency_loss(embeddings, targets):
    return (1 - F.cosine_similarity(embeddings, targets.detach(), dim=1)).mean()


def training_losses(model, clean_images, robust_images, labels, config):
    """Train on the degraded view and match its embedding to a stop-gradient clean view."""
    model.eval()
    with torch.no_grad():
        clean_embeddings = model.encoder(clean_images)
    model.train()
    logits, embeddings = model(robust_images)
    ce = F.cross_entropy(logits, labels, label_smoothing=config.label_smoothing)
    triplet = batch_hard_triplet_loss(embeddings, labels, config.triplet_margin)
    consistency = consistency_loss(embeddings, clean_embeddings)
    loss = ce + config.triplet_weight * triplet + config.consistency_weight * consistency
    return {"loss": loss, "cross_entropy": ce, "triplet": triplet, "consistency": consistency,
            "accuracy": (logits.argmax(1) == labels).float().mean()}


def train_one_epoch(model, loader, sampler, optimizer, device, config, epoch):
    model.train()
    sampler.set_epoch(epoch)
    totals = defaultdict(float)
    for clean_images, robust_images, labels, _ in tqdm(loader, desc=f"epoch {epoch + 1}"):
        clean_images, robust_images, labels = clean_images.to(device), robust_images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        losses = training_losses(model, clean_images, robust_images, labels, config)
        losses["loss"].backward()
        optimizer.step()
        for name, value in losses.items():
            totals[name] += value.item()
        totals["batches"] += 1
    return {key: value / totals["batches"] for key, value in totals.items() if key != "batches"}


@torch.inference_mode()
def encode_rows(model, rows, device, dataset=DATASET, batch_size=64, num_workers=0):
    prepared = [{**row, "label": 0} for row in rows]
    loader = DataLoader(VehicleDataset(prepared, dataset, augment=False), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    model.eval()
    result = []
    for images, _, _ in tqdm(loader, desc="embedding", leave=False):
        result.append(F.normalize(model.encoder(images.to(device)), dim=1).cpu().numpy())
    return np.concatenate(result).astype(np.float32)


def evaluate_split(model, rows, identities, device, dataset=DATASET, seed=SEED, threshold=None, num_workers=0):
    ranked = rank_split(model, rows, identities, device, dataset, seed, num_workers)
    threshold = calibrate(ranked) if threshold is None else threshold
    return metrics(ranked, threshold), threshold


def rank_split(model, rows, identities, device, dataset=DATASET, seed=SEED, num_workers=0):
    query, gallery = make_protocol(rows, identities, seed)
    unique = {row["image_id"]: row for row in query + gallery}
    selected = list(unique.values())
    vectors = encode_rows(model, selected, device, dataset, num_workers=num_workers)
    embeddings = dict(zip((row["image_id"] for row in selected), vectors))
    return ranked_queries(query, gallery, embeddings)


def mask_lower_center(images, width_fraction=.5, height_fraction=.16, center_y=.72):
    """Deterministic proxy mask for auditing; this is not a plate detector."""
    squeeze = images.ndim == 3
    if squeeze:
        images = images.unsqueeze(0)
    if images.ndim != 4:
        raise ValueError("Expected CHW or BCHW tensor")
    result = images.clone()
    _, _, height, width = result.shape
    patch_width = max(1, round(width * width_fraction))
    patch_height = max(1, round(height * height_fraction))
    left = (width - patch_width) // 2
    top = min(max(0, round(height * center_y - patch_height / 2)), height - patch_height)
    result[:, :, top:top + patch_height, left:left + patch_width] = 0
    return result[0] if squeeze else result


@torch.inference_mode()
def lower_center_invariance(model, rows, device, dataset=DATASET, max_samples=128, batch_size=64):
    selected = [{**row, "label": 0} for row in rows[:max_samples]]
    loader = DataLoader(VehicleDataset(selected, dataset, augment=False), batch_size=batch_size, shuffle=False)
    similarities = []
    model.eval()
    for images, _, _ in loader:
        images = images.to(device)
        original = F.normalize(model.encoder(images), dim=1)
        masked = F.normalize(model.encoder(mask_lower_center(images)), dim=1)
        similarities.extend(F.cosine_similarity(original, masked, dim=1).cpu().tolist())
    values = np.asarray(similarities, dtype=np.float32)
    return {"samples": len(values), "mean_cosine": float(values.mean()),
            "p05_cosine": float(np.percentile(values, 5)), "min_cosine": float(values.min()),
            "fraction_below_0_95": float(np.mean(values < .95))}


@torch.inference_mode()
def occlusion_sensitivity(model, query, reference, device, grid=6):
    """Return the cosine drop when each grid cell of a query is hidden."""
    if query.ndim != 3 or reference.ndim != 3:
        raise ValueError("query and reference must be CHW tensors")
    model.eval()
    query, reference = query.to(device), reference.to(device)
    reference_embedding = F.normalize(model.encoder(reference.unsqueeze(0)), dim=1)
    base_embedding = F.normalize(model.encoder(query.unsqueeze(0)), dim=1)
    base_similarity = F.cosine_similarity(base_embedding, reference_embedding).item()
    variants = []
    _, height, width = query.shape
    for row in range(grid):
        for column in range(grid):
            variant = query.clone()
            top, bottom = round(row * height / grid), round((row + 1) * height / grid)
            left, right = round(column * width / grid), round((column + 1) * width / grid)
            variant[:, top:bottom, left:right] = 0
            variants.append(variant)
    embeddings = F.normalize(model.encoder(torch.stack(variants)), dim=1)
    similarities = embeddings @ reference_embedding.T
    drops = base_similarity - similarities[:, 0].cpu().numpy()
    return base_similarity, drops.reshape(grid, grid)


def fit(model, rows, split, loader, sampler, device, config, output_dir=Path("artifacts/training")):
    if config.mode != "development":
        raise ValueError("Use fit_final for mode='final'; final training must not select epochs on validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    history, best_map = [], -1.0
    training_started = time.perf_counter()
    for epoch in range(config.epochs):
        epoch_started = time.perf_counter()
        train_result = train_one_epoch(model, loader, sampler, optimizer, device, config, epoch)
        train_seconds = time.perf_counter() - epoch_started
        calibration, threshold = evaluate_split(model, rows, split["identities"]["calibration"], device,
                                                 seed=config.seed, num_workers=config.num_workers)
        validation, _ = evaluate_split(model, rows, split["identities"]["validation"], device,
                                       seed=config.seed, threshold=threshold, num_workers=config.num_workers)
        epoch_seconds = time.perf_counter() - epoch_started
        elapsed_seconds = time.perf_counter() - training_started
        completed = epoch + 1
        timing = {"train_seconds": train_seconds, "evaluation_seconds": epoch_seconds - train_seconds,
                  "epoch_seconds": epoch_seconds, "elapsed_seconds": elapsed_seconds,
                  "estimated_remaining_seconds": elapsed_seconds / completed * (config.epochs - completed)}
        record = {"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"], "train": train_result,
                  "threshold": threshold, "calibration": calibration, "validation": validation,
                  "timing": timing}
        history.append(record)
        write_json(output_dir / "history.json", history)
        is_best = validation["mAP"] > best_map
        if is_best:
            best_map = validation["mAP"]
        print(progress_line(completed, config.epochs, timing, best_map), flush=True)
        print(json.dumps(record, ensure_ascii=False, indent=2))
        if is_best:
            torch.save({"epoch": epoch + 1, "model": model.state_dict(), "config": asdict(config),
                        "threshold": threshold, "calibration": calibration, "validation": validation},
                       output_dir / "best.pt")
        scheduler.step()
    return history


def fit_final(model, loader, sampler, device, config, epochs, output_dir=Path("artifacts/training/final")):
    if config.mode != "final":
        raise ValueError("Set mode='final' to train on train+validation identities")
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = []
    training_started = time.perf_counter()
    for epoch in range(epochs):
        epoch_started = time.perf_counter()
        train_result = train_one_epoch(model, loader, sampler, optimizer, device, config, epoch)
        epoch_seconds = time.perf_counter() - epoch_started
        elapsed_seconds = time.perf_counter() - training_started
        completed = epoch + 1
        timing = {"train_seconds": epoch_seconds, "epoch_seconds": epoch_seconds,
                  "elapsed_seconds": elapsed_seconds,
                  "estimated_remaining_seconds": elapsed_seconds / completed * (epochs - completed)}
        record = {"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"],
                  "train": train_result, "timing": timing}
        history.append(record)
        write_json(output_dir / "history.json", history)
        print(progress_line(completed, epochs, timing), flush=True)
        print(json.dumps(record, ensure_ascii=False, indent=2))
        scheduler.step()
    return history


def initialize_model(num_classes, device, onnx_path=STOCK_MODEL):
    model = ReIDTrainerModel(num_classes)
    count = load_encoder_from_onnx(model.encoder, onnx_path)
    return model.to(device), count


def save_final(model, rows, split, device, config, output_dir=Path("artifacts/training/final")):
    calibration, threshold = evaluate_split(model, rows, split["identities"]["calibration"], device,
                                             seed=config.seed, num_workers=config.num_workers)
    checkpoint = {"model": model.state_dict(), "config": asdict(config), "threshold": threshold,
                  "calibration": calibration}
    torch.save(checkpoint, output_dir / "final.pt")
    onnx_path = export_encoder_onnx(model.encoder, output_dir / "osnet_finetuned.onnx", "cpu")
    metadata = {"fine_tuned": True, "embedding_dim": 512, "preprocessing": "strict bbox resize 208x208 ImageNet",
                "threshold": threshold, "calibration": calibration, "training_config": asdict(config)}
    write_json(Path(str(onnx_path) + ".json"), metadata)
    return checkpoint, onnx_path


def parity_check(model, sample_row, device, dataset=DATASET, onnx_path=STOCK_MODEL):
    import onnxruntime as ort
    from backend.core import preprocess

    with Image.open(dataset / "images" / f"{sample_row['image_id']}.jpg") as image:
        array = preprocess(image, bbox(sample_row))[None]
    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])
    reference = session.run(["output"], {session.get_inputs()[0].name: array})[0]
    model.eval()
    with torch.inference_mode():
        actual = model.encoder(torch.from_numpy(array).to(device)).cpu().numpy()
    difference = float(np.max(np.abs(reference - actual)))
    cosine = float(torch.sum(F.normalize(torch.from_numpy(reference), dim=1) *
                             F.normalize(torch.from_numpy(actual), dim=1)))
    return {"max_absolute_difference": difference, "cosine_similarity": cosine}
