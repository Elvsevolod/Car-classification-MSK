"""v20 controlled training duration/schedule and post-training weight averaging."""
from dataclasses import asdict, replace

import torch
from torch.utils.data import DataLoader

from backend.core import sha256
from training.audit import digest
from training import overnight_training as training
from training.osnet_ablations import AblationDataset


def jobs():
    return {j.name: j for j in (
        training.TrainingJob("R1_control", "R1_resolution256", stop_step=800, lr_horizon=1700),
        training.TrainingJob("R1_cosine800", "R1_resolution256", stop_step=800, lr_horizon=800),
        training.TrainingJob("R1_full1700", "R1_resolution256", stop_step=1700, lr_horizon=1700),
    )}


CASES = ("R1_control", "R1_cosine800", "R1_full1700", "R1_full1700_bn", "R1_full1700_avg")


def job_name(case):
    if case not in CASES:
        raise ValueError("Unknown clock case")
    return "R1_full1700" if case.endswith(("_bn", "_avg")) else case


def job_context(context, name):
    job = jobs()[name]
    budget = replace(context["budget"], max_steps=job.lr_horizon)
    budget.validate()
    return {**context, "budget": budget,
            "signature": digest({"context": context["signature"], "clock_job": asdict(job)})}


def bn_batches(count, batch_size=32):
    if count < 2:
        raise ValueError("BN calibration needs at least two train images")
    batches = [list(range(i, min(i + batch_size, count))) for i in range(0, count, batch_size)]
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2].extend(batches.pop())
    return batches


def recalibrate_bn(model, batches, device):
    """Only BatchNorm trains; dropout/augmentation remain off. No held-out inputs.

    Follows torch.optim.swa_utils.update_bn's running-stat reset/momentum=None,
    but does not put unrelated stochastic modules in training mode.
    """
    model.eval()
    modules = {m: m.momentum for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)}
    for module in modules:
        module.reset_running_stats()
        module.momentum = None
        module.train()
    count = 0
    try:
        with torch.no_grad():
            for batch in batches:
                if len(batch) < 2:
                    raise ValueError("Singleton BN batch")
                model(batch.to(device))
                count += len(batch)
        if not count:
            raise ValueError("Empty BN calibration")
    finally:
        for module, momentum in modules.items():
            module.momentum = momentum
        model.eval()
    return count


def average_parameters(model, states):
    """Average only parameters; last-state buffers are later recalibrated on train."""
    if len(states) < 2:
        raise ValueError("Need at least two fixed checkpoints")
    model.load_state_dict(states[-1])
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(torch.stack([state[name] for state in states]).mean(0).to(parameter.device))


def derived_model(context, summary, case, directory):
    """Fixed end-state, or BN-only/average with identical train-only BN treatment."""
    if not directory.resolve().is_relative_to(context["output"].resolve()):
        raise ValueError("Derived weights must stay inside the new run")
    jc = job_context(context, job_name(case))
    model, variant = training.load_job_model(jc, summary)
    if not case.endswith(("_bn", "_avg")):
        return model, variant, {"case": case, "bn_reestimated": False}
    steps = sorted(map(int, summary["checkpoints"]))[-3:] if case.endswith("_avg") else [summary["stop_step"]]
    states, provenance = [], []
    for step in steps:
        entry = summary["checkpoints"][str(step)]
        path = context["output"] / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise ValueError("Averaging checkpoint changed")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["signature"] != summary["signature"] or saved["step"] != step:
            raise ValueError("Averaging checkpoint provenance mismatch")
        states.append(saved["model"])
        provenance.append({"step": step, **entry})
    if case.endswith("_avg"):
        average_parameters(model, states)
    allowed = training.fold_training_ids(context, summary["fold"])
    rows = [r for r in context["rows"] if r["vehicle_id"] in allowed]
    dataset = AblationDataset(rows, variant, context["dataset"], augment=False)
    loader = DataLoader(dataset, batch_sampler=bn_batches(len(rows)), num_workers=0)
    count = recalibrate_bn(model, (batch[0] for batch in loader), context["device"])
    path = directory / "derived.pt"
    metadata = {"case": case, "sources": provenance, "bn_reestimated": True, "bn_train_images": count,
                "bn_identity_digest": digest(sorted(allowed)), "context_signature": context["signature"]}
    training.review.old.save_checkpoint(path, {"model": model.state_dict(), "metadata": metadata})
    return model, variant, {**metadata, "path": str(path.relative_to(context["output"])), "sha256": sha256(path)}
