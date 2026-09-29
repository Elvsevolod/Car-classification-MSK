"""Train the four frozen v25 member recipes in a new output directory."""
import argparse
import json
import os
import platform
import sys
from pathlib import Path

from verify import ROOT, read_json, sha256, verify_dataset, verify_kit

MEMBERS = ("legacy", "r1_20260915", "r1_20260916", "r1_20260917")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def prepare(dataset, device, member):
    # Set before importing the historical modules: several default arguments bind DATASET.
    os.environ["DATASET_DIR"] = str(dataset)
    if member != "legacy":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.path.insert(0, str(ROOT / "source"))
    import torch
    from backend.core import read_rows
    from backend.evaluate import make_protocol
    from training.hpo import ExperimentConfig

    torch.set_num_threads(8)
    if member != "legacy":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    split = read_json(ROOT / "metadata/splits.json")
    rows = read_rows(dataset / "train.csv")
    for name, saved in split["protocols"].items():
        query, gallery = make_protocol(rows, split["identities"][name], split["seed"])
        if ([r["image_id"] for r in query] != saved["query_ids"] or
                [r["image_id"] for r in gallery] != saved["gallery_ids"]):
            raise ValueError(f"Protocol reconstruction changed: {name}")
    recipe = read_json(ROOT / "metadata/recipe.json")
    manifest_path = recipe["r1"]["config_manifest"]
    if member in ("r1_20260916", "r1_20260917"):
        manifest_path = ("source/OSNet-AIN-x1.0/variant_17_final_seed_confirmation/"
                         "runs/final_seeds_v1/manifest.json")
    manifest = read_json(ROOT / manifest_path)
    if manifest["outer"] != split["identities"] or manifest["protocols"] != split["protocols"]:
        raise ValueError("Selected training manifest uses a different outer split")
    return {"rows": rows, "split": split, "dataset": dataset, "device": torch.device(device),
            "recipe": recipe, "manifest": manifest,
            "base": ExperimentConfig(**manifest["base_recipe"])}


def train_legacy(context, output):
    """Same epoch body as hpo.fit_selected; stop at the already chosen epoch 5.

    Keep the 30-epoch LR horizon and both evaluations between epochs. Omitting
    those DataLoader iterations would change the random state of the next epoch.
    """
    import torch
    from training import hpo
    from training.pipeline import set_seed

    selected = context["recipe"]["legacy"]
    config = hpo.ExperimentConfig(**selected["config"])
    set_seed(config.seed)
    _, labels, sampler, loader = hpo.prepare_experiment(
        context["rows"], context["split"]["identities"]["train"], config, context["dataset"])
    model, _ = hpo.initialize_experiment(len(labels), config, context["device"])
    optimizer = hpo.make_optimizer(model, config)
    history = []
    for epoch in range(selected["selected_epoch"]):
        lrs = hpo.set_epoch_learning_rates(optimizer, config, epoch, config.epochs)
        trained = hpo.train_epoch(model, loader, sampler, optimizer, context["device"], config, epoch)
        calibration, threshold = hpo.evaluate_experiment(
            model, context["rows"], context["split"]["identities"]["calibration"], context["device"],
            seed=config.seed, num_workers=config.num_workers)
        validation, _ = hpo.evaluate_experiment(
            model, context["rows"], context["split"]["identities"]["validation"], context["device"],
            seed=config.seed, threshold=threshold, num_workers=config.num_workers)
        history.append({"epoch": epoch + 1, "lr": lrs, "train": trained, "threshold": threshold,
                        "calibration": calibration, "validation": validation})
        write_json(output / "history.json", history)
        torch.save(hpo._checkpoint_payload(epoch + 1, model, optimizer, config, threshold,
                                          calibration, validation), output / "last.pt")
        print(f"legacy epoch {epoch + 1}/{selected['selected_epoch']} (LR horizon {config.epochs})", flush=True)
    return output / "last.pt"


def r1_context(context, output):
    from training.audit import digest
    from training.osnet_ablation_suite import Budget
    from training.osnet_review_protocol import variants

    return {**context, "output": output, "signature": digest(context["manifest"]),
            "budget": Budget(**context["manifest"]["budget"]),
            "seeds": tuple(context["recipe"]["r1"]["seeds"]), "variants": variants()}


def train_r1(context, member, output):
    from training.osnet_review_protocol import fit

    context = r1_context(context, output)
    step = context["recipe"]["r1"]["selected_step"]
    summary = fit(context, context["recipe"]["r1"]["variant"], int(member[3:]), fold="final", stop_step=step)
    return output / summary["checkpoints"][str(step)]["path"]


def export(context, member, checkpoint, output):
    import numpy as np
    import onnxruntime as ort
    import torch
    from training.hpo import ExperimentConfig, initialize_experiment
    from training.osnet import export_encoder_onnx
    from training.osnet_ablations import AblationDataset, InferenceEncoder
    from training.osnet_review_protocol import initialize, variants
    from training.pipeline import VehicleDataset

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    classes = len(context["split"]["identities"]["train"])
    if member == "legacy":
        reference = context["recipe"]["legacy"]
        config = ExperimentConfig(**saved["config"])
        if (saved["epoch"] != reference["selected_epoch"] or
                config != ExperimentConfig(**reference["config"])):
            raise ValueError("Expected the frozen legacy recipe at epoch 5")
        model, _ = initialize_experiment(classes, config, "cpu")
        model.load_state_dict(saved["model"])
        encoder = model.inference_module().eval()
        export_encoder_onnx(encoder, output / "encoder.onnx", "cpu")
        data_class = lambda rows: VehicleDataset(rows, context["dataset"], augment=False)
        tolerance = 1e-3
    else:
        if saved["step"] != context["recipe"]["r1"]["selected_step"]:
            raise ValueError("Expected the frozen R1 checkpoint at step 800")
        from training.audit import digest
        allowed = set(context["split"]["identities"]["train"])
        labels = {identity: index for index, identity in enumerate(sorted(allowed))}
        training_rows = [{**r, "label": labels[r["vehicle_id"]]} for r in context["rows"]
                         if r["vehicle_id"] in allowed]
        expected_signature = digest({"context": digest(context["manifest"]),
                                     "variant": context["recipe"]["r1"]["variant"],
                                     "seed": int(member[3:]), "fold": "final",
                                     "stop": 800, "rows": training_rows})
        if saved["signature"] != expected_signature:
            raise ValueError("R1 checkpoint belongs to another seed, recipe or split")
        variant = variants()[context["recipe"]["r1"]["variant"]]
        model = initialize(classes, variant.recipe(context["base"], int(member[3:])), variant, "cpu")
        model.load_state_dict(saved["model"])
        encoder = InferenceEncoder(model).eval()
        torch.onnx.export(encoder, torch.zeros(1, 3, variant.size, variant.size), output / "encoder.onnx",
                          input_names=["input"], output_names=["output"],
                          dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                          opset_version=17, dynamo=False)
        data_class = lambda rows: AblationDataset(rows, variant, context["dataset"])
        tolerance = 2e-4
    lookup = {r["image_id"]: r for r in context["rows"]}
    data = data_class([{**lookup[i], "label": 0}
                       for i in context["split"]["protocols"]["calibration"]["query_ids"][:8]])
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    session = ort.InferenceSession(str(output / "encoder.onnx"), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    parity = {}
    for batch in (1, 3, 8):
        images = torch.stack([data[i][0] for i in range(batch)])
        with torch.inference_mode():
            expected = encoder(images).numpy()
        actual = session.run(None, {"input": images.numpy()})[0]
        error = float(np.max(np.abs(actual - expected)))
        if actual.shape != (batch, 512) or not np.isfinite(actual).all() or error > tolerance:
            raise ValueError(f"ONNX parity failed: batch={batch}, max_abs={error}")
        if member != "legacy" and not np.allclose(np.linalg.norm(actual, axis=1), 1, atol=1e-5):
            raise ValueError("R1 output is not unit normalized")
        parity[str(batch)] = error
    reference = (context["recipe"]["legacy"]["onnx_sha256"] if member == "legacy" else
                 read_json(ROOT / f"metadata/{member}_bundle.json")["model"]["sha256"])
    result = {"member": member, "checkpoint_sha256": sha256(checkpoint),
              "onnx_sha256": sha256(output / "encoder.onnx"), "reference_onnx_sha256": reference,
              "byte_identical_to_reference": sha256(output / "encoder.onnx") == reference,
              "cpu_parity_max_abs": parity, "gpu_parity": "not tested", "promoted": False}
    write_json(output / "export.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "train", "export"))
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--member", choices=MEMBERS, default="legacy")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    checked = {"kit": verify_kit(), "dataset": verify_dataset(dataset)}
    context = prepare(dataset, args.device, args.member)
    if args.action == "preflight":
        print(json.dumps(checked, indent=2))
        return
    if args.output is None or (args.action == "export" and args.checkpoint is None):
        parser.error("train/export require --output; export also requires --checkpoint")
    output = args.output.resolve()
    if output.is_relative_to(ROOT / "source") or output.is_relative_to(dataset):
        parser.error("Output must be outside the historical source and dataset")
    output.mkdir(parents=True, exist_ok=False)
    import torch
    write_json(output / "run.json", {**checked, "member": args.member, "device": args.device,
               "python": platform.python_version(), "torch": str(torch.__version__),
               "recipe_sha256": sha256(ROOT / "metadata/recipe.json"), "promoted": False})
    if args.action == "train":
        checkpoint = (train_legacy(context, output) if args.member == "legacy" else
                      train_r1(context, args.member, output))
    else:
        checkpoint = args.checkpoint.resolve()
    print(json.dumps(export(context, args.member, checkpoint, output), indent=2))


if __name__ == "__main__":
    main()
