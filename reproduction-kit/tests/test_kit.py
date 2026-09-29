import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify import ROOT, read_json, sha256, verify_dataset, verify_kit


class DatasetChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "metadata").mkdir()
        self.dataset = self.root / "dataset"
        (self.dataset / "images").mkdir(parents=True)
        with (self.dataset / "train.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image_id", "vehicle_id"])
            writer.writerows([("one", 1), ("two", 2)])
        for name in ("one", "two"):
            (self.dataset / "images" / f"{name}.jpg").write_bytes(name.encode())
        self.save_split()

    def save_split(self):
        self.split = {"train_csv_sha256": sha256(self.dataset / "train.csv"),
                      "identities": {"train": [1], "calibration": [2], "validation": []},
                      "frame_sha256": {n: sha256(self.dataset / "images" / f"{n}.jpg")
                                       for n in ("one", "two")}}
        (self.root / "metadata/splits.json").write_text(json.dumps(self.split))

    def test_original_data_passes_without_any_test_files(self):
        self.assertEqual(verify_dataset(self.dataset, self.root)["train_csv_rows"], 2)

    def test_changed_image_is_rejected(self):
        (self.dataset / "images/one.jpg").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Training image changed"):
            verify_dataset(self.dataset, self.root)

    def test_changed_annotations_are_rejected(self):
        with (self.dataset / "train.csv").open("a") as stream:
            stream.write("three,3\n")
        with self.assertRaisesRegex(ValueError, "train.csv differs"):
            verify_dataset(self.dataset, self.root)

    def test_identical_frame_across_partitions_is_rejected(self):
        (self.dataset / "images/two.jpg").write_bytes(b"one")
        self.save_split()
        with self.assertRaisesRegex(ValueError, "identical frame"):
            verify_dataset(self.dataset, self.root)


class SourceAndRecipeChecks(unittest.TestCase):
    def test_historical_sources_match_training_time_hashes(self):
        self.assertEqual(verify_kit()["historical_files"], 83)

    def test_legacy_prefix_matches_original_30_epoch_driver(self):
        # Lightweight model, real optimizer and RNG: compare orchestration against
        # the unchanged historical driver, including evaluation between epochs.
        import torch
        sys.path.insert(0, str(ROOT / "source"))
        from training import hpo
        from training.pipeline import set_seed
        from run_selected import train_legacy

        class StopAfterSelectedEpoch(Exception):
            pass

        recipe = read_json(ROOT / "metadata/recipe.json")
        config = hpo.ExperimentConfig(**recipe["legacy"]["config"])
        split = {"identities": {"train": [1, 2], "calibration": [3], "validation": [4]}}

        def initialize(*args):
            return torch.nn.Linear(2, 2), 0

        def optimizer(model, config):
            return torch.optim.SGD([{"params": list(model.parameters()), "base_lr": config.encoder_lr,
                                     "lr": config.encoder_lr, "name": "encoder"}])

        def train(model, loader, sampler, opt, device, config, epoch):
            if epoch == 5:
                raise StopAfterSelectedEpoch()
            opt.zero_grad()
            loss = model(torch.randn(2, 2)).square().mean()
            loss.backward()
            opt.step()
            return {"loss": loss.item()}

        def evaluate(*args, **kwargs):
            value = torch.rand(1).item()
            return {"mAP": value, "candidate_F1": value, "TNR": value}, .5

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(hpo, "initialize_experiment", initialize), \
                patch.object(hpo, "prepare_experiment", return_value=([], {1: 0, 2: 1}, None, None)), \
                patch.object(hpo, "make_optimizer", optimizer), \
                patch.object(hpo, "train_epoch", train), \
                patch.object(hpo, "evaluate_experiment", evaluate), patch("builtins.print"):
            root = Path(directory)
            original, selected = root / "original", root / "selected"
            selected.mkdir()
            set_seed(config.seed)
            model, _ = initialize()
            with self.assertRaises(StopAfterSelectedEpoch):
                hpo.fit_selected(model, [], split, None, None, "cpu", config, original, original)
            train_legacy({"recipe": recipe, "rows": [], "split": split,
                          "dataset": root, "device": "cpu"}, selected)
            a = torch.load(original / "last.pt", weights_only=True)
            b = torch.load(selected / "last.pt", weights_only=True)
            self.assertEqual(a["epoch"], 5)
            self.assertEqual(b["config"]["epochs"], 30)
            self.assertEqual(a["optimizer"]["param_groups"], b["optimizer"]["param_groups"])
            for name in a["model"]:
                self.assertTrue(torch.equal(a["model"][name], b["model"][name]), name)


if __name__ == "__main__":
    unittest.main()
