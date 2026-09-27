"""Run-All acceptance harness. Every measurement is new; failures never promote R1."""
import argparse
import csv
import datetime as dt
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from uuid import uuid4

import numpy as np

from backend.benchmark import weight_inventory
from backend.cache_spaces import FileGallerySpaces
from backend.core import ROOT, read_rows, sha256
from backend.images import ImageIndex
from backend.runtime import DEFAULT_PROFILE, PROFILE_NAMES, ExactScorer, Runtime, RuntimeGallery, encode_rows, write_json

PRIMARY = ("MVP_legacy", "MVP_dual_role_v24", DEFAULT_PROFILE)
OPEN_CHECKS = [
    "GPU CUDA / RTX A5000 and native Linux amd64 not established by Mac checks",
    "No OCR is used, but absence of residual plate-zone signal is not proven",
    "Official extractor.py signature not supplied; only the known three-file command is implemented",
    "Junk is filtered by evaluator before truncation; an exported top-10 cannot recover rank 11",
    "Presentation and final submission links require separate review",
]


def ids(decision):
    item = decision["accepted_candidate"]
    return ([x["image_id"] for x in decision["results"]], item["image_id"] if item else None,
            decision["refused"])


def compare_exports(left, right, atol=2e-4):
    a, b = (np.load(Path(p) / "embeddings.npy", allow_pickle=False) for p in (left, right))
    if a.shape != b.shape:
        return {"passed": False, "reason": "embedding shape mismatch", "shapes": [list(a.shape), list(b.shape)]}
    rankings = [list(csv.reader((Path(p) / "submission.csv").open())) for p in (left, right)]
    candidates = [{r["query_id"]: (r["gallery_id"], float(r["confidence"]))
                   for r in csv.DictReader((Path(p) / "candidates.csv").open())} for p in (left, right)]
    changed = [x[0] for x, y in zip(*rankings) if x != y]
    union = set(candidates[0]) | set(candidates[1])
    changed_candidates = [q for q in sorted(union)
                          if candidates[0].get(q, (None,))[0] != candidates[1].get(q, (None,))[0]]
    confidence_error = max([abs(candidates[0][q][1] - candidates[1][q][1])
                            for q in set(candidates[0]) & set(candidates[1])] or [0.])
    error = float(np.max(np.abs(a - b)))
    return {"passed": len(rankings[0]) == len(rankings[1]) and not changed and not changed_candidates
            and error <= atol and confidence_error <= atol, "max_abs_embedding_error": error,
            "max_abs_confidence_error": confidence_error, "atol": atol, "changed_top10": changed,
            "changed_candidate_or_refusal": changed_candidates, "queries": len(rankings[0])}


class Verification:
    def __init__(self, dataset, output=None, research=None, legacy_app=None, reference_python=None,
                 smoke=False, provider="CPUExecutionProvider"):
        self.research = Path(research or ROOT.parent / "Car-classification-MSK").resolve()
        self.legacy_app = Path(legacy_app or ROOT.parent / "Car-classification-MSK-review-fixes").resolve()
        self.reference_python = str(reference_python or self.research / ".venv/bin/python")
        name = dt.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]
        self.output = Path(output or ROOT / "artifacts/release_verification" / name).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.dataset = Path(dataset).resolve()
        self.smoke, self.provider = smoke, provider
        self.results = {}
        self.source_hashes = self.code_hashes()
        self.started = time.perf_counter()
        if smoke:
            self.dataset = self._smoke_dataset()
        write_json(self.output / "config.json", {
            "dataset": str(self.dataset), "research": str(self.research), "legacy_app": str(self.legacy_app),
            "reference_python": self.reference_python, "python": sys.executable, "provider": provider,
            "smoke": smoke, "overall_time_limit": None})
        write_json(self.output / "implementation_sha256.json", self.source_hashes)
        print(f"NEW RUN: {self.output}", flush=True)

    @staticmethod
    def code_hashes():
        paths = []
        for folder in ("backend", "tools", "tests", "alembic"):
            paths += list((ROOT / folder).rglob("*.py"))
        paths += list((ROOT / "web-ui/src").rglob("*.tsx")) + list((ROOT / "web-ui/e2e").glob("*.ts"))
        paths += [ROOT / "Dockerfile", ROOT / "requirements.txt", ROOT / "web-ui/package-lock.json"]
        return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}

    def _smoke_dataset(self):
        destination = self.output / "smoke_dataset"
        (destination / "images").mkdir(parents=True)
        images = ImageIndex(self.dataset / "images")
        for split, count in (("test_query", 32), ("test_gallery", 64)):
            rows = read_rows(self.dataset / f"{split}.csv")[:count]
            with (destination / f"{split}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            for row in rows:
                path = images.resolve(row["image_id"])
                shutil.copy2(path, destination / "images" / path.name)
        return destination

    def command(self, name, command, cwd=ROOT, env=None):
        environment = dict(os.environ, PYTHONUNBUFFERED="1")
        # Never implicitly target the user's application database from tests.
        environment.pop("DATABASE_URL", None)
        environment.pop("REID_TEST_DATABASE_URL", None)
        if env:
            environment.update(env)
        log = self.output / f"{name}.log"
        started = time.perf_counter()
        with log.open("w") as stream:
            process = subprocess.Popen([str(x) for x in command], cwd=cwd, env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                for line in process.stdout:
                    stream.write(line); stream.flush()
                    print(line, end="", flush=True)
                code = process.wait()
            except BaseException:
                process.terminate(); process.wait()
                raise
        if code:
            raise RuntimeError(f"{name} exited {code}; see {log}")
        result = {"log": log.name, "sha256": sha256(log), "exit_code": code,
                  "process_wall_seconds": time.perf_counter() - started}
        write_json(self.output / f"{name}.command.json", result)
        return result

    def stage(self, name, function):
        print(f"\n[{dt.datetime.now().isoformat(timespec='seconds')}] STAGE {name}", flush=True)
        started = time.perf_counter()
        try:
            details = function()
            status = "passed"
            if isinstance(details, dict) and details.get("passed") is False:
                status = "failed"
        except Exception as exc:
            status, details = "failed", {"error": str(exc), "traceback": traceback.format_exc()}
        self.results[name] = {"status": status, "seconds": time.perf_counter() - started, "details": details}
        write_json(self.output / "stages.json", self.results)
        print(f"{name}: {status}, {self.results[name]['seconds']:.1f}s", flush=True)
        return self.results[name]

    def preflight(self):
        weights = weight_inventory(ROOT / "models")
        if not weights["passed"]:
            raise ValueError("Delivery weight inventory exceeds 2 GB or is empty")
        rows = {split: read_rows(self.dataset / f"{split}.csv") for split in ("test_query", "test_gallery")}
        if len(rows["test_gallery"]) < 10:
            raise ValueError("Need at least 10 gallery items")
        images = ImageIndex(self.dataset / "images")
        for row in rows["test_query"] + rows["test_gallery"]:
            images.resolve(row["image_id"])
        profiles = {p: Runtime(p, self.provider).metadata() for p in PROFILE_NAMES}
        write_json(self.output / "weight_inventory.json", weights)
        return {"platform": platform.platform(), "profiles": profiles,
                "dataset_counts": {s: len(v) for s, v in rows.items()}, "weights_bytes": weights["total_bytes"],
                "official_gpu_verified": False}

    def components(self):
        return self.command("components", [sys.executable, "-m", "pytest", "-q",
            "tests/test_baseline.py", "tests/test_calibration.py", "tests/test_infer.py",
            "tests/test_official_evaluation.py", "tests/test_release_runtime.py", "tests/test_dataset_init.py"])

    def exports_and_reference(self):
        reports = {}
        for profile in PRIMARY:
            target, reference = self.output / f"export_{profile}", self.output / f"reference_{profile}"
            self.command(f"export_{profile}", [sys.executable, "-m", "backend.infer",
                "--dataset", self.dataset, "--output", target, "--profile", profile, "--provider", self.provider])
            if profile == "MVP_legacy":
                self.command(f"reference_{profile}", [sys.executable, "-m", "backend.infer",
                    "--dataset", self.dataset, "--output", reference], cwd=self.legacy_app,
                    env={"REID_PROFILE": "MVP_legacy", "REID_PROVIDER": "CPUExecutionProvider"})
            elif profile == "MVP_dual_role_v24":
                self.command(f"reference_{profile}", [self.reference_python, "-m", "training.dual_role_inference",
                    "--profile", self.research / "OSNet-AIN-x1.0/variant_24_dual_role/runs/dual_role_v1/dual_role_profile.json",
                    "--dataset", self.dataset, "--output", reference], cwd=self.research)
            elif profile == "MVP_fusion_v25":
                self.command(f"reference_{profile}", [self.reference_python, "-m", "training.map_inference",
                    "--profile", self.research / "OSNet-AIN-x1.0/variant_24_dual_role/runs/dual_role_v1/dual_role_profile.json",
                    "--selection", self.research / "OSNet-AIN-x1.0/variant_25_map_search/runs/map_search_v1/frozen_selection.json",
                    "--dataset", self.dataset, "--output", reference], cwd=self.research)
            else:
                spec = json.loads((ROOT / "models/profiles.json").read_text())[profile]
                relative = Path(spec["bundle"]).relative_to("models/frozen")
                bundle = self.research / relative
                if sha256(bundle) != spec["bundle_sha256"]:
                    raise ValueError("Reference policy bundle changed")
                self.command(f"reference_{profile}", [self.reference_python, "-m", "training.policy_inference",
                    "--bundle", bundle, "--dataset", self.dataset, "--output", reference], cwd=self.research)
            reports[profile] = compare_exports(reference, target)
        write_json(self.output / "reference_parity.json", reports)
        return {"passed": all(x["passed"] for x in reports.values()), "profiles": reports}

    def streaming_and_rollback(self):
        queries, rows = (read_rows(self.dataset / f"{split}.csv") for split in ("test_query", "test_gallery"))
        reports = {}
        cache = FileGallerySpaces(self.output / "cache_spaces")
        original = None
        for profile in (*PRIMARY, "MVP_legacy"):
            runtime = Runtime(profile, self.provider)
            gallery = RuntimeGallery(runtime, self.dataset, cache, progress=lambda n, t: print(f"{profile} gallery {n}/{t}", flush=True))
            base = encode_rows(runtime, queries, self.dataset, 1,
                               lambda n, t: print(f"{profile} batch1 {n}/{t}", flush=True) if n % 32 == 0 or n == t else None)
            decisions = [ids(gallery.decide(v)) for v in base]
            if profile == "MVP_legacy" and original is not None:
                assert gallery.cache_hit and original == decisions, "MVP rollback decisions changed"
                reports["rollback"] = {"passed": True, "cache_hit": gallery.cache_hit}
                continue
            if profile == "MVP_legacy":
                original = decisions
            checks = []
            for batch in (8, 16, 32):
                actual = encode_rows(runtime, queries, self.dataset, batch,
                                     lambda n, t: print(f"{profile} batch{batch} {n}/{t}", flush=True))
                error = float(np.max(np.abs(actual - base)))
                same = [ids(gallery.decide(v)) for v in actual] == decisions
                checks.append({"batch": batch, "max_abs_error": error, "same_decisions": same, "passed": error <= 2e-4 and same})
            reverse = encode_rows(runtime, queries[::-1], self.dataset, 16)[::-1]
            subset = encode_rows(runtime, queries[::2], self.dataset, 16)
            reordered = [ids(gallery.decide(v)) for v in reverse] == decisions
            removed = [ids(gallery.decide(v)) for v in subset] == decisions[::2]
            reports[profile] = {"batches": checks, "reordered": reordered, "removed_neighbors": removed,
                                "queries": len(queries), "passed": all(x["passed"] for x in checks) and reordered and removed}
        return {"passed": all(x["passed"] for x in reports.values()), "profiles": reports}

    def benchmarks(self):
        reports = {}
        for profile in PRIMARY:
            path = self.output / f"benchmark_{profile}.json"
            self.command(f"benchmark_{profile}", [sys.executable, "-m", "backend.benchmark",
                         "--profile", profile, "--provider", self.provider, "--dataset", self.dataset, "--output", path])
            report = json.loads(path.read_text())
            timing = self.output / f"export_{profile}.command.json"
            if timing.exists():
                full = json.loads(timing.read_text())["process_wall_seconds"]
                report["full_contest_runtime_seconds"] = full
                report["full_contest_runtime_scope"] = "fresh subprocess wall time including interpreter/imports/load/export/exit"
                report["within_runtime_guideline"] = full <= report["organizer_runtime_guideline_seconds"]
            write_json(path, report)
            reports[profile] = {"median_ms": report["median_ms"], "full_extract_seconds": report["full_extract_seconds"],
                                "full_contest_runtime_seconds": report.get("full_contest_runtime_seconds"),
                                "within_runtime_guideline": report.get("within_runtime_guideline"),
                                "official_gpu_verified": False}
        return reports

    def postgres(self):
        name = "reid-verification-db-" + uuid4().hex[:10]
        created = False
        try:
            self.command("database_create", ["docker", "run", "-d", "--rm", "--name", name,
                "-p", "127.0.0.1::5432", "-e", "POSTGRES_DB=reid_integration_test",
                "-e", "POSTGRES_USER=reid_test", "-e", "POSTGRES_PASSWORD=reid_test_local",
                "pgvector/pgvector:0.8.6-pg16-bookworm@sha256:ccc6e83d6e35e931dc7c5def2022729d5a6c370318d099181995567ff1fb4d6b"])
            created = True
            port = subprocess.check_output(["docker", "port", name, "5432/tcp"], text=True).strip().split(":")[-1]
            url = f"postgresql://reid_test:reid_test_local@127.0.0.1:{port}/reid_integration_test"
            while subprocess.run(["docker", "exec", name, "pg_isready", "-U", "reid_test", "-d", "reid_integration_test"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
                state = subprocess.check_output(["docker", "inspect", "-f", "{{.State.Running}}", name], text=True).strip()
                if state != "true":
                    raise RuntimeError("Disposable database exited")
                time.sleep(1)
            self.command("database_migrate", [sys.executable, "-m", "alembic", "upgrade", "head"], env={"DATABASE_URL": url})
            return self.command("database_tests", [sys.executable, "-m", "pytest", "-q",
                                "tests/test_release_postgres.py", "tests/test_postgres_integration.py",
                                "tests/test_postgres_inference_parity.py"],
                                env={"REID_TEST_DATABASE_URL": url, "DATABASE_URL": url})
        finally:
            if created:
                subprocess.run(["docker", "stop", name], check=True, stdout=subprocess.DEVNULL)

    def ui(self):
        self.command("ui_build", ["npm", "run", "build"], cwd=ROOT / "web-ui")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
        log = (self.output / "ui_server.log").open("w")
        server = subprocess.Popen([sys.executable, "-m", "tools.serve_verification", "--dataset", str(self.dataset),
            "--cache", str(self.output / "ui_cache"), "--port", str(port)], cwd=ROOT, stdout=log, stderr=log)
        try:
            while True:
                if server.poll() is not None:
                    raise RuntimeError("Test UI server exited; see ui_server.log")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        break
                except OSError:
                    time.sleep(1)
            return self.command("ui_tests", ["npm", "run", "test:e2e"], cwd=ROOT / "web-ui",
                                env={"E2E_BASE_URL": f"http://127.0.0.1:{port}"})
        finally:
            server.terminate(); server.wait(); log.close()

    def docker(self):
        tag = "vehicle-reid:verification-" + self.output.name.lower()
        self.command("docker_build", ["docker", "build", "-t", tag, "."])
        comparisons = {}
        for profile in PRIMARY:
            out = self.output / f"docker_{profile}"
            out.mkdir()
            self.command(f"docker_offline_{profile}", ["docker", "run", "--rm", "--network", "none",
                "--user", "0:0", "--entrypoint", "python", "-v", f"{self.dataset}:/data:ro",
                "-v", f"{out}:/out", tag, "-m", "backend.infer", "--dataset", "/data", "--output", "/out",
                "--profile", profile])
            comparisons[profile] = compare_exports(self.output / f"export_{profile}", out)
        architecture = subprocess.check_output(["docker", "image", "inspect", tag, "--format", "{{.Architecture}}"], text=True).strip()
        return {"passed": all(x["passed"] for x in comparisons.values()), "image": tag,
                "architecture": architecture, "network": "none", "comparisons": comparisons,
                "native_linux_amd64_verified": False}

    def finish(self):
        required = ("preflight", "components", "exports_reference", "streaming_rollback",
                    "postgres", "ui", "docker_offline", "fresh_benchmarks")
        passed = all(self.results.get(x, {}).get("status") == "passed" for x in required)
        changed = self.source_hashes != self.code_hashes()
        passed = passed and not changed
        decision = {"active_profile": DEFAULT_PROFILE, "rollback_profile": "MVP_dual_role_v24", "legacy_profile": "MVP_legacy", "candidate_profile": "RC_R1_equal3_v18",
                    "reserve_profile": "RC_R1_single_v18", "promoted": True,
                    "local_acceptance_passed": passed, "scope": "smoke" if self.smoke else "full",
                    "official_submission_ready": False, "open_checks": OPEN_CHECKS,
                    "implementation_changed_during_run": changed,
                    "stages": self.results, "elapsed_seconds": time.perf_counter() - self.started}
        write_json(self.output / "release_decision.json", decision)
        lines = ["# Release verification", "", f"Run: {self.output.name}",
                 f"Coverage: {'SMOKE, not full dataset' if self.smoke else 'FULL supplied dataset'}",
                 "", "User-approved v25 is active; v24 and MVP_legacy are preserved. No thresholds or labels were changed.", "",
                 "| Stage | Status | Seconds |", "|---|---|---:|"]
        lines.extend(f"| {name} | {value['status']} | {value['seconds']:.1f} |" for name, value in self.results.items())
        lines += ["", "## Open acceptance items", "", *["- " + x for x in OPEN_CHECKS],
                  "", "JSON reports contain concrete decision parity, checksums, provider and measurement details.",
                  "A successful Mac or emulated-container test is not a GPU performance certificate."]
        (self.output / "REPORT.md").write_text("\n".join(lines) + "\n")
        print(json.dumps({k: decision[k] for k in ("active_profile", "local_acceptance_passed", "scope", "elapsed_seconds")}, indent=2))
        return decision

    def run_all(self):
        for name, method in (("preflight", self.preflight), ("components", self.components),
                             ("exports_reference", self.exports_and_reference),
                             ("streaming_rollback", self.streaming_and_rollback), ("postgres", self.postgres),
                             ("ui", self.ui), ("docker_offline", self.docker), ("fresh_benchmarks", self.benchmarks)):
            self.stage(name, method)
        return self.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--provider", choices=("CPUExecutionProvider", "CUDAExecutionProvider"), default="CPUExecutionProvider")
    args = parser.parse_args()
    result = Verification(args.dataset, args.output, smoke=args.smoke, provider=args.provider).run_all()
    if not result["local_acceptance_passed"]:
        raise SystemExit(1)
