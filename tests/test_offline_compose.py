"""Compose CLI checks do not start Docker or require a running daemon."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_postgres_build_target_is_pinned_and_separate_from_runtime():
    source = (ROOT / "Dockerfile").read_text()
    assert re.search(r"^FROM pgvector/pgvector:0\.8\.6-pg16-bookworm@sha256:[0-9a-f]{64} AS postgres$",
                     source, re.MULTILINE)
    assert source.index(" AS postgres") < source.index(" AS frontend-builder") < source.index(" AS runtime")


@pytest.mark.skipif(shutil.which("docker") is None, reason="Compose CLI is checked on the Docker host")
@pytest.mark.parametrize("gpu", [False, True])
def test_offline_compose_builds_all_images_and_disables_pulls(gpu):
    command = ["docker", "compose", "--env-file", ".env.example", "-f", "docker-compose.yml"]
    if gpu:
        command += ["-f", "docker-compose.gpu.yml"]
    command += ["--profile", "inference", "config", "--format", "json"]
    env = dict(os.environ, REID_PROFILE="MVP_fusion_v25", REID_PROVIDER="CPUExecutionProvider")
    config = json.loads(subprocess.check_output(command, cwd=ROOT, env=env, text=True))
    services = config["services"]
    assert services["postgres"]["build"]["target"] == "postgres"
    assert services["postgres"]["image"] == "vehicle-reid-postgres:pg16-pgvector0.8.6"
    for service in services.values():
        assert service["build"]["dockerfile"] == "Dockerfile"
        assert service["pull_policy"] == "never"
    assert services["dataset-init"]["network_mode"] == "none"
    assert services["inference"]["network_mode"] == "none"
    assert not services["inference"].get("depends_on")
    for name in ("postgres", "vehicle-reid"):
        assert set(services[name]["networks"]) == {"default"}
    if gpu:
        assert all(service["platform"] == "linux/amd64" for service in services.values())
        assert services["vehicle-reid"]["environment"]["REID_PROVIDER"] == "CUDAExecutionProvider"
    else:
        assert services["vehicle-reid"]["environment"]["REID_PROVIDER"] == "CPUExecutionProvider"
