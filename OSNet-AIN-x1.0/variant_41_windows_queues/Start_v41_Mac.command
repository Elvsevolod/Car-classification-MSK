#!/bin/bash
set -euo pipefail
RESEARCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$RESEARCH_DIR/setup_mac.sh"
PROJECT_ROOT="$(cd "$RESEARCH_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
export ORT_DISABLE_TELEMETRY=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 XFORMERS_DISABLED=1
# Inhibit idle sleep only while this Jupyter process runs; no system settings are changed.
exec caffeinate -i "$PROJECT_ROOT/.venv-v41-m4/bin/python" -m jupyterlab "$RESEARCH_DIR/train_mac_m4_queues.ipynb"
