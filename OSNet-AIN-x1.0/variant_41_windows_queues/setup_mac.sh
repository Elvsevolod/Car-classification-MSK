#!/bin/bash
set -euo pipefail
RESEARCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$RESEARCH_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "Нужен Mac Apple Silicon и Terminal без Rosetta." >&2
  exit 1
fi
RESEARCH_PYTHON="${REID_PYTHON:-}"
if [[ -z "$RESEARCH_PYTHON" ]]; then
  RESEARCH_PYTHON="$(command -v python3.11 || true)"
fi
if [[ -z "$RESEARCH_PYTHON" && -x /opt/homebrew/bin/python3.11 ]]; then
  RESEARCH_PYTHON=/opt/homebrew/bin/python3.11
fi
if [[ -z "$RESEARCH_PYTHON" && -x /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 ]]; then
  RESEARCH_PYTHON=/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11
fi
if [[ -z "$RESEARCH_PYTHON" ]]; then
  echo "Установи Python 3.11 arm64 на ЭТОТ Mac (например, brew install python@3.11), затем повтори запуск." >&2
  exit 1
fi
"$RESEARCH_PYTHON" -c 'import sys,platform; assert sys.version_info[:2]==(3,11) and platform.machine()=="arm64", "Нужен Python 3.11 arm64"'
RESEARCH_ENV="$PROJECT_ROOT/.venv-v41-m4"
if [[ -e "$RESEARCH_ENV" ]]; then
  if [[ ! -f "$RESEARCH_ENV/.v41-location" ]] || [[ "$(< "$RESEARCH_ENV/.v41-location")" != "$RESEARCH_ENV" ]]; then
    echo "Окружение .venv-v41-m4 перенесено или создано другим способом. Переименуй его для сохранности и запусти снова; старую .venv не используй." >&2
    exit 1
  fi
else
  "$RESEARCH_PYTHON" -m venv "$RESEARCH_ENV"
  printf '%s\n' "$RESEARCH_ENV" > "$RESEARCH_ENV/.v41-location"
fi
"$RESEARCH_ENV/bin/python" -m pip install -r "$RESEARCH_DIR/requirements-mac.txt"
export ORT_DISABLE_TELEMETRY=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 XFORMERS_DISABLED=1
"$RESEARCH_ENV/bin/python" -m training.research_mac
if [[ ! -f "$PROJECT_ROOT/research_transfer/v41_inputs/inputs.json" ]]; then
  echo "Не найдены research_transfer/v41_inputs. Заверши перенос всей папки; заново скачивать NiVe не нужно." >&2
  exit 1
fi
"$RESEARCH_ENV/bin/python" -m ipykernel install --prefix "$RESEARCH_ENV" --name car-reid-m4 --display-name "Car ReID Mac M4"
echo "Готово: kernel Car ReID Mac M4. Старые .venv и .venv-rtx4060 не использовались."
