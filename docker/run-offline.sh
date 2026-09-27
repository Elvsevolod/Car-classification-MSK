#!/bin/sh
set -eu
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: sh run-offline.sh ABSOLUTE_DATASET_DIR NEW_OUTPUT_DIR [PROFILE]" >&2
  exit 2
fi
case "$1" in /*) ;; *) echo "Dataset path must be absolute" >&2; exit 2 ;; esac
case "$2" in /*) ;; *) echo "Output path must be absolute" >&2; exit 2 ;; esac
test -d "$1/images"
test -f "$1/test_query.csv"
test -f "$1/test_gallery.csv"
if [ -e "$2" ]; then
  echo "Use a NEW output directory. Existing outputs are never overwritten." >&2
  exit 2
fi
mkdir -p "$2"
PACKAGE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
docker load -i "$PACKAGE_DIR/vehicle-reid.tar.gz"
docker run --rm --network none --user 0:0 --entrypoint python \
  -v "$1:/data:ro" -v "$2:/out" vehicle-reid:release-integration \
  -m backend.infer --dataset /data --output /out --profile "${3:-MVP_fusion_v25}" --provider CPUExecutionProvider
