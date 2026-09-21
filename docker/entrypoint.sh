#!/bin/sh
set -eu

: "${DATASET_DIR:?DATASET_DIR must point to the mounted dataset}"
: "${DATABASE_URL:?DATABASE_URL must be configured}"

if [ ! -d "$DATASET_DIR" ]; then
    echo "Dataset directory does not exist: $DATASET_DIR" >&2
    exit 1
fi

for required in images test_query.csv test_gallery.csv; do
    if [ ! -e "$DATASET_DIR/$required" ]; then
        echo "Dataset is missing required path: $DATASET_DIR/$required" >&2
        exit 1
    fi
done

echo "Applying database migrations"
alembic upgrade head
echo "Starting Vehicle ReID API"
exec "$@"
