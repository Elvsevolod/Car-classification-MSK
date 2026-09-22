#!/bin/sh
set -eu

# Один раз переносит внешний dataset в Docker volume: runtime затем не зависит от прав bind mount на хосте.
SOURCE_DATASET_DIR=${SOURCE_DATASET_DIR:-/source}
TARGET_DATASET_DIR=${TARGET_DATASET_DIR:-/dataset-cache/dataset}
TARGET_PARENT=$(dirname "$TARGET_DATASET_DIR")
STAGING_DATASET_DIR="${TARGET_DATASET_DIR}.incomplete"

for required in images train.csv test_query.csv test_gallery.csv; do
    if [ ! -e "$SOURCE_DATASET_DIR/$required" ]; then
        echo "Dataset source is missing required path: $SOURCE_DATASET_DIR/$required" >&2
        exit 1
    fi
done

# CSV и список файлов достаточно отличают конкурсные наборы; изображения считаются неизменяемыми внутри одного dataset.
source_fingerprint=$(
    {
        sha256sum "$SOURCE_DATASET_DIR/train.csv" "$SOURCE_DATASET_DIR/test_query.csv" "$SOURCE_DATASET_DIR/test_gallery.csv"
        find "$SOURCE_DATASET_DIR/images" -type f -printf '%P %s\n' | LC_ALL=C sort
    } | sha256sum | awk '{print $1}'
)
marker="$TARGET_DATASET_DIR/.vehicle-reid-source-fingerprint"

if [ -f "$marker" ] && [ "$(cat "$marker")" = "$source_fingerprint" ]; then
    echo "Prepared dataset cache is current"
    exit 0
fi

echo "Preparing dataset cache inside Docker volume"
mkdir -p "$TARGET_PARENT"
rm -rf "$STAGING_DATASET_DIR"
mkdir -p "$STAGING_DATASET_DIR"
cp -a "$SOURCE_DATASET_DIR/." "$STAGING_DATASET_DIR/"

# Основной сервис запускается от appuser и читает только эту подготовленную копию.
chown -R 10001:10001 "$STAGING_DATASET_DIR"
chmod -R u+rwX,go+rX "$STAGING_DATASET_DIR"
printf '%s\n' "$source_fingerprint" > "$STAGING_DATASET_DIR/.vehicle-reid-source-fingerprint"

rm -rf "$TARGET_DATASET_DIR"
mv "$STAGING_DATASET_DIR" "$TARGET_DATASET_DIR"
echo "Prepared dataset cache is ready"
