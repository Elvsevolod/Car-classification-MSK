import os
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "docker" / "dataset-init.sh"
MARKER = ".vehicle-reid-source-fingerprint"


def make_source(path):
    (path / "images").mkdir(parents=True)
    (path / "test_query.csv").write_text("image_id,x,y,w,h\nquery,0,0,1,1\n")
    (path / "test_gallery.csv").write_text("image_id,x,y,w,h\ngallery,0,0,1,1\n")
    (path / "images" / "query.jpg").write_bytes(b"query-image")
    (path / "images" / "gallery.jpg").write_bytes(b"old-gallery")


def prepare(source, target):
    environment = dict(os.environ, SOURCE_DATASET_DIR=str(source), TARGET_DATASET_DIR=str(target))
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment["PATH"]
    return subprocess.run(["sh", str(SCRIPT)], env=environment, capture_output=True, text=True, timeout=30)


def test_test_only_dataset_cache_invalidates_equal_size_equal_mtime_image_changes(tmp_path):
    source, target = tmp_path / "source", tmp_path / "cache" / "dataset"
    make_source(source)
    first = prepare(source, target)
    assert first.returncode == 0, first.stderr
    assert not (target / "train.csv").exists()
    old_marker = (target / MARKER).read_text()
    second = prepare(source, target)
    assert second.returncode == 0 and "is current" in second.stdout

    image = source / "images" / "gallery.jpg"
    previous = image.stat()
    image.write_bytes(b"new-gallery")
    os.utime(image, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    assert image.stat().st_size == previous.st_size
    assert image.stat().st_mtime_ns == previous.st_mtime_ns
    changed = prepare(source, target)
    assert changed.returncode == 0, changed.stderr
    assert (target / "images" / "gallery.jpg").read_bytes() == b"new-gallery"
    assert (target / MARKER).read_text() != old_marker
    assert sorted(path.name for path in target.parent.iterdir()) == ["dataset"]


def test_failed_copy_preserves_previous_cache_and_cleans_staging(tmp_path):
    source, target = tmp_path / "source", tmp_path / "cache" / "dataset"
    make_source(source)
    assert prepare(source, target).returncode == 0
    old_marker = (target / MARKER).read_text()
    (source / "images" / "gallery.jpg").write_bytes(b"new-gallery")
    # A dangling extra source symlink makes copytree fail after fingerprinting.
    (source / "missing-extra").symlink_to(source / "does-not-exist")
    failure = prepare(source, target)
    assert failure.returncode != 0
    assert (target / "images" / "gallery.jpg").read_bytes() == b"old-gallery"
    assert (target / MARKER).read_text() == old_marker
    assert sorted(path.name for path in target.parent.iterdir()) == ["dataset"]


def test_optional_train_csv_participates_in_cache_fingerprint(tmp_path):
    source, target = tmp_path / "source", tmp_path / "cache" / "dataset"
    make_source(source)
    assert prepare(source, target).returncode == 0
    old_marker = (target / MARKER).read_text()
    (source / "train.csv").write_text("image_id,x,y,w,h\ntrain,0,0,1,1\n")
    changed = prepare(source, target)
    assert changed.returncode == 0, changed.stderr
    assert (target / "train.csv").read_text() == (source / "train.csv").read_text()
    assert (target / MARKER).read_text() != old_marker


def test_dataset_cache_cannot_replace_its_source(tmp_path):
    source = tmp_path / "source"
    make_source(source)
    rejected = prepare(source, source)
    assert rejected.returncode != 0
    assert "separate from the source" in rejected.stderr
    assert (source / "images" / "gallery.jpg").read_bytes() == b"old-gallery"
