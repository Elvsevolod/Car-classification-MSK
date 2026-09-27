"""Package exact source and selected protocol/config metadata; no data or run history."""
import argparse
import json
import shutil
from pathlib import Path

from backend.core import ROOT, sha256
from backend.runtime import write_json


def package(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    files = []
    for folder in ("training", "backend"):
        files += [p for p in (source / folder).rglob("*")
                  if p.is_file() and (p.suffix == ".py" or p.name.startswith("LICENSE"))]
    files += list(source.glob("requirements*.txt")) + list(source.glob("requirements*.in"))
    files += [source / "evaluate.py", source / "models/LICENSE.osnet", source / "models/README.md"]
    roots = ("variant_02_hpo_bnneck_supcon", "variant_16_review_protocol",
             "variant_17_final_seed_confirmation", "variant_18_retrieval_policy")
    manifest = {}
    for name in roots:
        folder = source / "OSNet-AIN-x1.0" / name
        files += list(folder.glob("*.md"))
        for path in folder.glob("*.ipynb"):
            dest = output / path.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            notebook = json.loads(path.read_text())
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    cell["outputs"], cell["execution_count"] = [], None
            write_json(dest, notebook)
            manifest[str(path.relative_to(source))] = {"source_sha256": sha256(path), "code_only_sha256": sha256(dest)}
    # Selected immutable recipes/splits; no logs, result grids or checkpoints.
    for variant, run in (("variant_16_review_protocol", "review_v1"),
                         ("variant_17_final_seed_confirmation", "final_seeds_v1"),
                         ("variant_18_retrieval_policy", "policy_v1")):
        base = source / "OSNet-AIN-x1.0" / variant / "runs" / run
        for name in ("manifest.json", "selection.json", "final_selection.json"):
            if (base / name).is_file():
                files.append(base / name)
    for path in sorted(set(files)):
        if not path.is_file():
            continue
        dest = output / path.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        manifest[str(path.relative_to(source))] = {"sha256": sha256(path)}
    write_json(output / "SOURCE_SHA256.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "reproduction/source")
    args = parser.parse_args()
    package(args.research, args.output)
