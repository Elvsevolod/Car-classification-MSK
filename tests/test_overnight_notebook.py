"""Notebook stays an unexecuted, explicit Run All entry point."""
import json

import nbformat

from training import overnight_system as experiment


def test_notebook_valid_compilable_and_unexecuted():
    path = experiment.VARIANT / "run_overnight_system.ipynb"
    raw = path.read_bytes()
    notebook = nbformat.reads(raw.decode(), as_version=4)
    nbformat.validate(notebook)
    code = [c for c in notebook.cells if c.cell_type == "code"]
    for index, cell in enumerate(code):
        assert cell.execution_count is None and cell.outputs == []
        compile(cell.source, f"overnight_cell_{index}", "exec")
    combined = "\n".join(c.source for c in code)
    assert "WALL_HOURS = None" in combined
    assert "RUN_TRAINING = True" in combined
    assert "AUTO_CONFIRM = True" in combined
    assert "experiment.run(context" in combined
    assert "allow_outer=True" not in combined
    assert path.read_bytes() == raw


def test_notebook_and_readme_do_not_claim_plate_or_gpu_proof():
    notebook = json.loads((experiment.VARIANT / "run_overnight_system.ipynb").read_text())
    text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "не plate-only" in text
    assert "не подтверждает баллы GPU" in text
    readme = (experiment.VARIANT / "README.md").read_text()
    assert "NiVe остаётся локальным" in readme
    assert "не проверены" in readme
