import json

import nbformat

from training import quality_experiment as experiment


def test_quality_notebook_is_unexecuted_compilable_run_all():
    path = experiment.VARIANT / "run_quality_experiments.ipynb"
    raw = path.read_bytes()
    notebook = nbformat.reads(raw.decode(), as_version=4)
    nbformat.validate(notebook)
    cells = [c for c in notebook.cells if c.cell_type == "code"]
    for i, cell in enumerate(cells):
        assert cell.execution_count is None and cell.outputs == []
        compile(cell.source, f"quality_cell_{i}", "exec")
    code = "\n".join(c.source for c in cells)
    for expected in ("RUN_HEAD = True", "RUN_CLOCK = True", "AUTO_CONFIRM = True", "experiment.run(context)"):
        assert expected in code
    assert "allow_outer=True" not in code
    assert path.read_bytes() == raw


def test_notebook_explains_quality_not_hardware_and_no_claimed_win():
    notebook = json.loads((experiment.VARIANT / "run_quality_experiments.ipynb").read_text())
    text = "\n".join("".join(c["source"]) for c in notebook["cells"])
    assert "не проверка Docker" in text
    assert "не EMA" in text
    assert "Общего ограничения времени нет" in text
    assert "не означает автоматическую победу" in text
    readme = (experiment.VARIANT / "README.md").read_text()
    assert "NiVe остаётся локальным" in readme
    assert "не OOF" in readme
