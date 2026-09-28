import ast
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))


def test_tutorial_notebooks_exist():
    assert [path.name[:2] for path in NOTEBOOKS] == ["01", "02", "03", "04"]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_code_parses_and_has_no_stored_errors(path):
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        source = "\n".join(line for line in cell.source.splitlines()
                           if not line.lstrip().startswith(("%", "!")))
        ast.parse(source)
        assert not any(output.get("output_type") == "error" for output in cell.get("outputs", []))
