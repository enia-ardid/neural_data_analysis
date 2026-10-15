"""Load notebook definitions for examples and tests without running analysis cells."""

import ast
import json
from pathlib import Path
import sys
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]


def load_definitions(notebook):
    """Execute top-level imports, functions and classes from a trusted local notebook.

    File reads, analysis calls and other top-level statements are skipped. This
    keeps the notebooks as the source of the functions exercised by the tests.
    """
    path = ROOT / notebook
    module = ModuleType(f"notebook_{path.stem}")
    sys.modules[module.__name__] = module
    cells = json.loads(path.read_text())["cells"]
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        tree = ast.parse(source)
        tree.body = [node for node in tree.body if isinstance(
            node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)
        )]
        exec(compile(tree, f"{path.name}:cell-{index}", "exec"), module.__dict__)
    return module
