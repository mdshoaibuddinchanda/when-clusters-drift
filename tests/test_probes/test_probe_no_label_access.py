"""Static AST audit confirming strict label-free boundary in alignment and probe modules."""

import ast
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_no_labels_parquet_in_probes_or_alignment():
    """Verify that neither src/clusterdrift/alignment nor src/clusterdrift/probes references labels.parquet."""
    target_dirs = [
        PROJECT_ROOT / "src" / "clusterdrift" / "alignment",
        PROJECT_ROOT / "src" / "clusterdrift" / "probes",
    ]

    forbidden_strings = ["labels.parquet", "labels_path"]

    for d in target_dirs:
        for py_file in d.glob("*.py"):
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content, filename=str(py_file))
            for node in ast.walk(tree):
                # Check string literals
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    for forbidden in forbidden_strings:
                        assert forbidden not in node.value, (
                            f"Forbidden label reference '{forbidden}' found in {py_file}"
                        )
                # Check function argument names
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    arg_names = [a.arg for a in node.args.args]
                    for forbidden_arg in ["y", "y_target", "y_source", "y_train", "labels"]:
                        assert forbidden_arg not in arg_names, (
                            f"Forbidden label argument '{forbidden_arg}' in {node.name}() at {py_file}"
                        )
