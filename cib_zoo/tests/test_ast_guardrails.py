"""Static AST guardrail tests enforcing zero database backdoors in bot behaviors."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


def get_behavior_python_files() -> list[Path]:
    """Retrieve all Python source files in primitives/ and patterns/."""
    oasis_root = Path(__file__).resolve().parents[2]
    cib_zoo_dir = oasis_root / "cib_zoo"
    
    files = []
    for sub in ("primitives", "patterns"):
        target_dir = cib_zoo_dir / sub
        if target_dir.exists():
            files.extend(target_dir.glob("*.py"))
    return sorted(files)


@pytest.mark.parametrize("file_path", get_behavior_python_files(), ids=lambda p: p.name)
def test_no_sqlite3_import_in_behavior_files(file_path: Path) -> None:
    """Verify that bot behavior files strictly do not import sqlite3."""
    code = file_path.read_text(encoding="utf-8")
    tree = ast.parse(code, filename=str(file_path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "sqlite3", (
                    f"Forbidden 'import sqlite3' found in {file_path.name}:{node.lineno}. "
                    "CIB behaviors must interact strictly via client actions."
                )
        elif isinstance(node, ast.ImportFrom):
            assert node.module != "sqlite3" and not (node.module or "").startswith("sqlite3."), (
                f"Forbidden 'from sqlite3 import ...' found in {file_path.name}:{node.lineno}. "
                "CIB behaviors must interact strictly via client actions."
            )


@pytest.mark.parametrize("file_path", get_behavior_python_files(), ids=lambda p: p.name)
def test_no_raw_sql_queries_in_behavior_files(file_path: Path) -> None:
    """Verify that no raw SQL statement string literals exist in primitives or patterns."""
    code = file_path.read_text(encoding="utf-8")
    tree = ast.parse(code, filename=str(file_path))

    forbidden_sql_fragments = [
        "SELECT ", "INSERT INTO ", "DELETE FROM ", "UPDATE ", "DROP TABLE", "CREATE TABLE"
    ]

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            val_upper = node.value.strip().upper()
            for fragment in forbidden_sql_fragments:
                assert fragment not in val_upper, (
                    f"Forbidden raw SQL query literal '{fragment}' detected in {file_path.name}:{node.lineno}."
                )
