# Owner: backbone (ALL)
"""Architectural checks (AST-based; not a security sandbox).

The three real Student module directories must not import the evaluation
oracle: perception/planner/executor implementations are forbidden from
reading ground truth.  Mocks (core/mocks.py) are exempt by design.
"""

from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
STUDENT_DIRS = ("perception", "planner", "executor")
FORBIDDEN_MODULES = ("core.oracle",)
FORBIDDEN_NAMES = ("EvalOracle",)


def _violations_in(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(alias.name == m or alias.name.startswith(m + ".") for m in FORBIDDEN_MODULES):
                    out.append(f"{path}:{node.lineno} imports {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if any(mod == m or mod.startswith(m + ".") for m in FORBIDDEN_MODULES):
                out.append(f"{path}:{node.lineno} imports from {mod}")
            if mod == "core" or mod.startswith("core."):
                for alias in node.names:
                    if alias.name in FORBIDDEN_NAMES:
                        out.append(f"{path}:{node.lineno} imports {alias.name}")
    return out


def test_student_modules_do_not_import_oracle():
    violations: list[str] = []
    for dirname in STUDENT_DIRS:
        for path in (ROOT / dirname).rglob("*.py"):
            violations.extend(_violations_in(path))
    assert violations == [], "\n".join(violations)


def test_check_actually_detects_violations(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("from core.oracle import EvalOracle\n")
    assert _violations_in(bad)
    bad.write_text("import core.oracle\n")
    assert _violations_in(bad)
    bad.write_text("from core import EvalOracle\n")
    assert _violations_in(bad)
    ok = tmp_path / "ok.py"
    ok.write_text("from core.types import Plan\nimport core.skills\n")
    assert not _violations_in(ok)
