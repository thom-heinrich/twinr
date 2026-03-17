"""Regression audit for silent broad exception handlers in Twinr runtime code."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "twinr"


def _is_broad_exception_handler(handler: ast.ExceptHandler) -> bool:
    """Return whether one ``except`` handler catches broadly enough to hide bugs."""

    exception_type = handler.type
    if exception_type is None:
        return True
    if isinstance(exception_type, ast.Name):
        return exception_type.id in {"Exception", "BaseException"}
    if isinstance(exception_type, ast.Tuple):
        return any(isinstance(item, ast.Name) and item.id in {"Exception", "BaseException"} for item in exception_type.elts)
    return False


def test_no_silent_broad_exception_handlers() -> None:
    """Broad catches must not swallow failures with naked ``pass`` or ``continue``."""

    silent_handlers: list[str] = []
    for path in SRC_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            for handler in node.handlers:
                if not _is_broad_exception_handler(handler):
                    continue
                if len(handler.body) != 1:
                    continue
                statement = handler.body[0]
                if isinstance(statement, ast.Pass):
                    silent_handlers.append(f"{path.relative_to(REPO_ROOT)}:{handler.lineno}: pass")
                elif isinstance(statement, ast.Continue):
                    silent_handlers.append(f"{path.relative_to(REPO_ROOT)}:{handler.lineno}: continue")

    assert not silent_handlers, "Silent broad exception handlers remain:\n" + "\n".join(sorted(silent_handlers))
