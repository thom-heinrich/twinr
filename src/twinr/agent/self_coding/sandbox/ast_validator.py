"""Validate generated skill source before it can run inside the sandbox."""

from __future__ import annotations

import ast

# AUDIT-FIX(#1): Block reflective, file-reading, and control builtins that enable sandbox escape or host-process termination.
_BANNED_CALL_NAMES = frozenset(
    {
        "__import__",
        "BaseException",
        "BaseExceptionGroup",
        "GeneratorExit",
        "KeyboardInterrupt",
        "SystemExit",
        "breakpoint",
        "compile",
        "copyright",
        "credits",
        "delattr",
        "dir",
        "eval",
        "exec",
        "exit",
        "globals",
        "help",
        "input",
        "license",
        "locals",
        "open",
        "quit",
        "setattr",
        "type",
        "vars",
    }
)
# AUDIT-FIX(#2): Reject banned builtins even when referenced outside direct call sites, so aliasing cannot bypass policy.
_BANNED_NAME_IDS = _BANNED_CALL_NAMES
_BANNED_NODE_TYPES = (
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.Await,
    ast.ClassDef,
    ast.Global,
    ast.Import,
    ast.Nonlocal,
    ast.Yield,
    ast.YieldFrom,
)
# AUDIT-FIX(#1): Block non-dunder runtime-introspection attributes that expose frames, globals, builtins, and tracebacks.
_BANNED_ATTRIBUTE_NAMES = frozenset(
    {
        "ag_await",
        "ag_code",
        "ag_frame",
        "cr_await",
        "cr_code",
        "cr_frame",
        "f_back",
        "f_builtins",
        "f_code",
        "f_globals",
        "f_lineno",
        "f_locals",
        "f_trace",
        "f_trace_lines",
        "f_trace_opcodes",
        "gi_code",
        "gi_frame",
        "tb_frame",
        "tb_next",
    }
)
# AUDIT-FIX(#3): Bound validation work so pathological input cannot pin the single-process RPi runtime.
_MAX_SOURCE_BYTES = 128 * 1024
_MAX_AST_NODES = 10_000
_MAX_AST_DEPTH = 100


class SelfCodingSandboxValidationError(RuntimeError):
    """Raised when generated skill code exceeds the first sandbox policy."""


def validate_skill_source(source_text: str, *, filename: str = "<skill>") -> ast.Module:
    """Return a parsed AST when one skill file fits the trusted sandbox subset."""

    if not isinstance(source_text, str):
        raise TypeError("skill source must be a string")
    # AUDIT-FIX(#3): Reject oversized source before parsing to avoid unnecessary CPU and memory pressure.
    _enforce_source_size_limit(source_text, filename=filename)
    try:
        tree = ast.parse(source_text, filename=filename, mode="exec")
    except SyntaxError as exc:
        # AUDIT-FIX(#5): Preserve line and column information so production diagnostics are actionable.
        raise SelfCodingSandboxValidationError(
            f"Sandbox rejected invalid Python in {filename}: {_format_syntax_error(exc)}."
        ) from exc
    except (RecursionError, ValueError) as exc:
        # AUDIT-FIX(#4): Normalize parser edge-case failures into the sandbox-specific exception type.
        detail = str(exc).strip() or type(exc).__name__
        raise SelfCodingSandboxValidationError(f"Sandbox rejected invalid Python in {filename}: {detail}.") from exc
    # AUDIT-FIX(#3): Cap AST complexity before recursive visiting to avoid validator recursion blowups.
    _enforce_ast_limits(tree, filename=filename)
    _SkillSourceValidator(filename=filename).visit(tree)
    return tree


def _enforce_source_size_limit(source_text: str, *, filename: str) -> None:
    """Reject source that is too large for safe validation on constrained hardware."""

    source_size = len(source_text.encode("utf-8", "surrogatepass"))
    if source_size > _MAX_SOURCE_BYTES:
        raise SelfCodingSandboxValidationError(
            f"{filename}: sandbox blocks files larger than {_MAX_SOURCE_BYTES} bytes."
        )


def _enforce_ast_limits(tree: ast.AST, *, filename: str) -> None:
    """Reject ASTs that are too large or too deep for safe recursive validation."""

    node_count = 0
    stack: list[tuple[ast.AST, int]] = [(tree, 1)]
    while stack:
        node, depth = stack.pop()
        node_count += 1
        if node_count > _MAX_AST_NODES:
            raise SelfCodingSandboxValidationError(
                f"{filename}: sandbox blocks files with more than {_MAX_AST_NODES} AST nodes."
            )
        if depth > _MAX_AST_DEPTH:
            line = getattr(node, "lineno", None)
            location = "" if line is None else f" line {line}"
            raise SelfCodingSandboxValidationError(
                f"{filename}{location}: sandbox blocks AST nesting deeper than {_MAX_AST_DEPTH} levels."
            )
        for child in ast.iter_child_nodes(node):
            stack.append((child, depth + 1))


def _format_syntax_error(exc: SyntaxError) -> str:
    """Render syntax errors with stable, concise location context."""

    message = exc.msg or "invalid syntax"
    if exc.lineno is None:
        return message
    if exc.offset is None:
        return f"{message} (line {exc.lineno})"
    return f"{message} (line {exc.lineno}, column {exc.offset})"


class _SkillSourceValidator(ast.NodeVisitor):
    """Reject imports and Python object-escape patterns while keeping normal logic available."""

    def __init__(self, *, filename: str) -> None:
        self.filename = filename

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, _BANNED_NODE_TYPES):
            self._fail(node, f"sandbox blocks {type(node).__name__}.")
        super().generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # AUDIT-FIX(#2): Function definitions must not shadow blocked builtins or use dunder protocol names.
        self._validate_identifier(node, node.name, "function name")
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        # AUDIT-FIX(#2): Parameters must not smuggle blocked builtins into local scope under reserved names.
        self._validate_identifier(node, node.arg, "parameter name")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if (
            node.module == "__future__"
            and node.level == 0
            and tuple(alias.name for alias in node.names) == ("annotations",)
        ):
            return
        self._fail(node, "Sandbox only allows 'from __future__ import annotations'.")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if "__" in str(node.attr):
            self._fail(node, "Sandbox blocks double-underscore attribute access.")
        if node.attr in _BANNED_ATTRIBUTE_NAMES:
            # AUDIT-FIX(#1): Frame/code/coroutine/traceback attributes expose execution internals and sandbox escape routes.
            self._fail(node, f"Sandbox blocks access to sensitive attribute '{node.attr}'.")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        self._validate_identifier(node, node.id, "name")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in _BANNED_CALL_NAMES:
            # AUDIT-FIX(#2): Keep the direct-call check as defense in depth for clearer error locality.
            self._fail(node, f"Sandbox blocks the builtin or control name '{node.func.id}'.")
        self.generic_visit(node)

    def _validate_identifier(self, node: ast.AST, identifier: str, kind: str) -> None:
        if identifier.startswith("__"):
            self._fail(node, f"Sandbox blocks dunder {kind}s.")
        if identifier in _BANNED_NAME_IDS:
            self._fail(node, f"Sandbox blocks the builtin or control name '{identifier}'.")

    def _fail(self, node: ast.AST, message: str) -> None:
        line = getattr(node, "lineno", None)
        location = "" if line is None else f" line {line}"
        raise SelfCodingSandboxValidationError(f"{self.filename}{location}: {message}")
