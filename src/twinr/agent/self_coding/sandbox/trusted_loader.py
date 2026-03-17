"""Load validated self-coding skill modules with a restricted builtin surface."""

from __future__ import annotations

import __future__ as _future_module
import ast
from dataclasses import dataclass, field
from types import CodeType, FunctionType, MappingProxyType
from typing import Any, Final, Mapping

from twinr.agent.self_coding.sandbox.ast_validator import validate_skill_source

_MISSING = object()

# AUDIT-FIX(#1): Block known frame/code/introspection attribute names even if the
# upstream AST validator regresses or direct attribute syntax slips through.
_BLOCKED_ATTRIBUTE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "ag_code",
        "ag_frame",
        "cell_contents",
        "cr_await",
        "cr_code",
        "cr_frame",
        "f_back",
        "f_builtins",
        "f_code",
        "f_globals",
        "f_locals",
        "func_globals",
        "gi_code",
        "gi_frame",
        "mro",
        "tb_frame",
        "tb_lasti",
        "tb_lineno",
        "tb_next",
    }
)

# AUDIT-FIX(#1): Re-check compiled code for forbidden names before exec so this
# loader does not rely on a single validator layer.
_BLOCKED_CODE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "__annotations__",
        "__bases__",
        "__builtins__",
        "__cached__",
        "__class__",
        "__closure__",
        "__code__",
        "__defaults__",
        "__dict__",
        "__doc__",
        "__file__",
        "__func__",
        "__globals__",
        "__kwdefaults__",
        "__loader__",
        "__module__",
        "__mro__",
        "__name__",
        "__package__",
        "__self__",
        "__spec__",
        "__subclasses__",
        "__traceback__",
        *_BLOCKED_ATTRIBUTE_NAMES,
    }
)

# AUDIT-FIX(#2): Bound source and AST size to reduce denial-of-service risk on
# the single-process Raspberry Pi runtime during skill loading.
_MAX_SOURCE_BYTES: Final[int] = 64 * 1024
_MAX_AST_NODES: Final[int] = 4_000
_SANDBOX_MODULE_NAME: Final[str] = "twinr_self_coding_sandbox"


# AUDIT-FIX(#3): Return a minimal proxy instead of the real __future__ module
# so sandboxed code never receives an unnecessary module object surface.
@dataclass(frozen=True, slots=True)
class _SafeFutureModule:
    annotations: Any


_SAFE_FUTURE_MODULE = _SafeFutureModule(annotations=_future_module.annotations)


# AUDIT-FIX(#4): Do not coerce attacker-controlled objects through str(); helper
# functions must accept explicit strings only.
def _require_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    return value


# AUDIT-FIX(#9): Normalize and validate filename labels before using them in
# compile/error paths to avoid control-character confusion.
def _normalize_filename(filename: object) -> str:
    normalized = _require_str(filename, field_name="filename")
    if not normalized:
        return "<sandbox-skill>"
    if "\x00" in normalized:
        raise ValueError("filename must not contain NUL bytes")
    if "\r" in normalized or "\n" in normalized:
        raise ValueError("filename must not contain line breaks")
    return normalized


# AUDIT-FIX(#2): Reject oversized source early before validation/compile work.
# AUDIT-FIX(#9): Normalize source_text input type at the module boundary.
def _normalize_source_text(source_text: object) -> str:
    normalized = _require_str(source_text, field_name="source_text")
    if len(normalized.encode("utf-8")) > _MAX_SOURCE_BYTES:
        raise ValueError("sandbox skill source is too large")
    return normalized


def _normalize_fromlist(fromlist: object) -> tuple[str, ...]:
    if fromlist in (None, ()):
        return ()
    if not isinstance(fromlist, (tuple, list)):
        raise ImportError("sandbox only allows standard import fromlist values")
    normalized: list[str] = []
    for item in fromlist:
        if not isinstance(item, str):
            raise ImportError("sandbox only allows string fromlist entries")
        normalized.append(item)
    return tuple(normalized)


def safe_import(
    name: object,
    globals: object | None = None,
    locals: object | None = None,
    fromlist: object = (),
    level: int = 0,
):
    """Allow only the future-annotations import needed by generated skill files."""

    del globals, locals
    # AUDIT-FIX(#3): Permit only the exact __future__/annotations import shape.
    # AUDIT-FIX(#4): Require a real string import target to avoid __str__ abuse.
    module_name = _require_str(name, field_name="import name")
    normalized_fromlist = _normalize_fromlist(fromlist)
    if module_name != "__future__" or level != 0:
        raise ImportError("sandbox only allows 'from __future__ import annotations'")
    if normalized_fromlist not in ((), ("annotations",)):
        raise ImportError("sandbox only allows 'from __future__ import annotations'")
    return _SAFE_FUTURE_MODULE


def _validate_attribute_name(name: object) -> str:
    # AUDIT-FIX(#1): Block private and introspection attributes, not only dunders.
    # AUDIT-FIX(#4): Require a real string attribute name.
    attribute_name = _require_str(name, field_name="attribute name")
    if attribute_name.startswith("_") or attribute_name in _BLOCKED_ATTRIBUTE_NAMES:
        raise AttributeError("sandbox blocks private or introspection attribute access")
    return attribute_name


def safe_getattr(target: object, name: object, default: object = _MISSING) -> object:
    """Return one attribute while blocking private and introspection access."""

    attribute_name = _validate_attribute_name(name)
    if default is _MISSING:
        return getattr(target, attribute_name)
    return getattr(target, attribute_name, default)


def safe_hasattr(target: object, name: object) -> bool:
    """Return whether a permitted attribute exists on one object."""

    try:
        safe_getattr(target, name)
    except AttributeError:
        return False
    return True


def _is_docstring_statement(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _is_future_annotations_import(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.ImportFrom):
        return False
    if statement.module != "__future__" or statement.level != 0:
        return False
    return (
        len(statement.names) == 1
        and statement.names[0].name == "annotations"
        and statement.names[0].asname is None
    )


def _is_load_time_safe_expr(node: ast.AST, *, literal_names: frozenset[str]) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Name):
        return node.id in literal_names
    if isinstance(node, ast.Tuple | ast.List | ast.Set):
        return all(
            _is_load_time_safe_expr(child, literal_names=literal_names)
            for child in node.elts
        )
    if isinstance(node, ast.Dict):
        return all(
            (key is None or _is_load_time_safe_expr(key, literal_names=literal_names))
            and _is_load_time_safe_expr(value, literal_names=literal_names)
            for key, value in zip(node.keys, node.values)
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
        return _is_load_time_safe_expr(node.operand, literal_names=literal_names)
    return False


def _is_immutable_load_time_safe_expr(
    node: ast.AST,
    *,
    immutable_names: frozenset[str],
) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Name):
        return node.id in immutable_names
    if isinstance(node, ast.Tuple):
        return all(
            _is_immutable_load_time_safe_expr(child, immutable_names=immutable_names)
            for child in node.elts
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
        return _is_immutable_load_time_safe_expr(
            node.operand,
            immutable_names=immutable_names,
        )
    return False


def _assert_load_time_safe_expr(
    node: ast.AST,
    *,
    filename: str,
    context: str,
    literal_names: frozenset[str],
) -> None:
    if not _is_load_time_safe_expr(node, literal_names=literal_names):
        raise RuntimeError(
            f"Sandbox skill {context} must be a literal-only expression in {filename}"
        )


def _assert_immutable_load_time_safe_expr(
    node: ast.AST,
    *,
    filename: str,
    context: str,
    immutable_names: frozenset[str],
) -> None:
    if not _is_immutable_load_time_safe_expr(node, immutable_names=immutable_names):
        raise RuntimeError(
            f"Sandbox skill {context} must be immutable at load time in {filename}"
        )


def _assert_function_definition_is_safe(
    statement: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    filename: str,
    immutable_names: frozenset[str],
) -> None:
    # AUDIT-FIX(#2): Function decorators and mutable defaults execute/create
    # shared state at load time, so reject them before exec.
    if statement.decorator_list:
        raise RuntimeError(f"Sandbox skill decorators are not allowed in {filename}")
    defaults = list(statement.args.defaults) + [
        default for default in statement.args.kw_defaults if default is not None
    ]
    for default in defaults:
        _assert_immutable_load_time_safe_expr(
            default,
            filename=filename,
            context="default argument",
            immutable_names=immutable_names,
        )


def _assert_safe_module_shape(tree: ast.AST, *, filename: str) -> ast.Module:
    # AUDIT-FIX(#1): Perform local AST defense-in-depth instead of trusting a
    # single upstream validator result.
    # AUDIT-FIX(#2): Forbid executable top-level statements that can hang the
    # voice agent while merely loading a skill.
    if not isinstance(tree, ast.Module):
        raise RuntimeError(
            f"Sandbox validator returned unexpected AST type for {filename}"
        )

    if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
        raise RuntimeError(f"Sandbox skill AST is too large: {filename}")

    body = tree.body
    first_non_docstring_index = 1 if body and _is_docstring_statement(body[0]) else 0
    if first_non_docstring_index >= len(body) or not _is_future_annotations_import(
        body[first_non_docstring_index]
    ):
        raise RuntimeError(
            f"Sandbox skill must start with 'from __future__ import annotations': "
            f"{filename}"
        )

    literal_names: set[str] = set()
    immutable_names: set[str] = set()

    for statement in body:
        if _is_docstring_statement(statement):
            continue

        if _is_future_annotations_import(statement):
            continue

        if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
            _assert_function_definition_is_safe(
                statement,
                filename=filename,
                immutable_names=frozenset(immutable_names),
            )
            continue

        if isinstance(statement, ast.Assign):
            assigned_names: list[str] = []
            for target in statement.targets:
                if not isinstance(target, ast.Name):
                    raise RuntimeError(
                        f"Sandbox skill assignments must target simple names in "
                        f"{filename}"
                    )
                if target.id in _BLOCKED_CODE_NAMES:
                    raise RuntimeError(
                        f"Sandbox skill cannot bind reserved name '{target.id}' in "
                        f"{filename}"
                    )
                assigned_names.append(target.id)

            _assert_load_time_safe_expr(
                statement.value,
                filename=filename,
                context="module assignment",
                literal_names=frozenset(literal_names),
            )
            literal_names.update(assigned_names)

            if _is_immutable_load_time_safe_expr(
                statement.value,
                immutable_names=frozenset(immutable_names),
            ):
                immutable_names.update(assigned_names)
            else:
                for assigned_name in assigned_names:
                    immutable_names.discard(assigned_name)
            continue

        if isinstance(statement, ast.AnnAssign):
            if not isinstance(statement.target, ast.Name):
                raise RuntimeError(
                    f"Sandbox skill annotated assignments must target simple names in "
                    f"{filename}"
                )
            if statement.target.id in _BLOCKED_CODE_NAMES:
                raise RuntimeError(
                    f"Sandbox skill cannot bind reserved name "
                    f"'{statement.target.id}' in {filename}"
                )
            if statement.value is not None:
                _assert_load_time_safe_expr(
                    statement.value,
                    filename=filename,
                    context="module annotated assignment",
                    literal_names=frozenset(literal_names),
                )
                literal_names.add(statement.target.id)
                if _is_immutable_load_time_safe_expr(
                    statement.value,
                    immutable_names=frozenset(immutable_names),
                ):
                    immutable_names.add(statement.target.id)
                else:
                    immutable_names.discard(statement.target.id)
            continue

        if isinstance(statement, ast.Pass):
            continue

        raise RuntimeError(
            f"Sandbox skill contains forbidden top-level statement "
            f"{type(statement).__name__} in {filename}"
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith("_") or node.attr in _BLOCKED_ATTRIBUTE_NAMES
        ):
            raise RuntimeError(
                f"Sandbox skill references forbidden attribute '{node.attr}' in "
                f"{filename}"
            )

        if isinstance(node, ast.Global | ast.Nonlocal):
            raise RuntimeError(
                f"Sandbox skill uses forbidden stateful scope statement "
                f"{type(node).__name__} in {filename}"
            )

        if isinstance(node, ast.Import):
            raise RuntimeError(
                f"Sandbox skill uses forbidden import statement in {filename}"
            )

        if isinstance(node, ast.ImportFrom) and not _is_future_annotations_import(node):
            raise RuntimeError(
                f"Sandbox skill uses forbidden import-from statement in {filename}"
            )

        if isinstance(node, ast.ClassDef):
            raise RuntimeError(
                f"Sandbox skill class definitions are not allowed in {filename}"
            )

    return tree


def _iter_code_objects(code: CodeType):
    yield code
    for const in code.co_consts:
        if isinstance(const, CodeType):
            yield from _iter_code_objects(const)


def _assert_safe_code_object_names(code: CodeType, *, filename: str) -> None:
    for nested_code in _iter_code_objects(code):
        for name in nested_code.co_names:
            if name == "__future__":
                continue
            if name.startswith("__") or name in _BLOCKED_CODE_NAMES:
                raise RuntimeError(
                    f"Sandbox skill references forbidden name '{name}' in {filename}"
                )


_SAFE_BUILTINS = MappingProxyType(
    {
        "Exception": Exception,
        "False": False,
        "None": None,
        "RuntimeError": RuntimeError,
        "True": True,
        "ValueError": ValueError,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "getattr": safe_getattr,
        "hasattr": safe_hasattr,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": range,
        "reversed": reversed,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        # AUDIT-FIX(#3): Keep the builtin surface read-only and expose only the
        # restricted import helper, not a mutable builtin dict copy.
        "__import__": safe_import,
    }
)


@dataclass(frozen=True, slots=True)
class TrustedSkillModule:
    """Hold the trusted execution namespace for one validated skill module."""

    filename: str
    # AUDIT-FIX(#5): Expose a read-only namespace view so callers cannot mutate
    # trusted globals after validation.
    namespace: Mapping[str, Any]
    # AUDIT-FIX(#5): Store only the globals identity for trust checks instead of
    # exposing the mutable dict itself on the dataclass.
    _globals_id: int = field(repr=False, compare=False)

    def get_handler(self, handler_name: str) -> Any:
        """Return one callable handler from the trusted namespace."""

        # AUDIT-FIX(#4): Reject non-string handler names instead of coercing them.
        normalized_handler_name = _require_str(handler_name, field_name="handler_name")
        # AUDIT-FIX(#6): Only public identifier-like function names are valid
        # handler entry points.
        if (
            not normalized_handler_name.isidentifier()
            or normalized_handler_name.startswith("_")
        ):
            raise RuntimeError(
                f"Sandbox skill handler name is invalid: {normalized_handler_name}"
            )

        candidate = self.namespace.get(normalized_handler_name)
        if not isinstance(candidate, FunctionType):
            raise RuntimeError(
                f"Sandbox skill handler is missing: {normalized_handler_name}"
            )
        if id(candidate.__globals__) != self._globals_id:
            raise RuntimeError(
                f"Sandbox skill handler is not trusted: {normalized_handler_name}"
            )
        return candidate


def load_trusted_skill_module(*, source_text: str, filename: str) -> TrustedSkillModule:
    """Compile and execute one validated skill file under restricted builtins."""

    # AUDIT-FIX(#9): Normalize inputs before they reach validator/compile paths.
    normalized_source = _normalize_source_text(source_text)
    normalized_filename = _normalize_filename(filename)

    try:
        tree = validate_skill_source(normalized_source, filename=normalized_filename)
    # AUDIT-FIX(#8): Add stage-specific error context so the caller can recover,
    # quarantine, and report the failure deterministically.
    except Exception as exc:
        detail = str(exc).strip() or type(exc).__name__
        raise RuntimeError(
            f"Sandbox skill validation failed: {normalized_filename}: {detail}"
        ) from exc

    module_tree = _assert_safe_module_shape(tree, filename=normalized_filename)

    try:
        # AUDIT-FIX(#7): Compile without inheriting ambient future flags from the
        # host module so sandbox semantics stay deterministic on Python 3.11.2.
        code = compile(
            module_tree,
            normalized_filename,
            "exec",
            dont_inherit=True,
            optimize=0,
        )
    except Exception as exc:
        detail = str(exc).strip() or type(exc).__name__
        raise RuntimeError(
            f"Sandbox skill compilation failed: {normalized_filename}: {detail}"
        ) from exc

    _assert_safe_code_object_names(code, filename=normalized_filename)

    namespace_dict: dict[str, Any] = {
        # AUDIT-FIX(#3): CPython exec() expects a real builtin dict here; using a
        # plain snapshot preserves the restricted surface without tripping the
        # interpreter-level MappingProxyType crash path.
        "__builtins__": dict(_SAFE_BUILTINS),
        "__name__": _SANDBOX_MODULE_NAME,
    }

    try:
        exec(code, namespace_dict, namespace_dict)
    except Exception as exc:
        raise RuntimeError(
            f"Sandbox skill execution failed during load: {normalized_filename}"
        ) from exc

    return TrustedSkillModule(
        filename=normalized_filename,
        namespace=MappingProxyType(namespace_dict),
        _globals_id=id(namespace_dict),
    )
