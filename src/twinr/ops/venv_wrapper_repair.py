"""Repair stale Python-wrapper shebangs inside preserved Pi virtualenvs.

The Pi runtime deploy preserves ``/twinr/.venv`` across repo mirrors so
runtime-only state and installed dependencies survive normal code rollouts.
Historically moved/copied venvs can still carry console-script wrappers whose
shebang points at an old absolute path such as ``/home/thh/twinr/.venv``.
Those wrappers fail even though the package itself is installed correctly in the
current venv and ``python -m ...`` still works.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PiVenvScriptRepairResult:
    """Summarize one venv wrapper-shebang repair pass."""

    checked_files: int
    rewritten_files: int
    sample_paths: tuple[str, ...]


def repair_venv_python_shebangs(
    *,
    bin_dir: Path,
    expected_interpreter: str,
    sample_limit: int = 12,
) -> PiVenvScriptRepairResult:
    """Rewrite relocated venv Python-wrapper shebangs to the in-place interpreter.

    Virtual environments record absolute interpreter paths in generated console
    scripts. If a venv was historically copied from another checkout root, the
    package imports may still work via ``python -m ...`` while direct wrapper
    execution fails because the first line points to the old path.
    """

    expected = str(expected_interpreter).strip()
    if not expected:
        raise ValueError("expected_interpreter must not be empty")
    if sample_limit <= 0:
        raise ValueError("sample_limit must be greater than zero")
    if not bin_dir.exists():
        return PiVenvScriptRepairResult(
            checked_files=0,
            rewritten_files=0,
            sample_paths=(),
        )

    checked_files = 0
    rewritten_files = 0
    sample_paths: list[str] = []
    for candidate in sorted(bin_dir.iterdir()):
        if candidate.is_symlink() or not candidate.is_file():
            continue
        original = candidate.read_bytes()
        first_line, separator, remainder = original.partition(b"\n")
        if not first_line.startswith(b"#!") or b"\0" in first_line:
            continue
        interpreter = _shebang_interpreter(first_line)
        if not interpreter:
            continue
        checked_files += 1
        if not _looks_like_venv_python(interpreter) or interpreter == expected:
            continue
        line_suffix = b"\r" if first_line.endswith(b"\r") else b""
        rewritten = b"#!" + expected.encode("utf-8") + line_suffix
        candidate.write_bytes(rewritten + (separator + remainder if separator else b""))
        rewritten_files += 1
        if len(sample_paths) < sample_limit:
            sample_paths.append(candidate.name)
    return PiVenvScriptRepairResult(
        checked_files=checked_files,
        rewritten_files=rewritten_files,
        sample_paths=tuple(sample_paths),
    )


def _shebang_interpreter(first_line: bytes) -> str:
    """Extract the interpreter path from one script shebang line."""

    try:
        text = first_line.decode("utf-8")
    except UnicodeDecodeError:
        return ""
    if not text.startswith("#!"):
        return ""
    payload = text[2:].strip()
    if not payload:
        return ""
    return payload.split(maxsplit=1)[0]


def _looks_like_venv_python(interpreter: str) -> bool:
    """Return whether one shebang points at a venv-local Python executable."""

    return "/.venv/bin/python" in str(interpreter or "").strip()


__all__ = [
    "PiVenvScriptRepairResult",
    "repair_venv_python_shebangs",
]
