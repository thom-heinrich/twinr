from __future__ import annotations

import asyncio
import logging
import math
import os
import shutil
import subprocess
import tempfile
import textwrap
import unicodedata
from pathlib import Path

from twinr.agent.base_agent.config import TwinrConfig

logger = logging.getLogger(__name__)

_DEFAULT_HEADER_TEXT = "TWINR.com"
_DEFAULT_PRINT_TIMEOUT_SECONDS = 10.0
_DEFAULT_MAX_PAYLOAD_BYTES = 1_048_576

_UNICODE_REPLACEMENTS = {
    "Ä": "Ae",
    "Ö": "Oe",
    "Ü": "Ue",
    "ä": "ae",
    "ö": "oe",
    "ü": "ue",
    "ß": "ss",
    "ẞ": "SS",
    "€": "EUR",
    "–": "-",
    "—": "-",
    "…": "...",
    "„": '"',
    "“": '"',
    "”": '"',
    "‚": "'",
    "‘": "'",
    "’": "'",
}


class PrinterError(RuntimeError):
    """Raised when receipt printing fails in a way that is safe to surface upstream."""


def _coerce_int(name: str, value: object, *, minimum: int) -> int:
    # AUDIT-FIX(#2): Accept sane .env-style numeric strings while still failing fast on invalid config values.
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got bool")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{name} must not be empty")
        try:
            parsed = int(stripped)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    else:
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    return max(minimum, parsed)


def _coerce_float(name: str, value: object, *, minimum: float) -> float:
    # AUDIT-FIX(#2): Accept sane .env-style float strings while bounding timeouts away from zero and negative values.
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a float, got bool")
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{name} must not be empty")
        try:
            parsed = float(stripped)
        except ValueError as exc:
            raise ValueError(f"{name} must be a float, got {value!r}") from exc
    else:
        raise ValueError(f"{name} must be a float, got {type(value).__name__}")
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return max(minimum, parsed)


def _coerce_text(name: str, value: object, *, default: str | None = None) -> str:
    if value is None:
        if default is not None:
            return default
        raise ValueError(f"{name} must not be None")
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string, got {type(value).__name__}")
    return value


def _sanitize_queue(queue: object) -> str:
    # AUDIT-FIX(#3): Reject empty and whitespace/control-tainted queue names before they reach lp.
    cleaned = _coerce_text("queue", queue).strip()
    if not cleaned:
        raise ValueError("queue must not be empty")
    if any(char.isspace() or ord(char) < 32 for char in cleaned):
        raise ValueError("queue contains invalid whitespace/control characters")
    return cleaned


def _resolve_lp_binary(lp_binary: object) -> str:
    # AUDIT-FIX(#3): Resolve the lp executable once at startup instead of relying on PATH lookups for every job.
    candidate = _coerce_text("lp_binary", lp_binary, default="").strip()
    if not candidate:
        resolved = shutil.which("lp")
    elif "/" in candidate:
        resolved = candidate
    else:
        resolved = shutil.which(candidate)

    if not resolved:
        raise ValueError("Could not find an executable lp binary")

    path = Path(resolved).expanduser()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"lp binary is not executable: {path}")
    return str(path.resolve())


class RawReceiptPrinter:
    def __init__(
        self,
        *,
        queue: str,
        header_text: str = "TWINR.com",
        feed_lines: int = 3,
        line_width: int = 30,
        max_lines: int = 8,
        max_chars: int = 320,
        print_timeout_seconds: float = _DEFAULT_PRINT_TIMEOUT_SECONDS,
        max_payload_bytes: int = _DEFAULT_MAX_PAYLOAD_BYTES,
        lp_binary: str | None = None,
    ) -> None:
        self.queue = _sanitize_queue(queue)
        self.header_text = _coerce_text("header_text", header_text, default=_DEFAULT_HEADER_TEXT).strip() or _DEFAULT_HEADER_TEXT
        self.feed_lines = _coerce_int("feed_lines", feed_lines, minimum=2)
        self.line_width = _coerce_int("line_width", line_width, minimum=16)
        self.max_lines = _coerce_int("max_lines", max_lines, minimum=1)
        self.max_chars = _coerce_int("max_chars", max_chars, minimum=16)
        self.print_timeout_seconds = _coerce_float("print_timeout_seconds", print_timeout_seconds, minimum=0.5)  # AUDIT-FIX(#1): Bound external printer waits so the service cannot hang forever.
        self.max_payload_bytes = _coerce_int("max_payload_bytes", max_payload_bytes, minimum=1)  # AUDIT-FIX(#7): Cap raw payload size to prevent tmpfs/printer abuse from oversized jobs.
        self.lp_binary = _resolve_lp_binary(lp_binary)  # AUDIT-FIX(#3): Pin execution to a validated lp binary path.

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "RawReceiptPrinter":
        return cls(
            queue=config.printer_queue,
            header_text=config.printer_header_text,
            feed_lines=config.printer_feed_lines,
            line_width=config.printer_line_width,
            max_lines=config.print_max_lines,
            max_chars=config.print_max_chars,
            print_timeout_seconds=getattr(config, "printer_timeout_seconds", _DEFAULT_PRINT_TIMEOUT_SECONDS),  # AUDIT-FIX(#1): Backward-compatible timeout config with a safe default.
            max_payload_bytes=getattr(config, "printer_max_payload_bytes", _DEFAULT_MAX_PAYLOAD_BYTES),  # AUDIT-FIX(#7): Backward-compatible payload cap with a generous default.
            lp_binary=getattr(config, "printer_lp_binary", None),  # AUDIT-FIX(#3): Allow explicit lp binary pinning without breaking existing config.
        )

    async def print_text_async(self, text: str) -> str:
        # AUDIT-FIX(#1): Provide an async-safe entry point so uvicorn callers can offload blocking printer I/O from the event loop.
        return await asyncio.to_thread(self.print_text, text)

    async def print_bytes_async(self, payload: bytes) -> str:
        # AUDIT-FIX(#1): Provide an async-safe entry point so uvicorn callers can offload blocking printer I/O from the event loop.
        return await asyncio.to_thread(self.print_bytes, payload)

    def print_text(self, text: str) -> str:
        if not isinstance(text, str):
            logger.error("Receipt text must be str, got %s", type(text).__name__)
            raise PrinterError("The printer could not prepare the printout.")  # AUDIT-FIX(#10): Fail deterministically on invalid caller input instead of deep attribute errors.
        return self.print_bytes(self._build_text_payload(text))

    def print_bytes(self, payload: bytes) -> str:
        payload_bytes = self._validate_payload(payload)
        spool_path: Path | None = None
        try:
            try:
                spool_path = self._write_spool_file(payload_bytes)  # AUDIT-FIX(#5): Keep temp-file lifecycle inside one protected scope so write failures do not leak spool files.
            except OSError as exc:
                logger.exception("Failed to create/write print spool file for queue=%s", self.queue)
                raise PrinterError("The printer could not prepare the printout.") from exc  # AUDIT-FIX(#4): Normalize local spool-file I/O failures into stable upstream errors.
            try:
                result = subprocess.run(
                    [*self._lp_command(), str(spool_path)],
                    capture_output=True,
                    check=False,
                    stdin=subprocess.DEVNULL,
                    timeout=self.print_timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                logger.warning(
                    "Printer command timed out after %.2fs for queue=%s",
                    self.print_timeout_seconds,
                    self.queue,
                    exc_info=True,
                )
                raise PrinterError("The printer is taking too long to respond.") from exc  # AUDIT-FIX(#1): Turn indefinite hangs into bounded, user-safe failures.
            except OSError as exc:
                logger.exception("Failed to execute printer command via %s for queue=%s", self.lp_binary, self.queue)
                raise PrinterError("The printer is not available right now.") from exc  # AUDIT-FIX(#4): Normalize subprocess execution failures into stable upstream errors.

            stdout = result.stdout.decode("utf-8", errors="ignore").strip()
            stderr = result.stderr.decode("utf-8", errors="ignore").strip()

            if result.returncode != 0:
                logger.warning(
                    "Print job rejected by CUPS: queue=%s returncode=%s stdout=%r stderr=%r",
                    self.queue,
                    result.returncode,
                    stdout,
                    stderr,
                )
                raise PrinterError("The printer could not complete the printout.")  # AUDIT-FIX(#6): Do not leak raw CUPS stderr or numeric codes into user-facing paths.
            return stdout
        finally:
            if spool_path is not None:
                self._safe_unlink(spool_path)

    def _validate_payload(self, payload: bytes) -> bytes:
        # AUDIT-FIX(#7): Reject empty or oversized raw jobs before they reach disk/CUPS.
        if isinstance(payload, bytes):
            payload_bytes = payload
        elif isinstance(payload, bytearray):
            payload_bytes = bytes(payload)
        elif isinstance(payload, memoryview):
            payload_bytes = payload.tobytes()
        else:
            logger.error("Receipt payload must be bytes-like, got %s", type(payload).__name__)
            raise PrinterError("The printer received an invalid document.")  # AUDIT-FIX(#10): Fail deterministically on invalid caller input instead of deep type errors.

        if not payload_bytes:
            raise PrinterError("The printer received an empty document.")  # AUDIT-FIX(#8): Stop empty jobs before submission instead of relying on locale-specific lp stdout text.
        if len(payload_bytes) > self.max_payload_bytes:
            logger.warning(
                "Rejected oversized print payload for queue=%s: %s bytes > %s bytes",
                self.queue,
                len(payload_bytes),
                self.max_payload_bytes,
            )
            raise PrinterError("The document is too large for this printer.")
        return payload_bytes

    def _write_spool_file(self, payload: bytes) -> Path:
        spool_path: Path | None = None
        fd = -1
        try:
            fd, raw_path = tempfile.mkstemp(prefix="twinr-print-", suffix=".bin")
            spool_path = Path(raw_path)
            with os.fdopen(fd, "wb") as handle:
                fd = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            return spool_path
        except Exception:
            if fd != -1:
                os.close(fd)
            if spool_path is not None:
                self._safe_unlink(spool_path)
            raise

    def _safe_unlink(self, path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to remove temporary print spool file: %s", path, exc_info=True)  # AUDIT-FIX(#5): Cleanup failures must not mask the original print outcome.

    def _build_text_payload(self, text: str) -> bytes:
        return self._compose_receipt_text(text).encode("ascii", errors="strict")

    def _compose_receipt_text(self, text: str) -> str:
        header = self._normalize_header_text(self.header_text)
        body = self._normalize_text(text)
        if body:
            lines = ["", header, "", *body.split("\n"), *([""] * self.feed_lines)]
        else:
            lines = ["", header, *([""] * (self.feed_lines + 1))]
        return "\r\n".join(lines)

    def _normalize_header_text(self, text: str) -> str:
        # AUDIT-FIX(#9): Collapse multi-line or overlong headers to one receipt-safe line.
        compact = " ".join(self._to_ascii_text(text).replace("\n", " ").split())
        if not compact:
            return _DEFAULT_HEADER_TEXT
        if len(compact) <= self.line_width:
            return compact
        allowed = max(self.line_width - 3, 1)
        shortened = compact[:allowed].rstrip()
        if self.line_width >= 4:
            return f"{shortened}..."
        return compact[: self.line_width]

    def _normalize_text(self, text: str) -> str:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
        wrapped_lines: list[str] = []
        for raw_line in normalized.split("\n"):
            ascii_line = self._to_ascii_text(raw_line)
            compact = " ".join(ascii_line.split())
            if not compact:
                continue
            wrapped_lines.extend(
                textwrap.wrap(
                    compact,
                    width=self.line_width,
                    break_long_words=True,
                    break_on_hyphens=False,
                )
                or [compact]
            )
        return self._limit_wrapped_lines(wrapped_lines)

    def _limit_wrapped_lines(self, wrapped_lines: list[str]) -> str:
        result_lines: list[str] = []
        budget = self.max_chars
        truncated = False
        for line in wrapped_lines:
            if len(result_lines) >= self.max_lines:
                truncated = True
                break
            separator = 1 if result_lines else 0
            if budget <= separator:
                truncated = True
                break
            allowed = budget - separator
            if len(line) > allowed:
                shortened = line[: max(allowed - 3, 0)].rstrip()
                if allowed >= 4:
                    line = f"{shortened}..."
                else:
                    line = line[:allowed]
                truncated = True
            result_lines.append(line)
            budget -= len(line) + separator
            if truncated:
                break

        if truncated and result_lines:
            last_line = result_lines[-1]
            if not last_line.endswith("...") and len(last_line) <= max(self.line_width - 3, 1):
                candidate = f"{last_line[: max(self.line_width - 3, 1)].rstrip()}..."
                candidate_budget = self.max_chars - sum(len(line) for line in result_lines[:-1]) - len(result_lines[:-1])
                if len(candidate) <= candidate_budget:
                    result_lines[-1] = candidate
        return "\n".join(result_lines)

    def _to_ascii_text(self, text: str) -> str:
        normalized = text
        for source, replacement in _UNICODE_REPLACEMENTS.items():
            normalized = normalized.replace(source, replacement)
        normalized = unicodedata.normalize("NFKD", normalized)
        ascii_text = normalized.encode("ascii", errors="ignore").decode("ascii")
        cleaned = "".join(char if char == "\n" or 32 <= ord(char) <= 126 else " " for char in ascii_text)
        return cleaned.strip()

    def _lp_command(self) -> list[str]:
        return [
            self.lp_binary,  # AUDIT-FIX(#3): Use the validated absolute lp path resolved at startup.
            "-d",
            self.queue,
            "-o",
            "raw",
            "-o",
            "document-format=application/vnd.cups-raw",
            "-o",
            "job-sheets=none,none",
        ]