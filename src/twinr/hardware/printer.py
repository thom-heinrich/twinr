"""Format and submit bounded Twinr receipt print jobs.

This module owns text normalization, payload limits, and ``lp`` execution for
Twinr's small receipt printers. The 2026 revision removes local disk spooling,
preserves caller-supplied receipt layout, and exposes structured job metadata
while staying drop-in compatible for existing callers of ``print_text`` and
``print_bytes``.
"""

# CHANGELOG: 2026-03-28
# BUG-1: Fixed receipt body normalization that collapsed multiple spaces, indentation, and blank lines,
#        which corrupted aligned totals and other preformatted receipt layouts.
# BUG-2: Replaced thread-offloaded async printing with true async subprocess handling so uvicorn/event-loop
#        callers no longer burn thread-pool capacity for every print and can cancel stalled jobs cleanly.
# SEC-1: Removed temp-file spooling of raw receipt payloads; jobs are now streamed to lp via stdin so
#        sensitive receipt text is not written to disk on the Raspberry Pi's SD card.
# SEC-2: Sanitized ambient CUPS-related environment overrides for lp subprocesses to prevent unintended
#        destination/server/user redirection through inherited process environment.
# IMP-1: Added structured print receipts (job_id/files_reported/stdout/stderr) and deterministic English
#        lp output parsing via LC_ALL=C.
# IMP-2: Added explicit job naming and CUPS server pinning to align the module with 2026 IPP/Printer
#        Application deployments while remaining compatible with classic queues.
# BREAKING: lp subprocesses no longer honor inherited CUPS_* / LPDEST / PRINTER / HOME overrides.
#            Configure printer_cups_server explicitly if you previously relied on environment-driven routing.
# BREAKING: text normalization now preserves intentional spacing and blank lines instead of collapsing them.

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import shutil
import subprocess
import textwrap
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from twinr.agent.base_agent.config import TwinrConfig

logger = logging.getLogger(__name__)

_DEFAULT_HEADER_TEXT = "TWINR.com"
_DEFAULT_PRINT_TIMEOUT_SECONDS = 10.0
_DEFAULT_MAX_PAYLOAD_BYTES = 1_048_576
_DEFAULT_JOB_NAME_PREFIX = "twinr-receipt"

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

_JOB_ID_RE = re.compile(
    r"request id is (?P<job_id>\S+)\s+\((?P<files>\d+)\s+file\(s\)\)",
    re.IGNORECASE,
)


class PrinterError(RuntimeError):
    """Raised when receipt printing fails in a way that is safe to surface upstream."""


@dataclass(frozen=True, slots=True)
class PrintJobReceipt:
    """Structured metadata about an accepted print job."""

    destination: str
    stdout: str
    stderr: str
    returncode: int
    bytes_submitted: int
    job_id: str | None
    files_reported: int | None
    backend: str = "cups-lp"


def _coerce_int(name: str, value: object, *, minimum: int) -> int:
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
    cleaned = _coerce_text("queue", queue).strip()
    if not cleaned:
        raise ValueError("queue must not be empty")
    if any(char.isspace() or ord(char) < 32 for char in cleaned):
        raise ValueError("queue contains invalid whitespace/control characters")
    return cleaned


def _sanitize_job_name(job_name: object, *, fallback: str) -> str:
    raw = _coerce_text("job_name", job_name, default="").replace("\r", " ").replace("\n", " ").strip()
    if not raw:
        raw = fallback
    compact = " ".join(raw.split())
    printable = "".join(char if 32 <= ord(char) <= 126 else " " for char in compact)
    normalized = " ".join(printable.split())
    return normalized[:80] or fallback


def _sanitize_cups_server(cups_server: object) -> str | None:
    if cups_server is None:
        return None
    cleaned = _coerce_text("cups_server", cups_server).strip()
    if not cleaned:
        return None
    if any(char.isspace() or ord(char) < 32 for char in cleaned):
        raise ValueError("cups_server contains invalid whitespace/control characters")
    return cleaned


def _resolve_lp_binary(lp_binary: object) -> str:
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
    """Print short Twinr receipts through a validated CUPS queue."""

    def __init__(
        self,
        *,
        queue: str,
        header_text: str = _DEFAULT_HEADER_TEXT,
        feed_lines: int = 3,
        line_width: int = 30,
        max_lines: int = 8,
        max_chars: int = 320,
        print_timeout_seconds: float = _DEFAULT_PRINT_TIMEOUT_SECONDS,
        max_payload_bytes: int = _DEFAULT_MAX_PAYLOAD_BYTES,
        lp_binary: str | None = None,
        job_name_prefix: str = _DEFAULT_JOB_NAME_PREFIX,
        cups_server: str | None = None,
    ) -> None:
        self.queue = _sanitize_queue(queue)
        self.header_text = _coerce_text("header_text", header_text, default=_DEFAULT_HEADER_TEXT).strip() or _DEFAULT_HEADER_TEXT
        self.feed_lines = _coerce_int("feed_lines", feed_lines, minimum=2)
        self.line_width = _coerce_int("line_width", line_width, minimum=16)
        self.max_lines = _coerce_int("max_lines", max_lines, minimum=1)
        self.max_chars = _coerce_int("max_chars", max_chars, minimum=16)
        self.print_timeout_seconds = _coerce_float("print_timeout_seconds", print_timeout_seconds, minimum=0.5)
        self.max_payload_bytes = _coerce_int("max_payload_bytes", max_payload_bytes, minimum=1)
        self.lp_binary = _resolve_lp_binary(lp_binary)
        self.job_name_prefix = _sanitize_job_name(job_name_prefix, fallback=_DEFAULT_JOB_NAME_PREFIX)
        self.cups_server = _sanitize_cups_server(cups_server)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "RawReceiptPrinter":
        """Build a receipt printer from ``TwinrConfig`` values."""

        return cls(
            queue=config.printer_queue,
            header_text=config.printer_header_text,
            feed_lines=config.printer_feed_lines,
            line_width=config.printer_line_width,
            max_lines=config.print_max_lines,
            max_chars=config.print_max_chars,
            print_timeout_seconds=getattr(config, "printer_timeout_seconds", _DEFAULT_PRINT_TIMEOUT_SECONDS),
            max_payload_bytes=getattr(config, "printer_max_payload_bytes", _DEFAULT_MAX_PAYLOAD_BYTES),
            lp_binary=getattr(config, "printer_lp_binary", None),
            job_name_prefix=getattr(config, "printer_job_name_prefix", _DEFAULT_JOB_NAME_PREFIX),
            cups_server=getattr(config, "printer_cups_server", None),
        )

    def print_text(self, text: str, *, job_name: str | None = None) -> str:
        """Normalize Twinr receipt text and submit it to the printer queue.

        Returns the legacy ``lp`` stdout string for drop-in compatibility.
        """

        return self.print_text_result(text, job_name=job_name).stdout

    def print_bytes(self, payload: bytes, *, job_name: str | None = None) -> str:
        """Submit a validated raw receipt payload and return legacy ``lp`` stdout."""

        return self.print_bytes_result(payload, job_name=job_name).stdout

    async def print_text_async(self, text: str, *, job_name: str | None = None) -> str:
        """Async variant of :meth:`print_text` using a real subprocess, not a worker thread."""

        return (await self.print_text_result_async(text, job_name=job_name)).stdout

    async def print_bytes_async(self, payload: bytes, *, job_name: str | None = None) -> str:
        """Async variant of :meth:`print_bytes` using a real subprocess, not a worker thread."""

        return (await self.print_bytes_result_async(payload, job_name=job_name)).stdout

    def print_text_result(self, text: str, *, job_name: str | None = None) -> PrintJobReceipt:
        """Normalize receipt text and return structured print-job metadata."""

        if not isinstance(text, str):
            logger.error("Receipt text must be str, got %s", type(text).__name__)
            raise PrinterError("The printer could not prepare the printout.")
        return self.print_bytes_result(self._build_text_payload(text), job_name=job_name)

    def print_bytes_result(self, payload: bytes, *, job_name: str | None = None) -> PrintJobReceipt:
        """Submit a validated raw receipt payload and return structured job metadata."""

        payload_bytes = self._validate_payload(payload)
        command = self._lp_command(job_name)
        env = self._subprocess_env()

        try:
            result = subprocess.run(
                command,
                input=payload_bytes,
                capture_output=True,
                check=False,
                timeout=self.print_timeout_seconds,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            logger.warning(
                "Printer command timed out after %.2fs for queue=%s",
                self.print_timeout_seconds,
                self.queue,
                exc_info=True,
            )
            raise PrinterError("The printer is taking too long to respond.") from exc
        except OSError as exc:
            logger.exception("Failed to execute printer command via %s for queue=%s", self.lp_binary, self.queue)
            raise PrinterError("The printer is not available right now.") from exc

        return self._handle_completed_process(
            returncode=result.returncode,
            stdout_bytes=result.stdout,
            stderr_bytes=result.stderr,
            payload_size=len(payload_bytes),
        )

    async def print_text_result_async(self, text: str, *, job_name: str | None = None) -> PrintJobReceipt:
        """Async text-print API returning structured job metadata."""

        if not isinstance(text, str):
            logger.error("Receipt text must be str, got %s", type(text).__name__)
            raise PrinterError("The printer could not prepare the printout.")
        return await self.print_bytes_result_async(self._build_text_payload(text), job_name=job_name)

    async def print_bytes_result_async(self, payload: bytes, *, job_name: str | None = None) -> PrintJobReceipt:
        """Async raw-print API returning structured job metadata."""

        payload_bytes = self._validate_payload(payload)
        command = self._lp_command(job_name)
        env = self._subprocess_env()

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except OSError as exc:
            logger.exception("Failed to execute printer command via %s for queue=%s", self.lp_binary, self.queue)
            raise PrinterError("The printer is not available right now.") from exc

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(payload_bytes),
                timeout=self.print_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.communicate()
            logger.warning(
                "Async printer command timed out after %.2fs for queue=%s",
                self.print_timeout_seconds,
                self.queue,
                exc_info=True,
            )
            raise PrinterError("The printer is taking too long to respond.") from exc
        except asyncio.CancelledError:
            process.kill()
            await process.communicate()
            raise

        return self._handle_completed_process(
            returncode=process.returncode,
            stdout_bytes=stdout_bytes,
            stderr_bytes=stderr_bytes,
            payload_size=len(payload_bytes),
        )

    def _handle_completed_process(
        self,
        *,
        returncode: int | None,
        stdout_bytes: bytes,
        stderr_bytes: bytes,
        payload_size: int,
    ) -> PrintJobReceipt:
        stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
        stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
        normalized_stdout = " ".join(stdout.split())
        parsed = _JOB_ID_RE.search(normalized_stdout)

        if returncode != 0:
            logger.warning(
                "Print job rejected by CUPS: queue=%s returncode=%s stdout=%r stderr=%r",
                self.queue,
                returncode,
                stdout,
                stderr,
            )
            raise PrinterError("The printer could not complete the printout.")

        receipt = PrintJobReceipt(
            destination=self.queue,
            stdout=stdout,
            stderr=stderr,
            returncode=int(returncode or 0),
            bytes_submitted=payload_size,
            job_id=parsed.group("job_id") if parsed else None,
            files_reported=int(parsed.group("files")) if parsed else None,
        )

        if receipt.files_reported == 0:
            logger.warning(
                "CUPS accepted a zero-file print job: queue=%s stdout=%r stderr=%r",
                self.queue,
                stdout,
                stderr,
            )
            raise PrinterError("CUPS accepted the job but received no document data")

        logger.info(
            "Submitted print job: queue=%s job_id=%s bytes=%s",
            receipt.destination,
            receipt.job_id or "unknown",
            receipt.bytes_submitted,
        )
        return receipt

    def _validate_payload(self, payload: bytes) -> bytes:
        if isinstance(payload, bytes):
            payload_bytes = payload
        elif isinstance(payload, bytearray):
            payload_bytes = bytes(payload)
        elif isinstance(payload, memoryview):
            payload_bytes = payload.tobytes()
        else:
            logger.error("Receipt payload must be bytes-like, got %s", type(payload).__name__)
            raise PrinterError("The printer received an invalid document.")

        if not payload_bytes:
            raise PrinterError("The printer received an empty document.")
        if len(payload_bytes) > self.max_payload_bytes:
            logger.warning(
                "Rejected oversized print payload for queue=%s: %s bytes > %s bytes",
                self.queue,
                len(payload_bytes),
                self.max_payload_bytes,
            )
            raise PrinterError("The document is too large for this printer.")
        return payload_bytes

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
        compact = " ".join(self._to_ascii_text(text, trim=True).replace("\n", " ").split())
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
        if not normalized:
            return ""

        wrapper = textwrap.TextWrapper(
            width=self.line_width,
            break_long_words=True,
            break_on_hyphens=False,
            expand_tabs=False,
            replace_whitespace=False,
            drop_whitespace=False,
        )

        wrapped_lines: list[str] = []
        for raw_line in normalized.split("\n"):
            ascii_line = self._to_ascii_text(raw_line, trim=False).expandtabs(4)
            if not ascii_line.strip():
                wrapped_lines.append("")
                continue
            wrapped_lines.extend(chunk.rstrip() for chunk in (wrapper.wrap(ascii_line) or [ascii_line]))

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
            candidate = line
            if len(candidate) > allowed:
                candidate = candidate[:allowed]
                truncated = True

            result_lines.append(candidate)
            budget -= len(candidate) + separator

            if len(candidate) < len(line):
                break

        if truncated:
            return self._with_ellipsis(result_lines)
        return "\n".join(result_lines)

    def _with_ellipsis(self, lines: list[str]) -> str:
        if not lines:
            return "." * min(3, self.max_chars, self.line_width)

        prefix_chars = sum(len(line) for line in lines[:-1]) + len(lines[:-1])
        remaining_budget = max(self.max_chars - prefix_chars, 0)
        line_budget = min(self.line_width, remaining_budget)

        if line_budget <= 0:
            return "\n".join(lines[:-1])

        ellipsis = "..."[:line_budget]
        if line_budget <= len(ellipsis):
            lines[-1] = ellipsis
            return "\n".join(lines)

        last_line = lines[-1][:line_budget]
        base = last_line[: max(line_budget - len(ellipsis), 0)].rstrip()
        lines[-1] = f"{base}{ellipsis}" if base else ellipsis
        if len(lines[-1]) > line_budget:
            lines[-1] = lines[-1][:line_budget]
        return "\n".join(lines)

    def _to_ascii_text(self, text: str, *, trim: bool) -> str:
        normalized = text
        for source, replacement in _UNICODE_REPLACEMENTS.items():
            normalized = normalized.replace(source, replacement)
        normalized = unicodedata.normalize("NFKD", normalized)
        ascii_text = normalized.encode("ascii", errors="ignore").decode("ascii")
        cleaned = "".join(char if 32 <= ord(char) <= 126 else " " for char in ascii_text)
        return cleaned.strip() if trim else cleaned

    def _build_job_name(self, job_name: str | None) -> str:
        return _sanitize_job_name(job_name, fallback=self.job_name_prefix)

    def _subprocess_env(self) -> dict[str, str]:
        env = os.environ.copy()
        for name in (
            "CUPS_ANYROOT",
            "CUPS_CACHEDIR",
            "CUPS_DATADIR",
            "CUPS_ENCRYPTION",
            "CUPS_EXPIREDCERTS",
            "CUPS_GSSSERVICENAME",
            "CUPS_SERVER",
            "CUPS_SERVERBIN",
            "CUPS_SERVERROOT",
            "CUPS_STATEDIR",
            "CUPS_USER",
            "IPP_PORT",
            "LOCALEDIR",
            "LPDEST",
            "PRINTER",
            "TMPDIR",
            "LANG",
            "LANGUAGE",
            "LC_ALL",
        ):
            env.pop(name, None)

        # BREAKING: do not let lp read user-scoped ~/.cups or XDG config for routing/options.
        env["HOME"] = "/"
        env["XDG_CONFIG_HOME"] = "/"
        env["TMPDIR"] = "/tmp"
        env["LANG"] = "C"
        env["LC_ALL"] = "C"
        return env

    def _lp_command(self, job_name: str | None) -> list[str]:
        command = [self.lp_binary]
        if self.cups_server:
            command.extend(["-h", self.cups_server])

        command.extend(
            [
                "-d",
                self.queue,
                "-o",
                "raw",
                "-o",
                "document-format=application/vnd.cups-raw",
                "-o",
                "job-sheets=none,none",
                "-t",
                self._build_job_name(job_name),
                "-",
            ]
        )
        return command