from __future__ import annotations

import subprocess
import textwrap
import unicodedata

from twinr.agent.base_agent.config import TwinrConfig

_ESC_INIT = b"\x1b@"
_ESC_STYLE_NORMAL = b"\x1b!\x00"
_ESC_FONT_A = b"\x1bM\x00"
_ESC_LINE_SPACING_DEFAULT = b"\x1b2"
_GS_TEXT_NORMAL = b"\x1d!\x00"

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
    ) -> None:
        self.queue = queue
        self.header_text = header_text.strip() or "TWINR.com"
        self.feed_lines = max(2, feed_lines)
        self.line_width = max(16, line_width)
        self.max_lines = max(1, max_lines)
        self.max_chars = max(16, max_chars)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "RawReceiptPrinter":
        return cls(
            queue=config.printer_queue,
            header_text=config.printer_header_text,
            feed_lines=config.printer_feed_lines,
            line_width=config.printer_line_width,
            max_lines=config.print_max_lines,
            max_chars=config.print_max_chars,
        )

    def print_text(self, text: str) -> str:
        payload = self._build_text_payload(text)
        result = subprocess.run(
            ["lp", "-d", self.queue, "-o", "raw"],
            input=payload,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"Printing failed: {stderr or result.returncode}")
        return result.stdout.decode("utf-8", errors="ignore").strip()

    def print_bytes(self, payload: bytes) -> str:
        result = subprocess.run(
            ["lp", "-d", self.queue, "-o", "raw"],
            input=payload,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"Printing failed: {stderr or result.returncode}")
        return result.stdout.decode("utf-8", errors="ignore").strip()

    def _build_text_payload(self, text: str) -> bytes:
        return b"".join(
            [
                _ESC_INIT,
                _ESC_STYLE_NORMAL,
                _ESC_FONT_A,
                _ESC_LINE_SPACING_DEFAULT,
                _GS_TEXT_NORMAL,
                self._compose_receipt_text(text).encode("ascii", errors="strict"),
            ]
        )

    def _compose_receipt_text(self, text: str) -> str:
        header = self._to_ascii_text(self.header_text).strip() or "TWINR.com"
        body = self._normalize_text(text)
        if body:
            return f"\n{header}\n\n{body}" + ("\n" * self.feed_lines)
        return f"\n{header}" + ("\n" * (self.feed_lines + 1))

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
