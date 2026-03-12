from __future__ import annotations

import subprocess

from twinr.config import TwinrConfig


class RawReceiptPrinter:
    def __init__(self, *, queue: str, feed_lines: int = 3) -> None:
        self.queue = queue
        self.feed_lines = max(2, feed_lines)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "RawReceiptPrinter":
        return cls(queue=config.printer_queue, feed_lines=config.printer_feed_lines)

    def print_text(self, text: str) -> str:
        payload = self._normalize_text(text).encode("utf-8")
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

    def _normalize_text(self, text: str) -> str:
        normalized = text.replace("\r\n", "\n").strip("\n")
        return normalized + ("\n" * self.feed_lines)
