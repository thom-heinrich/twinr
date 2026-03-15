from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.printer import RawReceiptPrinter


class RawReceiptPrinterTests(unittest.TestCase):
    def test_compose_receipt_text_adds_header_blank_lines_and_feed(self) -> None:
        printer = RawReceiptPrinter(queue="Test", header_text="TWINR.com", feed_lines=3, line_width=30)

        receipt = printer._compose_receipt_text("Termin Montag 14 Uhr")

        self.assertEqual(receipt, "\r\nTWINR.com\r\n\r\nTermin Montag 14 Uhr\r\n\r\n\r\n")

    def test_normalize_text_transliterates_umlauts_and_wraps(self) -> None:
        printer = RawReceiptPrinter(queue="Test", line_width=16)

        normalized = printer._normalize_text("Öl über Größen\nStraße zum Ärztehaus")

        self.assertEqual(
            normalized,
            "Oel ueber\nGroessen\nStrasse zum\nAerztehaus",
        )

    def test_build_text_payload_resets_printer_and_uses_ascii_only(self) -> None:
        printer = RawReceiptPrinter(queue="Test", line_width=32)

        payload = printer._build_text_payload("Grüße aus Köln")

        self.assertEqual(payload, b"\r\nTWINR.com\r\n\r\nGruesse aus Koeln\r\n\r\n\r\n")

    def test_normalize_text_enforces_physical_line_and_character_limits(self) -> None:
        printer = RawReceiptPrinter(queue="Test", line_width=10, max_lines=3, max_chars=24)

        normalized = printer._normalize_text(
            "Das ist eine sehr lange Zeile\nund noch eine zweite sehr lange Zeile"
        )

        self.assertLessEqual(len(normalized.splitlines()), 3)
        self.assertLessEqual(len(normalized), 24)

    def test_lp_command_forces_raw_without_banner_pages(self) -> None:
        printer = RawReceiptPrinter(queue="Test")

        command = printer._lp_command()

        self.assertEqual(
            command,
            [
                printer.lp_binary,
                "-d",
                "Test",
                "-o",
                "raw",
                "-o",
                "document-format=application/vnd.cups-raw",
                "-o",
                "job-sheets=none,none",
            ],
        )

    def test_print_bytes_spools_payload_as_temp_file(self) -> None:
        printer = RawReceiptPrinter(queue="Test")
        observed: dict[str, object] = {}

        def fake_run(command, *, capture_output: bool, check: bool, stdin, timeout):
            spool_path = Path(command[-1])
            observed["command"] = command
            observed["payload"] = spool_path.read_bytes()
            observed["path_exists_during_run"] = spool_path.exists()
            observed["stdin"] = stdin
            observed["timeout"] = timeout
            return type("CompletedProcess", (), {"returncode": 0, "stdout": b"request id is Test-2 (1 file(s))", "stderr": b""})()

        with patch("twinr.hardware.printer.subprocess.run", side_effect=fake_run):
            response = printer.print_bytes(b"HELLO\n")

        self.assertEqual(response, "request id is Test-2 (1 file(s))")
        self.assertEqual(observed["payload"], b"HELLO\n")
        self.assertTrue(observed["path_exists_during_run"])
        self.assertEqual(observed["stdin"], -3)
        self.assertEqual(observed["timeout"], printer.print_timeout_seconds)
        self.assertFalse(Path(observed["command"][-1]).exists())

    def test_print_bytes_rejects_zero_file_cups_response(self) -> None:
        printer = RawReceiptPrinter(queue="Test")

        with patch(
            "twinr.hardware.printer.subprocess.run",
            return_value=type("CompletedProcess", (), {"returncode": 0, "stdout": b"request id is Test-3 (0 file(s))", "stderr": b""})(),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "CUPS accepted the job but received no document data",
            ):
                printer.print_bytes(b"HELLO\n")


if __name__ == "__main__":
    unittest.main()
