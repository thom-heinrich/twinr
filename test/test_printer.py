from pathlib import Path
import sys
import unittest

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
                "lp",
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


if __name__ == "__main__":
    unittest.main()
