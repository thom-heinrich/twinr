from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"


def _run_memory_probe(source: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads((completed.stdout or "").strip() or "{}")


class MemoryPackageTests(unittest.TestCase):
    def test_context_exports_do_not_eager_import_longterm_or_chonkydb(self) -> None:
        payload = _run_memory_probe(
            textwrap.dedent(
                f"""
                import json
                import sys

                sys.path.insert(0, {str(_SRC_ROOT)!r})

                import twinr.memory as memory

                before = {{
                    "longterm_loaded": "twinr.memory.longterm" in sys.modules,
                    "chonkydb_loaded": "twinr.memory.chonkydb" in sys.modules,
                }}
                from twinr.memory import PersistentMemoryEntry

                after = {{
                    "persistent_memory_entry_name": PersistentMemoryEntry.__name__,
                    "longterm_loaded": "twinr.memory.longterm" in sys.modules,
                    "chonkydb_loaded": "twinr.memory.chonkydb" in sys.modules,
                }}
                print(json.dumps({{"before": before, "after": after}}, ensure_ascii=False))
                """
            )
        )

        self.assertEqual(payload["before"]["longterm_loaded"], False)
        self.assertEqual(payload["before"]["chonkydb_loaded"], False)
        self.assertEqual(payload["after"]["persistent_memory_entry_name"], "PersistentMemoryEntry")
        self.assertEqual(payload["after"]["longterm_loaded"], False)
        self.assertEqual(payload["after"]["chonkydb_loaded"], False)

    def test_longterm_exports_load_on_demand(self) -> None:
        payload = _run_memory_probe(
            textwrap.dedent(
                f"""
                import json
                import sys

                sys.path.insert(0, {str(_SRC_ROOT)!r})

                import twinr.memory as memory

                before = "twinr.memory.longterm" in sys.modules
                from twinr.memory import LongTermMemoryService

                after = {{
                    "longterm_loaded": "twinr.memory.longterm" in sys.modules,
                    "service_name": LongTermMemoryService.__name__,
                }}
                print(json.dumps({{"before": before, "after": after}}, ensure_ascii=False))
                """
            )
        )

        self.assertEqual(payload["before"], False)
        self.assertEqual(payload["after"]["longterm_loaded"], True)
        self.assertEqual(payload["after"]["service_name"], "LongTermMemoryService")
