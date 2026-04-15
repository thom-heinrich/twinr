"""Regression coverage for the Crazyflie deck-bus probe helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "hardware" / "bitcraze" / "probe_deck_bus.py"
_SPEC = importlib.util.spec_from_file_location("bitcraze_deck_bus_probe_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class _FakeMemoryElementType:
    TYPE_DECKCTRL = 0x21

    @staticmethod
    def type_to_string(value: int) -> str:
        if value == 1:
            return "1-wire"
        if value == 0x21:
            return "DeckCtrl"
        return "Unknown"


class _FakeMemory:
    def __init__(self) -> None:
        self.id = 4
        self.type = 1
        self.size = 112
        self.valid = True
        self.vid = 0xBC
        self.pid = 0x12
        self.name = "bcFlow2"
        self.revision = "A"
        self.addr: int | str = 0x12345678
        self.elements = {"Board name": "bcFlow2", "Board revision": "A"}


class _FakeOwMemory(_FakeMemory):
    def __init__(self) -> None:
        super().__init__()
        self.addr = "0DB0EC5D0100002A"


class BitcrazeDeckBusProbeTests(unittest.TestCase):
    def test_console_collector_accumulates_chunked_lines(self) -> None:
        collector = _MODULE._ConsoleCollector()

        collector.on_text("SYS: ok")
        collector.on_text("\nDECK_CORE: 0 deck")
        collector.on_text("(s) found\ntrailing")

        lines = collector.finalize()

        self.assertEqual(
            [line.text for line in lines],
            ["SYS: ok", "DECK_CORE: 0 deck(s) found", "trailing"],
        )

    def test_capture_console_markers_keeps_exact_known_markers_once(self) -> None:
        console_lines = (
            _MODULE.ConsoleLine(timestamp_s=0.1, text="i2c1 [FAIL]"),
            _MODULE.ConsoleLine(timestamp_s=0.2, text="random"),
            _MODULE.ConsoleLine(timestamp_s=0.3, text="EEPROM: I2C connection [FAIL]"),
            _MODULE.ConsoleLine(timestamp_s=0.4, text="i2c1 [FAIL]"),
        )

        markers = _MODULE._capture_console_markers(console_lines)

        self.assertEqual(markers, ("i2c1 [FAIL]", "EEPROM: I2C connection [FAIL]"))

    def test_summarize_memory_keeps_structured_fields(self) -> None:
        summary = _MODULE._summarize_memory(_FakeMemory(), _FakeMemoryElementType)

        self.assertEqual(summary.id, 4)
        self.assertEqual(summary.type_name, "1-wire")
        self.assertEqual(summary.name, "bcFlow2")
        self.assertEqual(summary.revision, "A")
        self.assertEqual(summary.address, 0x12345678)
        self.assertEqual(summary.elements["Board name"], "bcFlow2")

    def test_summarize_memory_keeps_string_onewire_address(self) -> None:
        summary = _MODULE._summarize_memory(_FakeOwMemory(), _FakeMemoryElementType)

        self.assertEqual(summary.address, "0DB0EC5D0100002A")
        self.assertEqual(summary.name, "bcFlow2")

    def test_should_stm_power_cycle_only_for_radio_by_default(self) -> None:
        self.assertTrue(
            _MODULE._should_stm_power_cycle(
                "radio://0/80/2M/E7E7E7E7E7",
                skip_stm_power_cycle=False,
            )
        )
        self.assertFalse(
            _MODULE._should_stm_power_cycle(
                "tcp://127.0.0.1:9000",
                skip_stm_power_cycle=False,
            )
        )
        self.assertFalse(
            _MODULE._should_stm_power_cycle(
                "radio://0/80/2M/E7E7E7E7E7",
                skip_stm_power_cycle=True,
            )
        )

    def test_stm_power_cycle_uses_power_switch_and_closes(self) -> None:
        events: list[str] = []

        class _FakePowerSwitch:
            def __init__(self, uri: str) -> None:
                events.append(f"init:{uri}")

            def stm_power_cycle(self) -> None:
                events.append("cycle")

            def close(self) -> None:
                events.append("close")

        original_module = sys.modules.get("cflib.utils.power_switch")
        fake_module = types.ModuleType("cflib.utils.power_switch")
        setattr(fake_module, "PowerSwitch", _FakePowerSwitch)
        sys.modules["cflib.utils.power_switch"] = fake_module
        try:
            _MODULE._stm_power_cycle("radio://0/80/2M/E7E7E7E7E7", settle_s=0.0)
        finally:
            if original_module is None:
                del sys.modules["cflib.utils.power_switch"]
            else:
                sys.modules["cflib.utils.power_switch"] = original_module

        self.assertEqual(
            events,
            ["init:radio://0/80/2M/E7E7E7E7E7", "cycle", "close"],
        )

if __name__ == "__main__":
    unittest.main()
