from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.ops.devices import collect_device_overview
from twinr.ops.events import TwinrOpsEventStore


class DeviceOverviewTests(unittest.TestCase):
    def test_collect_device_overview_reports_unknown_printer_paper_status(self) -> None:
        def fake_run(command, *, capture_output: bool, text: bool, check: bool, timeout: float):
            if command[:3] == ["lpstat", "-l", "-p"]:
                return type(
                    "CompletedProcess",
                    (),
                    {
                        "returncode": 0,
                        "stdout": (
                            "printer Thermal_GP58 is idle. enabled since Fri Mar 13 16:03:44 2026\n"
                            "\tDescription: Thermal_GP58\n"
                            "\tConnection: direct\n"
                            "\tAlerts: none\n"
                        ),
                        "stderr": "",
                    },
                )()
            if command[:2] == ["lpoptions", "-p"]:
                return type(
                    "CompletedProcess",
                    (),
                    {
                        "returncode": 0,
                        "stdout": (
                            "device-uri=usb://Gprinter/GP-58?serial=WTTING%20 "
                            "printer-is-accepting-jobs=true printer-state-reasons=none printer-state=3 "
                            "printer-info=Thermal_GP58"
                        ),
                        "stderr": "",
                    },
                )()
            if command[:2] == ["arecord", "-l"]:
                return type(
                    "CompletedProcess",
                    (),
                    {
                        "returncode": 0,
                        "stdout": "card 3: CameraB409241 [USB Camera-B4.09.24.1], device 0: USB Audio [USB Audio]\n",
                        "stderr": "",
                    },
                )()
            raise AssertionError(f"Unexpected command: {command}")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            camera_path = root / "video0"
            ffmpeg_path = root / "ffmpeg"
            camera_path.write_text("", encoding="utf-8")
            ffmpeg_path.write_text("", encoding="utf-8")
            config = TwinrConfig(
                project_root=temp_dir,
                printer_queue="Thermal_GP58",
                camera_device=str(camera_path),
                camera_ffmpeg_path=str(ffmpeg_path),
                audio_input_device="default",
                audio_output_device="default",
                green_button_gpio=23,
                yellow_button_gpio=22,
                pir_motion_gpio=26,
                pir_active_high=True,
                pir_bias="pull-down",
            )
            events = TwinrOpsEventStore(root / "events.jsonl")
            events.append(
                event="self_test_finished",
                message="Printer self-test finished.",
                data={"test_name": "printer", "status": "warn"},
            )
            events.append(
                event="proactive_observation",
                message="Proactive monitor recorded a changed observation.",
                data={"pir_motion_detected": True},
            )

            with patch("twinr.ops.devices.which", side_effect=lambda name: f"/usr/bin/{name}" if name in {"lpstat", "lpoptions", "arecord"} else None):
                with patch("twinr.ops.devices.subprocess.run", side_effect=fake_run):
                    overview = collect_device_overview(config, event_store=events)

        devices = {device.key: device for device in overview.devices}
        self.assertIn("printer", devices)
        self.assertIn("camera", devices)
        self.assertIn("audio_input", devices)
        self.assertIn("pir", devices)
        printer = devices["printer"]
        self.assertEqual(printer.status, "ok")
        self.assertIn(
            "unknown on the current raw USB/CUPS path",
            {fact.label: fact.value for fact in printer.facts}["Paper status"],
        )
        self.assertIn("warn at", {fact.label: fact.value for fact in printer.facts}["Last self-test"])
        self.assertEqual(devices["camera"].status, "ok")
        self.assertIn("card 3: CameraB409241", {fact.label: fact.value for fact in devices["audio_input"].facts}["Detected capture devices"])
        self.assertNotEqual(
            {fact.label: fact.value for fact in devices["pir"].facts}["Last motion seen"],
            "not recorded in recent ops events",
        )


if __name__ == "__main__":
    unittest.main()
