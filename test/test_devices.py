from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.ops.devices import collect_device_overview
from twinr.ops.events import TwinrOpsEventStore


def _fake_respeaker_snapshot(
    *,
    capture_ready: bool,
    usb_visible: bool,
    host_control_ready: bool,
    transport_reason: str | None = None,
    requires_elevated_permissions: bool = False,
):
    capture_device = (
        type("Capture", (), {"hw_identifier": "hw:CARD=Array,DEV=0", "card_label": "reSpeaker XVF3800 4-Mic Array"})()
        if capture_ready
        else None
    )
    usb_device = (
        type("Usb", (), {"description": "Seeed Technology Co., Ltd. reSpeaker XVF3800 4-Mic Array"})()
        if usb_visible
        else None
    )
    probe = type(
        "FakeProbe",
        (),
        {
            "capture_ready": capture_ready,
            "capture_device": capture_device,
            "usb_visible": usb_visible,
            "usb_device": usb_device,
            "arecord_available": True,
            "lsusb_available": True,
            "state": (
                "audio_ready"
                if capture_ready
                else "usb_visible_no_capture"
                if usb_visible
                else "not_detected"
            ),
        },
    )()
    return type(
        "FakeSnapshot",
        (),
        {
            "probe": probe,
            "host_control_ready": host_control_ready,
            "transport": type(
                "Transport",
                (),
                {
                    "reason": transport_reason,
                    "requires_elevated_permissions": requires_elevated_permissions,
                },
            )(),
            "firmware_version": (2, 0, 7),
            "direction": type(
                "Direction",
                (),
                {
                    "doa_degrees": 277,
                    "speech_detected": True,
                    "room_quiet": False,
                    "beam_azimuth_degrees": (90.0, 270.0, 180.0, 277.0),
                    "beam_speech_energies": (0.1, 0.2, 0.0, 0.0),
                    "selected_azimuth_degrees": (277.0, 277.0),
                },
            )(),
            "mute": type(
                "Mute",
                (),
                {
                    "mute_active": None,
                    "gpo_logic_levels": (0, 0, 0, 1, 0),
                },
            )(),
        },
    )()


class DeviceOverviewTests(unittest.TestCase):
    def test_collect_device_overview_reports_unknown_printer_paper_status(self) -> None:
        def fake_run(command, *, capture_output: bool, text: bool, check: bool, timeout: float, **_kwargs):
            executable = Path(command[0]).name
            if executable == "lpstat" and command[1:3] == ["-l", "-p"]:
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
            if executable == "lpstat" and command[1:3] == ["-a", "Thermal_GP58"]:
                return type(
                    "CompletedProcess",
                    (),
                    {
                        "returncode": 0,
                        "stdout": "Thermal_GP58 accepting requests since Fri Mar 13 16:03:44 2026\n",
                        "stderr": "",
                    },
                )()
            if executable == "lpstat" and command[1:3] == ["-v", "Thermal_GP58"]:
                return type(
                    "CompletedProcess",
                    (),
                    {
                        "returncode": 0,
                        "stdout": "device for Thermal_GP58: usb://Gprinter/GP-58?serial=WTTING%20\n",
                        "stderr": "",
                    },
                )()
            if executable == "lpoptions" and command[1:2] == ["-p"]:
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
            if executable == "arecord" and command[1:2] == ["-l"]:
                return type(
                    "CompletedProcess",
                    (),
                    {
                        "returncode": 0,
                        "stdout": "card 3: CameraB409241 [USB Camera-B4.09.24.1], device 0: USB Audio [USB Audio]\n",
                        "stderr": "",
                    },
                )()
            if executable == "arecord" and command[1:2] == ["-L"]:
                return type(
                    "CompletedProcess",
                    (),
                    {
                        "returncode": 0,
                        "stdout": "default\nsysdefault:CARD=CameraB409241\nplughw:CARD=CameraB409241,DEV=0\n",
                        "stderr": "",
                    },
                )()
            if executable == "aplay" and command[1:2] == ["-L"]:
                return type(
                    "CompletedProcess",
                    (),
                    {
                        "returncode": 0,
                        "stdout": "default\nsysdefault:CARD=ThermalSpeaker\n",
                        "stderr": "",
                    },
                )()
            raise AssertionError(f"Unexpected command: {command}")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=temp_dir,
                printer_queue="Thermal_GP58",
                camera_device="/dev/null",
                camera_ffmpeg_path="/bin/sh",
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

            with patch(
                "twinr.ops.devices._resolve_executable",
                side_effect=lambda name: (
                    name
                    if str(name).startswith("/")
                    else f"/usr/bin/{name}"
                    if name in {"lpstat", "lpoptions", "arecord", "aplay"}
                    else None
                ),
            ):
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
        self.assertIn("card 3: CameraB409241", {fact.label: fact.value for fact in devices["audio_input"].facts}["ALSA capture cards"])
        self.assertNotEqual(
            {fact.label: fact.value for fact in devices["pir"].facts}["Last motion seen"],
            "not recorded in recent ops events",
        )

    def test_collect_device_overview_includes_respeaker_runtime_state(self) -> None:
        fake_snapshot = _fake_respeaker_snapshot(
            capture_ready=False,
            usb_visible=True,
            host_control_ready=False,
            transport_reason="permission_denied_or_transport_blocked",
            requires_elevated_permissions=True,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                audio_input_device="plughw:CARD=Array,DEV=0",
                proactive_audio_enabled=True,
                proactive_audio_input_device="plughw:CARD=Array,DEV=0",
            )
            with patch("twinr.ops.devices.capture_respeaker_primitive_snapshot", return_value=fake_snapshot):
                overview = collect_device_overview(config, event_store=TwinrOpsEventStore(Path(temp_dir) / "events.jsonl"))

        devices = {device.key: device for device in overview.devices}
        self.assertIn("respeaker", devices)
        self.assertEqual(devices["respeaker"].status, "warn")
        facts = {fact.label: fact.value for fact in devices["respeaker"].facts}
        self.assertEqual(facts["Probe state"], "usb_visible_no_capture")
        self.assertEqual(facts["Host control"], "no")
        self.assertEqual(facts["Transport reason"], "permission_denied_or_transport_blocked")
        self.assertIn("reSpeaker XVF3800", facts["USB device"])


if __name__ == "__main__":
    unittest.main()
