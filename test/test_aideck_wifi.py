from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.aideck_wifi import AIDeckWifiConnectionManager, AIDeckWifiHandoverError


class AIDeckWifiConnectionManagerTests(unittest.TestCase):
    def test_ensure_stream_ready_is_noop_when_stream_is_already_reachable(self) -> None:
        manager = AIDeckWifiConnectionManager(
            reachability_probe=lambda _host, _port, _timeout: True,
        )

        with manager.ensure_stream_ready("192.168.4.1", 5000):
            pass

    def test_ensure_stream_ready_switches_to_aideck_ssid_and_restores_previous_connection(self) -> None:
        state = {"connection": "preconfigured"}
        commands: list[list[str]] = []

        def _runner(args, **_kwargs):
            commands.append(list(args))
            if args[:4] == ["nmcli", "-t", "-f", "DEVICE,TYPE,STATE,CONNECTION"]:
                stdout = f"wlan0:wifi:connected:{state['connection']}\n"
                return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")
            if args[:4] == ["nmcli", "connection", "show", "twinr-aideck-wlan0"]:
                return subprocess.CompletedProcess(args, 10, stdout="", stderr="not found")
            if args[:8] == [
                "nmcli",
                "connection",
                "add",
                "type",
                "wifi",
                "ifname",
                "wlan0",
                "con-name",
            ]:
                return subprocess.CompletedProcess(args, 0, stdout="connected\n", stderr="")
            if args[:4] == ["nmcli", "connection", "modify", "twinr-aideck-wlan0"]:
                return subprocess.CompletedProcess(args, 0, stdout="modified\n", stderr="")
            if args[:4] == ["nmcli", "connection", "up", "twinr-aideck-wlan0"]:
                state["connection"] = "WiFi streaming example"
                return subprocess.CompletedProcess(args, 0, stdout="connected\n", stderr="")
            if args[:3] == ["nmcli", "connection", "up"]:
                state["connection"] = args[3]
                return subprocess.CompletedProcess(args, 0, stdout="restored\n", stderr="")
            raise AssertionError(f"Unexpected nmcli command: {args}")

        manager = AIDeckWifiConnectionManager(
            subprocess_runner=_runner,
            sleep_fn=lambda _seconds: None,
            reachability_probe=lambda _host, _port, _timeout: state["connection"] == "WiFi streaming example",
        )

        with patch("twinr.hardware.aideck_wifi.shutil.which", return_value="/usr/bin/nmcli"):
            with manager.ensure_stream_ready("192.168.4.1", 5000):
                self.assertEqual(state["connection"], "WiFi streaming example")

        self.assertEqual(state["connection"], "preconfigured")
        self.assertEqual(
            commands,
            [
                ["nmcli", "-t", "-f", "DEVICE,TYPE,STATE,CONNECTION", "device", "status"],
                ["nmcli", "connection", "show", "twinr-aideck-wlan0"],
                [
                    "nmcli",
                    "connection",
                    "add",
                    "type",
                    "wifi",
                    "ifname",
                    "wlan0",
                    "con-name",
                    "twinr-aideck-wlan0",
                    "ssid",
                    "WiFi streaming example",
                ],
                [
                    "nmcli",
                    "connection",
                    "modify",
                    "twinr-aideck-wlan0",
                    "connection.autoconnect",
                    "no",
                    "802-11-wireless.hidden",
                    "yes",
                    "ipv4.method",
                    "auto",
                    "ipv6.method",
                    "ignore",
                ],
                ["nmcli", "connection", "up", "twinr-aideck-wlan0"],
                ["nmcli", "connection", "up", "preconfigured", "ifname", "wlan0"],
                ["nmcli", "-t", "-f", "DEVICE,TYPE,STATE,CONNECTION", "device", "status"],
            ],
        )

    def test_ensure_stream_ready_requires_a_connected_wifi_interface(self) -> None:
        def _runner(args, **_kwargs):
            del _kwargs
            return subprocess.CompletedProcess(
                args,
                0,
                stdout="eth0:ethernet:connected:uplink\n",
                stderr="",
            )

        manager = AIDeckWifiConnectionManager(
            subprocess_runner=_runner,
            sleep_fn=lambda _seconds: None,
            reachability_probe=lambda _host, _port, _timeout: False,
        )

        with patch("twinr.hardware.aideck_wifi.shutil.which", return_value="/usr/bin/nmcli"):
            with self.assertRaises(AIDeckWifiHandoverError):
                with manager.ensure_stream_ready("192.168.4.1", 5000):
                    pass

    def test_connect_falls_back_to_sudo_nmcli_when_direct_nmcli_is_not_authorized(self) -> None:
        state = {"connection": "preconfigured"}
        commands: list[list[str]] = []

        def _runner(args, **_kwargs):
            commands.append(list(args))
            if args[:4] == ["nmcli", "-t", "-f", "DEVICE,TYPE,STATE,CONNECTION"]:
                stdout = f"wlan0:wifi:connected:{state['connection']}\n"
                return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")
            if args[:4] == ["nmcli", "connection", "show", "twinr-aideck-wlan0"]:
                return subprocess.CompletedProcess(args, 0, stdout="profile\n", stderr="")
            if args[:4] == ["nmcli", "connection", "modify", "twinr-aideck-wlan0"]:
                return subprocess.CompletedProcess(
                    args,
                    10,
                    stdout="",
                    stderr="Error: Connection activation failed: Not authorized to control networking.",
                )
            if args[:6] == ["sudo", "-n", "nmcli", "connection", "modify", "twinr-aideck-wlan0"]:
                return subprocess.CompletedProcess(args, 0, stdout="modified\n", stderr="")
            if args[:4] == ["nmcli", "connection", "up", "twinr-aideck-wlan0"]:
                return subprocess.CompletedProcess(
                    args,
                    10,
                    stdout="",
                    stderr="Error: Connection activation failed: Not authorized to control networking.",
                )
            if args[:6] == ["sudo", "-n", "nmcli", "connection", "up", "twinr-aideck-wlan0"]:
                state["connection"] = "WiFi streaming example"
                return subprocess.CompletedProcess(args, 0, stdout="connected\n", stderr="")
            if args[:3] == ["nmcli", "connection", "up"]:
                state["connection"] = args[3]
                return subprocess.CompletedProcess(args, 0, stdout="restored\n", stderr="")
            raise AssertionError(f"Unexpected nmcli command: {args}")

        manager = AIDeckWifiConnectionManager(
            subprocess_runner=_runner,
            sleep_fn=lambda _seconds: None,
            reachability_probe=lambda _host, _port, _timeout: state["connection"] == "WiFi streaming example",
        )

        with patch("twinr.hardware.aideck_wifi.shutil.which", side_effect=lambda name: f"/usr/bin/{name}"):
            with manager.ensure_stream_ready("192.168.4.1", 5000):
                self.assertEqual(state["connection"], "WiFi streaming example")

        self.assertEqual(state["connection"], "preconfigured")
        self.assertIn(
            [
                "sudo",
                "-n",
                "nmcli",
                "connection",
                "up",
                "twinr-aideck-wlan0",
            ],
            commands,
        )


if __name__ == "__main__":
    unittest.main()
