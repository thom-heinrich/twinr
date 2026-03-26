"""Manage bounded WiFi handovers for Bitcraze AI-Deck still captures.

Twinr may need to reach the AI-Deck stream over the deck's own access point
while still using the regular home network for OpenAI and remote-memory calls.
This helper performs one short-lived `nmcli` handover to the AI-Deck AP when
the stream is not currently reachable, then restores the previous WiFi
connection before control returns to the caller.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import shutil
import socket
import subprocess
import time
from typing import Any, Callable, Iterator

from twinr.agent.base_agent.config import TwinrConfig

_SubprocessRunner = Any
_SleepFn = Callable[[float], None]
_MonotonicFn = Callable[[], float]
_ReachabilityProbe = Callable[[str, int, float], bool]
_ACTIVE_DEVICE_STATUS_FIELDS = ("DEVICE", "TYPE", "STATE", "CONNECTION")
_DEFAULT_AIDECK_SSID = "WiFi streaming example"
_DEFAULT_CONNECT_TIMEOUT_SECONDS = 15.0
_DEFAULT_RESTORE_TIMEOUT_SECONDS = 15.0
_DEFAULT_REACHABILITY_TIMEOUT_SECONDS = 1.0
_AUTHORIZATION_MARKERS = ("not authorized", "insufficient privileges", "permission denied")
_CONNECTION_PROFILE_PREFIX = "twinr-aideck"


class AIDeckWifiHandoverError(RuntimeError):
    """Raised when the AI-Deck AP handover could not be completed safely."""


@dataclass(frozen=True, slots=True)
class ActiveWifiConnection:
    """Describe one currently visible WiFi device/connection pair."""

    interface: str
    connection_name: str | None


class AIDeckWifiConnectionManager:
    """Temporarily join the AI-Deck AP and restore the previous WiFi link."""

    def __init__(
        self,
        *,
        ssid: str = _DEFAULT_AIDECK_SSID,
        password: str | None = None,
        hidden: bool = True,
        interface: str | None = None,
        connect_timeout_seconds: float = _DEFAULT_CONNECT_TIMEOUT_SECONDS,
        restore_timeout_seconds: float = _DEFAULT_RESTORE_TIMEOUT_SECONDS,
        reachability_timeout_seconds: float = _DEFAULT_REACHABILITY_TIMEOUT_SECONDS,
        subprocess_runner: _SubprocessRunner = subprocess.run,
        sleep_fn: _SleepFn = time.sleep,
        monotonic_fn: _MonotonicFn = time.monotonic,
        reachability_probe: _ReachabilityProbe | None = None,
    ) -> None:
        self.ssid = str(ssid or _DEFAULT_AIDECK_SSID).strip() or _DEFAULT_AIDECK_SSID
        self.password = str(password or "").strip() or None
        self.hidden = bool(hidden)
        self.interface = str(interface or "").strip() or None
        self.connect_timeout_seconds = float(connect_timeout_seconds)
        self.restore_timeout_seconds = float(restore_timeout_seconds)
        self.reachability_timeout_seconds = float(reachability_timeout_seconds)
        self._subprocess_runner = subprocess_runner
        self._sleep = sleep_fn
        self._monotonic = monotonic_fn
        self._reachability_probe = reachability_probe or _default_reachability_probe
        if self.connect_timeout_seconds <= 0:
            raise ValueError("connect_timeout_seconds must be greater than zero")
        if self.restore_timeout_seconds <= 0:
            raise ValueError("restore_timeout_seconds must be greater than zero")
        if self.reachability_timeout_seconds <= 0:
            raise ValueError("reachability_timeout_seconds must be greater than zero")

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "AIDeckWifiConnectionManager":
        """Build the handover manager from optional Twinr config fields."""

        return cls(
            ssid=str(getattr(config, "aideck_wifi_ssid", _DEFAULT_AIDECK_SSID) or _DEFAULT_AIDECK_SSID),
            password=str(getattr(config, "aideck_wifi_password", "") or "").strip() or None,
            hidden=bool(getattr(config, "aideck_wifi_hidden", True)),
            interface=str(getattr(config, "aideck_wifi_interface", "") or "").strip() or None,
            connect_timeout_seconds=float(
                getattr(config, "aideck_wifi_connect_timeout_seconds", _DEFAULT_CONNECT_TIMEOUT_SECONDS)
            ),
            restore_timeout_seconds=float(
                getattr(config, "aideck_wifi_restore_timeout_seconds", _DEFAULT_RESTORE_TIMEOUT_SECONDS)
            ),
            reachability_timeout_seconds=float(
                getattr(
                    config,
                    "aideck_wifi_reachability_timeout_seconds",
                    _DEFAULT_REACHABILITY_TIMEOUT_SECONDS,
                )
            ),
        )

    @contextmanager
    def ensure_stream_ready(self, stream_host: str, stream_port: int) -> Iterator[None]:
        """Ensure the AI-Deck TCP endpoint is reachable for the enclosed work."""

        if self._stream_is_reachable(stream_host, stream_port):
            yield
            return

        previous = self._active_wifi_connection()
        if previous.connection_name == self.ssid:
            self._wait_for_stream(stream_host, stream_port, timeout_s=self.connect_timeout_seconds)
            yield
            return

        should_restore = previous.connection_name is not None
        try:
            self._connect_stream_ssid(previous.interface)
            self._wait_for_stream(stream_host, stream_port, timeout_s=self.connect_timeout_seconds)
            yield
        finally:
            if should_restore and previous.connection_name is not None:
                self._restore_previous_connection(previous)

    def _stream_is_reachable(self, stream_host: str, stream_port: int) -> bool:
        return self._reachability_probe(
            stream_host,
            stream_port,
            self.reachability_timeout_seconds,
        )

    def _wait_for_stream(self, stream_host: str, stream_port: int, *, timeout_s: float) -> None:
        self._wait_until(
            lambda: self._stream_is_reachable(stream_host, stream_port),
            timeout_s=timeout_s,
            timeout_message=(
                f"AI-Deck stream {stream_host}:{stream_port} did not become reachable "
                f"within {timeout_s:.1f}s"
            ),
        )

    def _restore_previous_connection(self, previous: ActiveWifiConnection) -> None:
        connection_name = previous.connection_name
        if connection_name is None:
            return
        self._run_nmcli(["connection", "up", connection_name, "ifname", previous.interface])
        self._wait_until(
            lambda: self._active_connection_name(previous.interface) == connection_name,
            timeout_s=self.restore_timeout_seconds,
            timeout_message=(
                f"Previous WiFi connection `{connection_name}` on {previous.interface} "
                f"did not recover within {self.restore_timeout_seconds:.1f}s"
            ),
            timeout_error_cls=AIDeckWifiHandoverError,
        )

    def _connect_stream_ssid(self, interface: str) -> None:
        profile_name = self._connection_profile_name(interface)
        if not self._connection_profile_exists(profile_name):
            self._run_nmcli(
                [
                    "connection",
                    "add",
                    "type",
                    "wifi",
                    "ifname",
                    interface,
                    "con-name",
                    profile_name,
                    "ssid",
                    self.ssid,
                ]
            )
        modify_args = [
            "connection",
            "modify",
            profile_name,
            "connection.autoconnect",
            "no",
            "802-11-wireless.hidden",
            "yes" if self.hidden else "no",
            "ipv4.method",
            "auto",
            "ipv6.method",
            "ignore",
        ]
        if self.password is not None:
            modify_args.extend(["wifi-sec.key-mgmt", "wpa-psk", "wifi-sec.psk", self.password])
        self._run_nmcli(modify_args)
        self._run_nmcli(["connection", "up", profile_name])

    @staticmethod
    def _connection_profile_name(interface: str) -> str:
        return f"{_CONNECTION_PROFILE_PREFIX}-{interface}"

    def _connection_profile_exists(self, profile_name: str) -> bool:
        completed = self._run_command(["nmcli", "connection", "show", profile_name])
        return completed.returncode == 0

    def _active_wifi_connection(self) -> ActiveWifiConnection:
        lines = self._run_nmcli(
            [
                "-t",
                "-f",
                ",".join(_ACTIVE_DEVICE_STATUS_FIELDS),
                "device",
                "status",
            ]
        ).splitlines()
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            device, device_type, state, connection = self._parse_nmcli_line(line)
            if device_type != "wifi":
                continue
            if self.interface is not None and device != self.interface:
                continue
            if state == "connected":
                return ActiveWifiConnection(
                    interface=device,
                    connection_name=connection or None,
                )
        if self.interface is not None:
            return ActiveWifiConnection(interface=self.interface, connection_name=None)
        raise AIDeckWifiHandoverError(
            "No connected WiFi interface is available for AI-Deck capture."
        )

    def _active_connection_name(self, interface: str) -> str | None:
        lines = self._run_nmcli(
            [
                "-t",
                "-f",
                ",".join(_ACTIVE_DEVICE_STATUS_FIELDS),
                "device",
                "status",
            ]
        ).splitlines()
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            device, device_type, state, connection = self._parse_nmcli_line(line)
            if device != interface or device_type != "wifi" or state != "connected":
                continue
            return connection or None
        return None

    @staticmethod
    def _parse_nmcli_line(line: str) -> tuple[str, str, str, str]:
        parts = line.split(":", 3)
        if len(parts) != 4:
            raise AIDeckWifiHandoverError(f"Unexpected nmcli device-status line: {line!r}")
        return tuple(part.strip() for part in parts)  # type: ignore[return-value]

    def _run_nmcli(self, args: list[str]) -> str:
        if shutil.which("nmcli") is None:
            raise AIDeckWifiHandoverError(
                "nmcli is required for AI-Deck WiFi handover but is not installed."
            )
        completed = self._run_command(["nmcli", *args])
        if completed.returncode == 0:
            return str(completed.stdout or "")

        message = (completed.stderr or completed.stdout or "").strip() or "nmcli command failed"
        if self._looks_like_authorization_error(message) and shutil.which("sudo") is not None:
            sudo_completed = self._run_command(["sudo", "-n", "nmcli", *args])
            if sudo_completed.returncode == 0:
                return str(sudo_completed.stdout or "")
            message = (sudo_completed.stderr or sudo_completed.stdout or "").strip() or message
        raise AIDeckWifiHandoverError(message)

    def _run_command(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        return self._subprocess_runner(
            args,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
        )

    @staticmethod
    def _looks_like_authorization_error(message: str) -> bool:
        normalized = str(message or "").strip().lower()
        return any(marker in normalized for marker in _AUTHORIZATION_MARKERS)

    def _wait_until(
        self,
        predicate: Callable[[], bool],
        *,
        timeout_s: float,
        timeout_message: str,
        timeout_error_cls: type[Exception] = TimeoutError,
    ) -> None:
        deadline = self._monotonic() + timeout_s
        while self._monotonic() < deadline:
            if predicate():
                return
            self._sleep(0.5)
        if predicate():
            return
        raise timeout_error_cls(timeout_message)


def _default_reachability_probe(host: str, port: int, timeout_s: float) -> bool:
    """Return whether one TCP endpoint can be opened within the timeout."""

    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False
