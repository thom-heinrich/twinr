# CHANGELOG: 2026-03-28
# BUG-1: Bound nmcli activation/deactivation with --wait so configured timeouts actually apply.
# BUG-2: Replaced fragile localized/escaped `nmcli device status` parsing with C-locale structured queries and UUID-based restore.
# BUG-3: The AI-Deck handover profile is now rebuilt from the live config on every handover, so SSID/security changes take effect immediately.
# BUG-4: Added process/thread handover locking to prevent concurrent WiFi flaps and restore races.
# SEC-1: WiFi secrets are no longer passed on the process command line.
# SEC-2: AI-Deck profiles are now created in-memory only and WPA secrets are marked not-saved to avoid persisting credentials on disk.
# IMP-1: Added targeted SSID rescans for hidden APs and optional BSSID pinning for more reliable Pi-class associations.
# IMP-2: Temporary AI-Deck profiles default to never becoming the default route, preserving other uplinks when present.
# IMP-3: Cleanup no longer force-restores the previous network if some other actor changed WiFi while the context was active.
# BREAKING: The helper now uses ephemeral in-memory AI-Deck profiles instead of persistent `twinr-aideck-*` profiles.
# BREAKING: Temporary AI-Deck profiles set ipv4/ipv6.never-default=yes by default; set `aideck_wifi_never_default=False`
# BREAKING: in TwinrConfig (or `never_default_routes=False` in code) to restore the old route-stealing behavior.

"""Manage bounded WiFi handovers for Bitcraze AI-Deck still captures.

Twinr may need to reach the AI-Deck stream over the deck's own access point
while still using the regular home network for OpenAI and remote-memory calls.
This helper performs one short-lived NetworkManager handover to the AI-Deck AP
when the stream is not currently reachable, then restores the previous WiFi
connection before control returns to the caller.

Compared to the earlier version, this implementation is designed for real Pi
deployments:

* it bounds NetworkManager operations with explicit timeouts;
* it serializes handovers across threads/processes to prevent WiFi flapping;
* it uses in-memory profiles and activation secret files instead of leaking PSKs
  via process arguments or persisting them unnecessarily on disk;
* it restores by UUID when possible, which is safer than restoring by
  connection name.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import math
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable, IO, Iterator

from twinr.agent.base_agent.config import TwinrConfig

_SubprocessRunner = Any
_SleepFn = Callable[[float], None]
_MonotonicFn = Callable[[], float]
_ReachabilityProbe = Callable[[str, int, float], bool]

_DEFAULT_AIDECK_SSID = "WiFi streaming example"
_DEFAULT_CONNECT_TIMEOUT_SECONDS = 15.0
_DEFAULT_RESTORE_TIMEOUT_SECONDS = 15.0
_DEFAULT_REACHABILITY_TIMEOUT_SECONDS = 1.0
_DEFAULT_RESCAN_BEFORE_CONNECT = True
_DEFAULT_NEVER_DEFAULT_ROUTES = True
_DEFAULT_CLEANUP_PROFILE = True
_DEFAULT_ALLOW_SUDO_FALLBACK = True
_AUTHORIZATION_MARKERS = ("not authorized", "insufficient privileges", "permission denied")
_CONNECTION_PROFILE_PREFIX = "twinr-aideck"
_DEFAULT_LOCK_PATH = os.path.join(tempfile.gettempdir(), "twinr-aideck-wifi.lock")
_NM_DEVICE_TYPE_WIFI = "wifi"
_NM_DEVICE_STATE_ACTIVATED = 100
_NM_RESCAN_TIMEOUT_SECONDS = 5.0


class AIDeckWifiHandoverError(RuntimeError):
    """Raised when the AI-Deck AP handover could not be completed safely."""


@dataclass(frozen=True, slots=True)
class ActiveWifiConnection:
    """Describe one WiFi interface as NetworkManager currently sees it."""

    interface: str
    connection_name: str | None
    connection_uuid: str | None
    ssid: str | None
    state_code: int | None

    @property
    def is_connected(self) -> bool:
        return self.state_code == _NM_DEVICE_STATE_ACTIVATED


@dataclass(slots=True)
class _SharedLockState:
    thread_lock: threading.RLock
    handle: IO[str] | None = None
    depth: int = 0


class AIDeckWifiConnectionManager:
    """Temporarily join the AI-Deck AP and restore the previous WiFi link."""

    _lock_registry_guard = threading.Lock()
    _lock_registry: dict[str, _SharedLockState] = {}

    def __init__(
        self,
        *,
        ssid: str = _DEFAULT_AIDECK_SSID,
        password: str | None = None,
        hidden: bool = True,
        interface: str | None = None,
        bssid: str | None = None,
        connect_timeout_seconds: float = _DEFAULT_CONNECT_TIMEOUT_SECONDS,
        restore_timeout_seconds: float = _DEFAULT_RESTORE_TIMEOUT_SECONDS,
        reachability_timeout_seconds: float = _DEFAULT_REACHABILITY_TIMEOUT_SECONDS,
        rescan_before_connect: bool = _DEFAULT_RESCAN_BEFORE_CONNECT,
        never_default_routes: bool = _DEFAULT_NEVER_DEFAULT_ROUTES,
        cleanup_profile: bool = _DEFAULT_CLEANUP_PROFILE,
        allow_sudo_fallback: bool = _DEFAULT_ALLOW_SUDO_FALLBACK,
        lock_path: str = _DEFAULT_LOCK_PATH,
        subprocess_runner: _SubprocessRunner = subprocess.run,
        sleep_fn: _SleepFn = time.sleep,
        monotonic_fn: _MonotonicFn = time.monotonic,
        reachability_probe: _ReachabilityProbe | None = None,
    ) -> None:
        self.ssid = str(ssid or _DEFAULT_AIDECK_SSID).strip() or _DEFAULT_AIDECK_SSID
        self.password = str(password or "").strip() or None
        self.hidden = bool(hidden)
        self.interface = str(interface or "").strip() or None
        self.bssid = str(bssid or "").strip() or None
        self.connect_timeout_seconds = float(connect_timeout_seconds)
        self.restore_timeout_seconds = float(restore_timeout_seconds)
        self.reachability_timeout_seconds = float(reachability_timeout_seconds)
        self.rescan_before_connect = bool(rescan_before_connect)
        self.never_default_routes = bool(never_default_routes)
        self.cleanup_profile = bool(cleanup_profile)
        self.allow_sudo_fallback = bool(allow_sudo_fallback)
        self.lock_path = str(lock_path or _DEFAULT_LOCK_PATH).strip() or _DEFAULT_LOCK_PATH
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
        if self.password is not None and ("\n" in self.password or "\r" in self.password):
            raise ValueError("password must not contain line breaks")

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "AIDeckWifiConnectionManager":
        """Build the handover manager from optional Twinr config fields."""

        return cls(
            ssid=str(getattr(config, "aideck_wifi_ssid", _DEFAULT_AIDECK_SSID) or _DEFAULT_AIDECK_SSID),
            password=str(getattr(config, "aideck_wifi_password", "") or "").strip() or None,
            hidden=bool(getattr(config, "aideck_wifi_hidden", True)),
            interface=str(getattr(config, "aideck_wifi_interface", "") or "").strip() or None,
            bssid=str(getattr(config, "aideck_wifi_bssid", "") or "").strip() or None,
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
            rescan_before_connect=bool(
                getattr(config, "aideck_wifi_rescan_before_connect", _DEFAULT_RESCAN_BEFORE_CONNECT)
            ),
            never_default_routes=bool(
                getattr(config, "aideck_wifi_never_default", _DEFAULT_NEVER_DEFAULT_ROUTES)
            ),
            cleanup_profile=bool(
                getattr(config, "aideck_wifi_cleanup_profile", _DEFAULT_CLEANUP_PROFILE)
            ),
            allow_sudo_fallback=bool(
                getattr(config, "aideck_wifi_allow_sudo_fallback", _DEFAULT_ALLOW_SUDO_FALLBACK)
            ),
        )

    @contextmanager
    def ensure_stream_ready(self, stream_host: str, stream_port: int) -> Iterator[None]:
        """Ensure the AI-Deck TCP endpoint is reachable for the enclosed work."""

        with self._handover_lock():
            if self._stream_is_reachable(stream_host, stream_port):
                yield
                return

            previous = self._active_wifi_connection()
            if self._is_target_connection(previous):
                self._wait_for_stream(stream_host, stream_port, timeout_s=self.connect_timeout_seconds)
                yield
                return

            should_restore = previous.connection_uuid is not None or previous.connection_name is not None
            managed_profile_name = self._connection_profile_name(previous.interface)
            managed_profile_uuid: str | None = None

            try:
                managed_profile_uuid = self._connect_stream_ssid(
                    previous.interface,
                    profile_name=managed_profile_name,
                    timeout_s=self.connect_timeout_seconds,
                )
                self._wait_for_stream(stream_host, stream_port, timeout_s=self.connect_timeout_seconds)
                yield
            finally:
                current_before_restore = self._safe_wifi_connection(previous.interface)
                try:
                    if should_restore and self._should_restore_previous(
                        previous=previous,
                        current=current_before_restore,
                        managed_profile_name=managed_profile_name,
                        managed_profile_uuid=managed_profile_uuid,
                    ):
                        self._restore_previous_connection(previous)
                finally:
                    if self.cleanup_profile and managed_profile_uuid is not None:
                        current_after_restore = self._safe_wifi_connection(previous.interface)
                        if (
                            current_after_restore is None
                            or current_after_restore.connection_uuid != managed_profile_uuid
                        ):
                            self._delete_connection_profile(managed_profile_uuid)

    def _stream_is_reachable(self, stream_host: str, stream_port: int) -> bool:
        return self._reachability_probe(
            stream_host,
            int(stream_port),
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
        if previous.connection_uuid is None and previous.connection_name is None:
            return

        current = self._safe_wifi_connection(previous.interface)
        if current is not None:
            if previous.connection_uuid and current.connection_uuid == previous.connection_uuid:
                return
            if previous.connection_uuid is None and previous.connection_name == current.connection_name:
                return

        restore_timeout_int = _nmcli_wait_seconds(self.restore_timeout_seconds)
        last_error: AIDeckWifiHandoverError | None = None

        if previous.connection_uuid is not None:
            try:
                self._run_nmcli(
                    [
                        "--wait",
                        str(restore_timeout_int),
                        "connection",
                        "up",
                        "uuid",
                        previous.connection_uuid,
                        "ifname",
                        previous.interface,
                    ]
                )
            except AIDeckWifiHandoverError as exc:
                last_error = exc
            else:
                last_error = None

        if last_error is not None and previous.connection_name is not None:
            self._run_nmcli(
                [
                    "--wait",
                    str(restore_timeout_int),
                    "connection",
                    "up",
                    "id",
                    previous.connection_name,
                    "ifname",
                    previous.interface,
                ]
            )
            last_error = None

        if last_error is not None:
            raise last_error

        self._wait_until(
            lambda: self._matches_previous(self._safe_wifi_connection(previous.interface), previous),
            timeout_s=self.restore_timeout_seconds,
            timeout_message=(
                f"Previous WiFi connection `{previous.connection_name or previous.connection_uuid}` "
                f"on {previous.interface} did not recover within {self.restore_timeout_seconds:.1f}s"
            ),
            timeout_error_cls=AIDeckWifiHandoverError,
        )

    def _connect_stream_ssid(
        self,
        interface: str,
        *,
        profile_name: str,
        timeout_s: float,
    ) -> str:
        self._best_effort_rescan(interface)
        self._best_effort_delete_connection_profile_by_name(profile_name)

        add_args = [
            "connection",
            "add",
            "save",
            "no",
            "type",
            "wifi",
            "ifname",
            interface,
            "con-name",
            profile_name,
            "ssid",
            self.ssid,
            "connection.autoconnect",
            "no",
            "802-11-wireless.hidden",
            "yes" if self.hidden else "no",
            "ipv4.method",
            "auto",
            "ipv6.method",
            "ignore",
        ]
        if self.never_default_routes:
            add_args.extend(
                [
                    "ipv4.never-default",
                    "yes",
                    "ipv6.never-default",
                    "yes",
                ]
            )
        if self.password is not None:
            add_args.extend(
                [
                    "wifi-sec.key-mgmt",
                    "wpa-psk",
                    "wifi-sec.psk-flags",
                    "0x2",
                ]
            )

        self._run_nmcli(add_args)
        profile_uuid = self._connection_uuid_for_profile(profile_name)
        connect_timeout_int = _nmcli_wait_seconds(timeout_s)

        with self._activation_secret_file() as passwd_file:
            up_args = [
                "--wait",
                str(connect_timeout_int),
                "connection",
                "up",
                "uuid",
                profile_uuid,
                "ifname",
                interface,
            ]
            if self.bssid is not None:
                up_args.extend(["ap", self.bssid])
            if passwd_file is not None:
                up_args.extend(["passwd-file", passwd_file])
            self._run_nmcli(up_args)

        self._wait_until(
            lambda: self._matches_profile(self._safe_wifi_connection(interface), profile_uuid, profile_name),
            timeout_s=timeout_s,
            timeout_message=(
                f"AI-Deck WiFi profile `{profile_name}` on {interface} did not activate "
                f"within {timeout_s:.1f}s"
            ),
            timeout_error_cls=AIDeckWifiHandoverError,
        )
        return profile_uuid

    def _active_wifi_connection(self) -> ActiveWifiConnection:
        candidate_interfaces = self._candidate_wifi_interfaces()
        first_wifi: ActiveWifiConnection | None = None

        for interface in candidate_interfaces:
            connection = self._wifi_connection_state(interface)
            if connection.is_connected:
                return connection
            if first_wifi is None:
                first_wifi = connection

        if first_wifi is not None:
            return first_wifi

        if self.interface is not None:
            raise AIDeckWifiHandoverError(
                f"WiFi interface `{self.interface}` is not available or not managed by NetworkManager."
            )
        raise AIDeckWifiHandoverError(
            "No WiFi interface managed by NetworkManager is available for AI-Deck capture."
        )

    def _safe_wifi_connection(self, interface: str) -> ActiveWifiConnection | None:
        try:
            return self._wifi_connection_state(interface)
        except AIDeckWifiHandoverError:
            return None

    def _candidate_wifi_interfaces(self) -> list[str]:
        if self.interface is not None:
            return [self.interface]

        lines = self._run_nmcli(["-g", "DEVICE,TYPE", "device", "status"]).splitlines()
        interfaces: list[str] = []
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            device, _, device_type = line.partition(":")
            if device_type.strip() != _NM_DEVICE_TYPE_WIFI:
                continue
            interface = device.strip()
            if interface:
                interfaces.append(interface)

        return interfaces

    def _wifi_connection_state(self, interface: str) -> ActiveWifiConnection:
        data = self._nmcli_properties(
            [
                "-m",
                "multiline",
                "-f",
                "GENERAL.DEVICE,GENERAL.TYPE,GENERAL.STATE,GENERAL.CONNECTION,GENERAL.CON-UUID",
                "device",
                "show",
                interface,
            ]
        )
        device = _none_if_blank(data.get("GENERAL.DEVICE")) or interface
        device_type = (_none_if_blank(data.get("GENERAL.TYPE")) or "").strip().lower()
        if device_type != _NM_DEVICE_TYPE_WIFI:
            raise AIDeckWifiHandoverError(
                f"Interface `{interface}` is not a NetworkManager-managed WiFi device."
            )

        connection_uuid = _none_if_blank(data.get("GENERAL.CON-UUID"))
        ssid = self._connection_ssid(connection_uuid) if connection_uuid is not None else None

        return ActiveWifiConnection(
            interface=device,
            connection_name=_none_if_blank(data.get("GENERAL.CONNECTION")),
            connection_uuid=connection_uuid,
            ssid=ssid,
            state_code=_parse_nm_state_code(data.get("GENERAL.STATE")),
        )

    def _connection_uuid_for_profile(self, profile_name: str) -> str:
        output = self._run_nmcli(["-g", "connection.uuid", "connection", "show", "id", profile_name])
        for raw_line in output.splitlines():
            uuid = raw_line.strip()
            if uuid:
                return uuid
        raise AIDeckWifiHandoverError(f"Could not resolve UUID for WiFi profile `{profile_name}`")

    def _connection_ssid(self, connection_uuid: str) -> str | None:
        output = self._run_nmcli(
            ["-g", "802-11-wireless.ssid", "connection", "show", "uuid", connection_uuid]
        )
        for raw_line in output.splitlines():
            if raw_line == "":
                continue
            return _none_if_blank(raw_line)
        return None

    def _best_effort_rescan(self, interface: str) -> None:
        if not self.rescan_before_connect:
            return

        args = [
            "--wait",
            str(_nmcli_wait_seconds(min(self.connect_timeout_seconds, _NM_RESCAN_TIMEOUT_SECONDS))),
            "device",
            "wifi",
            "rescan",
            "ifname",
            interface,
        ]
        if self.ssid:
            args.extend(["ssid", self.ssid])

        try:
            self._run_nmcli(args)
        except AIDeckWifiHandoverError:
            # A targeted rescan materially improves hidden-SSID reliability, but
            # a failed rescan does not necessarily prevent activation.
            return

    def _best_effort_delete_connection_profile_by_name(self, profile_name: str) -> None:
        try:
            self._run_nmcli(["--wait", "5", "connection", "delete", "id", profile_name])
        except AIDeckWifiHandoverError:
            return

    def _delete_connection_profile(self, connection_uuid: str) -> None:
        try:
            self._run_nmcli(["--wait", "5", "connection", "delete", "uuid", connection_uuid])
        except AIDeckWifiHandoverError:
            return

    def _connection_profile_name(self, interface: str) -> str:
        # Keep the profile name deterministic per interface and target AP, but
        # decouple it from passwords so credential rotation does not rename the
        # profile.
        digest = hashlib.sha256(f"{interface}\0{self.ssid}\0{self.bssid or ''}".encode("utf-8")).hexdigest()[:8]
        return f"{_CONNECTION_PROFILE_PREFIX}-{interface}-{digest}"

    def _is_target_connection(self, connection: ActiveWifiConnection) -> bool:
        return connection.ssid == self.ssid

    def _should_restore_previous(
        self,
        *,
        previous: ActiveWifiConnection,
        current: ActiveWifiConnection | None,
        managed_profile_name: str,
        managed_profile_uuid: str | None,
    ) -> bool:
        if current is None:
            return True

        if self._matches_previous(current, previous):
            return False

        if managed_profile_uuid is not None and current.connection_uuid == managed_profile_uuid:
            return True
        if current.connection_name == managed_profile_name:
            return True
        if current.connection_uuid is None and current.connection_name is None:
            return True
        if current.ssid == self.ssid:
            return True
        return False

    @staticmethod
    def _matches_previous(current: ActiveWifiConnection | None, previous: ActiveWifiConnection) -> bool:
        if current is None:
            return False
        if previous.connection_uuid is not None:
            return current.connection_uuid == previous.connection_uuid
        if previous.connection_name is not None:
            return current.connection_name == previous.connection_name
        return False

    @staticmethod
    def _matches_profile(
        current: ActiveWifiConnection | None,
        profile_uuid: str,
        profile_name: str,
    ) -> bool:
        if current is None or not current.is_connected:
            return False
        if current.connection_uuid == profile_uuid:
            return True
        return current.connection_name == profile_name

    @contextmanager
    def _activation_secret_file(self) -> Iterator[str | None]:
        if self.password is None:
            yield None
            return

        fd, path = tempfile.mkstemp(prefix="twinr-aideck-", suffix=".passwd")
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(f"802-11-wireless-security.psk:{self.password}\n")
            yield path
        finally:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    @contextmanager
    def _handover_lock(self) -> Iterator[None]:
        state = self._shared_lock_state(self.lock_path)
        state.thread_lock.acquire()
        try:
            self._enter_process_lock(state)
            try:
                yield
            finally:
                self._leave_process_lock(state)
        finally:
            state.thread_lock.release()

    @classmethod
    def _shared_lock_state(cls, lock_path: str) -> _SharedLockState:
        with cls._lock_registry_guard:
            state = cls._lock_registry.get(lock_path)
            if state is None:
                state = _SharedLockState(thread_lock=threading.RLock())
                cls._lock_registry[lock_path] = state
            return state

    def _enter_process_lock(self, state: _SharedLockState) -> None:
        with self._lock_registry_guard:
            if state.depth == 0:
                os.makedirs(os.path.dirname(self.lock_path) or ".", exist_ok=True)
                handle = open(self.lock_path, "a+", encoding="utf-8")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                except OSError as exc:
                    handle.close()
                    raise AIDeckWifiHandoverError(
                        f"Could not acquire AI-Deck WiFi handover lock `{self.lock_path}`: {exc}"
                    ) from exc
                state.handle = handle
            state.depth += 1

    def _leave_process_lock(self, state: _SharedLockState) -> None:
        with self._lock_registry_guard:
            state.depth -= 1
            if state.depth != 0:
                return
            handle = state.handle
            state.handle = None
            if handle is None:
                return
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    def _nmcli_properties(self, args: list[str]) -> dict[str, str]:
        output = self._run_nmcli(args)
        data: dict[str, str] = {}
        for raw_line in output.splitlines():
            if not raw_line.strip() or ":" not in raw_line:
                continue
            key, value = raw_line.split(":", 1)
            data[key.strip()] = value.lstrip()
        return data

    def _run_nmcli(self, args: list[str]) -> str:
        nmcli_path = shutil.which("nmcli")
        if nmcli_path is None:
            if self._subprocess_runner is subprocess.run:
                raise AIDeckWifiHandoverError(
                    "nmcli is required for AI-Deck WiFi handover but is not installed."
                )
            nmcli_path = "nmcli"

        completed = self._run_command([nmcli_path, *args])
        if completed.returncode == 0:
            return str(completed.stdout or "")

        message = (completed.stderr or completed.stdout or "").strip() or "nmcli command failed"
        if (
            self.allow_sudo_fallback
            and self._looks_like_authorization_error(message)
        ):
            sudo_path = shutil.which("sudo")
            if sudo_path is not None or self._subprocess_runner is not subprocess.run:
                sudo_completed = self._run_command([sudo_path or "sudo", "-n", nmcli_path, *args])
                if sudo_completed.returncode == 0:
                    return str(sudo_completed.stdout or "")
                message = (sudo_completed.stderr or sudo_completed.stdout or "").strip() or message
        raise AIDeckWifiHandoverError(message)

    def _run_command(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["LC_ALL"] = "C"
        env["LANG"] = "C"
        env.setdefault("NO_COLOR", "1")
        return self._subprocess_runner(
            args,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
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


def _parse_nm_state_code(value: str | None) -> int | None:
    token = str(value or "").strip().split(" ", 1)[0]
    try:
        return int(token)
    except ValueError:
        return None


def _none_if_blank(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized or normalized == "--":
        return None
    return normalized


def _nmcli_wait_seconds(timeout_s: float) -> int:
    return max(1, int(math.ceil(timeout_s)))


def _default_reachability_probe(host: str, port: int, timeout_s: float) -> bool:
    """Return whether one TCP endpoint can be opened within the timeout."""

    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False