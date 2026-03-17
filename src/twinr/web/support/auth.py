"""Manage Twinr's file-backed web sign-in state and signed login cookies.

This module keeps the operator web login separate from route orchestration.
It owns the bootstrap ``admin/admin`` credential, password hashing, the
mandatory first-login password-change flag, and cookie signing helpers used by
the web app.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import base64
import binascii
import hashlib
import hmac
import json
from pathlib import Path
import secrets
import time
from typing import Final

from twinr.web.support.store import read_text_file, write_text_file

_SCHEMA_VERSION: Final[int] = 1
_DEFAULT_USERNAME: Final[str] = "admin"
_DEFAULT_PASSWORD: Final[str] = "admin"
_PASSWORD_HASH_ITERATIONS: Final[int] = 240_000
_PASSWORD_MIN_LENGTH: Final[int] = 8
_SESSION_MAX_AGE_SECONDS: Final[int] = 12 * 60 * 60
_SESSION_SCOPE: Final[bytes] = b"twinr.web.session.v1"
_SESSION_COOKIE_NAME: Final[str] = "twinr_control_session"


@dataclass(frozen=True, slots=True)
class WebAuthState:
    """Persisted web sign-in credential and policy state."""

    schema_version: int
    username: str
    password_hash: str
    password_salt: str
    must_change_password: bool
    updated_at: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready representation."""

        return {
            "schema_version": self.schema_version,
            "username": self.username,
            "password_hash": self.password_hash,
            "password_salt": self.password_salt,
            "must_change_password": self.must_change_password,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "WebAuthState":
        """Validate and normalize one persisted auth payload."""

        if not isinstance(payload, dict):
            raise ValueError("Web auth state payload must be an object.")
        schema_version = int(payload.get("schema_version", 0) or 0)
        username = str(payload.get("username", "") or "").strip()
        password_hash = str(payload.get("password_hash", "") or "").strip().lower()
        password_salt = str(payload.get("password_salt", "") or "").strip().lower()
        must_change_password = bool(payload.get("must_change_password", False))
        updated_at = str(payload.get("updated_at", "") or "").strip()
        if schema_version != _SCHEMA_VERSION:
            raise ValueError("Web auth state schema_version is invalid.")
        if not username:
            raise ValueError("Web auth username is required.")
        if not _is_hex_token(password_hash):
            raise ValueError("Web auth password_hash is invalid.")
        if not _is_hex_token(password_salt):
            raise ValueError("Web auth password_salt is invalid.")
        if not updated_at:
            raise ValueError("Web auth updated_at is required.")
        return cls(
            schema_version=schema_version,
            username=username,
            password_hash=password_hash,
            password_salt=password_salt,
            must_change_password=must_change_password,
            updated_at=updated_at,
        )


class FileBackedWebAuthStore:
    """Persist and update Twinr's managed web sign-in state."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()

    @classmethod
    def from_project_root(cls, project_root: str | Path) -> "FileBackedWebAuthStore":
        """Return the canonical managed web-auth store for one Twinr project."""

        return cls(Path(project_root).expanduser() / "state" / "web_auth.json")

    def load(self) -> WebAuthState | None:
        """Load the saved auth state, or ``None`` when it does not exist yet."""

        raw_payload = read_text_file(self.path)
        if not raw_payload.strip():
            return None
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Web auth state is not valid JSON: {self.path}") from exc
        try:
            return WebAuthState.from_dict(payload)
        except ValueError as exc:
            raise RuntimeError(f"Web auth state is malformed: {self.path}") from exc

    def save(self, state: WebAuthState) -> None:
        """Persist one complete auth state atomically."""

        payload = json.dumps(state.to_dict(), indent=2, sort_keys=True) + "\n"
        write_text_file(self.path, payload)

    def load_or_bootstrap(self) -> WebAuthState:
        """Load the auth state, creating the default bootstrap credential when missing."""

        existing = self.load()
        if existing is not None:
            return existing
        bootstrap_state = build_bootstrap_web_auth_state()
        self.save(bootstrap_state)
        return bootstrap_state

    def update_password(
        self,
        *,
        current_password: str,
        new_password: str,
        confirm_password: str,
    ) -> WebAuthState:
        """Verify the current password and persist one replacement password."""

        state = self.load_or_bootstrap()
        if not verify_web_auth_password(state, username=state.username, password=current_password):
            raise ValueError("The current password was not correct.")
        validate_new_web_auth_password(
            current_password=current_password,
            new_password=new_password,
            confirm_password=confirm_password,
        )
        updated = replace_web_auth_password(state, new_password=new_password)
        self.save(updated)
        return updated


def default_web_auth_username() -> str:
    """Return the fixed bootstrap username exposed to operators."""

    return _DEFAULT_USERNAME


def web_auth_password_min_length() -> int:
    """Return the minimum accepted managed web password length."""

    return _PASSWORD_MIN_LENGTH


def web_auth_session_cookie_name() -> str:
    """Return the managed-session cookie name used by the web portal."""

    return _SESSION_COOKIE_NAME


def web_auth_session_max_age_seconds() -> int:
    """Return the maximum managed-session age in seconds."""

    return _SESSION_MAX_AGE_SECONDS


def build_bootstrap_web_auth_state() -> WebAuthState:
    """Build the first-login ``admin/admin`` credential state."""

    salt = secrets.token_hex(16)
    return WebAuthState(
        schema_version=_SCHEMA_VERSION,
        username=_DEFAULT_USERNAME,
        password_hash=_hash_password(_DEFAULT_PASSWORD, salt=salt),
        password_salt=salt,
        must_change_password=True,
        updated_at=_utc_now_iso(),
    )


def verify_web_auth_password(state: WebAuthState, *, username: str, password: str) -> bool:
    """Return whether one username/password pair matches the saved credential."""

    normalized_username = str(username or "").strip()
    if not secrets.compare_digest(normalized_username, state.username):
        return False
    candidate_hash = _hash_password(password, salt=state.password_salt)
    return secrets.compare_digest(candidate_hash, state.password_hash)


def validate_new_web_auth_password(
    *,
    current_password: str,
    new_password: str,
    confirm_password: str,
) -> None:
    """Validate a new password before it is persisted."""

    if not new_password:
        raise ValueError("Enter a new password.")
    if len(new_password) < _PASSWORD_MIN_LENGTH:
        raise ValueError(f"The new password must be at least {_PASSWORD_MIN_LENGTH} characters long.")
    if new_password != confirm_password:
        raise ValueError("The new passwords did not match.")
    if secrets.compare_digest(new_password, current_password):
        raise ValueError("Choose a password that is different from the current one.")


def replace_web_auth_password(state: WebAuthState, *, new_password: str) -> WebAuthState:
    """Return a copy of ``state`` with one new password and cleared bootstrap flag."""

    new_salt = secrets.token_hex(16)
    return WebAuthState(
        schema_version=_SCHEMA_VERSION,
        username=state.username,
        password_hash=_hash_password(new_password, salt=new_salt),
        password_salt=new_salt,
        must_change_password=False,
        updated_at=_utc_now_iso(),
    )


def build_web_auth_session_cookie(state: WebAuthState, *, username: str, issued_at: int | None = None) -> str:
    """Build one signed stateless session cookie value for the managed login."""

    issued_timestamp = int(time.time() if issued_at is None else issued_at)
    normalized_username = str(username or "").strip()
    payload = f"{normalized_username}\n{issued_timestamp}"
    signature = hmac.new(_session_key(state), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    token = f"{payload}\n{signature}".encode("utf-8")
    return base64.urlsafe_b64encode(token).decode("ascii")


def load_authenticated_web_session(
    state: WebAuthState,
    cookie_value: str | None,
    *,
    now: int | None = None,
) -> str | None:
    """Return the authenticated username from one signed session cookie."""

    if not cookie_value:
        return None
    try:
        decoded = base64.urlsafe_b64decode(cookie_value.encode("ascii"))
        payload = decoded.decode("utf-8")
    except (ValueError, binascii.Error, UnicodeDecodeError):
        return None
    username, issued_raw, signature = payload.split("\n", 2) if payload.count("\n") == 2 else ("", "", "")
    if not username or not issued_raw or not signature:
        return None
    try:
        issued_timestamp = int(issued_raw)
    except ValueError:
        return None
    observed_now = int(time.time() if now is None else now)
    if issued_timestamp > observed_now + 60:
        return None
    if observed_now - issued_timestamp > _SESSION_MAX_AGE_SECONDS:
        return None
    signed_payload = f"{username}\n{issued_timestamp}"
    expected_signature = hmac.new(
        _session_key(state),
        signed_payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not secrets.compare_digest(signature, expected_signature):
        return None
    if not secrets.compare_digest(username, state.username):
        return None
    return username


def _hash_password(password: str, *, salt: str) -> str:
    """Return the stable PBKDF2 hash used for one password."""

    salt_bytes = bytes.fromhex(salt)
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        str(password).encode("utf-8"),
        salt_bytes,
        _PASSWORD_HASH_ITERATIONS,
    )
    return derived.hex()


def _session_key(state: WebAuthState) -> bytes:
    """Derive the HMAC key used for stateless managed login cookies."""

    return hashlib.sha256(
        _SESSION_SCOPE
        + b"\n"
        + state.username.encode("utf-8")
        + b"\n"
        + state.password_salt.encode("ascii")
        + b"\n"
        + state.password_hash.encode("ascii")
    ).digest()


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 form."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _is_hex_token(value: str) -> bool:
    """Return whether ``value`` is a non-empty lowercase/uppercase hex token."""

    if not value or len(value) % 2:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True
