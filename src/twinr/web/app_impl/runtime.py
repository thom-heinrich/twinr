"""Shared runtime state for the refactored Twinr web app package."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from twinr.web.context import WebAppContext
from twinr.web.support.channel_onboarding import InProcessChannelPairingRegistry
from twinr.web.support.whatsapp import WhatsAppPairingCoordinator


class SurfaceProxy:
    """Resolve compatibility-surface attributes from the legacy wrapper module."""

    def __init__(self, module: ModuleType) -> None:
        self._module = module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)


@dataclass(frozen=True)
class SecurityConfig:
    """Request-guard configuration assembled during app startup."""

    allowed_hosts: tuple[str, ...]
    allow_remote: bool
    require_auth: bool
    auth_username: str
    auth_password_value: str
    managed_auth_enabled: bool
    managed_auth_store: Any | None


@dataclass(frozen=True)
class LockSet:
    """Per-process locks that serialize web mutations and bounded jobs."""

    state_write_lock: asyncio.Lock
    ops_job_lock: asyncio.Lock
    conversation_lab_lock: asyncio.Lock
    voice_profile_lock: asyncio.Lock
    managed_auth_write_lock: asyncio.Lock


@dataclass(frozen=True)
class AppRuntime:
    """Shared request-time dependencies used by route registration modules."""

    ctx: WebAppContext
    surface: SurfaceProxy
    max_form_bytes: int
    security: SecurityConfig
    locks: LockSet
    channel_pairing_registry: InProcessChannelPairingRegistry
    whatsapp_pairing: WhatsAppPairingCoordinator
