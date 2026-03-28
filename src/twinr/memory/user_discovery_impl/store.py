"""State persistence for guided user discovery."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

from twinr.agent.base_agent.config import TwinrConfig

from .common import LOGGER, _DEFAULT_STATE_PATH, _atomic_write_text
from .models import UserDiscoveryState


@dataclass(slots=True)
class UserDiscoveryStateStore:
    """Read and write the file-backed user-discovery state."""

    path: Path

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "UserDiscoveryStateStore":
        project_root = Path(config.project_root).expanduser().resolve()
        configured = Path(_DEFAULT_STATE_PATH)
        resolved = configured if configured.is_absolute() else project_root / configured
        return cls(path=resolved)

    def load(self) -> UserDiscoveryState:
        if not self.path.exists():
            return UserDiscoveryState.empty()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            LOGGER.warning("Failed to read user-discovery state from %s.", self.path, exc_info=True)
            return UserDiscoveryState.empty()
        if not isinstance(payload, Mapping):
            return UserDiscoveryState.empty()
        try:
            return UserDiscoveryState.from_dict(payload)
        except Exception:
            LOGGER.warning("Failed to normalize user-discovery state from %s.", self.path, exc_info=True)
            return UserDiscoveryState.empty()

    def save(self, state: UserDiscoveryState) -> UserDiscoveryState:
        payload = json.dumps(state.to_dict(), ensure_ascii=False, indent=2) + "\n"
        _atomic_write_text(self.path, payload)
        return state
