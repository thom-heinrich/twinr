from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
import json


def integration_store_path(project_root: str | Path) -> Path:
    return Path(project_root).resolve() / "artifacts" / "stores" / "integrations" / "integrations.json"


@dataclass(frozen=True, slots=True)
class ManagedIntegrationConfig:
    integration_id: str
    enabled: bool = False
    settings: dict[str, str] = field(default_factory=dict)
    updated_at: str | None = None

    @classmethod
    def from_dict(cls, integration_id: str, payload: dict[str, object]) -> "ManagedIntegrationConfig":
        settings = payload.get("settings", {})
        if not isinstance(settings, dict):
            settings = {}
        return cls(
            integration_id=integration_id,
            enabled=bool(payload.get("enabled", False)),
            settings={str(key): str(value) for key, value in settings.items()},
            updated_at=str(payload.get("updated_at") or "") or None,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "settings": dict(sorted(self.settings.items())),
            "updated_at": self.updated_at,
        }

    def value(self, key: str, default: str = "") -> str:
        return self.settings.get(key, default)


class TwinrIntegrationStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def from_project_root(cls, project_root: str | Path) -> "TwinrIntegrationStore":
        return cls(integration_store_path(project_root))

    def load_all(self) -> dict[str, ManagedIntegrationConfig]:
        payload = self._read_payload()
        integrations = payload.get("integrations", {})
        if not isinstance(integrations, dict):
            return {}
        return {
            str(integration_id): ManagedIntegrationConfig.from_dict(str(integration_id), record)
            for integration_id, record in integrations.items()
            if isinstance(record, dict)
        }

    def get(self, integration_id: str) -> ManagedIntegrationConfig:
        return self.load_all().get(integration_id, ManagedIntegrationConfig(integration_id=integration_id))

    def save(self, record: ManagedIntegrationConfig) -> ManagedIntegrationConfig:
        payload = self._read_payload()
        integrations = payload.setdefault("integrations", {})
        if not isinstance(integrations, dict):
            integrations = {}
            payload["integrations"] = integrations

        saved_record = ManagedIntegrationConfig(
            integration_id=record.integration_id,
            enabled=record.enabled,
            settings=dict(record.settings),
            updated_at=datetime.now(UTC).isoformat(),
        )
        integrations[record.integration_id] = saved_record.to_dict()
        self._write_payload(payload)
        return saved_record

    def _read_payload(self) -> dict[str, object]:
        if not self.path.exists():
            return {"version": 1, "integrations": {}}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"version": 1, "integrations": {}}
        if not isinstance(payload, dict):
            return {"version": 1, "integrations": {}}
        payload.setdefault("version", 1)
        payload.setdefault("integrations", {})
        return payload

    def _write_payload(self, payload: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
