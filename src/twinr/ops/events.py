from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

from twinr.agent.base_agent.config import TwinrConfig
from twinr.ops.paths import resolve_ops_paths, resolve_ops_paths_for_config


def _utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compact_text(value: str | None, *, limit: int = 160) -> str:
    compact = " ".join((value or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: max(limit - 3, 0)].rstrip() + "..."


class TwinrOpsEventStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "TwinrOpsEventStore":
        return cls(resolve_ops_paths_for_config(config).events_path)

    @classmethod
    def from_project_root(cls, project_root: str | Path) -> "TwinrOpsEventStore":
        return cls(resolve_ops_paths(project_root).events_path)

    def append(
        self,
        *,
        event: str,
        message: str,
        level: str = "info",
        data: dict[str, object] | None = None,
    ) -> dict[str, object]:
        entry = {
            "created_at": _utc_now_iso_z(),
            "level": level.strip().lower() or "info",
            "event": event.strip() or "unknown",
            "message": message.strip() or event.strip() or "unknown",
            "data": dict(data or {}),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        return entry

    def tail(self, *, limit: int = 100) -> list[dict[str, object]]:
        if limit <= 0 or not self.path.exists():
            return []
        entries: list[dict[str, object]] = []
        for raw_line in self.path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                entries.append(parsed)
        return entries[-limit:]
