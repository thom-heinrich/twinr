from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import json

from twinr.agent.base_agent.config import TwinrConfig
from twinr.ops.checks import check_summary, run_config_checks
from twinr.ops.events import TwinrOpsEventStore
from twinr.ops.health import collect_system_health
from twinr.ops.paths import resolve_ops_paths_for_config
from twinr.ops.usage import TwinrUsageStore

_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass(frozen=True, slots=True)
class SupportBundleInfo:
    bundle_name: str
    bundle_path: str
    created_at: str
    file_count: int
    includes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_support_bundle(
    config: TwinrConfig,
    *,
    env_path: str | Path,
    event_limit: int = 100,
) -> SupportBundleInfo:
    paths = resolve_ops_paths_for_config(config)
    paths.bundles_root.mkdir(parents=True, exist_ok=True)

    env_values = _read_env_values(Path(env_path))
    redacted_env = redact_env_values(env_values)
    checks = run_config_checks(config)
    summary = check_summary(checks)
    event_store = TwinrOpsEventStore.from_config(config)
    events = event_store.tail(limit=event_limit)
    errors = [entry for entry in events if str(entry.get("level", "")).lower() == "error"][-20:]
    snapshot_path = Path(config.runtime_state_path)
    snapshot_payload = _read_json(snapshot_path) if snapshot_path.exists() else None
    usage_store = TwinrUsageStore.from_config(config)
    usage_summary = usage_store.summary().to_dict()
    recent_usage = [record.to_dict() for record in usage_store.tail(limit=50)]
    health_payload = collect_system_health(config).to_dict()

    bundle_name = f"twinr-support-{_utc_stamp()}.zip"
    bundle_path = paths.bundles_root / bundle_name

    includes = [
        "summary.json",
        "redacted_env.json",
        "config_checks.json",
        "events.json",
        "errors.json",
        "system_health.json",
        "usage_summary.json",
        "recent_usage.json",
    ]

    with ZipFile(bundle_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "summary.json",
            json.dumps(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "bundle_name": bundle_name,
                    "env_path": str(Path(env_path).resolve()),
                    "runtime_state_path": str(snapshot_path),
                    "check_summary": summary,
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
        )
        archive.writestr(
            "redacted_env.json",
            json.dumps(redacted_env, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        )
        archive.writestr(
            "config_checks.json",
            json.dumps([check.to_dict() for check in checks], indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        )
        archive.writestr(
            "events.json",
            json.dumps(events, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        )
        archive.writestr(
            "errors.json",
            json.dumps(errors, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        )
        archive.writestr(
            "system_health.json",
            json.dumps(health_payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        )
        archive.writestr(
            "usage_summary.json",
            json.dumps(usage_summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        )
        archive.writestr(
            "recent_usage.json",
            json.dumps(recent_usage, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        )
        if snapshot_payload is not None:
            archive.writestr(
                "runtime_snapshot.json",
                json.dumps(snapshot_payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            )
            includes.append("runtime_snapshot.json")
        for artifact_path in _latest_self_test_artifacts(paths.self_tests_root):
            archive.write(artifact_path, arcname=f"self_tests/{artifact_path.name}")
            includes.append(f"self_tests/{artifact_path.name}")

    event_store.append(
        event="support_bundle_created",
        message="Support bundle created.",
        data={"bundle_name": bundle_name, "file_count": len(includes)},
    )
    return SupportBundleInfo(
        bundle_name=bundle_name,
        bundle_path=str(bundle_path),
        created_at=datetime.now(timezone.utc).isoformat(),
        file_count=len(includes),
        includes=tuple(includes),
    )


def redact_env_values(values: dict[str, str]) -> dict[str, str]:
    redacted: dict[str, str] = {}
    for key, value in sorted(values.items()):
        if not _is_relevant_key(key):
            continue
        if _is_secret_key(key):
            redacted[key] = _mask_secret(value)
        else:
            redacted[key] = value
    return redacted


def _read_env_values(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _read_json(path: Path) -> dict[str, object] | list[object] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _latest_self_test_artifacts(root: Path, *, limit: int = 4) -> tuple[Path, ...]:
    if not root.exists():
        return ()
    files = sorted(
        [path for path in root.iterdir() if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return tuple(files[:limit])


def _is_relevant_key(key: str) -> bool:
    return key.startswith("TWINR_") or key.startswith("OPENAI_") or key in {
        "DEEPINFRA_API_KEY",
        "OPENROUTER_API_KEY",
    }


def _is_secret_key(key: str) -> bool:
    upper = key.upper()
    return any(marker in upper for marker in _SECRET_MARKERS)


def _mask_secret(value: str | None) -> str:
    if not value:
        return "Not configured"
    if len(value) <= 8:
        return "Configured"
    return f"{value[:4]}…{value[-4:]}"
