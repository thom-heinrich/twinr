from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

from twinr.agent.base_agent.config import TwinrConfig
from twinr.ops.paths import resolve_ops_paths, resolve_ops_paths_for_config


def _utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _field_value(source: object, key: str) -> object | None:
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


def _coerce_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_iso_datetime(value: str | None) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        normalized = raw.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    audio_input_tokens: int | None = None
    audio_output_tokens: int | None = None

    @property
    def has_values(self) -> bool:
        return any(
            value is not None
            for value in (
                self.input_tokens,
                self.output_tokens,
                self.total_tokens,
                self.cached_input_tokens,
                self.reasoning_tokens,
                self.audio_input_tokens,
                self.audio_output_tokens,
            )
        )

    @property
    def total_tokens_estimate(self) -> int | None:
        if self.total_tokens is not None:
            return self.total_tokens
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return int(self.input_tokens or 0) + int(self.output_tokens or 0)

    def to_dict(self) -> dict[str, int]:
        payload = {key: value for key, value in asdict(self).items() if value is not None}
        estimated_total = self.total_tokens_estimate
        if "total_tokens" not in payload and estimated_total is not None:
            payload["total_tokens"] = estimated_total
        return payload


@dataclass(frozen=True, slots=True)
class UsageRecord:
    created_at: str
    source: str
    request_kind: str
    model: str | None = None
    response_id: str | None = None
    request_id: str | None = None
    used_web_search: bool | None = None
    token_usage: TokenUsage | None = None
    metadata: dict[str, str] | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.token_usage is None:
            return None
        return self.token_usage.total_tokens_estimate

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "created_at": self.created_at,
            "source": self.source,
            "request_kind": self.request_kind,
            "model": self.model,
            "response_id": self.response_id,
            "request_id": self.request_id,
            "used_web_search": self.used_web_search,
            "metadata": dict(self.metadata or {}),
        }
        if self.token_usage is not None and self.token_usage.has_values:
            payload["token_usage"] = self.token_usage.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "UsageRecord":
        usage_payload = payload.get("token_usage")
        token_usage = None
        if isinstance(usage_payload, dict):
            token_usage = TokenUsage(
                input_tokens=_coerce_int(usage_payload.get("input_tokens")),
                output_tokens=_coerce_int(usage_payload.get("output_tokens")),
                total_tokens=_coerce_int(usage_payload.get("total_tokens")),
                cached_input_tokens=_coerce_int(usage_payload.get("cached_input_tokens")),
                reasoning_tokens=_coerce_int(usage_payload.get("reasoning_tokens")),
                audio_input_tokens=_coerce_int(usage_payload.get("audio_input_tokens")),
                audio_output_tokens=_coerce_int(usage_payload.get("audio_output_tokens")),
            )
        metadata_payload = payload.get("metadata")
        metadata = None
        if isinstance(metadata_payload, dict):
            metadata = {
                str(key): str(value)
                for key, value in metadata_payload.items()
                if str(value or "").strip()
            }
        return cls(
            created_at=str(payload.get("created_at", "")),
            source=str(payload.get("source", "runtime")),
            request_kind=str(payload.get("request_kind", "unknown")),
            model=(
                str(payload.get("model")).strip()
                if payload.get("model") is not None and str(payload.get("model")).strip()
                else None
            ),
            response_id=(
                str(payload.get("response_id")).strip()
                if payload.get("response_id") is not None and str(payload.get("response_id")).strip()
                else None
            ),
            request_id=(
                str(payload.get("request_id")).strip()
                if payload.get("request_id") is not None and str(payload.get("request_id")).strip()
                else None
            ),
            used_web_search=(
                bool(payload.get("used_web_search"))
                if payload.get("used_web_search") is not None
                else None
            ),
            token_usage=token_usage if token_usage is not None and token_usage.has_values else None,
            metadata=metadata,
        )


@dataclass(frozen=True, slots=True)
class UsageSummary:
    requests_total: int = 0
    requests_with_token_data: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    latest_model: str | None = None
    latest_request_kind: str | None = None
    latest_created_at: str | None = None
    by_kind: dict[str, int] | None = None
    by_model: dict[str, int] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "requests_total": self.requests_total,
            "requests_with_token_data": self.requests_with_token_data,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "latest_model": self.latest_model,
            "latest_request_kind": self.latest_request_kind,
            "latest_created_at": self.latest_created_at,
            "by_kind": dict(self.by_kind or {}),
            "by_model": dict(self.by_model or {}),
        }


class TwinrUsageStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "TwinrUsageStore":
        return cls(resolve_ops_paths_for_config(config).usage_path)

    @classmethod
    def from_project_root(cls, project_root: str | Path) -> "TwinrUsageStore":
        return cls(resolve_ops_paths(project_root).usage_path)

    def append(
        self,
        *,
        source: str,
        request_kind: str,
        model: str | None = None,
        response_id: str | None = None,
        request_id: str | None = None,
        used_web_search: bool | None = None,
        token_usage: TokenUsage | None = None,
        metadata: dict[str, object] | None = None,
    ) -> UsageRecord:
        record = UsageRecord(
            created_at=_utc_now_iso_z(),
            source=source.strip() or "runtime",
            request_kind=request_kind.strip() or "unknown",
            model=(str(model).strip() if model is not None and str(model).strip() else None),
            response_id=(str(response_id).strip() if response_id is not None and str(response_id).strip() else None),
            request_id=(str(request_id).strip() if request_id is not None and str(request_id).strip() else None),
            used_web_search=used_web_search,
            token_usage=token_usage if token_usage is not None and token_usage.has_values else None,
            metadata={
                str(key): str(value)
                for key, value in dict(metadata or {}).items()
                if str(value or "").strip()
            }
            or None,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        return record

    def tail(self, *, limit: int = 100) -> list[UsageRecord]:
        if limit <= 0 or not self.path.exists():
            return []
        records: list[UsageRecord] = []
        for raw_line in self.path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                records.append(UsageRecord.from_dict(parsed))
        return records[-limit:]

    def summary(self, *, within_hours: int | None = None) -> UsageSummary:
        records = self.tail(limit=10000)
        cutoff = None
        if within_hours is not None and within_hours > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=within_hours)
        filtered: list[UsageRecord] = []
        for record in records:
            if cutoff is None:
                filtered.append(record)
                continue
            created_at = _parse_iso_datetime(record.created_at)
            if created_at is not None and created_at >= cutoff:
                filtered.append(record)
        if not filtered:
            return UsageSummary(by_kind={}, by_model={})

        by_kind: dict[str, int] = {}
        by_model: dict[str, int] = {}
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        cached_input_tokens = 0
        reasoning_tokens = 0
        requests_with_token_data = 0
        for record in filtered:
            by_kind[record.request_kind] = by_kind.get(record.request_kind, 0) + 1
            model_key = record.model or "unknown"
            by_model[model_key] = by_model.get(model_key, 0) + 1
            if record.token_usage is None:
                continue
            requests_with_token_data += 1
            input_tokens += int(record.token_usage.input_tokens or 0)
            output_tokens += int(record.token_usage.output_tokens or 0)
            total_tokens += int(record.token_usage.total_tokens_estimate or 0)
            cached_input_tokens += int(record.token_usage.cached_input_tokens or 0)
            reasoning_tokens += int(record.token_usage.reasoning_tokens or 0)

        latest = filtered[-1]
        return UsageSummary(
            requests_total=len(filtered),
            requests_with_token_data=requests_with_token_data,
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_input_tokens=cached_input_tokens,
            reasoning_tokens=reasoning_tokens,
            latest_model=latest.model,
            latest_request_kind=latest.request_kind,
            latest_created_at=latest.created_at,
            by_kind=by_kind,
            by_model=by_model,
        )


def extract_model_name(source: object, fallback: str | None = None) -> str | None:
    value = _field_value(source, "model")
    if value is None or not str(value).strip():
        return (str(fallback).strip() if fallback is not None and str(fallback).strip() else None)
    return str(value).strip()


def extract_token_usage(source: object) -> TokenUsage | None:
    usage = _field_value(source, "usage")
    if usage is None:
        return None
    input_details = _field_value(usage, "input_tokens_details") or _field_value(usage, "input_token_details")
    output_details = _field_value(usage, "output_tokens_details") or _field_value(usage, "output_token_details")
    token_usage = TokenUsage(
        input_tokens=_coerce_int(_field_value(usage, "input_tokens")),
        output_tokens=_coerce_int(_field_value(usage, "output_tokens")),
        total_tokens=_coerce_int(_field_value(usage, "total_tokens")),
        cached_input_tokens=_coerce_int(_field_value(input_details, "cached_tokens")),
        reasoning_tokens=_coerce_int(_field_value(output_details, "reasoning_tokens")),
        audio_input_tokens=_coerce_int(
            _field_value(input_details, "audio_tokens") or _field_value(input_details, "input_audio_tokens")
        ),
        audio_output_tokens=_coerce_int(
            _field_value(output_details, "audio_tokens") or _field_value(output_details, "output_audio_tokens")
        ),
    )
    if not token_usage.has_values:
        return None
    return token_usage
