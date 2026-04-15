# CHANGELOG: 2026-03-27
# BUG-1: Explicit durable-memory writes now force a durable flush before returning success.
# BUG-2: Added retry-safe recent-write journaling so persist/flush retries do not duplicate memories.
# BUG-3: Search-result source normalization now truncates overlong source lists to the bounded persisted set
#        instead of aborting successful tool turns.
# SEC-1: Suspicious untrusted memory writes are blocked by default unless explicitly user-confirmed.
# SEC-2: Added hard bounds for payload sizes, metadata, and source lists to prevent practical Pi-side
#        memory/disk exhaustion from oversized inputs.
# IMP-1: Canonicalize contact emails/phones and normalize identity fields to reduce duplicate graph entities.
# IMP-2: Stamp note writes with provenance/trust metadata, richer ops events, and optional strict flush for
#        personality tool-history writes.

"""Mediate structured memory, graph memory, and durable memory mutations."""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parseaddr
from hashlib import blake2b
from threading import Lock, RLock
from typing import Any

from twinr.agent.base_agent.contracts import AgentToolCall, AgentToolResult
from twinr.agent.personality.intelligence import WorldIntelligenceConfigRequest
from twinr.memory.context_store import ManagedContextEntry, PersistentMemoryEntry
from twinr.memory.on_device import MemoryLedgerItem, SearchMemoryEntry
from twinr.ops.events import compact_text


LOGGER = logging.getLogger(__name__)
_LOCK_INIT_GUARD = Lock()
_PERSIST_RETRIES = 2
_PERSIST_RETRY_DELAY_S = 0.05
_NON_STORING_GRAPH_STATUSES = frozenset({"needs_clarification", "validation_error", "error", "rejected"})

_SCHEMA_VERSION = "2026-03-27"
_INTERNAL_META_PREFIX = "_twinr_"
_MAX_SHORT_TEXT_CHARS = 256
_MAX_LONG_TEXT_CHARS = 8_192
_MAX_NOTE_CONTENT_CHARS = 32_768
_MAX_SUMMARY_CHARS = 4_096
_MAX_DETAILS_CHARS = 32_768
_MAX_SEARCH_QUESTION_CHARS = 8_192
_MAX_SEARCH_ANSWER_CHARS = 32_768
_MAX_SOURCE_CHARS = 2_048
_MAX_SOURCES = 16
_MAX_METADATA_ENTRIES = 64
_MAX_METADATA_KEY_CHARS = 128
_MAX_METADATA_VALUE_CHARS = 2_048
_MAX_SELECT_LIMIT = 256
_MAX_PHONE_CHARS = 32
_MAX_EMAIL_CHARS = 320
_RECENT_WRITE_TTL_S = 10.0
# BREAKING: identical write payloads are treated as idempotent inside the short retry window
# to prevent duplicate memories after partial persist/flush failures and rapid tool retries.
_RECENT_WRITE_CACHE_SIZE = 512

_TRUSTED_SOURCES = frozenset(
    {
        "user",
        "caregiver",
        "operator",
        "system",
        "calendar",
        "gcal",
        "contacts",
        "gcontacts",
        "medical_device",
        "trusted_import",
    }
)
_ASSISTANT_GENERATED_SOURCES = frozenset({"assistant", "agent", "model", "llm"})
_UNTRUSTED_SOURCES = frozenset(
    {
        "tool",
        "browser",
        "web",
        "search",
        "rss",
        "email",
        "imap",
        "ocr",
        "scraper",
        "import",
        "api",
        "feed",
    }
)

_PHONE_EXTENSION_RE = re.compile(r"(?i)\b(?:ext|extension|x)\b\.?\s*\d+\s*$")
_PHONE_PUNCTUATION_RE = re.compile(r"[\s()./\-]+")
_PERSISTENCE_RISK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ignore_previous_instructions",
        re.compile(
            r"\b(?:ignore|disregard|forget)\b.{0,40}\b(?:previous|prior|above|all)\b.{0,40}\b(?:instruction|directive|prompt)s?\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "system_prompt_override",
        re.compile(
            r"(?:\b(?:system prompt|developer message|hidden instruction)\b.{0,40}\b(?:follow|obey|override|reveal|print|dump|ignore)\b)|(?:\b(?:reveal|print|dump)\b.{0,40}\b(?:system prompt|developer message)\b)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "unsafe_tool_autonomy",
        re.compile(
            r"\b(?:always|whenever|next time|regardless)\b.{0,40}\b(?:use|call|invoke|run|execute)\b.{0,40}\b(?:tool|browser|shell|terminal|python|api)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "secrets_exfiltration",
        re.compile(
            r"\b(?:password|token|secret|api key|credential|ssh key|cookie)\b.{0,40}\b(?:send|reveal|export|print|dump|upload|exfiltrate)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "disable_safety_controls",
        re.compile(
            r"\b(?:bypass|disable|ignore)\b.{0,40}\b(?:safety|guard|policy|validation|verification|approval)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
)


@dataclass
class _RecentWriteRecord:
    created_monotonic_s: float
    result: Any
    persisted: bool


class TwinrRuntimeMemoryMixin:
    """Provide the runtime-facing memory mutation and flush API."""

    def _memory_runtime_lock(self) -> RLock:
        lock = getattr(self, "_twinr_runtime_memory_lock", None)
        if lock is None:
            with _LOCK_INIT_GUARD:
                lock = getattr(self, "_twinr_runtime_memory_lock", None)
                if lock is None:
                    lock = RLock()
                    setattr(self, "_twinr_runtime_memory_lock", lock)
        return lock

    def _recent_write_cache(self) -> OrderedDict[str, _RecentWriteRecord]:
        cache = getattr(self, "_twinr_recent_memory_writes", None)
        if cache is None:
            with _LOCK_INIT_GUARD:
                cache = getattr(self, "_twinr_recent_memory_writes", None)
                if cache is None:
                    cache = OrderedDict()
                    setattr(self, "_twinr_recent_memory_writes", cache)
        return cache

    def _prune_recent_write_cache(self) -> None:
        cache = self._recent_write_cache()
        now = time.monotonic()
        expired_keys: list[str] = []
        for key, record in cache.items():
            if now - record.created_monotonic_s > _RECENT_WRITE_TTL_S:
                expired_keys.append(key)
        for key in expired_keys:
            cache.pop(key, None)
        while len(cache) > _RECENT_WRITE_CACHE_SIZE:
            cache.popitem(last=False)

    def _get_recent_write(self, fingerprint: str) -> _RecentWriteRecord | None:
        cache = self._recent_write_cache()
        self._prune_recent_write_cache()
        record = cache.get(fingerprint)
        if record is not None:
            cache.move_to_end(fingerprint)
        return record

    def _remember_recent_write(self, *, fingerprint: str, result: Any) -> None:
        cache = self._recent_write_cache()
        self._prune_recent_write_cache()
        cache[fingerprint] = _RecentWriteRecord(
            created_monotonic_s=time.monotonic(),
            result=result,
            persisted=False,
        )
        cache.move_to_end(fingerprint)
        while len(cache) > _RECENT_WRITE_CACHE_SIZE:
            cache.popitem(last=False)

    def _mark_recent_write_persisted(self, *, fingerprint: str) -> None:
        record = self._recent_write_cache().get(fingerprint)
        if record is not None:
            record.persisted = True

    def _run_snapshot_mutation(
        self,
        *,
        operation: str,
        fingerprint_payload: Mapping[str, object],
        mutate,
        after_persist=None,
    ):
        fingerprint = self._make_write_fingerprint(operation=operation, payload=fingerprint_payload)
        with self._memory_runtime_lock():
            record = self._get_recent_write(fingerprint)
            if record is not None:
                if not record.persisted:
                    self._persist_snapshot_or_raise(operation=operation)
                    self._mark_recent_write_persisted(fingerprint=fingerprint)
                    if after_persist is not None:
                        after_persist(record.result)
                return record.result

            result = mutate()
            self._remember_recent_write(fingerprint=fingerprint, result=result)
            self._persist_snapshot_or_raise(operation=operation)
            self._mark_recent_write_persisted(fingerprint=fingerprint)
            if after_persist is not None:
                after_persist(result)
            return result

    def _run_strict_long_term_mutation(
        self,
        *,
        operation: str,
        fingerprint_payload: Mapping[str, object],
        mutate,
        after_flush=None,
        timeout_s: float = 2.0,
    ):
        fingerprint = self._make_write_fingerprint(operation=operation, payload=fingerprint_payload)
        with self._memory_runtime_lock():
            record = self._get_recent_write(fingerprint)
            if record is not None:
                if not record.persisted:
                    self._flush_long_term_memory_strict(operation=operation, timeout_s=timeout_s)
                    self._mark_recent_write_persisted(fingerprint=fingerprint)
                    if after_flush is not None:
                        after_flush(record.result)
                return record.result

            result = mutate()
            self._remember_recent_write(fingerprint=fingerprint, result=result)
            # BREAKING: explicit durable mutations now wait for a durable flush before returning success.
            self._flush_long_term_memory_strict(operation=operation, timeout_s=timeout_s)
            self._mark_recent_write_persisted(fingerprint=fingerprint)
            if after_flush is not None:
                after_flush(result)
            return result

    # BREAKING: oversized text payloads are rejected early to protect low-memory edge deployments.
    def _normalize_required_text(
        self,
        field_name: str,
        value: str,
        *,
        max_chars: int | None = None,
        collapse_internal_ws: bool = False,
    ) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string")
        normalized = value.strip()
        if collapse_internal_ws:
            normalized = " ".join(normalized.split())
        if "\x00" in normalized:
            raise ValueError(f"{field_name} must not contain NUL bytes")
        if not normalized:
            raise ValueError(f"{field_name} must not be empty")
        if max_chars is not None and len(normalized) > max_chars:
            raise ValueError(f"{field_name} must be <= {max_chars} characters")
        return normalized

    def _normalize_optional_text(
        self,
        field_name: str,
        value: str | None,
        *,
        max_chars: int | None = None,
        collapse_internal_ws: bool = False,
    ) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string when provided")
        normalized = value.strip()
        if collapse_internal_ws:
            normalized = " ".join(normalized.split())
        if "\x00" in normalized:
            raise ValueError(f"{field_name} must not contain NUL bytes")
        if not normalized:
            return None
        if max_chars is not None and len(normalized) > max_chars:
            raise ValueError(f"{field_name} must be <= {max_chars} characters")
        return normalized

    def _normalize_identity_text(
        self,
        field_name: str,
        value: str,
        *,
        max_chars: int = _MAX_SHORT_TEXT_CHARS,
    ) -> str:
        return self._normalize_required_text(
            field_name,
            value,
            max_chars=max_chars,
            collapse_internal_ws=True,
        )

    def _normalize_optional_identity_text(
        self,
        field_name: str,
        value: str | None,
        *,
        max_chars: int = _MAX_SHORT_TEXT_CHARS,
    ) -> str | None:
        return self._normalize_optional_text(
            field_name,
            value,
            max_chars=max_chars,
            collapse_internal_ws=True,
        )

    def _normalize_metadata(self, metadata: Mapping[str, object] | None) -> dict[str, str] | None:
        if metadata is None:
            return None
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping when provided")

        normalized: dict[str, str] = {}
        for index, (raw_key, raw_value) in enumerate(metadata.items()):
            if index >= _MAX_METADATA_ENTRIES:
                raise ValueError(f"metadata must contain at most {_MAX_METADATA_ENTRIES} entries")

            key = self._normalize_required_text(
                "metadata key",
                str(raw_key),
                max_chars=_MAX_METADATA_KEY_CHARS,
                collapse_internal_ws=True,
            )
            if raw_value is None:
                continue

            value = str(raw_value).strip().replace("\r", " ").replace("\n", " ")
            if not value:
                continue
            if "\x00" in value:
                raise ValueError("metadata values must not contain NUL bytes")
            if len(value) > _MAX_METADATA_VALUE_CHARS:
                raise ValueError(f"metadata values must be <= {_MAX_METADATA_VALUE_CHARS} characters")
            normalized[key] = value
        return normalized or None

    def _normalize_sources(self, sources: Iterable[str] | str | None) -> tuple[str, ...]:
        if sources is None:
            return ()
        if isinstance(sources, str):
            source = self._normalize_optional_text("sources", sources, max_chars=_MAX_SOURCE_CHARS)
            return (source,) if source is not None else ()

        try:
            iterator = iter(sources)
        except TypeError as exc:
            raise TypeError("sources must be an iterable of strings") from exc

        max_sources = self._search_source_limit()
        normalized: list[str] = []
        seen: set[str] = set()
        for index, item in enumerate(iterator):
            source = self._normalize_optional_text(
                f"sources[{index}]",
                item,
                max_chars=_MAX_SOURCE_CHARS,
            )
            if source is None or source in seen:
                continue
            normalized.append(source)
            seen.add(source)
            if len(normalized) >= max_sources:
                break
        return tuple(normalized)

    def _search_source_limit(self) -> int:
        memory_limit = getattr(getattr(self, "memory", None), "_MAX_SOURCE_COUNT", _MAX_SOURCES)
        try:
            limit = int(memory_limit)
        except (TypeError, ValueError):
            return _MAX_SOURCES
        return max(1, min(limit, _MAX_SOURCES))

    def _normalize_timeout(self, timeout_s: float) -> float:
        try:
            timeout = float(timeout_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("timeout_s must be a positive finite float") from exc
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_s must be a positive finite float")
        return timeout

    def _normalize_strict_long_term_flush_timeout(self, timeout_s: float) -> float:
        timeout = self._normalize_timeout(timeout_s)
        if self.config.long_term_memory_mode == "remote_primary":
            timeout = max(timeout, float(self.config.long_term_memory_remote_flush_timeout_s))
        return timeout

    def _validate_aware_datetime(self, field_name: str, value: Any) -> Any:
        if isinstance(value, datetime):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{field_name} must be timezone-aware")
        return value

    def _utc_now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _safe_event_text(self, value: str | None, *, max_chars: int = 160) -> str:
        if value is None:
            return ""
        sanitized = value.replace("\r", r"\r").replace("\n", r"\n")
        if len(sanitized) > max_chars:
            sanitized = sanitized[: max_chars - 1] + "…"
        return compact_text(sanitized)

    def _content_digest(self, value: str) -> str:
        return blake2b(value.encode("utf-8"), digest_size=16).hexdigest()

    def _fingerprintable_value(self, value: object) -> object:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Mapping):
            return {
                str(key): self._fingerprintable_value(subvalue)
                for key, subvalue in sorted(value.items(), key=lambda item: str(item[0]))
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [self._fingerprintable_value(item) for item in value]
        if hasattr(value, "model_dump"):
            try:
                return self._fingerprintable_value(value.model_dump())
            except Exception:
                return repr(value)
        if hasattr(value, "dict"):
            try:
                return self._fingerprintable_value(value.dict())
            except Exception:
                return repr(value)
        if hasattr(value, "__dict__"):
            try:
                return self._fingerprintable_value(vars(value))
            except Exception:
                return repr(value)
        return repr(value)

    def _make_write_fingerprint(self, *, operation: str, payload: Mapping[str, object]) -> str:
        encoded = json.dumps(
            {
                "operation": operation,
                "payload": self._fingerprintable_value(payload),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return blake2b(encoded.encode("utf-8"), digest_size=16).hexdigest()

    def _normalize_email(self, field_name: str, value: str | None) -> str | None:
        email = self._normalize_optional_identity_text(field_name, value, max_chars=_MAX_EMAIL_CHARS)
        if email is None:
            return None
        _, parsed = parseaddr(email)
        candidate = (parsed or email).strip().lower()
        if not candidate or "@" not in candidate or candidate.startswith("@") or candidate.endswith("@"):
            raise ValueError(f"{field_name} must be a valid email address")
        if len(candidate) > _MAX_EMAIL_CHARS:
            raise ValueError(f"{field_name} must be <= {_MAX_EMAIL_CHARS} characters")
        return candidate

    def _normalize_phone(self, field_name: str, value: str | None) -> str | None:
        phone = self._normalize_optional_text(field_name, value, max_chars=64)
        if phone is None:
            return None
        label = _PHONE_EXTENSION_RE.sub("", phone).strip()
        if label.lower().startswith("tel:"):
            label = label[4:].strip()

        candidate = _PHONE_PUNCTUATION_RE.sub("", label)
        if candidate.startswith("00"):
            candidate = "+" + candidate[2:]
        if candidate.startswith("+"):
            digits = candidate[1:]
            if not digits.isdigit():
                raise ValueError(f"{field_name} must contain only digits and an optional leading '+'")
        else:
            if not candidate.isdigit():
                raise ValueError(f"{field_name} must contain only digits and an optional leading '+'")
        if len(candidate.lstrip("+")) < 3:
            raise ValueError(f"{field_name} must contain at least 3 digits")
        if len(candidate) > _MAX_PHONE_CHARS:
            raise ValueError(f"{field_name} must be <= {_MAX_PHONE_CHARS} characters")
        return label

    def _classify_source_trust(
        self,
        *,
        source: str | None = None,
        sources: Sequence[str] = (),
    ) -> str:
        normalized_source = None if source is None else source.strip().lower()
        if normalized_source in _TRUSTED_SOURCES:
            return "trusted"
        if normalized_source in _ASSISTANT_GENERATED_SOURCES:
            return "assistant_generated"
        if normalized_source in _UNTRUSTED_SOURCES:
            return "external_unverified"
        if sources:
            return "external_unverified"
        if normalized_source:
            return "external_unverified"
        return "assistant_generated"

    def _assess_persistence_risks(self, text_fields: Mapping[str, str | None]) -> tuple[str, ...]:
        haystacks = [value for value in text_fields.values() if isinstance(value, str) and value]
        if not haystacks:
            return ()
        merged = "\n".join(haystacks)
        hits = [name for name, pattern in _PERSISTENCE_RISK_PATTERNS if pattern.search(merged)]
        return tuple(sorted(set(hits)))

    def _guard_memory_persistence(
        self,
        *,
        operation: str,
        text_fields: Mapping[str, str | None],
        source: str | None = None,
        sources: Sequence[str] = (),
        confirmed_by_user: bool = False,
    ) -> tuple[str, tuple[str, ...]]:
        trust_tier = self._classify_source_trust(source=source, sources=sources)
        risk_tags = self._assess_persistence_risks(text_fields)
        # BREAKING: suspicious untrusted memory content is rejected unless explicitly user-confirmed.
        if risk_tags and trust_tier != "trusted" and not confirmed_by_user:
            self._append_memory_ops_event(
                event="memory_write_blocked",
                message="Blocked suspicious untrusted memory persistence.",
                data={
                    "operation": operation,
                    "trust_tier": trust_tier,
                    "risk_tags": ",".join(risk_tags),
                },
            )
            raise ValueError(
                f"{operation} blocked suspicious untrusted content; set confirmed_by_user=True after explicit user review."
            )
        return trust_tier, risk_tags

    def _build_internal_note_metadata(
        self,
        *,
        content: str,
        source: str,
        trust_tier: str,
        confirmed_by_user: bool,
        risk_tags: Sequence[str],
        extra: Mapping[str, object] | None = None,
    ) -> dict[str, str] | None:
        merged: dict[str, object] = {}
        if extra:
            merged.update(extra)
        merged.update(
            {
                f"{_INTERNAL_META_PREFIX}schema": _SCHEMA_VERSION,
                f"{_INTERNAL_META_PREFIX}recorded_at": self._utc_now_iso(),
                f"{_INTERNAL_META_PREFIX}source": source,
                f"{_INTERNAL_META_PREFIX}trust_tier": trust_tier,
                f"{_INTERNAL_META_PREFIX}confirmed_by_user": "true" if confirmed_by_user else "false",
                f"{_INTERNAL_META_PREFIX}content_hash": self._content_digest(content),
                f"{_INTERNAL_META_PREFIX}risk_tags": ",".join(risk_tags) if risk_tags else None,
            }
        )
        return self._normalize_metadata(merged)

    def _append_memory_ops_event(
        self,
        *,
        event: str,
        message: str,
        data: Mapping[str, object] | None = None,
    ) -> None:
        ops_events = getattr(self, "ops_events", None)
        if ops_events is None:
            return

        payload = {
            "event": self._normalize_required_text("event", event, max_chars=96, collapse_internal_ws=True),
            "message": self._normalize_required_text("message", message, max_chars=512),
            "data": dict(data or {}),
        }

        try:
            append = ops_events.append
        except AttributeError:
            LOGGER.warning("ops_events has no append() method; dropping event '%s'", payload["event"])
            return

        try:
            append(
                event=payload["event"],
                message=payload["message"],
                data=payload["data"],
            )
            return
        except TypeError:
            try:
                append(payload)
                return
            except Exception:
                LOGGER.exception("Failed to append ops event '%s'", payload["event"])
                return
        except Exception:
            LOGGER.exception("Failed to append ops event '%s'", payload["event"])
            return

    def _persist_snapshot_or_raise(self, *, operation: str) -> None:
        persist = getattr(self, "_persist_snapshot", None)
        if not callable(persist):
            return
        persist_fn = persist
        for attempt in range(_PERSIST_RETRIES + 1):
            try:
                persist_fn()  # pylint: disable=not-callable
                return
            except Exception as exc:
                if attempt >= _PERSIST_RETRIES:
                    LOGGER.exception("Failed to persist runtime snapshot after %s", operation)
                    raise RuntimeError(
                        f"Runtime memory changed during {operation}, but the updated snapshot could not be persisted."
                    ) from exc
                time.sleep(_PERSIST_RETRY_DELAY_S * (attempt + 1))

    def _flush_long_term_memory_strict(self, *, operation: str, timeout_s: float = 2.0) -> None:
        timeout = self._normalize_strict_long_term_flush_timeout(timeout_s)
        try:
            flushed = self.long_term_memory.flush(timeout_s=timeout)
        except Exception as exc:
            LOGGER.exception("Failed to flush long-term memory after %s", operation)
            raise RuntimeError(
                f"Long-term memory changed during {operation}, but the update could not be flushed to durable storage."
            ) from exc
        if not flushed:
            raise TimeoutError(
                f"Timed out after {timeout:.2f}s while flushing long-term memory after {operation}."
            )

    def _graph_result_status(self, result: Any) -> str | None:
        status = getattr(result, "status", None)
        if status is None:
            return None
        return str(status)

    def _graph_result_label(self, result: Any) -> str | None:
        label = getattr(result, "label", None)
        if label is None:
            return None
        return str(label)

    def _graph_result_node_id(self, result: Any) -> str | None:
        node_id = getattr(result, "node_id", None)
        if node_id is None:
            return None
        return str(node_id)

    def _graph_result_edge_type(self, result: Any) -> str:
        edge_type = getattr(result, "edge_type", None)
        return "" if edge_type is None else str(edge_type)

    def _should_record_graph_storage(self, result: Any) -> bool:
        status = self._graph_result_status(result)
        return status not in _NON_STORING_GRAPH_STATUSES

    def _normalize_limit(self, limit: int | None) -> int | None:
        if limit is None:
            return None
        if not isinstance(limit, int):
            raise TypeError("limit must be an integer when provided")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        return min(limit, _MAX_SELECT_LIMIT)

    def remember_search_result(
        self,
        *,
        question: str,
        answer: str,
        sources: tuple[str, ...] = (),
        location_hint: str | None = None,
        date_context: str | None = None,
        confirmed_by_user: bool = False,
    ) -> SearchMemoryEntry:
        """Store a verified search result in structured on-device memory."""

        question = self._normalize_required_text(
            "question",
            question,
            max_chars=_MAX_SEARCH_QUESTION_CHARS,
        )
        answer = self._normalize_required_text(
            "answer",
            answer,
            max_chars=_MAX_SEARCH_ANSWER_CHARS,
        )
        normalized_sources = self._normalize_sources(sources)
        location_hint = self._normalize_optional_text(
            "location_hint",
            location_hint,
            max_chars=_MAX_SHORT_TEXT_CHARS,
        )
        date_context = self._normalize_optional_text(
            "date_context",
            date_context,
            max_chars=_MAX_SHORT_TEXT_CHARS,
        )

        trust_tier, risk_tags = self._guard_memory_persistence(
            operation="remember_search_result",
            text_fields={
                "answer": answer,
                "location_hint": location_hint,
                "date_context": date_context,
            },
            sources=normalized_sources,
            confirmed_by_user=confirmed_by_user,
        )

        return self._run_snapshot_mutation(
            operation="remember_search_result",
            fingerprint_payload={
                "question": question,
                "answer": answer,
                "sources": normalized_sources,
                "location_hint": location_hint,
                "date_context": date_context,
                "confirmed_by_user": confirmed_by_user,
            },
            mutate=lambda: self.memory.remember_search(
                question=question,
                answer=answer,
                sources=normalized_sources,
                location_hint=location_hint,
                date_context=date_context,
            ),
            after_persist=lambda entry: self._append_memory_ops_event(
                event="search_result_stored",
                message="Search result stored in structured on-device memory.",
                data={
                    "question_chars": len(question),
                    "answer_chars": len(answer),
                    "sources": len(entry.sources),
                    "trust_tier": trust_tier,
                    "confirmed_by_user": confirmed_by_user,
                    "risk_tags": ",".join(risk_tags),
                    "answer_hash": self._content_digest(answer),
                },
            ),
        )

    def record_personality_tool_history(
        self,
        *,
        tool_calls: Sequence[AgentToolCall],
        tool_results: Sequence[AgentToolResult],
        flush: bool = False,
        timeout_s: float = 2.0,
    ) -> None:
        """Record tool-history signals for background personality learning."""

        normalized_tool_calls = tuple(tool_calls)
        normalized_tool_results = tuple(tool_results)
        if not normalized_tool_calls and not normalized_tool_results:
            return
        try:
            with self._memory_runtime_lock():
                self.long_term_memory.enqueue_personality_tool_history(
                    tool_calls=normalized_tool_calls,
                    tool_results=normalized_tool_results,
                )
                if flush:
                    self._flush_long_term_memory_strict(
                        operation="record_personality_tool_history",
                        timeout_s=timeout_s,
                    )
        except Exception:
            LOGGER.exception("Failed to record personality tool-history signals.")
            if flush:
                raise

    def store_durable_memory(
        self,
        *,
        kind: str,
        summary: str,
        details: str | None = None,
    ) -> PersistentMemoryEntry:
        """Write an explicit durable memory entry through the prompt-context store."""

        kind = self._normalize_identity_text("kind", kind, max_chars=128)
        summary = self._normalize_required_text("summary", summary, max_chars=_MAX_SUMMARY_CHARS)
        details = self._normalize_optional_text("details", details, max_chars=_MAX_DETAILS_CHARS)

        return self._run_snapshot_mutation(
            operation="store_durable_memory",
            fingerprint_payload={
                "kind": kind,
                "summary": summary,
                "details": details,
            },
            mutate=lambda: self.long_term_memory.store_explicit_memory(
                kind=kind,
                summary=summary,
                details=details,
            ),
            after_persist=lambda _entry: self._append_memory_ops_event(
                event="durable_memory_stored",
                message="Explicit durable memory entry stored.",
                data={
                    "kind": self._safe_event_text(kind),
                    "summary_chars": len(summary),
                    "details_chars": 0 if details is None else len(details),
                },
            ),
        )

    def delete_durable_memory_entry(
        self,
        *,
        entry_id: str,
    ) -> PersistentMemoryEntry | None:
        """Delete one explicit durable-memory entry through the prompt-context store."""

        entry_id = self._normalize_identity_text("entry_id", entry_id, max_chars=256)

        return self._run_snapshot_mutation(
            operation="delete_durable_memory_entry",
            fingerprint_payload={"entry_id": entry_id},
            mutate=lambda: self.long_term_memory.delete_explicit_memory(entry_id=entry_id),
            after_persist=lambda _entry: self._append_memory_ops_event(
                event="durable_memory_deleted",
                message="Explicit durable memory entry deleted.",
                data={"entry_id": self._safe_event_text(entry_id)},
            ),
        )

    def review_saved_memories(
        self,
        *,
        kind: str | None = None,
        limit: int | None = None,
    ) -> tuple[PersistentMemoryEntry, ...]:
        """Return explicit durable memories saved through the prompt-context store."""

        kind = self._normalize_optional_identity_text("kind", kind, max_chars=128)
        limit = self._normalize_limit(limit)

        with self._memory_runtime_lock():
            entries = self.long_term_memory.prompt_context_store.memory_store.load_entries()

        if kind is not None:
            entries = tuple(entry for entry in entries if entry.kind == kind)
        if limit is not None:
            entries = entries[:limit]
        return entries

    def update_user_profile_context(
        self,
        *,
        category: str,
        instruction: str,
    ) -> ManagedContextEntry:
        """Update managed user-profile context through the prompt-context store."""

        category = self._normalize_identity_text("category", category, max_chars=128)
        instruction = self._normalize_required_text(
            "instruction",
            instruction,
            max_chars=_MAX_LONG_TEXT_CHARS,
        )

        return self._run_snapshot_mutation(
            operation="update_user_profile_context",
            fingerprint_payload={
                "category": category,
                "instruction": instruction,
            },
            mutate=lambda: self.long_term_memory.update_user_profile(
                category=category,
                instruction=instruction,
            ),
            after_persist=lambda _entry: self._append_memory_ops_event(
                event="user_profile_context_updated",
                message="Managed user-profile context updated.",
                data={
                    "category": self._safe_event_text(category),
                    "instruction_chars": len(instruction),
                },
            ),
        )

    def remove_user_profile_context(
        self,
        *,
        category: str,
    ) -> ManagedContextEntry | None:
        """Delete one managed user-profile instruction through the prompt-context store."""

        category = self._normalize_identity_text("category", category, max_chars=128)

        return self._run_snapshot_mutation(
            operation="remove_user_profile_context",
            fingerprint_payload={"category": category},
            mutate=lambda: self.long_term_memory.remove_user_profile(category=category),
            after_persist=lambda _entry: self._append_memory_ops_event(
                event="user_profile_context_removed",
                message="Managed user-profile context removed.",
                data={"category": self._safe_event_text(category)},
            ),
        )

    def update_personality_context(
        self,
        *,
        category: str,
        instruction: str,
    ) -> ManagedContextEntry:
        """Update managed personality context through the prompt-context store."""

        category = self._normalize_identity_text("category", category, max_chars=128)
        instruction = self._normalize_required_text(
            "instruction",
            instruction,
            max_chars=_MAX_LONG_TEXT_CHARS,
        )

        return self._run_snapshot_mutation(
            operation="update_personality_context",
            fingerprint_payload={
                "category": category,
                "instruction": instruction,
            },
            mutate=lambda: self.long_term_memory.update_personality(
                category=category,
                instruction=instruction,
            ),
            after_persist=lambda _entry: self._append_memory_ops_event(
                event="personality_context_updated",
                message="Managed personality context updated.",
                data={
                    "category": self._safe_event_text(category),
                    "instruction_chars": len(instruction),
                },
            ),
        )

    def remove_personality_context(
        self,
        *,
        category: str,
    ) -> ManagedContextEntry | None:
        """Delete one managed personality instruction through the prompt-context store."""

        category = self._normalize_identity_text("category", category, max_chars=128)

        return self._run_snapshot_mutation(
            operation="remove_personality_context",
            fingerprint_payload={"category": category},
            mutate=lambda: self.long_term_memory.remove_personality(category=category),
            after_persist=lambda _entry: self._append_memory_ops_event(
                event="personality_context_removed",
                message="Managed personality context removed.",
                data={"category": self._safe_event_text(category)},
            ),
        )

    def configure_world_intelligence(
        self,
        *,
        request: WorldIntelligenceConfigRequest,
        search_backend: object | None = None,
    ):
        """Configure RSS/world-intelligence feeds for ongoing personality awareness."""

        return self._run_strict_long_term_mutation(
            operation="configure_world_intelligence",
            fingerprint_payload={
                "request": request,
                "search_backend": repr(search_backend),
            },
            mutate=lambda: self.long_term_memory.configure_world_intelligence(
                request=request,
                search_backend=search_backend,
            ),
            after_flush=lambda _entry: self._append_memory_ops_event(
                event="world_intelligence_configured",
                message="World intelligence configuration updated and flushed.",
                data={},
            ),
        )

    def flush_long_term_memory(self, *, timeout_s: float = 2.0) -> bool:
        """Flush queued long-term memory work within the caller-provided timeout.

        This path is intentionally best-effort: unlike durability-critical runtime
        mutations, it must not silently widen a UI/operator timeout to the
        remote-primary default floor.
        """

        timeout = self._normalize_timeout(timeout_s)
        with self._memory_runtime_lock():
            return self.long_term_memory.flush(timeout_s=timeout)

    def remember_note(
        self,
        *,
        kind: str,
        content: str,
        source: str = "tool",
        metadata: Mapping[str, object] | None = None,
        confirmed_by_user: bool = False,
    ) -> MemoryLedgerItem:
        """Store a structured note in on-device runtime memory."""

        kind = self._normalize_identity_text("kind", kind, max_chars=128)
        content = self._normalize_required_text("content", content, max_chars=_MAX_NOTE_CONTENT_CHARS)
        source = self._normalize_identity_text("source", source, max_chars=64)
        normalized_metadata = self._normalize_metadata(metadata)

        trust_tier, risk_tags = self._guard_memory_persistence(
            operation="remember_note",
            text_fields={"kind": kind, "content": content},
            source=source,
            confirmed_by_user=confirmed_by_user,
        )
        note_metadata = self._build_internal_note_metadata(
            content=content,
            source=source,
            trust_tier=trust_tier,
            confirmed_by_user=confirmed_by_user,
            risk_tags=risk_tags,
            extra=normalized_metadata,
        )

        return self._run_snapshot_mutation(
            operation="remember_note",
            fingerprint_payload={
                "kind": kind,
                "content": content,
                "source": source,
                "metadata": note_metadata,
            },
            mutate=lambda: self.memory.remember_note(
                kind=kind,
                content=content,
                source=source,
                metadata=note_metadata,
            ),
            after_persist=lambda _item: self._append_memory_ops_event(
                event="memory_note_stored",
                message="Structured memory note stored in on-device memory.",
                data={
                    "kind": self._safe_event_text(kind),
                    "content_chars": len(content),
                    "trust_tier": trust_tier,
                    "confirmed_by_user": confirmed_by_user,
                    "risk_tags": ",".join(risk_tags),
                },
            ),
        )

    def remember_contact(
        self,
        *,
        given_name: str,
        family_name: str | None = None,
        phone: str | None = None,
        email: str | None = None,
        role: str | None = None,
        relation: str | None = None,
        notes: str | None = None,
        source: str = "tool",
        confirmed_by_user: bool = False,
    ):
        """Store or update a contact in personal graph memory."""

        given_name = self._normalize_identity_text("given_name", given_name, max_chars=128)
        family_name = self._normalize_optional_identity_text("family_name", family_name, max_chars=128)
        phone = self._normalize_phone("phone", phone)
        email = self._normalize_email("email", email)
        role = self._normalize_optional_identity_text("role", role, max_chars=128)
        relation = self._normalize_optional_identity_text("relation", relation, max_chars=128)
        notes = self._normalize_optional_text("notes", notes, max_chars=_MAX_LONG_TEXT_CHARS)
        source = self._normalize_identity_text("source", source, max_chars=64)

        trust_tier, risk_tags = self._guard_memory_persistence(
            operation="remember_contact",
            text_fields={
                "given_name": given_name,
                "family_name": family_name,
                "role": role,
                "relation": relation,
                "notes": notes,
            },
            source=source,
            confirmed_by_user=confirmed_by_user,
        )

        state: dict[str, Any] = {
            "status": None,
            "label": None,
            "note_stored": False,
            "note_content": None,
        }

        def mutate():
            result = self.graph_memory.remember_contact(
                given_name=given_name,
                family_name=family_name,
                phone=phone,
                email=email,
                role=role,
                relation=relation,
                notes=notes,
            )

            status = self._graph_result_status(result)
            label = self._graph_result_label(result)
            state["status"] = status
            state["label"] = label

            if self.config.long_term_memory_mode != "remote_primary" and self._should_record_graph_storage(result):
                note_content = f"Stored contact: {label}" if label else "Stored contact."
                note_metadata = self._build_internal_note_metadata(
                    content=note_content,
                    source=source,
                    trust_tier=trust_tier,
                    confirmed_by_user=confirmed_by_user,
                    risk_tags=risk_tags,
                    extra={
                        "graph_status": status or "",
                        "graph_node_id": self._graph_result_node_id(result),
                        "graph_email": email,
                        "graph_phone": phone,
                    },
                )
                self.memory.remember_note(
                    kind="contact",
                    content=note_content,
                    source=source,
                    metadata=note_metadata,
                )
                state["note_stored"] = True
                state["note_content"] = note_content
            return result

        def after_persist(_result):
            if state["note_stored"] and isinstance(state["note_content"], str):
                self._append_memory_ops_event(
                    event="memory_note_stored",
                    message="Structured memory note stored in on-device memory.",
                    data={
                        "kind": "contact",
                        "content_chars": len(state["note_content"]),
                        "trust_tier": trust_tier,
                    },
                )
            self._append_memory_ops_event(
                event="graph_contact_saved",
                message="Structured contact memory was stored in the personal graph.",
                data={
                    "status": self._safe_event_text(state["status"] or "unknown"),
                    "has_label": state["label"] is not None,
                    "trust_tier": trust_tier,
                    "confirmed_by_user": confirmed_by_user,
                    "risk_tags": ",".join(risk_tags),
                    "canonical_email": email is not None,
                    "canonical_phone": phone is not None,
                },
            )

        return self._run_snapshot_mutation(
            operation="remember_contact",
            fingerprint_payload={
                "given_name": given_name,
                "family_name": family_name,
                "phone": phone,
                "email": email,
                "role": role,
                "relation": relation,
                "notes": notes,
                "source": source,
                "confirmed_by_user": confirmed_by_user,
            },
            mutate=mutate,
            after_persist=after_persist,
        )

    def lookup_contact(
        self,
        *,
        name: str,
        family_name: str | None = None,
        role: str | None = None,
        contact_label: str | None = None,
    ):
        """Look up a stored contact in graph memory."""

        name = self._normalize_identity_text("name", name, max_chars=128)
        family_name = self._normalize_optional_identity_text("family_name", family_name, max_chars=128)
        role = self._normalize_optional_identity_text("role", role, max_chars=128)
        contact_label = self._normalize_optional_identity_text("contact_label", contact_label, max_chars=256)

        with self._memory_runtime_lock():
            return self.graph_memory.lookup_contact(
                name=name,
                family_name=family_name,
                role=role,
                contact_label=contact_label,
            )

    def remember_preference(
        self,
        *,
        category: str,
        value: str,
        for_product: str | None = None,
        sentiment: str = "prefer",
        details: str | None = None,
        source: str = "tool",
        confirmed_by_user: bool = False,
    ):
        """Store or update a preference in personal graph memory."""

        category = self._normalize_identity_text("category", category, max_chars=128)
        value = self._normalize_required_text("value", value, max_chars=_MAX_LONG_TEXT_CHARS)
        for_product = self._normalize_optional_identity_text("for_product", for_product, max_chars=256)
        sentiment = self._normalize_identity_text("sentiment", sentiment, max_chars=64)
        details = self._normalize_optional_text("details", details, max_chars=_MAX_LONG_TEXT_CHARS)
        source = self._normalize_identity_text("source", source, max_chars=64)

        trust_tier, risk_tags = self._guard_memory_persistence(
            operation="remember_preference",
            text_fields={
                "category": category,
                "value": value,
                "for_product": for_product,
                "sentiment": sentiment,
                "details": details,
            },
            source=source,
            confirmed_by_user=confirmed_by_user,
        )

        state: dict[str, Any] = {
            "status": None,
            "label": None,
            "edge_type": "",
            "note_stored": False,
            "note_content": None,
        }

        def mutate():
            result = self.graph_memory.remember_preference(
                category=category,
                value=value,
                for_product=for_product,
                sentiment=sentiment,
                details=details,
            )

            status = self._graph_result_status(result)
            label = self._graph_result_label(result)
            edge_type = self._graph_result_edge_type(result)
            state["status"] = status
            state["label"] = label
            state["edge_type"] = edge_type

            if self.config.long_term_memory_mode != "remote_primary" and self._should_record_graph_storage(result):
                note_content = f"Stored preference: {label}" if label else "Stored preference."
                note_metadata = self._build_internal_note_metadata(
                    content=note_content,
                    source=source,
                    trust_tier=trust_tier,
                    confirmed_by_user=confirmed_by_user,
                    risk_tags=risk_tags,
                    extra={
                        "graph_edge_type": edge_type,
                        "graph_status": status or "",
                        "graph_node_id": self._graph_result_node_id(result),
                    },
                )
                self.memory.remember_note(
                    kind="preference",
                    content=note_content,
                    source=source,
                    metadata=note_metadata,
                )
                state["note_stored"] = True
                state["note_content"] = note_content
            return result

        def after_persist(_result):
            if state["note_stored"] and isinstance(state["note_content"], str):
                self._append_memory_ops_event(
                    event="memory_note_stored",
                    message="Structured memory note stored in on-device memory.",
                    data={
                        "kind": "preference",
                        "content_chars": len(state["note_content"]),
                        "trust_tier": trust_tier,
                    },
                )
            self._append_memory_ops_event(
                event="graph_preference_saved",
                message="Structured preference memory was stored in the personal graph.",
                data={
                    "status": self._safe_event_text(state["status"] or "unknown"),
                    "edge_type": self._safe_event_text(state["edge_type"]),
                    "trust_tier": trust_tier,
                    "confirmed_by_user": confirmed_by_user,
                    "risk_tags": ",".join(risk_tags),
                },
            )

        return self._run_snapshot_mutation(
            operation="remember_preference",
            fingerprint_payload={
                "category": category,
                "value": value,
                "for_product": for_product,
                "sentiment": sentiment,
                "details": details,
                "source": source,
                "confirmed_by_user": confirmed_by_user,
            },
            mutate=mutate,
            after_persist=after_persist,
        )

    def remember_plan(
        self,
        *,
        summary: str,
        when_text: str | None = None,
        details: str | None = None,
        source: str = "tool",
        confirmed_by_user: bool = False,
    ):
        """Store or update a future plan in personal graph memory."""

        summary = self._normalize_required_text("summary", summary, max_chars=2_048)
        when_text = self._normalize_optional_text("when_text", when_text, max_chars=_MAX_SHORT_TEXT_CHARS)
        details = self._normalize_optional_text("details", details, max_chars=_MAX_LONG_TEXT_CHARS)
        source = self._normalize_identity_text("source", source, max_chars=64)

        trust_tier, risk_tags = self._guard_memory_persistence(
            operation="remember_plan",
            text_fields={
                "summary": summary,
                "when_text": when_text,
                "details": details,
            },
            source=source,
            confirmed_by_user=confirmed_by_user,
        )

        state: dict[str, Any] = {
            "status": None,
            "label": None,
            "edge_type": "",
            "note_stored": False,
            "note_content": None,
        }

        def mutate():
            result = self.graph_memory.remember_plan(summary=summary, when_text=when_text, details=details)

            status = self._graph_result_status(result)
            label = self._graph_result_label(result)
            edge_type = self._graph_result_edge_type(result)
            state["status"] = status
            state["label"] = label
            state["edge_type"] = edge_type

            if self.config.long_term_memory_mode != "remote_primary" and self._should_record_graph_storage(result):
                note_content = f"Stored plan: {label}" if label else "Stored plan."
                note_metadata = self._build_internal_note_metadata(
                    content=note_content,
                    source=source,
                    trust_tier=trust_tier,
                    confirmed_by_user=confirmed_by_user,
                    risk_tags=risk_tags,
                    extra={
                        "graph_edge_type": edge_type,
                        "graph_status": status or "",
                        "graph_node_id": self._graph_result_node_id(result),
                    },
                )
                self.memory.remember_note(
                    kind="plan",
                    content=note_content,
                    source=source,
                    metadata=note_metadata,
                )
                state["note_stored"] = True
                state["note_content"] = note_content
            return result

        def after_persist(_result):
            if state["note_stored"] and isinstance(state["note_content"], str):
                self._append_memory_ops_event(
                    event="memory_note_stored",
                    message="Structured memory note stored in on-device memory.",
                    data={
                        "kind": "plan",
                        "content_chars": len(state["note_content"]),
                        "trust_tier": trust_tier,
                    },
                )
            self._append_memory_ops_event(
                event="graph_plan_saved",
                message="Structured plan memory was stored in the personal graph.",
                data={
                    "status": self._safe_event_text(state["status"] or "unknown"),
                    "edge_type": self._safe_event_text(state["edge_type"]),
                    "trust_tier": trust_tier,
                    "confirmed_by_user": confirmed_by_user,
                    "risk_tags": ",".join(risk_tags),
                },
            )

        return self._run_snapshot_mutation(
            operation="remember_plan",
            fingerprint_payload={
                "summary": summary,
                "when_text": when_text,
                "details": details,
                "source": source,
                "confirmed_by_user": confirmed_by_user,
            },
            mutate=mutate,
            after_persist=after_persist,
        )

    def delete_contact(
        self,
        *,
        node_id: str,
    ):
        """Delete one contact from graph memory."""

        node_id = self._normalize_identity_text("node_id", node_id, max_chars=256)

        return self._run_snapshot_mutation(
            operation="delete_contact",
            fingerprint_payload={"node_id": node_id},
            mutate=lambda: self.graph_memory.delete_contact(node_id=node_id),
        )

    def delete_preference(
        self,
        *,
        node_id: str,
        edge_type: str | None = None,
    ):
        """Delete one preference from graph memory."""

        node_id = self._normalize_identity_text("node_id", node_id, max_chars=256)
        edge_type = self._normalize_optional_identity_text("edge_type", edge_type, max_chars=128)

        return self._run_snapshot_mutation(
            operation="delete_preference",
            fingerprint_payload={"node_id": node_id, "edge_type": edge_type},
            mutate=lambda: self.graph_memory.delete_preference(node_id=node_id, edge_type=edge_type),
        )

    def delete_plan(
        self,
        *,
        node_id: str,
    ):
        """Delete one future plan from graph memory."""

        node_id = self._normalize_identity_text("node_id", node_id, max_chars=256)

        return self._run_snapshot_mutation(
            operation="delete_plan",
            fingerprint_payload={"node_id": node_id},
            mutate=lambda: self.graph_memory.delete_plan(node_id=node_id),
        )

    def select_long_term_memory_conflicts(
        self,
        *,
        query_text: str | None = None,
        limit: int | None = None,
    ):
        """Return pending long-term memory conflicts for review."""

        query_text = self._normalize_optional_text("query_text", query_text, max_chars=_MAX_LONG_TEXT_CHARS)
        limit = self._normalize_limit(limit)

        with self._memory_runtime_lock():
            return self.long_term_memory.select_conflict_queue(
                query_text=query_text,
                limit=limit,
            )

    def resolve_long_term_memory_conflict(
        self,
        *,
        slot_key: str,
        selected_memory_id: str,
    ):
        """Resolve a queued long-term memory conflict and flush it."""

        slot_key = self._normalize_identity_text("slot_key", slot_key, max_chars=256)
        selected_memory_id = self._normalize_identity_text("selected_memory_id", selected_memory_id, max_chars=256)

        return self._run_strict_long_term_mutation(
            operation="resolve_long_term_memory_conflict",
            fingerprint_payload={
                "slot_key": slot_key,
                "selected_memory_id": selected_memory_id,
            },
            mutate=lambda: self.long_term_memory.resolve_conflict(
                slot_key=slot_key,
                selected_memory_id=selected_memory_id,
            ),
        )

    def reserve_long_term_proactive_candidate(self, *, now=None, live_facts=None):
        """Reserve the next proactive long-term memory candidate, if any."""

        now = self._validate_aware_datetime("now", now)
        with self._memory_runtime_lock():
            result = self.long_term_memory.reserve_proactive_candidate(now=now, live_facts=live_facts)
            if result is not None:
                self._flush_long_term_memory_strict(operation="reserve_long_term_proactive_candidate")
        return result

    def preview_long_term_proactive_candidate(self, *, now=None, live_facts=None):
        """Preview the next proactive long-term memory candidate without reserving it."""

        now = self._validate_aware_datetime("now", now)
        with self._memory_runtime_lock():
            return self.long_term_memory.preview_proactive_candidate(now=now, live_facts=live_facts)

    def reserve_specific_long_term_proactive_candidate(self, candidate, *, now=None):
        """Reserve a specific proactive long-term memory candidate."""

        now = self._validate_aware_datetime("now", now)
        with self._memory_runtime_lock():
            result = self.long_term_memory.reserve_specific_proactive_candidate(candidate, now=now)
            if result is not None:
                self._flush_long_term_memory_strict(
                    operation="reserve_specific_long_term_proactive_candidate"
                )
        return result

    def mark_long_term_proactive_candidate_delivered(
        self,
        reservation,
        *,
        delivered_at=None,
        prompt_text: str | None = None,
    ):
        """Mark a reserved proactive candidate as delivered and flush it."""

        delivered_at = self._validate_aware_datetime("delivered_at", delivered_at)
        prompt_text = self._normalize_optional_text("prompt_text", prompt_text, max_chars=_MAX_LONG_TEXT_CHARS)

        with self._memory_runtime_lock():
            result = self.long_term_memory.mark_proactive_candidate_delivered(
                reservation,
                delivered_at=delivered_at,
                prompt_text=prompt_text,
            )
            self._flush_long_term_memory_strict(
                operation="mark_long_term_proactive_candidate_delivered"
            )
        return result

    def mark_long_term_proactive_candidate_skipped(
        self,
        reservation,
        *,
        reason: str,
        skipped_at=None,
    ):
        """Mark a reserved proactive candidate as skipped and flush it."""

        reason = self._normalize_required_text("reason", reason, max_chars=512)
        skipped_at = self._validate_aware_datetime("skipped_at", skipped_at)

        with self._memory_runtime_lock():
            result = self.long_term_memory.mark_proactive_candidate_skipped(
                reservation,
                reason=reason,
                skipped_at=skipped_at,
            )
            self._flush_long_term_memory_strict(
                operation="mark_long_term_proactive_candidate_skipped"
            )
        return result
