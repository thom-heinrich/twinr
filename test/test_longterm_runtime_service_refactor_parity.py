from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.longterm.core.models import (
    LongTermConsolidationResultV1,
    LongTermMemoryConflictV1,
    LongTermMemoryObjectV1,
    LongTermReflectionResultV1,
    LongTermSourceRefV1,
)
from twinr.memory.longterm.runtime import service as public_service_module
from twinr.memory.longterm.runtime.service import LongTermMemoryService
from twinr.memory.longterm.runtime.service_impl import compat as compat_module
from twinr.memory.longterm.runtime.service_impl.main import (
    LongTermMemoryService as LongTermMemoryServiceImpl,
)
from twinr.memory.longterm.storage.remote_state import LongTermRemoteStatus

_EXPECTED_GOLDEN_DIGESTS = {
    "helpers": "76d8fefbf5687d471938da4b3e49e6d67a2fba7d052b3aa948c1964ac0c1a6fd",
    "service": "56b10da4ac200b318399b578ba8b9ce9487087d60e88c077a2fee30653058a8d",
}


def _normalize_payload(value: object) -> object:
    if is_dataclass(value):
        return {
            field.name: _normalize_payload(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, MappingProxyType):
        return {str(key): _normalize_payload(item) for key, item in sorted(value.items())}
    if isinstance(value, dict):
        return {str(key): _normalize_payload(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize_payload(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "__dict__") and not isinstance(value, (int, float, bool, str, type(None))):
        return {
            key: _normalize_payload(item)
            for key, item in value.__dict__.items()
            if not key.startswith("_")
        }
    return value


def _payload_digest(payload: object) -> str:
    rendered = json.dumps(
        _normalize_payload(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return sha256(rendered.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _SortableObject:
    memory_id: str


@dataclass(frozen=True)
class _SortableConflict:
    slot_key: str
    candidate_memory_id: str


class _MergeObjectStore:
    def _merge_object(
        self,
        *,
        existing: LongTermMemoryObjectV1 | None,
        incoming: LongTermMemoryObjectV1,
        increment_support: bool,
    ) -> LongTermMemoryObjectV1:
        attrs = dict(existing.attributes or {}) if existing is not None and existing.attributes else {}
        if incoming.attributes:
            attrs.update(incoming.attributes)
        if increment_support:
            support_count = attrs.get("support_count", 1)
            attrs["support_count"] = (
                support_count if isinstance(support_count, int) else 1
            ) + 1
        return replace(incoming, attributes=attrs or None)


class _BuilderSentinel(RuntimeError):
    """Raised by the regression guard to prove builder wiring reaches runtime deps."""


def _source_ref(event_id: str) -> LongTermSourceRefV1:
    return LongTermSourceRefV1(
        source_type="conversation_turn",
        event_ids=(event_id,),
        speaker="user",
        modality="voice",
    )


def _helper_payload(module) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        safe_file = root / "ops.json"
        safe_file.write_text("{}", encoding="utf-8")
        readiness_step = module.LongTermRemoteReadinessStep(
            name="probe",
            status="ok",
            latency_ms=12.5,
            detail="ready",
        )
        readiness_result = module.LongTermRemoteReadinessResult(
            ready=True,
            detail="ready",
            remote_status=LongTermRemoteStatus(mode="remote_primary", ready=True, detail="ok"),
            steps=(readiness_step,),
            total_latency_ms=15.0,
        )
        return {
            "normalize_text": module._normalize_text("  Hallo \n Welt  ", limit=9),
            "positive_ints": [
                module._coerce_positive_int("7", default=1, maximum=5),
                module._coerce_positive_int("-4", default=3),
                module._coerce_positive_int("oops", default=2),
            ],
            "timeouts": [
                module._coerce_timeout_s("1.25", default=0.0),
                module._coerce_timeout_s(float("nan"), default=2.0),
                module._coerce_timeout_s(-3, default=4.0),
            ],
            "sanitize_jsonish": module._sanitize_jsonish(
                {
                    "captured_at": datetime(2026, 3, 27, 12, 34, tzinfo=timezone.utc),
                    "values": [1, float("nan"), b"raw", {"deep": {"deeper": {"deepest": "x"}}}],
                    "message": "  Raum ruhig  ",
                },
                timezone_name="Europe/Berlin",
            ),
            "serialize_datetime": module._serialize_datetime(
                datetime(2026, 3, 27, 12, 34, tzinfo=timezone.utc),
                timezone_name="Europe/Berlin",
            ),
            "path_validation": {
                "safe_name": None if module._validate_regular_file_path(safe_file, allow_missing=False) is None else safe_file.name,
                "missing_name": None
                if module._validate_regular_file_path(root / "missing.json", allow_missing=True) is None
                else "missing.json",
                "dir_rejected": module._validate_regular_file_path(root, allow_missing=False) is None,
            },
            "sorting": {
                "objects": [
                    item.memory_id
                    for item in module._sort_objects_by_memory_id(
                        (_SortableObject("b"), _SortableObject("a"))
                    )
                ],
                "conflicts": [
                    f"{item.slot_key}:{item.candidate_memory_id}"
                    for item in module._sort_conflicts(
                        (
                            _SortableConflict("slot:b", "cand:2"),
                            _SortableConflict("slot:a", "cand:1"),
                        )
                    )
                ],
            },
            "readiness_step": readiness_step.to_dict(),
            "readiness_result": readiness_result.to_dict(),
        }


def _service_payload(service_cls: type[LongTermMemoryService]) -> dict[str, object]:
    service = object.__new__(service_cls)
    service.config = TwinrConfig(
        project_root=".",
        personality_dir="personality",
    )

    base = datetime(2026, 3, 27, 10, 0, tzinfo=timezone.utc)
    existing = LongTermMemoryObjectV1(
        memory_id="fact:tea",
        kind="fact",
        summary="Corinna drinks mint tea.",
        source=_source_ref("turn:1"),
        status="candidate",
        slot_key="drink:tea",
        value_key="mint",
        created_at=base,
        updated_at=base,
        attributes={"support_count": 2, "origin": "seed"},
    )
    incoming = LongTermMemoryObjectV1(
        memory_id="fact:tea",
        kind="fact",
        summary="Corinna drinks mint tea.",
        source=_source_ref("turn:2"),
        status="active",
        slot_key="drink:tea",
        value_key="mint",
        created_at=base,
        updated_at=base,
        attributes={"note": "confirmed"},
    )
    episode = LongTermMemoryObjectV1(
        memory_id="episode:lunch",
        kind="episode",
        summary="Lunch was calm.",
        source=_source_ref("turn:3"),
        status="active",
        created_at=base,
        updated_at=base,
    )
    conflict = LongTermMemoryConflictV1(
        slot_key="drink:tea",
        candidate_memory_id="fact:tea",
        existing_memory_ids=("fact:old_tea",),
        question="Which tea is correct?",
        reason="Conflicting tea memory.",
    )
    consolidation = LongTermConsolidationResultV1(
        turn_id="turn:2",
        occurred_at=base,
        episodic_objects=(episode,),
        durable_objects=(incoming,),
        deferred_objects=(),
        conflicts=(conflict,),
        graph_edges=(),
    )
    merged_objects, merged_conflicts = service_cls._merge_consolidation_state(
        object_store=_MergeObjectStore(),
        existing_objects=(existing,),
        existing_conflicts=(),
        result=consolidation,
    )
    reflection = LongTermReflectionResultV1(
        reflected_objects=(
            LongTermMemoryObjectV1(
                memory_id="fact:tea",
                kind="fact",
                summary="Corinna prefers mint tea.",
                source=_source_ref("reflect:1"),
                status="active",
                slot_key="drink:tea",
                value_key="mint",
                created_at=base,
                updated_at=base,
                attributes={"reflected": True},
            ),
        ),
        created_summaries=(
            LongTermMemoryObjectV1(
                memory_id="summary:tea",
                kind="summary",
                summary="Tea preferences stabilized.",
                source=_source_ref("summary:1"),
                status="active",
                created_at=base,
                updated_at=base,
            ),
        ),
        midterm_packets=(),
    )
    merged_reflection = service_cls._merge_reflection_into_consolidation(
        result=consolidation,
        reflection_batches=(reflection,),
    )
    evidence = service._build_multimodal_evidence(
        event_name="  camera_seen_person  ",
        modality=" vision ",
        source=" device_event ",
        message="  Person visible near window.  ",
        data={
            "captured_at": datetime(2026, 3, 27, 10, 15, tzinfo=timezone.utc),
            "bytes": b"frame",
            "scores": [0.8, float("nan")],
            "nested": {"status": " calm "},
        },
    )
    evidence = replace(
        evidence,
        created_at=datetime(2026, 3, 27, 10, 15, tzinfo=timezone.utc),
    )
    keep_result = service_cls._apply_retention_or_keep(
        retention_policy=types.SimpleNamespace(apply=lambda *, objects: types.SimpleNamespace(
            kept_objects=objects,
            expired_objects=(),
            pruned_memory_ids=(),
            archived_objects=(),
        )),
        objects=merged_objects,
    )
    fallback_result = service_cls._apply_retention_or_keep(
        retention_policy=types.SimpleNamespace(apply=lambda *, objects: (_ for _ in ()).throw(RuntimeError("boom"))),
        objects=merged_objects,
    )
    archived = service_cls._merge_archived_objects(
        existing_archived=(episode,),
        archived_updates=(replace(episode, memory_id="episode:lunch:archived"),),
    )
    return {
        "evidence": _normalize_payload(evidence),
        "merged_objects": [item.memory_id for item in merged_objects],
        "merged_support_counts": [None if item.attributes is None else item.attributes.get("support_count") for item in merged_objects],
        "merged_conflicts": [f"{item.slot_key}:{item.candidate_memory_id}" for item in merged_conflicts],
        "reflection_merge": {
            "durable_ids": [item.memory_id for item in merged_reflection.durable_objects],
            "deferred_ids": [item.memory_id for item in merged_reflection.deferred_objects],
        },
        "include_midterm": {
            "episode_only": service_cls._should_include_midterm_in_multimodal_reflection(
                LongTermConsolidationResultV1(
                    turn_id="episode-only",
                    occurred_at=base,
                    episodic_objects=(episode,),
                    durable_objects=(),
                    deferred_objects=(),
                    conflicts=(),
                    graph_edges=(),
                )
            ),
            "mixed": service_cls._should_include_midterm_in_multimodal_reflection(consolidation),
        },
        "reflection_helpers": {
            "empty_has_payload": service_cls._has_reflection_payload(service_cls._empty_reflection_result()),
            "real_has_payload": service_cls._has_reflection_payload(reflection),
        },
        "retention": {
            "keep_len": len(keep_result.kept_objects),
            "fallback_len": len(fallback_result.kept_objects),
            "fallback_archived_len": len(fallback_result.archived_objects),
        },
        "archived_ids": [item.memory_id for item in archived],
    }


class LongTermRuntimeServiceRefactorParityTests(unittest.TestCase):
    def test_public_wrapper_preserves_class_module(self) -> None:
        self.assertEqual(
            LongTermMemoryService.__module__,
            "twinr.memory.longterm.runtime.service",
        )

    def test_public_wrapper_matches_internal_implementation_payloads(self) -> None:
        wrapped_helpers = _normalize_payload(_helper_payload(public_service_module))
        internal_helpers = _normalize_payload(_helper_payload(compat_module))
        self.assertEqual(wrapped_helpers, internal_helpers)

        wrapped_service = _normalize_payload(_service_payload(LongTermMemoryService))
        internal_service = _normalize_payload(_service_payload(LongTermMemoryServiceImpl))
        self.assertEqual(wrapped_service, internal_service)

    def test_builder_from_config_reaches_first_runtime_dependency(self) -> None:
        with patch(
            "twinr.memory.longterm.runtime.service_impl.builder.PromptContextStore.from_config",
            side_effect=_BuilderSentinel("sentinel"),
        ):
            with self.assertRaises(_BuilderSentinel):
                LongTermMemoryService.from_config(object())

    def test_golden_master_hashes_remain_stable(self) -> None:
        payloads = {
            "helpers": _helper_payload(public_service_module),
            "service": _service_payload(LongTermMemoryService),
        }
        for name, payload in payloads.items():
            with self.subTest(case=name):
                self.assertEqual(_payload_digest(payload), _EXPECTED_GOLDEN_DIGESTS[name])


if __name__ == "__main__":
    unittest.main()
