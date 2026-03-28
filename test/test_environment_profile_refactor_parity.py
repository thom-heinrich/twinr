from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, fields, is_dataclass
from datetime import date, datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import MappingProxyType, ModuleType, SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.memory.longterm.core.models import LongTermMemoryObjectV1, LongTermSourceRefV1
from twinr.memory.longterm.ingestion import environment_profile as public_module

impl_module: ModuleType | None
try:
    from twinr.memory.longterm.ingestion import environment_profile_impl as impl_module
except (ImportError, ModuleNotFoundError):  # pragma: no cover - pre-refactor collection path
    impl_module = None


_EXPECTED_GOLDEN_DIGESTS = {
    "helpers": "5983eb3ea6d7def46778893ffc2c684a601a56403ad81580a9f815c63ac12742",
    "from_config": "95c6917e31a4981d95a960076ceba8c814acf0d111560e5c6da415038c5ba1dd",
    "activity_compile": "56ddf3147a077d66ad0be8c681fabffa7d4b2df039d414a706c84d83715e43de",
    "regime_compile": "d8afe03ff0fc2d1b8ae58f917b255c963beb122ce5d2f6af07af5c6e338befbe",
}


def _smart_home_event_id(occurred_at: datetime, *, node_token: str, suffix: str) -> str:
    return f"smart_home_env:{occurred_at.strftime('%Y%m%dT%H%M%S%f%z')}:{node_token}:{suffix}"


def _smart_home_motion_pattern(
    *,
    day: date,
    node_id: str,
    event_hours: tuple[int, ...],
) -> LongTermMemoryObjectV1:
    node_token = node_id.replace(":", "_").replace("-", "_")
    event_datetimes = tuple(
        datetime(day.year, day.month, day.day, hour, 0, tzinfo=timezone.utc)
        for hour in event_hours
    )
    return LongTermMemoryObjectV1(
        memory_id=f"pattern:smart_home_node_activity:{node_token}:{day.isoformat()}",
        kind="pattern",
        summary="Synthetic smart-home motion node activity pattern.",
        details="Synthetic room-agnostic smart-home motion seed for tests.",
        source=LongTermSourceRefV1(
            source_type="smart_home_sensor",
            event_ids=tuple(
                _smart_home_event_id(occurred_at, node_token=node_token, suffix=f"{index:02d}")
                for index, occurred_at in enumerate(event_datetimes)
            ),
            modality="sensor",
        ),
        status="active",
        confidence=0.66,
        sensitivity="low",
        slot_key=f"pattern:smart_home_node_activity:{node_id}:{day.isoformat()}",
        value_key="smart_home_motion_node_activity",
        valid_from=day.isoformat(),
        valid_to=day.isoformat(),
        attributes={
            "memory_domain": "smart_home_environment",
            "environment_id": "home:main",
            "environment_signal_type": "motion_node_activity",
            "node_id": node_id,
            "provider": "hue",
            "route_id": "192.168.178.22",
            "source_entity_id": node_id,
            "provider_label": node_id,
            "provider_area_label": "Erdgeschoss",
        },
    )


def _smart_home_health_pattern(
    *,
    day: date,
    node_id: str,
    state: str,
    hour: int,
) -> LongTermMemoryObjectV1:
    node_token = node_id.replace(":", "_").replace("-", "_")
    occurred_at = datetime(day.year, day.month, day.day, hour, 30, tzinfo=timezone.utc)
    return LongTermMemoryObjectV1(
        memory_id=f"pattern:smart_home_node_health:{node_token}:{state}:{day.isoformat()}",
        kind="pattern",
        summary="Synthetic smart-home node health pattern.",
        details="Synthetic room-agnostic smart-home health seed for tests.",
        source=LongTermSourceRefV1(
            source_type="smart_home_sensor",
            event_ids=(
                _smart_home_event_id(occurred_at, node_token=node_token, suffix=state),
            ),
            modality="sensor",
        ),
        status="active",
        confidence=0.68,
        sensitivity="low",
        slot_key=f"pattern:smart_home_node_health:{node_id}:{day.isoformat()}",
        value_key="smart_home_node_health",
        valid_from=day.isoformat(),
        valid_to=day.isoformat(),
        attributes={
            "memory_domain": "smart_home_environment",
            "environment_id": "home:main",
            "environment_signal_type": "node_health",
            "health_state": state,
            "node_id": node_id,
            "provider": "hue",
            "route_id": "192.168.178.22",
            "source_entity_id": node_id,
            "provider_label": node_id,
            "provider_area_label": "Erdgeschoss",
        },
    )


def _normalize_payload(value: object) -> object:
    if is_dataclass(value):
        return {
            field.name: (
                "<dynamic-datetime>"
                if field.name in {"created_at", "updated_at"}
                and isinstance(getattr(value, field.name), datetime)
                else _normalize_payload(getattr(value, field.name))
            )
            for field in fields(value)
        }
    if isinstance(value, (MappingProxyType, Mapping)):
        return {str(key): _normalize_payload(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize_payload(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float):
        if value != value:
            return "NaN"
        return round(value, 6)
    return value


def _payload_digest(payload: object) -> str:
    rendered = json.dumps(
        _normalize_payload(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return sha256(rendered.encode("utf-8")).hexdigest()


def _helper_payload(module) -> dict[str, object]:
    return {
        "normalize_text": module._normalize_text("  Raum\n ruhig  heute  "),
        "normalize_slug": module._normalize_slug(" Wohn-Zimmer Süd ", fallback="fallback"),
        "tokenize_short": module._tokenize_identifier(
            "route:192.168.178.22:node-a",
            fallback="environment",
        ),
        "tokenize_long": module._tokenize_identifier("x" * 80, fallback="environment"),
        "coerce_mapping": module._coerce_mapping({"a": 1, 2: "b"}),
        "coerce_float": [
            module._coerce_float("1.25", default=0.0),
            module._coerce_float("oops", default=2.5),
        ],
        "weekday_class": [
            module._weekday_class(date(2026, 3, 27)),
            module._weekday_class(date(2026, 3, 28)),
        ],
        "normalize_datetime": [
            module._normalize_datetime(datetime(2026, 3, 27, 12, 0), timezone=timezone.utc),
            module._normalize_datetime(
                datetime(2026, 3, 27, 12, 0, tzinfo=timezone.utc),
                timezone=module._resolve_timezone("Europe/Berlin"),
            ),
        ],
        "parse_source_event_datetime": [
            module._parse_source_event_datetime(
                "smart_home_env:20260327T120000000000+0000:node_a:00"
            ),
            module._parse_source_event_datetime("invalid"),
        ],
        "ewma": module._ewma([1.0, 2.0, 5.0], alpha=0.4),
        "entropy": module._entropy_from_counts({"a": 2, "b": 1}),
        "cosine_similarity": module._cosine_similarity(
            [1.0, 0.0, 1.0],
            [0.5, 0.0, 0.5],
        ),
    }


def _from_config_payload(module) -> dict[str, object]:
    compiler = module.LongTermEnvironmentProfileCompiler.from_config(
        SimpleNamespace(
            local_timezone_name="America/New_York",
            long_term_memory_sensor_memory_enabled=True,
            long_term_memory_sensor_baseline_days=9,
            long_term_memory_sensor_min_days_observed=4,
            long_term_memory_environment_short_baseline_days=11,
            long_term_memory_environment_long_baseline_days=20,
            long_term_memory_environment_min_baseline_days=6,
            long_term_memory_environment_acute_z_threshold=2.25,
            long_term_memory_environment_acute_empirical_q=0.03,
            long_term_memory_environment_drift_min_sigma=1.75,
            long_term_memory_environment_drift_min_days=6,
            long_term_memory_environment_regime_accept_days=12,
            long_term_memory_environment_min_coverage_ratio=0.72,
        )
    )
    return asdict(compiler)


def _activity_history() -> tuple[LongTermMemoryObjectV1, ...]:
    return (
        _smart_home_motion_pattern(
            day=date(2026, 3, 11),
            node_id="route:192.168.178.22:node-a",
            event_hours=(7, 8, 12, 13, 18, 19),
        ),
        _smart_home_motion_pattern(
            day=date(2026, 3, 11),
            node_id="route:192.168.178.22:node-b",
            event_hours=(7, 12, 18),
        ),
        _smart_home_motion_pattern(
            day=date(2026, 3, 12),
            node_id="route:192.168.178.22:node-a",
            event_hours=(7, 8, 12, 13, 18, 19),
        ),
        _smart_home_motion_pattern(
            day=date(2026, 3, 12),
            node_id="route:192.168.178.22:node-b",
            event_hours=(7, 12, 18),
        ),
        _smart_home_motion_pattern(
            day=date(2026, 3, 13),
            node_id="route:192.168.178.22:node-a",
            event_hours=(7, 8, 12, 13, 18, 19),
        ),
        _smart_home_motion_pattern(
            day=date(2026, 3, 13),
            node_id="route:192.168.178.22:node-b",
            event_hours=(7, 12, 18),
        ),
        _smart_home_motion_pattern(
            day=date(2026, 3, 17),
            node_id="route:192.168.178.22:node-a",
            event_hours=(7, 8, 12, 13, 18, 19),
        ),
        _smart_home_motion_pattern(
            day=date(2026, 3, 17),
            node_id="route:192.168.178.22:node-b",
            event_hours=(7, 12, 18),
        ),
        _smart_home_motion_pattern(
            day=date(2026, 3, 18),
            node_id="route:192.168.178.22:node-a",
            event_hours=(7,),
        ),
        _smart_home_health_pattern(
            day=date(2026, 3, 18),
            node_id="route:192.168.178.22:node-b",
            state="offline",
            hour=9,
        ),
    )


def _regime_history() -> tuple[LongTermMemoryObjectV1, ...]:
    return tuple(
        _smart_home_motion_pattern(
            day=date(2026, 3, day),
            node_id="route:192.168.178.22:node-a",
            event_hours=(7, 8, 12, 13, 18, 19),
        )
        for day in range(1, 21)
    ) + tuple(
        _smart_home_motion_pattern(
            day=date(2026, 3, day),
            node_id="route:192.168.178.22:node-a",
            event_hours=(7, 18),
        )
        for day in range(21, 31)
    )


def _activity_compile_payload(module) -> object:
    compiler = module.LongTermEnvironmentProfileCompiler(
        enabled=True,
        baseline_days=7,
        history_days=21,
        min_baseline_days=4,
    )
    return compiler.compile(
        objects=_activity_history(),
        now=datetime(2026, 3, 18, 18, 0, tzinfo=timezone.utc),
    )


def _regime_compile_payload(module) -> object:
    compiler = module.LongTermEnvironmentProfileCompiler(
        enabled=True,
        baseline_days=14,
        short_baseline_days=14,
        long_baseline_days=56,
        history_days=56,
        min_baseline_days=7,
        drift_min_days=5,
        regime_accept_days=8,
    )
    return compiler.compile(
        objects=_regime_history(),
        now=datetime(2026, 3, 30, 18, 0, tzinfo=timezone.utc),
    )


class EnvironmentProfileRefactorGoldenMasterTests(unittest.TestCase):
    def test_public_module_matches_golden_master(self) -> None:
        payloads = {
            "helpers": _helper_payload(public_module),
            "from_config": _from_config_payload(public_module),
            "activity_compile": _activity_compile_payload(public_module),
            "regime_compile": _regime_compile_payload(public_module),
        }
        for name, payload in payloads.items():
            with self.subTest(name=name):
                self.assertEqual(_payload_digest(payload), _EXPECTED_GOLDEN_DIGESTS[name])

    def test_public_wrapper_exposes_legacy_module_names(self) -> None:
        self.assertEqual(
            public_module.LongTermEnvironmentProfileCompiler.__module__,
            public_module.__name__,
        )
        self.assertEqual(public_module.SmartHomeEnvironmentNode.__module__, public_module.__name__)
        self.assertEqual(
            public_module.SmartHomeEnvironmentBaseline.__module__,
            public_module.__name__,
        )

    def test_impl_package_matches_public_module(self) -> None:
        if impl_module is None:
            self.skipTest("environment_profile_impl is not present before the refactor lands")

        self.assertEqual(_helper_payload(public_module), _helper_payload(impl_module))
        self.assertEqual(_from_config_payload(public_module), _from_config_payload(impl_module))
        self.assertEqual(
            _normalize_payload(_activity_compile_payload(public_module)),
            _normalize_payload(_activity_compile_payload(impl_module)),
        )
        self.assertEqual(
            _normalize_payload(_regime_compile_payload(public_module)),
            _normalize_payload(_regime_compile_payload(impl_module)),
        )


if __name__ == "__main__":
    unittest.main()
