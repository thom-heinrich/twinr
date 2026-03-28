from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.user_discovery import (
    UserDiscoveryCommitCallbacks,
    UserDiscoveryFact,
    UserDiscoveryMemoryRoute,
    UserDiscoveryService,
    UserDiscoveryState,
    UserDiscoveryStoredFact,
    UserDiscoveryTopicState,
)

user_discovery_impl: ModuleType | None
try:
    import twinr.memory.user_discovery_impl as _user_discovery_impl
except ImportError:  # pragma: no cover - pre-refactor baseline
    user_discovery_impl = None
else:
    user_discovery_impl = _user_discovery_impl


_EXPECTED_GOLDEN_DIGESTS = {
    "initial": "88e851c165dd8cb67bfaef9abdc222fa1a92f4ab96092a19dca055427d5b2921",
    "review": "6ee9b904950d3b2c571e5b64387db3caabb6236d5cb5486f68796433804b6d13",
    "invites": "df34f72b9ed942ebc0892b9946aef60f9a729e0738ef1c6c6b8b6e625838a7e7",
}


def _json_safe(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _payload_digest(payload: object) -> str:
    rendered = json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return sha256(rendered.encode("utf-8")).hexdigest()


class _CallbackRecorder:
    """Capture deterministic commit side effects for golden-master flows."""

    def __init__(self) -> None:
        self.events: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def _record(self, name: str, *args: object, **kwargs: object) -> None:
        self.events.append((name, args, kwargs))

    def update_user_profile(self, category: str, instruction: str) -> SimpleNamespace:
        self._record("update_user_profile", category, instruction)
        return SimpleNamespace(status="ok")

    def delete_user_profile(self, category: str) -> SimpleNamespace:
        self._record("delete_user_profile", category)
        return SimpleNamespace(status="ok")

    def update_personality(self, category: str, instruction: str) -> SimpleNamespace:
        self._record("update_personality", category, instruction)
        return SimpleNamespace(status="ok")

    def delete_personality(self, category: str) -> SimpleNamespace:
        self._record("delete_personality", category)
        return SimpleNamespace(status="ok")

    def remember_contact(self, **kwargs: object) -> SimpleNamespace:
        self._record("remember_contact", **kwargs)
        given_name = str(kwargs.get("given_name") or "").lower() or "unknown"
        relation = str(kwargs.get("relation") or kwargs.get("role") or "person").lower()
        return SimpleNamespace(
            status="stored",
            node_id=f"contact:{given_name}:{relation}",
            edge_type="relationship",
        )

    def delete_contact(self, node_id: str) -> SimpleNamespace:
        self._record("delete_contact", node_id)
        return SimpleNamespace(status="deleted")

    def remember_preference(self, **kwargs: object) -> SimpleNamespace:
        self._record("remember_preference", **kwargs)
        category = str(kwargs.get("category") or "category").lower().replace(" ", "_")
        value = str(kwargs.get("value") or "value").lower().replace(" ", "_")
        return SimpleNamespace(
            status="stored",
            node_id=f"preference:{category}:{value}",
            edge_type="prefers",
        )

    def delete_preference(self, node_id: str, edge_type: str | None) -> SimpleNamespace:
        self._record("delete_preference", node_id, edge_type=edge_type)
        return SimpleNamespace(status="deleted")

    def remember_plan(self, **kwargs: object) -> SimpleNamespace:
        self._record("remember_plan", **kwargs)
        summary = str(kwargs.get("summary") or "plan").lower().replace(" ", "_")
        return SimpleNamespace(status="stored", node_id=f"plan:{summary}", edge_type="plans")

    def delete_plan(self, node_id: str) -> SimpleNamespace:
        self._record("delete_plan", node_id)
        return SimpleNamespace(status="deleted")

    def store_durable_memory(self, **kwargs: object) -> SimpleNamespace:
        self._record("store_durable_memory", **kwargs)
        kind = str(kwargs.get("kind") or "memory").lower().replace(" ", "_")
        summary = str(kwargs.get("summary") or "entry").lower().replace(" ", "_")
        return SimpleNamespace(entry_id=f"durable:{kind}:{summary}")

    def delete_durable_memory(self, entry_id: str) -> SimpleNamespace:
        self._record("delete_durable_memory", entry_id)
        return SimpleNamespace(status="deleted")

    def bundle(self) -> UserDiscoveryCommitCallbacks:
        return UserDiscoveryCommitCallbacks(
            update_user_profile=self.update_user_profile,
            delete_user_profile=self.delete_user_profile,
            update_personality=self.update_personality,
            delete_personality=self.delete_personality,
            remember_contact=self.remember_contact,
            delete_contact=self.delete_contact,
            remember_preference=self.remember_preference,
            delete_preference=self.delete_preference,
            remember_plan=self.remember_plan,
            delete_plan=self.delete_plan,
            store_durable_memory=self.store_durable_memory,
            delete_durable_memory=self.delete_durable_memory,
        )


def _capture_initial_progression() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as temp_dir:
        personality_dir = Path(temp_dir) / "personality"
        personality_dir.mkdir(parents=True, exist_ok=True)
        (personality_dir / "USER.md").write_text("Base user profile\n", encoding="utf-8")
        (personality_dir / "PERSONALITY.md").write_text("Base personality\n", encoding="utf-8")
        service = UserDiscoveryService.from_config(
            TwinrConfig(project_root=temp_dir, personality_dir="personality")
        )
        recorder = _CallbackRecorder()
        now = datetime(2026, 3, 27, 12, 0, tzinfo=timezone.utc)
        start = service.manage(action="start_or_resume", now=now)
        answer = service.manage(
            action="answer",
            learned_facts=(
                UserDiscoveryFact(
                    storage="user_profile",
                    text="User prefers to be called Thom.",
                ),
            ),
            topic_complete=True,
            callbacks=recorder.bundle(),
            now=now + timedelta(minutes=1),
        )
        status = service.manage(action="status", now=now + timedelta(minutes=2))
        return {
            "start": start.to_dict(),
            "answer": answer.to_dict(),
            "status": status.to_dict(),
            "state": service.load_state().to_dict(),
            "events": recorder.events,
            "user_md": (personality_dir / "USER.md").read_text(encoding="utf-8"),
        }


def _capture_review_replace_delete_flow() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as temp_dir:
        personality_dir = Path(temp_dir) / "personality"
        personality_dir.mkdir(parents=True, exist_ok=True)
        (personality_dir / "USER.md").write_text("Base user profile\n", encoding="utf-8")
        (personality_dir / "PERSONALITY.md").write_text("Base personality\n", encoding="utf-8")
        service = UserDiscoveryService.from_config(
            TwinrConfig(project_root=temp_dir, personality_dir="personality")
        )
        recorder = _CallbackRecorder()
        callbacks = recorder.bundle()
        now = datetime(2026, 3, 27, 15, 0, tzinfo=timezone.utc)
        service.manage(action="start_or_resume", topic_id="social", now=now)
        answer = service.manage(
            action="answer",
            topic_id="social",
            memory_routes=(
                UserDiscoveryMemoryRoute(route_kind="contact", given_name="Anna", relation="daughter"),
                UserDiscoveryMemoryRoute(
                    route_kind="preference",
                    category="drink",
                    value="Melitta",
                    sentiment="prefer",
                ),
                UserDiscoveryMemoryRoute(
                    route_kind="plan",
                    summary="Call Anna",
                    when_text="tomorrow",
                ),
                UserDiscoveryMemoryRoute(
                    route_kind="durable_memory",
                    kind="boundary",
                    summary="Do not call after 9 pm.",
                ),
            ),
            callbacks=callbacks,
            now=now + timedelta(minutes=1),
        )
        review = service.manage(action="review_profile", now=now + timedelta(minutes=2))
        contact_fact_id = next(item.fact_id for item in review.review_items if item.route_kind == "contact")
        plan_fact_id = next(item.fact_id for item in review.review_items if item.route_kind == "plan")
        replaced = service.manage(
            action="replace_fact",
            fact_id=contact_fact_id,
            memory_routes=(
                UserDiscoveryMemoryRoute(
                    route_kind="contact",
                    given_name="Anne",
                    relation="daughter",
                ),
            ),
            callbacks=callbacks,
            now=now + timedelta(minutes=3),
        )
        deleted = service.manage(
            action="delete_fact",
            fact_id=plan_fact_id,
            callbacks=callbacks,
            now=now + timedelta(minutes=4),
        )
        final_review = service.manage(action="review_profile", now=now + timedelta(minutes=5))
        return {
            "answer": answer.to_dict(),
            "review": review.to_dict(),
            "replaced": replaced.to_dict(),
            "deleted": deleted.to_dict(),
            "final_review": final_review.to_dict(),
            "state": service.load_state().to_dict(),
            "events": recorder.events,
        }


def _capture_invitation_selection() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as temp_dir:
        personality_dir = Path(temp_dir) / "personality"
        personality_dir.mkdir(parents=True, exist_ok=True)
        (personality_dir / "USER.md").write_text(
            "User: Thom.\nLives in Schwarzenbek.\n",
            encoding="utf-8",
        )
        service = UserDiscoveryService.from_config(
            TwinrConfig(project_root=temp_dir, personality_dir="personality")
        )
        now = datetime(2026, 3, 27, 10, 0, tzinfo=timezone.utc)
        curated_invite = service.build_invitation(now=now)
        service.store.save(
            UserDiscoveryState(
                phase="lifelong_learning",
                session_state="idle",
                setup_completed_at=(now - timedelta(days=20)).isoformat(),
                last_reviewed_at=(now - timedelta(days=3)).isoformat(),
                topics=(
                    UserDiscoveryTopicState(
                        topic_id="basics",
                        correction_count=1,
                        last_answer_at=(now - timedelta(days=1)).isoformat(),
                        stored_facts=(
                            UserDiscoveryStoredFact(
                                fact_id="fact-basics",
                                route_kind="user_profile",
                                review_text="User prefers to be called Thom.",
                                created_at=(now - timedelta(days=10)).isoformat(),
                                updated_at=(now - timedelta(days=1)).isoformat(),
                            ),
                        ),
                    ),
                    UserDiscoveryTopicState(
                        topic_id="social",
                        last_answer_at=(now - timedelta(days=2)).isoformat(),
                        stored_facts=(
                            UserDiscoveryStoredFact(
                                fact_id="fact-social",
                                route_kind="contact",
                                review_text="Anna is the user's daughter.",
                                created_at=(now - timedelta(days=9)).isoformat(),
                                updated_at=(now - timedelta(days=2)).isoformat(),
                            ),
                        ),
                    ),
                    UserDiscoveryTopicState(
                        topic_id="interests",
                        last_answer_at=(now - timedelta(days=4)).isoformat(),
                        stored_facts=(
                            UserDiscoveryStoredFact(
                                fact_id="fact-interests",
                                route_kind="preference",
                                review_text="User prefers Melitta coffee.",
                                created_at=(now - timedelta(days=8)).isoformat(),
                                updated_at=(now - timedelta(days=4)).isoformat(),
                            ),
                        ),
                    ),
                ),
            )
        )
        review_due_invite = service.build_invitation(now=now)
        return {
            "curated_invite": None if curated_invite is None else asdict(curated_invite),
            "review_due_invite": None if review_due_invite is None else asdict(review_due_invite),
        }


class UserDiscoveryRefactorParityTests(unittest.TestCase):
    def test_public_wrapper_preserves_service_module(self) -> None:
        self.assertEqual(UserDiscoveryService.__module__, "twinr.memory.user_discovery")

    def test_public_wrapper_matches_internal_exports_when_impl_package_exists(self) -> None:
        if user_discovery_impl is None:
            self.skipTest("internal implementation package does not exist yet")
        self.assertIs(UserDiscoveryService, user_discovery_impl.UserDiscoveryService)
        self.assertIs(UserDiscoveryFact, user_discovery_impl.UserDiscoveryFact)
        self.assertIs(UserDiscoveryMemoryRoute, user_discovery_impl.UserDiscoveryMemoryRoute)

    def test_initial_progression_golden_master(self) -> None:
        self.assertEqual(
            _payload_digest(_capture_initial_progression()),
            _EXPECTED_GOLDEN_DIGESTS["initial"],
        )

    def test_review_replace_delete_golden_master(self) -> None:
        self.assertEqual(
            _payload_digest(_capture_review_replace_delete_flow()),
            _EXPECTED_GOLDEN_DIGESTS["review"],
        )

    def test_invitation_selection_golden_master(self) -> None:
        self.assertEqual(
            _payload_digest(_capture_invitation_selection()),
            _EXPECTED_GOLDEN_DIGESTS["invites"],
        )


if __name__ == "__main__":
    unittest.main()
