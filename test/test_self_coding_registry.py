from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.self_coding import CapabilityStatus, SelfCodingCapabilityRegistry


@dataclass(frozen=True, slots=True)
class _FakeIntegrationReadiness:
    integration_id: str
    label: str
    status: str
    summary: str
    detail: str


@dataclass(frozen=True, slots=True)
class _FakeManagedIntegrationsRuntime:
    readiness: tuple[_FakeIntegrationReadiness, ...] = ()


class SelfCodingCapabilityRegistryTests(unittest.TestCase):
    def test_registry_exposes_expected_mvp_capabilities(self) -> None:
        registry = SelfCodingCapabilityRegistry(
            project_root=".",
            integration_runtime_factory=lambda *args, **kwargs: _FakeManagedIntegrationsRuntime(),
        )

        definitions = registry.definitions()

        self.assertEqual(
            tuple(definition.capability_id for definition in definitions),
            (
                "camera",
                "pir",
                "speaker",
                "llm_call",
                "memory",
                "scheduler",
                "rules",
                "safety",
                "email",
                "calendar",
            ),
        )

    def test_registry_maps_integration_readiness_to_capability_status(self) -> None:
        runtime = _FakeManagedIntegrationsRuntime(
            readiness=(
                _FakeIntegrationReadiness(
                    integration_id="email_mailbox",
                    label="Email",
                    status="ok",
                    summary="Email is ready.",
                    detail="Configured mailbox is ready.",
                ),
                _FakeIntegrationReadiness(
                    integration_id="calendar_agenda",
                    label="Calendar",
                    status="warn",
                    summary="Calendar needs setup.",
                    detail="Missing ICS source.",
                ),
            )
        )
        registry = SelfCodingCapabilityRegistry(
            project_root=".",
            integration_runtime_factory=lambda *args, **kwargs: runtime,
        )

        email = registry.availability_for("email")
        calendar = registry.availability_for("calendar")

        assert email is not None
        assert calendar is not None
        self.assertEqual(email.status, CapabilityStatus.READY)
        self.assertTrue(email.configured)
        self.assertEqual(calendar.status, CapabilityStatus.UNCONFIGURED)
        self.assertFalse(calendar.configured)
        self.assertIn("Missing ICS source.", calendar.detail)

    def test_registry_blocks_integration_capabilities_when_runtime_load_fails(self) -> None:
        def failing_factory(*args, **kwargs):
            raise RuntimeError("integration store unavailable")

        with tempfile.TemporaryDirectory() as temp_dir:
            registry = SelfCodingCapabilityRegistry(
                project_root=Path(temp_dir),
                integration_runtime_factory=failing_factory,
            )

            email = registry.availability_for("email")
            calendar = registry.availability_for("calendar")

        assert email is not None
        assert calendar is not None
        self.assertEqual(email.status, CapabilityStatus.BLOCKED)
        self.assertEqual(calendar.status, CapabilityStatus.BLOCKED)
        self.assertIn("integration store unavailable", email.detail)


if __name__ == "__main__":
    unittest.main()
