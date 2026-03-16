from dataclasses import dataclass
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.self_coding import (
    CapabilityStatus,
    CompileTarget,
    FeasibilityOutcome,
    SelfCodingCapabilityRegistry,
    SelfCodingFeasibilityChecker,
    SkillSpec,
    SkillTriggerSpec,
)


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


class SelfCodingFeasibilityCheckerTests(unittest.TestCase):
    def make_checker(self, runtime: _FakeManagedIntegrationsRuntime | None = None) -> SelfCodingFeasibilityChecker:
        registry = SelfCodingCapabilityRegistry(
            project_root=".",
            integration_runtime_factory=lambda *args, **kwargs: runtime or _FakeManagedIntegrationsRuntime(
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
                        status="ok",
                        summary="Calendar is ready.",
                        detail="Configured calendar is ready.",
                    ),
                )
            ),
        )
        return SelfCodingFeasibilityChecker(registry)

    def test_push_skill_inside_automation_envelope_is_green(self) -> None:
        checker = self.make_checker()
        spec = SkillSpec(
            name="Read New Emails",
            action="Read new email aloud",
            trigger=SkillTriggerSpec(mode="push", conditions=("new_email", "user_visible")),
            scope={"channel": "email", "selection": "all"},
            constraints=("not_during_night_mode",),
            capabilities=("email", "speaker", "safety"),
        )

        result = checker.check(spec)

        self.assertEqual(result.outcome, FeasibilityOutcome.GREEN)
        self.assertEqual(result.suggested_target, CompileTarget.AUTOMATION_MANIFEST)
        self.assertIn("automation-first", result.summary)

    def test_pull_skill_is_yellow_and_requires_skill_package_path(self) -> None:
        checker = self.make_checker()
        spec = SkillSpec(
            name="Check Calendar On Request",
            action="Read upcoming appointments",
            trigger=SkillTriggerSpec(mode="pull", conditions=("on_request",)),
            scope={"channel": "calendar"},
            capabilities=("calendar", "speaker"),
        )

        result = checker.check(spec)

        self.assertEqual(result.outcome, FeasibilityOutcome.YELLOW)
        self.assertEqual(result.suggested_target, CompileTarget.SKILL_PACKAGE)
        self.assertTrue(any("Pull-style" in reason for reason in result.reasons))

    def test_unconfigured_capability_is_red(self) -> None:
        checker = self.make_checker(
            runtime=_FakeManagedIntegrationsRuntime(
                readiness=(
                    _FakeIntegrationReadiness(
                        integration_id="calendar_agenda",
                        label="Calendar",
                        status="warn",
                        summary="Calendar needs setup.",
                        detail="Missing ICS source.",
                    ),
                )
            )
        )
        spec = SkillSpec(
            name="Read Calendar",
            action="Read the calendar aloud",
            trigger=SkillTriggerSpec(mode="push", conditions=("calendar_event_due",)),
            scope={"channel": "calendar"},
            capabilities=("calendar", "speaker"),
        )

        result = checker.check(spec)

        self.assertEqual(result.outcome, FeasibilityOutcome.RED)
        self.assertEqual(result.missing_capabilities, ("calendar",))
        self.assertIn("Missing ICS source.", result.reasons[0])
        self.assertEqual(result.suggested_target, None)


if __name__ == "__main__":
    unittest.main()
