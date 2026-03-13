from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.integrations import (
    CalendarEvent,
    CallableIntegrationAdapter,
    ConfirmationMode,
    IntegrationDecision,
    IntegrationDomain,
    IntegrationPolicyError,
    IntegrationRegistry,
    IntegrationRegistryError,
    IntegrationRequest,
    IntegrationResult,
    RequestOrigin,
    SafeIntegrationPolicy,
    builtin_manifests,
    manifest_for_id,
)
from twinr.integrations.models import IntegrationAction, IntegrationManifest, IntegrationOperation, RiskLevel, SafetyProfile


def requested_now():
    from datetime import UTC, datetime

    return datetime(2026, 3, 13, 10, 0, tzinfo=UTC)


class IntegrationCatalogTests(unittest.TestCase):
    def test_builtin_catalog_covers_requested_domains(self) -> None:
        manifests = builtin_manifests()
        self.assertEqual(
            {manifest.domain for manifest in manifests},
            {
                IntegrationDomain.CALENDAR,
                IntegrationDomain.EMAIL,
                IntegrationDomain.MESSENGER,
                IntegrationDomain.SMART_HOME,
                IntegrationDomain.SECURITY,
                IntegrationDomain.HEALTH,
            },
        )

    def test_manifest_lookup_returns_expected_manifest(self) -> None:
        manifest = manifest_for_id("calendar_agenda")
        self.assertIsNotNone(manifest)
        self.assertEqual(manifest.title, "Calendar Agenda")


class IntegrationRequestTests(unittest.TestCase):
    def test_redacted_parameters_hide_message_body(self) -> None:
        request = IntegrationRequest(
            integration_id="messenger_bridge",
            operation_id="send_message",
            parameters={"body": "Hallo", "thread_id": "family"},
        )

        self.assertEqual(
            request.redacted_parameters(),
            {"body": "<redacted>", "thread_id": "family"},
        )


class SafeIntegrationPolicyTests(unittest.TestCase):
    def test_calendar_queries_do_not_need_confirmation_by_default(self) -> None:
        manifest = manifest_for_id("calendar_agenda")
        assert manifest is not None
        policy = SafeIntegrationPolicy(enabled_integrations=frozenset({"calendar_agenda"}))
        request = IntegrationRequest(
            integration_id="calendar_agenda",
            operation_id="read_today",
        )

        decision = policy.evaluate(manifest, request)

        self.assertTrue(decision.allowed)

    def test_email_send_requires_explicit_user_confirmation(self) -> None:
        manifest = manifest_for_id("email_mailbox")
        assert manifest is not None
        policy = SafeIntegrationPolicy(enabled_integrations=frozenset({"email_mailbox"}))
        request = IntegrationRequest(
            integration_id="email_mailbox",
            operation_id="send_message",
            parameters={"to": "caregiver@example.com", "body": "Hallo"},
        )

        decision = policy.evaluate(manifest, request)

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.required_confirmation, ConfirmationMode.USER)

    def test_security_requests_reject_remote_origin_by_default(self) -> None:
        manifest = manifest_for_id("security_monitor")
        assert manifest is not None
        policy = SafeIntegrationPolicy(enabled_integrations=frozenset({"security_monitor"}))
        request = IntegrationRequest(
            integration_id="security_monitor",
            operation_id="read_status",
            origin=RequestOrigin.REMOTE_SERVICE,
            explicit_user_confirmation=True,
        )

        decision = policy.evaluate(manifest, request)

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "Remote-triggered integration requests are disabled.")

    def test_health_queries_require_user_confirmation(self) -> None:
        manifest = manifest_for_id("health_records")
        assert manifest is not None
        policy = SafeIntegrationPolicy(enabled_integrations=frozenset({"health_records"}))
        request = IntegrationRequest(
            integration_id="health_records",
            operation_id="read_daily_summary",
        )

        decision = policy.evaluate(manifest, request)

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.required_confirmation, ConfirmationMode.USER)

    def test_critical_operations_are_denied_by_default(self) -> None:
        manifest = IntegrationManifest(
            integration_id="custom_security",
            domain=IntegrationDomain.SECURITY,
            title="Custom Security",
            summary="Custom security adapter.",
            operations=(
                IntegrationOperation(
                    operation_id="critical_action",
                    label="Critical action",
                    action=IntegrationAction.CONTROL,
                    summary="Dangerous action.",
                    safety=SafetyProfile(
                        risk=RiskLevel.CRITICAL,
                        confirmation=ConfirmationMode.CAREGIVER,
                    ),
                ),
            ),
        )
        policy = SafeIntegrationPolicy(enabled_integrations=frozenset({"custom_security"}))
        request = IntegrationRequest(
            integration_id="custom_security",
            operation_id="critical_action",
            explicit_caregiver_confirmation=True,
        )

        decision = policy.evaluate(manifest, request)

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "Critical integration operations are disabled.")


class IntegrationRegistryTests(unittest.TestCase):
    def test_registry_can_dispatch_calendar_adapter(self) -> None:
        manifest = manifest_for_id("calendar_agenda")
        assert manifest is not None
        adapter = CallableIntegrationAdapter(
            manifest=manifest,
            handler=lambda request: IntegrationResult(
                ok=True,
                summary="agenda",
                details={"events": [CalendarEvent("evt-1", "Arzt", starts_at=requested_now()).as_dict()]},
            ),
        )
        registry = IntegrationRegistry((adapter,))
        policy = SafeIntegrationPolicy(enabled_integrations=frozenset({"calendar_agenda"}))
        request = IntegrationRequest(
            integration_id="calendar_agenda",
            operation_id="read_today",
        )

        result = registry.dispatch(request, policy=policy)

        self.assertTrue(result.ok)
        self.assertEqual(result.summary, "agenda")

    def test_duplicate_registration_is_rejected(self) -> None:
        manifest = manifest_for_id("email_mailbox")
        assert manifest is not None
        adapter = CallableIntegrationAdapter(
            manifest=manifest,
            handler=lambda request: IntegrationResult(ok=True, summary=request.audit_label()),
        )
        registry = IntegrationRegistry((adapter,))

        with self.assertRaises(IntegrationRegistryError):
            registry.register(adapter)

    def test_dispatch_runs_adapter_when_policy_allows(self) -> None:
        manifest = manifest_for_id("email_mailbox")
        assert manifest is not None
        adapter = CallableIntegrationAdapter(
            manifest=manifest,
            handler=lambda request: IntegrationResult(
                ok=True,
                summary="sent",
                details={"request": request.redacted_parameters()},
            ),
        )
        registry = IntegrationRegistry((adapter,))
        policy = SafeIntegrationPolicy(enabled_integrations=frozenset({"email_mailbox"}))
        request = IntegrationRequest(
            integration_id="email_mailbox",
            operation_id="send_message",
            parameters={"to": "caregiver@example.com", "body": "Hallo"},
            explicit_user_confirmation=True,
        )

        result = registry.dispatch(request, policy=policy)

        self.assertTrue(result.ok)
        self.assertEqual(result.summary, "sent")
        self.assertEqual(result.details["request"]["body"], "<redacted>")

    def test_dispatch_raises_policy_error_when_confirmation_is_missing(self) -> None:
        manifest = manifest_for_id("messenger_bridge")
        assert manifest is not None
        adapter = CallableIntegrationAdapter(
            manifest=manifest,
            handler=lambda request: IntegrationResult(ok=True, summary="sent"),
        )
        registry = IntegrationRegistry((adapter,))
        policy = SafeIntegrationPolicy(enabled_integrations=frozenset({"messenger_bridge"}))
        request = IntegrationRequest(
            integration_id="messenger_bridge",
            operation_id="send_message",
            parameters={"body": "Hallo"},
        )

        with self.assertRaises(IntegrationPolicyError):
            registry.dispatch(request, policy=policy)

    def test_dry_run_skips_adapter_execution(self) -> None:
        manifest = manifest_for_id("smart_home_hub")
        assert manifest is not None

        called = {"value": False}

        def _handler(request: IntegrationRequest) -> IntegrationResult:
            called["value"] = True
            return IntegrationResult(ok=True, summary="scene run")

        adapter = CallableIntegrationAdapter(manifest=manifest, handler=_handler)
        registry = IntegrationRegistry((adapter,))
        policy = SafeIntegrationPolicy(enabled_integrations=frozenset({"smart_home_hub"}))
        request = IntegrationRequest(
            integration_id="smart_home_hub",
            operation_id="run_safe_scene",
            explicit_user_confirmation=True,
            dry_run=True,
        )

        result = registry.dispatch(request, policy=policy)

        self.assertTrue(result.ok)
        self.assertFalse(called["value"])
        self.assertIn("Dry run approved", result.summary)


if __name__ == "__main__":
    unittest.main()
