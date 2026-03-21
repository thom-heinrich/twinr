from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.tools.runtime.broker_policy import AutomationToolBrokerPolicy


class _FakeToolExecutor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.smart_home_calls: list[dict[str, object]] = []

    def handle_print_receipt(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(payload)
        return {"status": "printed"}

    def handle_control_smart_home_entities(self, payload: dict[str, object]) -> dict[str, object]:
        self.smart_home_calls.append(payload)
        return {"status": "ok"}


class AutomationToolBrokerPolicyTests(unittest.TestCase):
    def test_policy_allows_only_safe_tools_and_resolves_handlers(self) -> None:
        policy = AutomationToolBrokerPolicy()
        executor = _FakeToolExecutor()

        self.assertTrue(policy.is_allowed("print_receipt"))
        self.assertFalse(policy.is_allowed("delete_automation"))
        handler = policy.resolve_handler(executor, "print_receipt")
        result = handler({"text": "Hallo"})

        self.assertEqual(result["status"], "printed")
        self.assertEqual(executor.calls, [{"text": "Hallo"}])
        with self.assertRaises(RuntimeError):
            policy.resolve_handler(executor, "delete_automation")

    def test_policy_allows_low_risk_smart_home_control_for_background_automations(self) -> None:
        policy = AutomationToolBrokerPolicy()
        executor = _FakeToolExecutor()

        self.assertTrue(policy.is_allowed("control_smart_home_entities"))
        handler = policy.resolve_handler(executor, "control_smart_home_entities")
        result = handler({"entity_ids": ["kitchen-light"], "command": "turn_on", "confirmed": True})

        self.assertEqual(result["status"], "ok")
        self.assertEqual(
            executor.smart_home_calls,
            [{"entity_ids": ["kitchen-light"], "command": "turn_on", "confirmed": True}],
        )


if __name__ == "__main__":
    unittest.main()
