from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.self_coding.codex_driver.app_server import CodexAppServerRunCollector
from twinr.agent.self_coding.codex_driver.exec_fallback import CodexExecRunCollector


class CodexExecRunCollectorTests(unittest.TestCase):
    def test_exec_collector_tracks_final_message_and_events(self) -> None:
        collector = CodexExecRunCollector()

        collector.consume({"type": "thread.started", "thread_id": "thread-123"})
        collector.consume({"type": "turn.started"})
        collector.consume({"type": "item.completed", "item": {"id": "item_0", "type": "agent_message", "text": "{\"status\":\"ok\"}"}})
        collector.consume({"type": "turn.completed", "usage": {"output_tokens": 12}})

        result = collector.build_result()

        self.assertEqual(result.thread_id, "thread-123")
        self.assertEqual(result.final_message, "{\"status\":\"ok\"}")
        self.assertEqual(result.events[-1].kind, "turn_completed")
        self.assertEqual(result.events[-1].metadata["output_tokens"], 12)


class CodexAppServerRunCollectorTests(unittest.TestCase):
    def test_app_server_collector_tracks_agent_message_deltas_and_completion(self) -> None:
        collector = CodexAppServerRunCollector()

        collector.consume({"method": "thread/started", "params": {"thread": {"id": "thread-123"}}})
        collector.consume({"method": "turn/started", "params": {"threadId": "thread-123", "turn": {"id": "turn-123", "status": "inProgress", "items": []}}})
        collector.consume(
            {
                "method": "item/started",
                "params": {
                    "threadId": "thread-123",
                    "turnId": "turn-123",
                    "item": {"type": "agentMessage", "id": "msg-1", "text": "", "phase": "final_answer"},
                },
            }
        )
        collector.consume(
            {
                "method": "item/agentMessage/delta",
                "params": {"threadId": "thread-123", "turnId": "turn-123", "itemId": "msg-1", "delta": "{\"status\":\"ok\"}"},
            }
        )
        collector.consume(
            {
                "method": "item/completed",
                "params": {
                    "threadId": "thread-123",
                    "turnId": "turn-123",
                    "item": {"type": "agentMessage", "id": "msg-1", "text": "{\"status\":\"ok\"}", "phase": "final_answer"},
                },
            }
        )
        collector.consume({"method": "turn/completed", "params": {"threadId": "thread-123", "turn": {"id": "turn-123", "status": "completed", "items": [], "error": None}}})

        result = collector.build_result()

        self.assertEqual(result.thread_id, "thread-123")
        self.assertEqual(result.turn_id, "turn-123")
        self.assertEqual(result.final_message, "{\"status\":\"ok\"}")
        self.assertEqual(result.events[-1].kind, "turn_completed")


if __name__ == "__main__":
    unittest.main()
