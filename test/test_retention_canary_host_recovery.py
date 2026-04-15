from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.ops.retention_canary_host_recovery import (
    retention_canary_host_recovery_eligible,
    stabilize_retention_canary_host,
)


class RetentionCanaryHostRecoveryTests(unittest.TestCase):
    def test_queue_saturated_exception_chain_is_contention_eligible(self) -> None:
        payload = {
            "status": "failed",
            "failure_stage": "seed_retention_objects",
            "error_message": (
                "LongTermRemoteUnavailableError: Failed to persist fine-grained "
                "remote long-term memory items."
            ),
            "root_cause_message": "ChonkyDBError: ChonkyDB request failed for POST /v1/external/records/bulk (status=429)",
            "remote_write_context": {
                "operation": "store_records_bulk",
                "request_path": "/v1/external/records/bulk",
                "request_execution_mode": "async",
            },
            "exception_chain": [
                {
                    "type": "LongTermRemoteUnavailableError",
                    "detail": "Failed to persist fine-grained remote long-term memory items.",
                },
                {
                    "type": "ChonkyDBError",
                    "detail": "ChonkyDB request failed for POST /v1/external/records/bulk (status=429)",
                    "status_code": 429,
                    "response_json": {
                        "detail": "queue_saturated",
                        "error": "queue_saturated",
                    },
                },
            ],
        }
        diagnosis = {
            "available": True,
            "contention_detected": True,
            "contention_signals": [
                "public_query_unhealthy",
                "backend_query_unhealthy",
                "backend_memory_current_high",
                "active_system_conflicts",
            ],
        }

        self.assertTrue(
            retention_canary_host_recovery_eligible(
                canary_payload=payload,
                diagnosis=diagnosis,
            )
        )

    def test_stabilize_retention_canary_host_repairs_backend_when_public_stays_unhealthy(self) -> None:
        settings = object()
        stabilization_result = SimpleNamespace(
            ok=False,
            diagnosis="public_still_unhealthy_after_host_stabilization",
            to_dict=lambda: {
                "ok": False,
                "diagnosis": "public_still_unhealthy_after_host_stabilization",
                "public_after": {"ready": False},
            },
        )
        repair_result = SimpleNamespace(
            ok=True,
            diagnosis="backend_local_unhealthy",
            to_dict=lambda: {
                "ok": True,
                "diagnosis": "backend_local_unhealthy",
                "action_taken": "restart_backend_service",
            },
        )
        diagnosis = {
            "backend_service": {
                "load_state": "loaded",
                "active_state": "deactivating",
                "sub_state": "stop-sigterm",
            },
            "backend_probe": {"ready": False},
            "public_probe": {"ready": False},
        }
        diagnosis_after_backend_repair = {
            "available": True,
            "contention_detected": False,
            "contention_signals": [],
            "backend_probe": {"ready": True},
            "public_probe": {"ready": True},
        }

        with (
            patch(
                "twinr.ops.retention_canary_host_recovery.load_remote_chonkydb_ops_settings",
                return_value=settings,
            ),
            patch(
                "twinr.ops.retention_canary_host_recovery.stabilize_remote_chonkydb_host",
                return_value=stabilization_result,
            ),
            patch(
                "twinr.ops.retention_canary_host_recovery.diagnose_retention_canary_host_contention",
                side_effect=[diagnosis, diagnosis_after_backend_repair],
            ),
            patch(
                "twinr.ops.retention_canary_host_recovery.repair_remote_chonkydb",
                return_value=repair_result,
            ) as repair_mock,
        ):
            result = stabilize_retention_canary_host(
                project_root="/tmp/twinr",
                probe_timeout_s=10.0,
                ssh_timeout_s=60.0,
                settle_s=8.0,
                repair_wait_ready_s=120.0,
                repair_poll_interval_s=3.0,
                diagnosis=diagnosis,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(
            result["diagnosis"],
            "public_recovered_after_host_stabilization_and_backend_repair",
        )
        backend_repair = cast(dict[str, object], result["backend_repair"])
        self.assertIsInstance(backend_repair, dict)
        self.assertEqual(
            backend_repair["action_taken"],
            "restart_backend_service",
        )
        repair_mock.assert_called_once_with(
            settings=settings,
            probe_timeout_s=10.0,
            ssh_timeout_s=60.0,
            wait_ready_s=120.0,
            poll_interval_s=3.0,
            restart_if_needed=True,
        )

    def test_stabilize_retention_canary_host_rediagnoses_after_stabilization_before_repair(self) -> None:
        settings = object()
        stabilization_result = SimpleNamespace(
            ok=False,
            diagnosis="public_still_unhealthy_after_host_stabilization",
            to_dict=lambda: {
                "ok": False,
                "diagnosis": "public_still_unhealthy_after_host_stabilization",
                "public_after": {"ready": False},
            },
        )
        repair_result = SimpleNamespace(
            ok=True,
            diagnosis="backend_local_unhealthy",
            to_dict=lambda: {
                "ok": True,
                "diagnosis": "backend_local_unhealthy",
                "action_taken": "restart_backend_service",
            },
        )
        initial_diagnosis = {
            "backend_service": {
                "load_state": "loaded",
                "active_state": "active",
                "sub_state": "running",
            },
            "backend_probe": {"ready": True},
            "public_probe": {"ready": True},
        }
        refreshed_diagnosis = {
            "backend_service": {
                "load_state": "loaded",
                "active_state": "active",
                "sub_state": "running",
            },
            "backend_probe": {"ready": False},
            "public_probe": {"ready": False},
            "contention_detected": True,
            "contention_signals": [
                "public_query_unhealthy",
                "backend_query_unhealthy",
                "backend_memory_current_high",
            ],
        }
        diagnosis_after_backend_repair = {
            "available": True,
            "contention_detected": False,
            "contention_signals": [],
            "backend_probe": {"ready": True},
            "public_probe": {"ready": True},
        }

        with (
            patch(
                "twinr.ops.retention_canary_host_recovery.load_remote_chonkydb_ops_settings",
                return_value=settings,
            ),
            patch(
                "twinr.ops.retention_canary_host_recovery.stabilize_remote_chonkydb_host",
                return_value=stabilization_result,
            ),
            patch(
                "twinr.ops.retention_canary_host_recovery.diagnose_retention_canary_host_contention",
                side_effect=[refreshed_diagnosis, diagnosis_after_backend_repair],
            ) as rediagnose_mock,
            patch(
                "twinr.ops.retention_canary_host_recovery.repair_remote_chonkydb",
                return_value=repair_result,
            ) as repair_mock,
        ):
            result = stabilize_retention_canary_host(
                project_root="/tmp/twinr",
                probe_timeout_s=10.0,
                ssh_timeout_s=60.0,
                settle_s=8.0,
                repair_wait_ready_s=120.0,
                repair_poll_interval_s=3.0,
                diagnosis=initial_diagnosis,
            )

        self.assertTrue(result["ok"])
        diagnosis_after = cast(dict[str, object], result["diagnosis_after_stabilization"])
        self.assertIsInstance(diagnosis_after, dict)
        self.assertEqual(
            diagnosis_after["contention_signals"],
            [
                "public_query_unhealthy",
                "backend_query_unhealthy",
                "backend_memory_current_high",
            ],
        )
        self.assertEqual(rediagnose_mock.call_count, 2)
        self.assertEqual(
            rediagnose_mock.call_args_list[0].kwargs,
            {
                "project_root": Path("/tmp/twinr").resolve(),
                "probe_timeout_s": 10.0,
                "ssh_timeout_s": 60.0,
            },
        )
        self.assertEqual(
            rediagnose_mock.call_args_list[1].kwargs,
            {
                "project_root": Path("/tmp/twinr").resolve(),
                "probe_timeout_s": 10.0,
                "ssh_timeout_s": 60.0,
            },
        )
        repair_mock.assert_called_once()

    def test_stabilize_retention_canary_host_restabilizes_after_repair_when_conflicts_return(self) -> None:
        settings = object()
        initial_stabilization = SimpleNamespace(
            ok=False,
            diagnosis="public_still_unhealthy_after_host_stabilization",
            to_dict=lambda: {
                "ok": False,
                "diagnosis": "public_still_unhealthy_after_host_stabilization",
                "public_after": {"ready": False},
            },
        )
        post_repair_stabilization = SimpleNamespace(
            ok=True,
            diagnosis="public_ready_after_host_stabilization",
            to_dict=lambda: {
                "ok": True,
                "diagnosis": "public_ready_after_host_stabilization",
                "public_after": {"ready": True},
            },
        )
        repair_result = SimpleNamespace(
            ok=True,
            diagnosis="backend_local_unhealthy",
            to_dict=lambda: {
                "ok": True,
                "diagnosis": "backend_local_unhealthy",
                "action_taken": "restart_backend_service",
            },
        )
        initial_diagnosis = {
            "backend_service": {
                "load_state": "loaded",
                "active_state": "deactivating",
                "sub_state": "stop-sigterm",
            },
            "backend_probe": {"ready": False},
            "public_probe": {"ready": False},
        }
        refreshed_before_repair = {
            "available": True,
            "contention_detected": True,
            "contention_signals": ["public_query_unhealthy", "backend_query_unhealthy"],
            "backend_service": {
                "load_state": "loaded",
                "active_state": "deactivating",
                "sub_state": "stop-sigterm",
            },
            "backend_probe": {"ready": False},
            "public_probe": {"ready": False},
        }
        diagnosis_after_backend_repair = {
            "available": True,
            "contention_detected": True,
            "contention_signals": ["active_system_conflicts"],
            "backend_probe": {"ready": True},
            "public_probe": {"ready": True},
        }
        diagnosis_after_post_repair_stabilization = {
            "available": True,
            "contention_detected": False,
            "contention_signals": [],
            "backend_probe": {"ready": True},
            "public_probe": {"ready": True},
        }

        with (
            patch(
                "twinr.ops.retention_canary_host_recovery.load_remote_chonkydb_ops_settings",
                return_value=settings,
            ),
            patch(
                "twinr.ops.retention_canary_host_recovery.stabilize_remote_chonkydb_host",
                side_effect=[initial_stabilization, post_repair_stabilization],
            ) as stabilize_mock,
            patch(
                "twinr.ops.retention_canary_host_recovery.diagnose_retention_canary_host_contention",
                side_effect=[
                    refreshed_before_repair,
                    diagnosis_after_backend_repair,
                    diagnosis_after_post_repair_stabilization,
                ],
            ) as diagnose_mock,
            patch(
                "twinr.ops.retention_canary_host_recovery.repair_remote_chonkydb",
                return_value=repair_result,
            ) as repair_mock,
        ):
            result = stabilize_retention_canary_host(
                project_root="/tmp/twinr",
                probe_timeout_s=10.0,
                ssh_timeout_s=60.0,
                settle_s=8.0,
                repair_wait_ready_s=120.0,
                repair_poll_interval_s=3.0,
                diagnosis=initial_diagnosis,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(
            cast(dict[str, object], result["diagnosis_after_backend_repair"])["contention_signals"],
            ["active_system_conflicts"],
        )
        self.assertIn("post_repair_stabilization", result)
        self.assertEqual(stabilize_mock.call_count, 2)
        self.assertEqual(diagnose_mock.call_count, 3)
        repair_mock.assert_called_once()

    def test_stabilize_retention_canary_host_fails_when_post_repair_contention_persists(self) -> None:
        settings = object()
        initial_stabilization = SimpleNamespace(
            ok=False,
            diagnosis="public_still_unhealthy_after_host_stabilization",
            to_dict=lambda: {
                "ok": False,
                "diagnosis": "public_still_unhealthy_after_host_stabilization",
                "public_after": {"ready": False},
            },
        )
        post_repair_stabilization = SimpleNamespace(
            ok=True,
            diagnosis="public_ready_after_host_stabilization",
            to_dict=lambda: {
                "ok": True,
                "diagnosis": "public_ready_after_host_stabilization",
                "public_after": {"ready": True},
            },
        )
        repair_result = SimpleNamespace(
            ok=True,
            diagnosis="backend_local_unhealthy",
            to_dict=lambda: {
                "ok": True,
                "diagnosis": "backend_local_unhealthy",
                "action_taken": "restart_backend_service",
            },
        )
        initial_diagnosis = {
            "backend_service": {
                "load_state": "loaded",
                "active_state": "deactivating",
                "sub_state": "stop-sigterm",
            },
            "backend_probe": {"ready": False},
            "public_probe": {"ready": False},
        }
        refreshed_before_repair = {
            "available": True,
            "contention_detected": True,
            "contention_signals": ["public_query_unhealthy", "backend_query_unhealthy"],
            "backend_service": {
                "load_state": "loaded",
                "active_state": "deactivating",
                "sub_state": "stop-sigterm",
            },
            "backend_probe": {"ready": False},
            "public_probe": {"ready": False},
        }
        diagnosis_after_backend_repair = {
            "available": True,
            "contention_detected": True,
            "contention_signals": ["active_system_conflicts"],
            "backend_probe": {"ready": True},
            "public_probe": {"ready": True},
        }
        diagnosis_after_post_repair_stabilization = {
            "available": True,
            "contention_detected": True,
            "contention_signals": ["active_system_conflicts"],
            "backend_probe": {"ready": True},
            "public_probe": {"ready": True},
        }

        with (
            patch(
                "twinr.ops.retention_canary_host_recovery.load_remote_chonkydb_ops_settings",
                return_value=settings,
            ),
            patch(
                "twinr.ops.retention_canary_host_recovery.stabilize_remote_chonkydb_host",
                side_effect=[initial_stabilization, post_repair_stabilization],
            ),
            patch(
                "twinr.ops.retention_canary_host_recovery.diagnose_retention_canary_host_contention",
                side_effect=[
                    refreshed_before_repair,
                    diagnosis_after_backend_repair,
                    diagnosis_after_post_repair_stabilization,
                ],
            ),
            patch(
                "twinr.ops.retention_canary_host_recovery.repair_remote_chonkydb",
                return_value=repair_result,
            ),
        ):
            result = stabilize_retention_canary_host(
                project_root="/tmp/twinr",
                probe_timeout_s=10.0,
                ssh_timeout_s=60.0,
                settle_s=8.0,
                repair_wait_ready_s=120.0,
                repair_poll_interval_s=3.0,
                diagnosis=initial_diagnosis,
            )

        self.assertFalse(result["ok"])
        self.assertEqual(
            result["diagnosis"],
            "post_repair_contention_persisted_after_host_restabilization",
        )
        self.assertIn("post_repair_stabilization", result)


if __name__ == "__main__":
    unittest.main()
