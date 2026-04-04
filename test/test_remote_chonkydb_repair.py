"""Regression tests for remote ChonkyDB outage diagnosis and repair planning."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from twinr.ops.remote_chonkydb_repair import (
    BackendQuerySurfaceReadinessContract,
    BackendQuerySurfaceReadinessRepairStatus,
    BackendDataOwnershipState,
    ChonkyDBHttpProbeResult,
    ChonkyDBRemoteServiceState,
    ForeignBackendConsumer,
    RemoteChonkyDBOpsSettings,
    _backend_query_surface_readiness_contract_from_env,
    _query_probe_empty_but_healthy,
    ensure_backend_query_surface_readiness_contract,
    plan_remote_chonkydb_repair,
    probe_public_chonkydb,
    _parse_systemd_environment_output,
    _query_surface_canary_payload,
    _require_query_surface_ready,
)
from twinr.ops.remote_systemd_restart_guard import RemoteManualRestartProtectionStatus
from twinr.ops.self_coding_pi import PiConnectionSettings


class RemoteChonkyDBRepairPlanTests(unittest.TestCase):
    """Cover the no-blind-restart decision logic for remote ChonkyDB repair."""

    def test_plan_skips_restart_when_public_endpoint_is_ready(self) -> None:
        plan = plan_remote_chonkydb_repair(
            public_probe=ChonkyDBHttpProbeResult(
                label="public",
                ok=True,
                status_code=200,
                ready=True,
                detail="ready",
            ),
            backend_service=ChonkyDBRemoteServiceState(
                active_state="active",
                sub_state="running",
                service_result="success",
            ),
            backend_probe=ChonkyDBHttpProbeResult(
                label="backend",
                ok=True,
                status_code=200,
                ready=True,
                detail="ready",
            ),
        )

        self.assertEqual(plan.action, "none")
        self.assertEqual(plan.reason, "public_ready")

    def test_plan_restarts_when_backend_service_is_not_active(self) -> None:
        plan = plan_remote_chonkydb_repair(
            public_probe=ChonkyDBHttpProbeResult(
                label="public",
                ok=False,
                status_code=503,
                ready=False,
                detail="public unavailable",
            ),
            backend_service=ChonkyDBRemoteServiceState(
                active_state="failed",
                sub_state="failed",
                service_result="exit-code",
            ),
            backend_probe=ChonkyDBHttpProbeResult(
                label="backend",
                ok=False,
                status_code=0,
                ready=False,
                detail="not probed",
            ),
        )

        self.assertEqual(plan.action, "restart_backend_service")
        self.assertEqual(plan.reason, "backend_service_inactive")

    def test_plan_avoids_restart_when_backend_data_ownership_drift_is_present(self) -> None:
        plan = plan_remote_chonkydb_repair(
            public_probe=ChonkyDBHttpProbeResult(
                label="public",
                ok=False,
                status_code=503,
                ready=False,
                detail="public unavailable",
            ),
            backend_service=ChonkyDBRemoteServiceState(
                active_state="failed",
                sub_state="failed",
                service_result="exit-code",
            ),
            backend_probe=ChonkyDBHttpProbeResult(
                label="backend",
                ok=False,
                status_code=503,
                ready=False,
                detail="backend unavailable",
            ),
            backend_data_ownership=BackendDataOwnershipState(
                data_dir="/srv/chonkydb/data",
                expected_user="thh",
                expected_group="thh",
                mismatched_entry_count=2,
                sample_entries=("root:root /srv/chonkydb/data/wal/chonkdb.wal.lock",),
            ),
        )

        self.assertEqual(plan.action, "none")
        self.assertEqual(plan.reason, "backend_data_permission_drift")

    def test_plan_avoids_blind_restart_when_public_proxy_is_the_only_failure(self) -> None:
        plan = plan_remote_chonkydb_repair(
            public_probe=ChonkyDBHttpProbeResult(
                label="public",
                ok=False,
                status_code=503,
                ready=False,
                detail="Upstream unavailable or restarting",
            ),
            backend_service=ChonkyDBRemoteServiceState(
                active_state="active",
                sub_state="running",
                service_result="success",
            ),
            backend_probe=ChonkyDBHttpProbeResult(
                label="backend",
                ok=True,
                status_code=200,
                ready=True,
                detail="backend ready",
            ),
        )

        self.assertEqual(plan.action, "none")
        self.assertEqual(plan.reason, "public_proxy_unhealthy")

    def test_plan_restarts_when_backend_local_probe_is_unhealthy(self) -> None:
        plan = plan_remote_chonkydb_repair(
            public_probe=ChonkyDBHttpProbeResult(
                label="public",
                ok=False,
                status_code=503,
                ready=False,
                detail="public unavailable",
            ),
            backend_service=ChonkyDBRemoteServiceState(
                active_state="active",
                sub_state="running",
                service_result="success",
            ),
            backend_probe=ChonkyDBHttpProbeResult(
                label="backend",
                ok=False,
                status_code=503,
                ready=False,
                detail="backend unavailable",
            ),
        )

        self.assertEqual(plan.action, "restart_backend_service")
        self.assertEqual(plan.reason, "backend_local_unhealthy")

    def test_plan_avoids_restart_when_active_foreign_consumers_contend_for_backend(self) -> None:
        plan = plan_remote_chonkydb_repair(
            public_probe=ChonkyDBHttpProbeResult(
                label="public",
                ok=False,
                status_code=503,
                ready=False,
                detail="public unavailable",
            ),
            backend_service=ChonkyDBRemoteServiceState(
                active_state="active",
                sub_state="running",
                service_result="success",
            ),
            backend_probe=ChonkyDBHttpProbeResult(
                label="backend",
                ok=False,
                status_code=504,
                ready=False,
                detail="query_payloads_advanced_timeout timeout_s=30.000",
            ),
            foreign_consumers=(
                ForeignBackendConsumer(
                    unit_name="caia-consumer-portal.service",
                    active_state="active",
                    sub_state="running",
                    configured_base_url="http://127.0.0.1:3044",
                    fragment_path="/etc/systemd/system/caia-consumer-portal.service",
                    coupled_to_backend_service=True,
                ),
            ),
        )

        self.assertEqual(plan.action, "none")
        self.assertEqual(plan.reason, "backend_foreign_consumer_contention")


class RemoteChonkyDBEnvironmentParsingTests(unittest.TestCase):
    """Verify the service-environment parser keeps quoted values intact."""

    def test_parse_systemd_environment_output_handles_quotes(self) -> None:
        environment = (
            'Environment=CHONKDB_API_KEY=secret '
            'CHONKDB_API_KEY_HEADER=x-api-key '
            'CHONKY_API_FULLTEXT_WARMUP_PROBE_QUERY="twinr dedicated instance"\n'
        )

        parsed = _parse_systemd_environment_output(environment)

        self.assertEqual(parsed["CHONKDB_API_KEY"], "secret")
        self.assertEqual(parsed["CHONKDB_API_KEY_HEADER"], "x-api-key")
        self.assertEqual(
            parsed["CHONKY_API_FULLTEXT_WARMUP_PROBE_QUERY"],
            "twinr dedicated instance",
        )

    def test_backend_query_surface_contract_marks_rebuild_without_gate_unsafe(self) -> None:
        contract = _backend_query_surface_readiness_contract_from_env(
            {
                "CHONKY_FT_REBUILD_ON_OPEN": "1",
                "CHONKY_API_WARMUP_FULLTEXT_GATE": "0",
                "CHONKY_API_WARMUP_WAIT_FOR_READY": "1",
                "CHONKY_API_WARMUP_WAIT_READY_TIMEOUT_S": "180",
            }
        )

        self.assertTrue(contract.unsafe)
        self.assertEqual(contract.unsafe_reason, "fulltext_rebuild_on_open_without_query_gate")


class RemoteChonkyDBReadinessContractRepairTests(unittest.TestCase):
    """Verify the remote readiness-contract hardening flow."""

    def test_ensure_backend_query_surface_readiness_contract_writes_override_when_unsafe(self) -> None:
        executor = unittest.mock.Mock()
        executor.run_sudo_ssh.side_effect = [
            unittest.mock.Mock(
                stdout=(
                    "Environment="
                    "CHONKY_FT_REBUILD_ON_OPEN=1 "
                    "CHONKY_API_WARMUP_FULLTEXT_GATE=0 "
                    "CHONKY_API_WARMUP_WAIT_FOR_READY=1 "
                    "CHONKY_API_WARMUP_WAIT_READY_TIMEOUT_S=180\n"
                )
            ),
            unittest.mock.Mock(
                stdout=(
                    '{"path": "/etc/systemd/system/caia-twinr-chonkydb-alt.service.d/'
                    '40-twinr-query-surface-readiness.conf", "changed": true}'
                )
            ),
        ]

        result = ensure_backend_query_surface_readiness_contract(
            executor=executor,
            service_name="caia-twinr-chonkydb-alt.service",
        )

        self.assertTrue(result.changed)
        self.assertTrue(result.contract.unsafe)
        write_script = executor.run_sudo_ssh.call_args_list[1].args[0]
        self.assertIn("CHONKY_API_WARMUP_FULLTEXT_GATE=1", write_script)
        self.assertIn("CHONKY_API_WARMUP_WAIT_FOR_READY=1", write_script)
        self.assertIn("systemctl daemon-reload", write_script)

    def test_ensure_backend_query_surface_readiness_contract_skips_override_when_safe(self) -> None:
        executor = unittest.mock.Mock()
        executor.run_sudo_ssh.return_value = unittest.mock.Mock(
            stdout=(
                "Environment="
                "CHONKY_FT_REBUILD_ON_OPEN=1 "
                "CHONKY_API_WARMUP_FULLTEXT_GATE=1 "
                "CHONKY_API_WARMUP_WAIT_FOR_READY=1 "
                "CHONKY_API_WARMUP_WAIT_READY_TIMEOUT_S=180\n"
            )
        )

        result = ensure_backend_query_surface_readiness_contract(
            executor=executor,
            service_name="caia-twinr-chonkydb-alt.service",
        )

        self.assertFalse(result.changed)
        self.assertFalse(result.contract.unsafe)
        self.assertEqual(executor.run_sudo_ssh.call_count, 1)


class RemoteChonkyDBQuerySurfaceProbeTests(unittest.TestCase):
    """Verify readiness only turns green when the live query surface works too."""

    def test_require_query_surface_ready_marks_instance_probe_unhealthy_when_query_canary_fails(self) -> None:
        result = _require_query_surface_ready(
            instance_probe=ChonkyDBHttpProbeResult(
                label="public",
                ok=True,
                status_code=200,
                ready=True,
                detail="ready",
                url="https://tessairact.com:2149/v1/external/instance",
            ),
            query_probe=ChonkyDBHttpProbeResult(
                label="public",
                ok=False,
                status_code=503,
                ready=False,
                detail="Service warmup in progress",
                url="https://tessairact.com:2149/v1/external/retrieve/topk_records",
            ),
        )

        self.assertFalse(result.ok)
        self.assertFalse(result.ready)
        self.assertEqual(result.status_code, 503)
        self.assertIn("query_surface_unhealthy", result.detail)

    def test_require_query_surface_ready_accepts_document_not_found_on_empty_scope(self) -> None:
        result = _require_query_surface_ready(
            instance_probe=ChonkyDBHttpProbeResult(
                label="public",
                ok=True,
                status_code=200,
                ready=True,
                detail="ready",
                url="https://tessairact.com:2149/v1/external/instance",
            ),
            query_probe=ChonkyDBHttpProbeResult(
                label="public",
                ok=False,
                status_code=404,
                ready=False,
                detail="document_not_found",
                url="https://tessairact.com:2149/v1/external/retrieve/topk_records",
                payload={"detail": "document_not_found", "error": "document_not_found"},
            ),
        )

        self.assertTrue(result.ok)
        self.assertTrue(result.ready)
        self.assertEqual(result.status_code, 200)

    def test_query_probe_empty_but_healthy_is_strict(self) -> None:
        self.assertTrue(
            _query_probe_empty_but_healthy(
                ChonkyDBHttpProbeResult(
                    label="public",
                    ok=False,
                    status_code=404,
                    ready=False,
                    detail="document_not_found",
                    payload={"detail": "document_not_found"},
                )
            )
        )
        self.assertFalse(
            _query_probe_empty_but_healthy(
                ChonkyDBHttpProbeResult(
                    label="public",
                    ok=False,
                    status_code=503,
                    ready=False,
                    detail="Upstream unavailable or restarting",
                )
            )
        )

    def test_probe_public_chonkydb_requires_query_surface_success(self) -> None:
        settings = RemoteChonkyDBOpsSettings(
            public_base_url="https://tessairact.com:2149",
            public_api_key="secret",
            public_api_key_header="x-api-key",
            ops_public_base_url="https://tessairact.com:2149",
            backend_local_base_url="http://127.0.0.1:3044",
            backend_service="caia-twinr-chonkydb-alt.service",
            runtime_namespace="twinr_longterm_v1:twinr:deff13356ec6",
            ssh=PiConnectionSettings(
                host="thh1986.ddns.net",
                user="thh",
                password="secret",
                port=22,
            ),
        )
        with patch(
            "twinr.ops.remote_chonkydb_repair._probe_http_json",
            side_effect=[
                ChonkyDBHttpProbeResult(
                    label="public",
                    ok=True,
                    status_code=200,
                    ready=True,
                    detail="ready",
                    url="https://tessairact.com:2149/v1/external/instance",
                ),
                ChonkyDBHttpProbeResult(
                    label="public",
                    ok=False,
                    status_code=503,
                    ready=False,
                    detail="Service warmup in progress",
                    url="https://tessairact.com:2149/v1/external/retrieve/topk_records",
                ),
            ],
        ) as mock_probe:
            result = probe_public_chonkydb(settings=settings, timeout_s=20.0)

        self.assertFalse(result.ready)
        self.assertEqual(result.status_code, 503)
        self.assertIn("query_surface_unhealthy", result.detail)
        self.assertEqual(mock_probe.call_count, 2)
        _, query_call = mock_probe.call_args_list
        self.assertEqual(query_call.kwargs["method"], "POST")
        self.assertEqual(
            query_call.kwargs["json_body"],
            _query_surface_canary_payload(runtime_namespace=settings.runtime_namespace),
        )

    def test_probe_public_chonkydb_accepts_empty_current_scope(self) -> None:
        settings = RemoteChonkyDBOpsSettings(
            public_base_url="https://tessairact.com:2149",
            public_api_key="secret",
            public_api_key_header="x-api-key",
            ops_public_base_url="https://tessairact.com:2149",
            backend_local_base_url="http://127.0.0.1:3044",
            backend_service="caia-twinr-chonkydb-alt.service",
            runtime_namespace="twinr_longterm_v1:twinr:deff13356ec6",
            ssh=PiConnectionSettings(
                host="thh1986.ddns.net",
                user="thh",
                password="secret",
                port=22,
            ),
        )
        with patch(
            "twinr.ops.remote_chonkydb_repair._probe_http_json",
            side_effect=[
                ChonkyDBHttpProbeResult(
                    label="public",
                    ok=True,
                    status_code=200,
                    ready=True,
                    detail="ready",
                    url="https://tessairact.com:2149/v1/external/instance",
                ),
                ChonkyDBHttpProbeResult(
                    label="public",
                    ok=False,
                    status_code=404,
                    ready=False,
                    detail="document_not_found",
                    url="https://tessairact.com:2149/v1/external/retrieve/topk_records",
                    payload={"detail": "document_not_found"},
                ),
            ],
        ):
            result = probe_public_chonkydb(settings=settings, timeout_s=20.0)

        self.assertTrue(result.ok)
        self.assertTrue(result.ready)
        self.assertEqual(result.status_code, 200)


class RemoteChonkyDBForeignConsumerTests(unittest.TestCase):
    """Verify repair output preserves detected foreign dedicated-backend consumers."""

    def test_repair_result_carries_foreign_consumers(self) -> None:
        settings = RemoteChonkyDBOpsSettings(
            public_base_url="https://tessairact.com:2149",
            public_api_key="secret",
            public_api_key_header="x-api-key",
            ops_public_base_url="https://tessairact.com:2149",
            backend_local_base_url="http://127.0.0.1:3044",
            backend_service="caia-twinr-chonkydb-alt.service",
            runtime_namespace="twinr_longterm_v1:twinr:deff13356ec6",
            ssh=PiConnectionSettings(
                host="thh1986.ddns.net",
                user="thh",
                password="secret",
                port=22,
            ),
        )
        foreign_consumers = (
            ForeignBackendConsumer(
                unit_name="caia-consumer-portal.service",
                active_state="active",
                sub_state="running",
                configured_base_url="http://127.0.0.1:3044",
                fragment_path="/etc/systemd/system/caia-consumer-portal.service",
                coupled_to_backend_service=True,
            ),
        )

        with (
            patch(
                "twinr.ops.remote_chonkydb_repair.ensure_remote_service_manual_restart_protection",
                return_value=RemoteManualRestartProtectionStatus(
                    service_name="caia-twinr-chonkydb-alt.service",
                    protected_dropin_path="/etc/systemd/system/caia-twinr-chonkydb-alt.service.d/10.conf",
                    protection_changed=False,
                    refuse_manual_start=True,
                    refuse_manual_stop=True,
                ),
            ),
            patch(
                "twinr.ops.remote_chonkydb_repair.probe_public_chonkydb",
                return_value=ChonkyDBHttpProbeResult(
                    label="public",
                    ok=False,
                    status_code=503,
                    ready=False,
                    detail="Upstream unavailable or restarting",
                ),
            ),
            patch(
                "twinr.ops.remote_chonkydb_repair.fetch_backend_service_state",
                return_value=ChonkyDBRemoteServiceState(
                    active_state="active",
                    sub_state="running",
                    service_result="success",
                ),
            ),
            patch(
                "twinr.ops.remote_chonkydb_repair.probe_backend_local_chonkydb",
                return_value=ChonkyDBHttpProbeResult(
                    label="backend",
                    ok=False,
                    status_code=504,
                    ready=False,
                    detail="query_payloads_advanced_timeout timeout_s=30.000",
                ),
            ),
            patch(
                "twinr.ops.remote_chonkydb_repair.inspect_backend_data_ownership",
                return_value=None,
            ),
            patch(
                "twinr.ops.remote_chonkydb_repair.inspect_foreign_backend_consumers",
                return_value=foreign_consumers,
            ),
        ):
            from twinr.ops.remote_chonkydb_repair import repair_remote_chonkydb

            result = repair_remote_chonkydb(
                settings=settings,
                probe_timeout_s=2.0,
                ssh_timeout_s=2.0,
                wait_ready_s=2.0,
                poll_interval_s=0.1,
                restart_if_needed=False,
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.diagnosis, "backend_foreign_consumer_contention")
        self.assertEqual(result.plan.action, "none")
        self.assertEqual(result.foreign_consumers, foreign_consumers)
        self.assertEqual(
            result.to_dict()["foreign_consumers"][0]["unit_name"],
            "caia-consumer-portal.service",
        )

    def test_repair_result_carries_backend_data_ownership_drift(self) -> None:
        settings = RemoteChonkyDBOpsSettings(
            public_base_url="https://tessairact.com:2149",
            public_api_key="secret",
            public_api_key_header="x-api-key",
            ops_public_base_url="https://tessairact.com:2149",
            backend_local_base_url="http://127.0.0.1:3044",
            backend_service="caia-twinr-chonkydb-alt.service",
            runtime_namespace="twinr_longterm_v1:twinr:deff13356ec6",
            ssh=PiConnectionSettings(
                host="thh1986.ddns.net",
                user="thh",
                password="secret",
                port=22,
            ),
        )
        ownership = BackendDataOwnershipState(
            data_dir="/srv/chonkydb/data",
            expected_user="thh",
            expected_group="thh",
            mismatched_entry_count=1,
            sample_entries=("root:root /srv/chonkydb/data/wal/chonkdb.wal.lock",),
        )

        with (
            patch(
                "twinr.ops.remote_chonkydb_repair.ensure_remote_service_manual_restart_protection",
                return_value=RemoteManualRestartProtectionStatus(
                    service_name="caia-twinr-chonkydb-alt.service",
                    protected_dropin_path="/etc/systemd/system/caia-twinr-chonkydb-alt.service.d/10.conf",
                    protection_changed=False,
                    refuse_manual_start=True,
                    refuse_manual_stop=True,
                ),
            ),
            patch(
                "twinr.ops.remote_chonkydb_repair.probe_public_chonkydb",
                return_value=ChonkyDBHttpProbeResult(
                    label="public",
                    ok=False,
                    status_code=503,
                    ready=False,
                    detail="Upstream unavailable or restarting",
                ),
            ),
            patch(
                "twinr.ops.remote_chonkydb_repair.fetch_backend_service_state",
                return_value=ChonkyDBRemoteServiceState(
                    active_state="failed",
                    sub_state="failed",
                    service_result="exit-code",
                ),
            ),
            patch(
                "twinr.ops.remote_chonkydb_repair.probe_backend_local_chonkydb",
                return_value=ChonkyDBHttpProbeResult(
                    label="backend",
                    ok=False,
                    status_code=503,
                    ready=False,
                    detail="backend unavailable",
                ),
            ),
            patch(
                "twinr.ops.remote_chonkydb_repair.inspect_backend_data_ownership",
                return_value=ownership,
            ),
            patch(
                "twinr.ops.remote_chonkydb_repair.inspect_foreign_backend_consumers",
                return_value=(),
            ),
        ):
            from twinr.ops.remote_chonkydb_repair import repair_remote_chonkydb

            result = repair_remote_chonkydb(
                settings=settings,
                probe_timeout_s=2.0,
                ssh_timeout_s=2.0,
                wait_ready_s=2.0,
                poll_interval_s=0.1,
                restart_if_needed=False,
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.diagnosis, "backend_data_permission_drift")
        self.assertEqual(result.plan.action, "none")
        self.assertEqual(result.backend_data_ownership, ownership)
        self.assertEqual(
            result.to_dict()["backend_data_ownership"]["sample_entries"][0],
            "root:root /srv/chonkydb/data/wal/chonkdb.wal.lock",
        )

    def test_repair_applies_readiness_contract_before_restart_when_backend_is_unhealthy(self) -> None:
        settings = RemoteChonkyDBOpsSettings(
            public_base_url="https://tessairact.com:2149",
            public_api_key="secret",
            public_api_key_header="x-api-key",
            ops_public_base_url="https://tessairact.com:2149",
            backend_local_base_url="http://127.0.0.1:3044",
            backend_service="caia-twinr-chonkydb-alt.service",
            runtime_namespace="twinr_longterm_v1:twinr:deff13356ec6",
            ssh=PiConnectionSettings(
                host="thh1986.ddns.net",
                user="thh",
                password="secret",
                port=22,
            ),
        )

        def _restart_side_effect(**_: object) -> None:
            self.assertTrue(mock_readiness.called)

        with (
            patch(
                "twinr.ops.remote_chonkydb_repair.ensure_remote_service_manual_restart_protection",
                return_value=RemoteManualRestartProtectionStatus(
                    service_name="caia-twinr-chonkydb-alt.service",
                    protected_dropin_path="/etc/systemd/system/caia-twinr-chonkydb-alt.service.d/10.conf",
                    protection_changed=False,
                    refuse_manual_start=True,
                    refuse_manual_stop=True,
                ),
            ),
            patch(
                "twinr.ops.remote_chonkydb_repair.probe_public_chonkydb",
                return_value=ChonkyDBHttpProbeResult(
                    label="public",
                    ok=False,
                    status_code=503,
                    ready=False,
                    detail="Upstream unavailable or restarting",
                ),
            ),
            patch(
                "twinr.ops.remote_chonkydb_repair.fetch_backend_service_state",
                return_value=ChonkyDBRemoteServiceState(
                    active_state="active",
                    sub_state="running",
                    service_result="success",
                ),
            ),
            patch(
                "twinr.ops.remote_chonkydb_repair.probe_backend_local_chonkydb",
                return_value=ChonkyDBHttpProbeResult(
                    label="backend",
                    ok=False,
                    status_code=504,
                    ready=False,
                    detail="query_payloads_advanced_timeout timeout_s=30.000",
                ),
            ),
            patch(
                "twinr.ops.remote_chonkydb_repair.inspect_backend_data_ownership",
                return_value=None,
            ),
            patch(
                "twinr.ops.remote_chonkydb_repair.inspect_foreign_backend_consumers",
                return_value=(),
            ),
            patch(
                "twinr.ops.remote_chonkydb_repair.ensure_backend_query_surface_readiness_contract",
                return_value=BackendQuerySurfaceReadinessRepairStatus(
                    dropin_path=(
                        "/etc/systemd/system/caia-twinr-chonkydb-alt.service.d/"
                        "40-twinr-query-surface-readiness.conf"
                    ),
                    changed=True,
                    contract=BackendQuerySurfaceReadinessContract(
                        fulltext_rebuild_on_open=True,
                        warmup_fulltext_gate=False,
                        warmup_wait_for_ready=True,
                        warmup_wait_ready_timeout_s=180.0,
                    ),
                ),
            ) as mock_readiness,
            patch(
                "twinr.ops.remote_chonkydb_repair.restart_backend_service",
                side_effect=_restart_side_effect,
            ) as mock_restart,
            patch(
                "twinr.ops.remote_chonkydb_repair.wait_for_backend_recovery",
                return_value=(
                    ChonkyDBHttpProbeResult(
                        label="public",
                        ok=True,
                        status_code=200,
                        ready=True,
                        detail="ready",
                    ),
                    ChonkyDBRemoteServiceState(
                        active_state="active",
                        sub_state="running",
                        service_result="success",
                    ),
                    ChonkyDBHttpProbeResult(
                        label="backend",
                        ok=True,
                        status_code=200,
                        ready=True,
                        detail="ready",
                    ),
                ),
            ),
        ):
            from twinr.ops.remote_chonkydb_repair import repair_remote_chonkydb

            result = repair_remote_chonkydb(
                settings=settings,
                probe_timeout_s=2.0,
                ssh_timeout_s=2.0,
                wait_ready_s=2.0,
                poll_interval_s=0.1,
                restart_if_needed=True,
            )

        self.assertTrue(result.ok)
        mock_readiness.assert_called_once()
        mock_restart.assert_called_once()


if __name__ == "__main__":
    unittest.main()
