"""Regression tests for remote ChonkyDB outage diagnosis and repair planning."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from twinr.ops.remote_chonkydb_repair import (
    ChonkyDBHttpProbeResult,
    ChonkyDBRemoteServiceState,
    RemoteChonkyDBOpsSettings,
    _query_probe_empty_but_healthy,
    plan_remote_chonkydb_repair,
    probe_public_chonkydb,
    _parse_systemd_environment_output,
    _query_surface_canary_payload,
    _require_query_surface_ready,
)
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


if __name__ == "__main__":
    unittest.main()
