from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote, unquote

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from chonkydb.api.models.admin import (
    OptimizeIndexesRequest,
    WarmStartRequest,
)
from chonkydb.api.models.external import (
    ExternalAdminProvisionRequest,
    ExternalGraphNeighborsRequest,
    ExternalGraphPathRequest,
    ExternalGraphPatternsRequest,
    ExternalInstanceResponse,
    ExternalJobResponse,
    ExternalRecordBulkRequest,
    ExternalRecordRequest,
    ExternalRetrieveRequest,
    ExternalTopKRecordsRequest,
)
from chonkydb.api.models.graph import (
    AddEdgeRequest,
    AddEdgeSmartRequest,
)
from chonkydb.api.routers import (
    admin as admin_router,
    graph as graph_router,
    health as health_router,
)
from chonkydb.api.services.chonky_service import (
    ChonkService,
)
from chonkydb.api.utils.responses import (
    ensure_query_response_or_http_error,
    to_json_safe,
)
from chonkydb.file_operations.router.types_config import (
    DataType,
    FileType,
)
from chonkydb.request_context import (
    update_request_context,
)

router = APIRouter(prefix="/external", tags=["external"])
# Bound by `api.app:create_app()` via app.state + request ContextVar.
# Tests may monkeypatch `_get_svc()` directly.
svc = None

_EXPLICIT_ERROR_STATUS_BY_TYPE = {
    "not_found": 404,
    "keyerror": 404,
    "documentnotfounderror": 404,
    "notimplementederror": 501,
    "timeouterror": 504,
    "asyncio.timeouterror": 504,
    "cancellederror": 408,
    "validationerror": 400,
    "valueerror": 400,
    "typeerror": 400,
    "toomanyrequests": 429,
    "queue_saturated": 429,
    "serverbusy": 503,
    "serviceunavailable": 503,
    "warmup_pending": 503,
}
_EXPLICIT_ERROR_STATUS_BY_CODE = {
    "queue_saturated": 429,
    "warmup_pending": 503,
}
_READY_OK_STATUSES = {"healthy", "ok", "ready"}
_TRUE_ENV = {"1", "true", "yes", "y", "on"}
_API_DOCS_PATH = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "chonkydb"
    / "API_DOCUMENTATION.md"
)
_TWINR_REMOTE_SNAPSHOT_SCHEMA = "twinr_remote_snapshot_v1"
_TWINR_SNAPSHOT_POINTER_SCHEMA = "twinr_remote_snapshot_pointer_v1"
_TWINR_SNAPSHOT_POINTER_VERSION = 1
_TWINR_SNAPSHOT_POINTER_PREFIX = "__pointer__:"
_TWINR_REMOTE_CATALOG_VERSION = 3
_TWINR_REMOTE_LEGACY_CATALOG_VERSION = 2
_TWINR_REMOTE_SEGMENT_VERSION = 1
_SCOPE_DOC_IDS_CACHE_TTL_DEFAULT_S = 30.0
_TWINR_LONGTERM_SCOPE_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "objects": {
        "snapshot_kind": "objects",
        "uri_segment": "objects",
        "catalog_schema": "twinr_memory_object_catalog_v3",
        "legacy_catalog_schema": "twinr_memory_object_catalog_v2",
        "segment_schema": "twinr_memory_object_catalog_segment_v1",
    },
    "conflicts": {
        "snapshot_kind": "conflicts",
        "uri_segment": "conflicts",
        "catalog_schema": "twinr_memory_conflict_catalog_v3",
        "legacy_catalog_schema": "twinr_memory_conflict_catalog_v2",
        "segment_schema": "twinr_memory_conflict_catalog_segment_v1",
    },
    "archive": {
        "snapshot_kind": "archive",
        "uri_segment": "archive",
        "catalog_schema": "twinr_memory_archive_catalog_v3",
        "legacy_catalog_schema": "twinr_memory_archive_catalog_v2",
        "segment_schema": "twinr_memory_archive_catalog_segment_v1",
    },
}
_SCOPE_ALLOWED_DOC_IDS_CACHE: Dict[tuple[str, str], Dict[str, Any]] = {}
_SCOPE_ALLOWED_DOC_IDS_CACHE_LOCK = Lock()
_SCOPE_ALLOWED_DOC_IDS_REFRESH_FUTURES: Dict[
    tuple[str, str], asyncio.Future[List[str]]
] = {}
_SCOPE_ALLOWED_DOC_IDS_REFRESH_LOCK = Lock()
_TWINR_CURRENT_DOCUMENT_CACHE_BY_ORIGIN_URI: Dict[str, Dict[str, Any]] = {}
_TWINR_CURRENT_DOCUMENT_CACHE_BY_DOCUMENT_ID: Dict[str, Dict[str, Any]] = {}
_TWINR_CURRENT_DOCUMENT_CACHE_LOCK = Lock()
_LOGGER = logging.getLogger(__name__)
_TWINR_SCOPE_POINTER_PHASE = "pointer_origin"
_TWINR_SCOPE_SNAPSHOT_PHASE = "snapshot_catalog"
_TWINR_SCOPE_SEGMENT_PHASE = "catalog_segment"
_TWINR_SCOPE_READINESS_SCOPE = "token_fast"
_TWINR_SCOPE_WARMUP_RETRY_ATTEMPTS = 4
_TWINR_SCOPE_WARMUP_RETRY_DELAY_S = 0.25
_TWINR_SCOPE_PHASE_CONTENT_CAP_DEFAULTS = {
    _TWINR_SCOPE_POINTER_PHASE: 32_768,
    _TWINR_SCOPE_SNAPSHOT_PHASE: 512_000,
    _TWINR_SCOPE_SEGMENT_PHASE: 512_000,
}
_TWINR_SCOPE_PHASE_CONTENT_CAP_ENVS = {
    _TWINR_SCOPE_POINTER_PHASE: "CHONKY_TWINR_POINTER_MAX_CONTENT_CHARS",
    _TWINR_SCOPE_SNAPSHOT_PHASE: "CHONKY_TWINR_SNAPSHOT_MAX_CONTENT_CHARS",
    _TWINR_SCOPE_SEGMENT_PHASE: "CHONKY_TWINR_SEGMENT_MAX_CONTENT_CHARS",
}
_INSTANCE_COMPONENT_DEFAULT_TIMEOUT_S = 3.0


def _model_dump(model: Any) -> Dict[str, Any]:
    if isinstance(model, dict):
        return dict(model)
    dump = getattr(model, "model_dump", None)
    if callable(dump):
        try:
            data = dump(mode="python")
        except TypeError:
            data = dump()
        if isinstance(data, dict):
            return dict(data)
    data_dict = getattr(model, "dict", None)
    if callable(data_dict):
        data = data_dict()
        if isinstance(data, dict):
            return dict(data)
    if hasattr(model, "__dict__"):
        raw = getattr(model, "__dict__", None)
        if isinstance(raw, dict):
            return dict(raw)
    return {}


def _drop_none_top_level(data: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


def _contract_http_exception(
    *,
    status_code: int,
    error_code: str,
    error: str,
    contract: str,
    original_exception: Optional[Exception] = None,
    context: Optional[Mapping[str, Any]] = None,
) -> HTTPException:
    payload: Dict[str, Any] = {
        "success": False,
        "error_code": str(error_code),
        "error": str(error),
        "contract": str(contract),
    }
    if original_exception is not None:
        payload["error_type"] = type(original_exception).__name__
    if isinstance(context, Mapping):
        payload.update(
            {
                str(key): value
                for key, value in context.items()
                if value is not None
            }
        )
    return HTTPException(
        status_code=int(status_code),
        detail=_drop_none_top_level(payload),
    )


def _reraise_contract_http_exception(exc: HTTPException) -> None:
    detail = getattr(exc, "detail", None)
    if not isinstance(detail, Mapping):
        return
    error_code = str(detail.get("error_code") or "").strip().lower()
    if error_code.endswith("_contract_mismatch"):
        raise exc


def _env_truthy(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in _TRUE_ENV


def _scope_doc_ids_cache_ttl_s() -> float:
    raw = os.getenv("CHONKY_SCOPE_DOC_IDS_CACHE_TTL_S")
    if raw is None:
        return _SCOPE_DOC_IDS_CACHE_TTL_DEFAULT_S
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        return _SCOPE_DOC_IDS_CACHE_TTL_DEFAULT_S
    return parsed if parsed > 0.0 else 0.0


def _instance_component_timeout_s() -> float:
    raw = os.getenv("CHONKY_EXTERNAL_INSTANCE_COMPONENT_TIMEOUT_S")
    if raw is None:
        return _INSTANCE_COMPONENT_DEFAULT_TIMEOUT_S
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        return _INSTANCE_COMPONENT_DEFAULT_TIMEOUT_S
    if parsed <= 0.0:
        return _INSTANCE_COMPONENT_DEFAULT_TIMEOUT_S
    return max(0.1, min(float(parsed), 30.0))


def _instance_component_error_payload(
    *,
    component: str,
    error: str,
    error_type: str,
    timeout_seconds: Optional[float] = None,
    status_code: Optional[int] = None,
) -> Dict[str, Any]:
    payload = {
        "success": False,
        "degraded": True,
        "component": str(component),
        "error": str(error),
        "error_type": str(error_type),
    }
    if timeout_seconds is not None:
        payload["timeout_seconds"] = float(timeout_seconds)
    if status_code is not None:
        payload["status_code"] = int(status_code)
    return payload


async def _await_instance_component(
    component: str,
    awaitable: Any,
    *,
    timeout_seconds: float,
) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        payload = await asyncio.wait_for(awaitable, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        elapsed_s = max(0.0, time.perf_counter() - started)
        error = (
            f"instance_component_timeout component={str(component)} "
            f"timeout_s={float(timeout_seconds):.3f}"
        )
        _LOGGER.warning(
            "instance_overview component timeout component=%s elapsed_s=%.3f timeout_s=%.3f",
            str(component),
            elapsed_s,
            float(timeout_seconds),
        )
        return _instance_component_error_payload(
            component=component,
            error=error,
            error_type="TimeoutError",
            timeout_seconds=timeout_seconds,
        )
    except HTTPException as exc:
        elapsed_s = max(0.0, time.perf_counter() - started)
        detail = exc.detail if isinstance(exc.detail, Mapping) else {}
        error = str(
            detail.get("error")
            or getattr(exc, "detail", "")
            or f"{str(component)}_failed"
        )
        error_type = str(detail.get("error_type") or type(exc).__name__)
        payload = dict(detail) if isinstance(detail, Mapping) else {}
        payload.update(
            _instance_component_error_payload(
                component=component,
                error=error,
                error_type=error_type,
                timeout_seconds=timeout_seconds,
                status_code=int(exc.status_code),
            )
        )
        _LOGGER.warning(
            "instance_overview component http_error component=%s elapsed_s=%.3f status_code=%s error_type=%s",
            str(component),
            elapsed_s,
            int(exc.status_code),
            error_type,
        )
        return payload
    except Exception as exc:
        elapsed_s = max(0.0, time.perf_counter() - started)
        error = str(exc).strip() or type(exc).__name__
        _LOGGER.warning(
            "instance_overview component exception component=%s elapsed_s=%.3f error_type=%s",
            str(component),
            elapsed_s,
            type(exc).__name__,
        )
        return _instance_component_error_payload(
            component=component,
            error=error,
            error_type=type(exc).__name__,
            timeout_seconds=timeout_seconds,
        )

    elapsed_s = max(0.0, time.perf_counter() - started)
    if not isinstance(payload, Mapping):
        _LOGGER.warning(
            "instance_overview component invalid response component=%s elapsed_s=%.3f response_type=%s",
            str(component),
            elapsed_s,
            type(payload).__name__,
        )
        return _instance_component_error_payload(
            component=component,
            error="invalid_response_type",
            error_type=type(payload).__name__,
            timeout_seconds=timeout_seconds,
        )

    result = dict(payload)
    result.setdefault("component", str(component))
    if result.get("success") is False:
        result.setdefault("degraded", True)
        _LOGGER.warning(
            "instance_overview component degraded component=%s elapsed_s=%.3f error_type=%s",
            str(component),
            elapsed_s,
            str(result.get("error_type") or "RuntimeError"),
        )
    return result


def _public_base_url(request: Optional[Request] = None) -> str:
    configured = str(os.getenv("CHONKDB_PUBLIC_BASE_URL", "") or "").strip().rstrip("/")
    if configured:
        return configured
    if request is None:
        return ""
    forwarded_proto = str(request.headers.get("x-forwarded-proto") or "").strip()
    forwarded_host = str(
        request.headers.get("x-forwarded-host") or request.headers.get("host") or ""
    ).strip()
    if forwarded_proto and forwarded_host:
        return f"{forwarded_proto}://{forwarded_host}".rstrip("/")
    return str(request.base_url).rstrip("/")


def _join_public_url(base_url: str, path: str) -> Optional[str]:
    normalized_base = str(base_url or "").strip().rstrip("/")
    normalized_path = "/" + str(path or "").lstrip("/")
    if not normalized_base:
        return None
    return f"{normalized_base}{normalized_path}"


def _load_api_documentation_markdown() -> str:
    try:
        return _API_DOCS_PATH.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "error": "api_documentation_missing",
                "error_type": type(exc).__name__,
                "path": str(_API_DOCS_PATH),
            },
        ) from exc


def _looks_like_external_service(candidate: Any) -> bool:
    required = (
        "store_payload",
        "store_payloads_sync_bulk",
        "start_ingest_job",
        "start_ingest_job_bulk",
        "ingest_document",
        "query_payloads",
        "query_payloads_advanced",
        "graph_semantic_filter",
        "get_payload_info",
        "get_full_document",
        "list_all_payloads",
        "delete_payload",
        "job_status",
        "cancel_job",
        "add_graph_edge",
        "add_graph_edge_smart",
        "graph_neighbors",
        "graph_path",
        "query_graph_patterns",
        "list_available_indexes",
        "get_system_stats",
        "get_basic_metrics",
    )
    return all(callable(getattr(candidate, name, None)) for name in required)


def _get_svc(request: Optional[Request] = None) -> ChonkService:
    if request is not None:
        app = getattr(request, "app", None)
        state = getattr(app, "state", None)
        if state is not None and hasattr(state, "chonky_service"):
            candidate = getattr(state, "chonky_service", None)
            if candidate is not None:
                if _looks_like_external_service(candidate):
                    return candidate
                raise RuntimeError("app.state chonky_service is invalid")
    global svc
    if svc is None:
        svc = ChonkService()
    return svc


def _error_status_from_result(
    result: Mapping[str, Any], *, default_status: int = 500
) -> int:
    for key in ("status_code", "status"):
        raw_status = result.get(key)
        try:
            status = int(raw_status)
        except (TypeError, ValueError):
            continue
        if 400 <= status <= 599:
            return status

    error_type = str(result.get("error_type") or "").strip().lower()
    if error_type in _EXPLICIT_ERROR_STATUS_BY_TYPE:
        return _EXPLICIT_ERROR_STATUS_BY_TYPE[error_type]

    error_code = str(result.get("error_code") or result.get("error") or "").strip().lower()
    if error_code in _EXPLICIT_ERROR_STATUS_BY_CODE:
        return _EXPLICIT_ERROR_STATUS_BY_CODE[error_code]

    return int(default_status)


def _raise_for_failed_result(
    result: Any, *, default_status: int = 500
) -> Dict[str, Any]:
    if not isinstance(result, dict):
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "invalid_response_type",
                "error_type": type(result).__name__,
            },
        )
    if result.get("success") is False:
        raise HTTPException(
            status_code=_error_status_from_result(
                result, default_status=default_status
            ),
            detail=to_json_safe(result),
        )
    return dict(result)


def _query_response_or_http_error(result: Any) -> Dict[str, Any]:
    if not isinstance(result, dict):
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "invalid_response_type",
                "error_type": type(result).__name__,
            },
        )
    if result.get("success") is False:
        status = _error_status_from_result(result, default_status=500)
        if status == 503:
            raise HTTPException(status_code=503, detail=to_json_safe(result))
    return ensure_query_response_or_http_error(
        result,
        treat_error_fields_as_failure=True,
    )


def _graph_response_or_http_error(result: Any) -> Dict[str, Any]:
    if not isinstance(result, dict):
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "invalid_response_type",
                "error_type": type(result).__name__,
            },
        )
    if result.get("success") is False:
        error_type = str(result.get("error_type") or "").strip()
        error_msg = (
            str(result.get("error") or result.get("detail") or "").strip().lower()
        )
        if graph_router._is_graph_service_unavailable_error(
            error_type=error_type, error_msg=error_msg
        ):
            raise HTTPException(status_code=503, detail=to_json_safe(result))
        if error_type == "TimeoutError":
            raise HTTPException(status_code=504, detail=to_json_safe(result))
        raise HTTPException(
            status_code=_error_status_from_result(result, default_status=500),
            detail=to_json_safe(result),
        )
    return dict(result)


def _require_callable(service: Any, method_name: str) -> Any:
    method = getattr(service, method_name, None)
    if callable(method):
        return method
    raise HTTPException(
        status_code=501,
        detail={
            "success": False,
            "error": f"{method_name}_unsupported",
            "error_type": "NotImplementedError",
        },
    )


async def _call_service(
    service: Any, method_name: str, /, *args: Any, **kwargs: Any
) -> Any:
    method = _require_callable(service, method_name)
    return await method(*args, **kwargs)


def _api_key_exempt_paths() -> List[str]:
    raw = str(os.getenv("CHONKDB_API_KEY_EXEMPT_PATHS", "/v1/health,/v1/ready") or "")
    return [token for token in {str(item).strip() for item in raw.split(",")} if token]


def _auth_config_payload() -> Dict[str, Any]:
    api_key = str(os.getenv("CHONKDB_API_KEY", "") or "").strip()
    header_name = (
        str(os.getenv("CHONKDB_API_KEY_HEADER", "x-api-key") or "x-api-key").strip()
        or "x-api-key"
    )
    allow_bearer = _env_truthy("CHONKDB_API_KEY_ALLOW_BEARER", True)
    exempt_paths = sorted(_api_key_exempt_paths())
    return {
        "success": True,
        "auth_enabled": bool(api_key),
        "scheme": "api_key" if api_key else "loopback_or_exempt_only",
        "header_name": header_name,
        "allow_bearer": bool(allow_bearer),
        "exempt_paths": exempt_paths,
        "api_key_configured": bool(api_key),
        "in_band_key_rotation_supported": False,
    }


def _is_ready_payload(payload: Mapping[str, Any]) -> bool:
    status = str(payload.get("status") or "").strip().lower()
    return bool(payload.get("ok")) and status in _READY_OK_STATUSES


def _health_payload_or_http_error(result: Any) -> Dict[str, Any]:
    if not isinstance(result, dict):
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "invalid_response_type",
                "error_type": type(result).__name__,
            },
        )
    return dict(result)


def _storage_payload_from_stats(stats: Mapping[str, Any]) -> Dict[str, Any]:
    disk = stats.get("disk")
    index_files = stats.get("index_files")
    disk_out = dict(disk) if isinstance(disk, Mapping) else {}
    index_files_out = dict(index_files) if isinstance(index_files, Mapping) else {}
    tracked_index_bytes = 0
    for item in index_files_out.values():
        if isinstance(item, Mapping):
            try:
                tracked_index_bytes += int(item.get("bytes") or 0)
            except (TypeError, ValueError):
                continue
    return {
        "success": True,
        "backend": stats.get("backend"),
        "data_dir": stats.get("data_dir"),
        "disk": to_json_safe(disk_out),
        "index_files": to_json_safe(index_files_out),
        "index_file_count": int(len(index_files_out)),
        "tracked_index_bytes": int(tracked_index_bytes),
    }


def _build_record_payload(
    req: ExternalRecordRequest,
) -> tuple[str, str, Optional[str], Dict[str, Any]]:
    payload = _drop_none_top_level(_model_dump(req))
    operation = (
        str(payload.pop("operation", "store_payload") or "store_payload")
        .strip()
        .lower()
    )
    execution_mode = (
        str(payload.pop("execution_mode", "sync") or "sync").strip().lower()
    )
    client_request_id = (
        str(payload.pop("client_request_id", None) or "").strip() or None
    )
    return operation, execution_mode, client_request_id, payload


def _build_bulk_payload(
    req: ExternalRecordBulkRequest,
) -> tuple[str, str, Optional[str], Optional[float], bool, List[Dict[str, Any]]]:
    operation = str(req.operation or "store_payload").strip().lower()
    execution_mode = str(req.execution_mode or "sync").strip().lower()
    client_request_id = str(req.client_request_id or "").strip() or None
    timeout_seconds = req.timeout_seconds
    finalize_vector_segments = bool(req.finalize_vector_segments)
    items = [_drop_none_top_level(_model_dump(item)) for item in list(req.items or [])]
    return (
        operation,
        execution_mode,
        client_request_id,
        timeout_seconds,
        finalize_vector_segments,
        items,
    )


async def _start_custom_job(*, name: str, coro_factory: Any) -> Dict[str, Any]:
    from chonkydb.api.utils.jobs import JOB_MANAGER

    rec = await JOB_MANAGER.start_job(name=name, coro_factory=coro_factory)
    return {"success": True, "job_id": rec.job_id, "status": rec.status}


async def _execute_bulk_document_ingest(
    service: Any,
    *,
    items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    succeeded = 0
    failed = 0

    for index, item in enumerate(items):
        try:
            res = await service.ingest_document(**item)
        except Exception as exc:  # noqa: BLE001 - external facade must return structured item failures.
            res = {
                "success": False,
                "error": str(exc) or type(exc).__name__,
                "error_type": type(exc).__name__,
            }
        row = (
            dict(res)
            if isinstance(res, Mapping)
            else {
                "success": False,
                "error": "invalid_response_type",
                "error_type": type(res).__name__,
            }
        )
        row["item_index"] = int(index)
        if bool(row.get("success")):
            succeeded += 1
        else:
            failed += 1
        results.append(row)

    return {
        "success": failed == 0,
        "all_succeeded": failed == 0,
        "count": int(len(items)),
        "succeeded": int(succeeded),
        "failed": int(failed),
        "items": results,
    }


def _single_item_bulk_result_or_http_error(
    result: Any,
    *,
    default_status: int = 503,
) -> Dict[str, Any]:
    bulk_result = _raise_for_failed_result(result, default_status=default_status)
    raw_items = bulk_result.get("items")
    if not isinstance(raw_items, list) or len(raw_items) != 1:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "invalid_single_item_bulk_response",
                "error_type": "RuntimeError",
                "item_count": len(raw_items) if isinstance(raw_items, list) else None,
            },
        )
    raw_item = raw_items[0]
    if not isinstance(raw_item, Mapping):
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "invalid_single_item_bulk_entry",
                "error_type": type(raw_item).__name__,
            },
        )
    item_result = _raise_for_failed_result(
        dict(raw_item), default_status=default_status
    )
    if (
        not str(item_result.get("payload_id") or "").strip()
        and str(item_result.get("document_id") or "").strip()
    ):
        item_result["payload_id"] = item_result["document_id"]
    if (
        not str(item_result.get("document_id") or "").strip()
        and str(item_result.get("payload_id") or "").strip()
    ):
        item_result["document_id"] = item_result["payload_id"]
    if "request_reuse" in bulk_result and "request_reuse" not in item_result:
        item_result["request_reuse"] = bulk_result.get("request_reuse")
    return item_result


def _advanced_query_payload(req: ExternalRetrieveRequest) -> Dict[str, Any]:
    payload = _drop_none_top_level(_model_dump(req))
    payload.pop("mode", None)
    payload.pop("namespace", None)
    payload.pop("scope_ref", None)
    payload.pop("seed_label", None)
    payload.pop("theme", None)
    payload.pop("exclude_keyword", None)
    payload.pop("question_type", None)
    if payload.get("graph_seed_label") is None and getattr(req, "seed_label", None):
        payload["graph_seed_label"] = str(req.seed_label)
    return payload


def _graph_query_payload(req: ExternalRetrieveRequest) -> Dict[str, Any]:
    seed_label = str(req.seed_label or req.graph_seed_label or "").strip()
    if not seed_label:
        raise HTTPException(
            status_code=400,
            detail={"success": False, "error": "seed_label required for graph mode"},
        )
    payload = {
        "seed_label": seed_label,
        "max_hops": int(req.graph_max_hops),
        "theme": req.theme,
        "exclude_keyword": req.exclude_keyword,
        "allowed_doc_ids": req.allowed_doc_ids,
        "days_back": int(req.graph_days_back),
        "question_type": req.question_type,
        "temporal_start": req.temporal_start,
        "temporal_end": req.temporal_end,
        "enforce_time_filter": bool(req.enforce_time_filter)
        if req.enforce_time_filter is not None
        else True,
        "index_name": str(req.graph_index_name or "graph_index"),
        "graph_edge_types": req.graph_edge_types,
        "graph_direction": req.graph_direction,
        "graph_weight": req.graph_weight,
        "result_limit": int(req.result_limit),
        "include_content": bool(req.include_content),
        "timeout_seconds": req.timeout_seconds,
    }
    return _drop_none_top_level(payload)


def _copy_optional_mapping(value: Any) -> Optional[Dict[str, Any]]:
    return dict(value) if isinstance(value, Mapping) else None


def _parse_json_mapping_text(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    return dict(parsed) if isinstance(parsed, dict) else None


def _normalize_optional_text(value: Any) -> Optional[str]:
    normalized = str(value or "").strip()
    return normalized or None


def _scope_definition_for_ref(scope_ref: Any) -> tuple[Optional[str], Optional[Dict[str, str]]]:
    normalized = _normalize_optional_text(scope_ref)
    if normalized is None:
        return None, None
    parts = normalized.split(":")
    if len(parts) != 3 or parts[0] != "longterm" or parts[2] != "current":
        return None, None
    definition = _TWINR_LONGTERM_SCOPE_DEFINITIONS.get(parts[1])
    if definition is None:
        return None, None
    return normalized, definition


def _scope_cache_key(*, namespace: str, scope_ref: str) -> tuple[str, str]:
    return namespace, scope_ref


def _get_scope_refresh_future(
    *,
    namespace: str,
    scope_ref: str,
) -> Optional[asyncio.Future[List[str]]]:
    cache_key = _scope_cache_key(namespace=namespace, scope_ref=scope_ref)
    with _SCOPE_ALLOWED_DOC_IDS_REFRESH_LOCK:
        future = _SCOPE_ALLOWED_DOC_IDS_REFRESH_FUTURES.get(cache_key)
        if future is None:
            return None
        if future.done():
            _SCOPE_ALLOWED_DOC_IDS_REFRESH_FUTURES.pop(cache_key, None)
            return None
        return future


def _begin_scope_refresh_future(
    *,
    namespace: str,
    scope_ref: str,
) -> tuple[asyncio.Future[List[str]], bool]:
    cache_key = _scope_cache_key(namespace=namespace, scope_ref=scope_ref)
    with _SCOPE_ALLOWED_DOC_IDS_REFRESH_LOCK:
        existing = _SCOPE_ALLOWED_DOC_IDS_REFRESH_FUTURES.get(cache_key)
        if existing is not None and not existing.done():
            return existing, False
        future: asyncio.Future[List[str]] = asyncio.get_running_loop().create_future()
        _SCOPE_ALLOWED_DOC_IDS_REFRESH_FUTURES[cache_key] = future
        return future, True


def _finish_scope_refresh_future(
    *,
    namespace: str,
    scope_ref: str,
    future: asyncio.Future[List[str]],
    result: Optional[List[str]] = None,
    exc: Optional[BaseException] = None,
) -> None:
    cache_key = _scope_cache_key(namespace=namespace, scope_ref=scope_ref)
    with _SCOPE_ALLOWED_DOC_IDS_REFRESH_LOCK:
        current = _SCOPE_ALLOWED_DOC_IDS_REFRESH_FUTURES.get(cache_key)
        if current is future:
            _SCOPE_ALLOWED_DOC_IDS_REFRESH_FUTURES.pop(cache_key, None)
    if future.done():
        return
    if exc is not None:
        future.set_exception(exc)
        return
    future.set_result(list(result or ()))


def _load_cached_scope_allowed_doc_ids(
    *,
    namespace: str,
    scope_ref: str,
) -> Optional[List[str]]:
    ttl_s = _scope_doc_ids_cache_ttl_s()
    if ttl_s <= 0.0:
        return None
    now = time.monotonic()
    cache_key = _scope_cache_key(namespace=namespace, scope_ref=scope_ref)
    with _SCOPE_ALLOWED_DOC_IDS_CACHE_LOCK:
        cached = _SCOPE_ALLOWED_DOC_IDS_CACHE.get(cache_key)
        if cached is None:
            return None
        expires_at = float(cached.get("expires_at_monotonic") or 0.0)
        if expires_at <= now:
            _SCOPE_ALLOWED_DOC_IDS_CACHE.pop(cache_key, None)
            return None
        raw_doc_ids = cached.get("doc_ids")
        if not isinstance(raw_doc_ids, list):
            _SCOPE_ALLOWED_DOC_IDS_CACHE.pop(cache_key, None)
            return None
        return [str(value) for value in raw_doc_ids if str(value or "").strip()]


def _store_cached_scope_allowed_doc_ids(
    *,
    namespace: str,
    scope_ref: str,
    doc_ids: List[str],
) -> None:
    ttl_s = _scope_doc_ids_cache_ttl_s()
    if ttl_s <= 0.0:
        return
    cache_key = _scope_cache_key(namespace=namespace, scope_ref=scope_ref)
    normalized_doc_ids = [str(value) for value in doc_ids if str(value or "").strip()]
    with _SCOPE_ALLOWED_DOC_IDS_CACHE_LOCK:
        _SCOPE_ALLOWED_DOC_IDS_CACHE[cache_key] = {
            "doc_ids": normalized_doc_ids,
            "expires_at_monotonic": time.monotonic() + ttl_s,
        }


def _scope_ref_for_snapshot_kind(snapshot_kind: Any) -> Optional[str]:
    normalized_kind = _normalize_optional_text(snapshot_kind)
    if normalized_kind is None:
        return None
    if normalized_kind.startswith(_TWINR_SNAPSHOT_POINTER_PREFIX):
        normalized_kind = normalized_kind[len(_TWINR_SNAPSHOT_POINTER_PREFIX) :]
    if normalized_kind not in _TWINR_LONGTERM_SCOPE_DEFINITIONS:
        return None
    return f"longterm:{normalized_kind}:current"


def _scope_ref_for_catalog_schema(schema: Any) -> Optional[str]:
    normalized_schema = _normalize_optional_text(schema)
    if normalized_schema is None:
        return None
    for snapshot_kind, definition in _TWINR_LONGTERM_SCOPE_DEFINITIONS.items():
        if normalized_schema in {
            definition["catalog_schema"],
            definition["legacy_catalog_schema"],
            definition["segment_schema"],
        }:
            return f"longterm:{snapshot_kind}:current"
    return None


def _namespace_from_longterm_origin_uri(uri: Any) -> Optional[str]:
    normalized_uri = _normalize_optional_text(uri)
    if normalized_uri is None:
        return None
    prefix = "twinr://longterm/"
    if not normalized_uri.startswith(prefix):
        return None
    remainder = normalized_uri[len(prefix) :]
    raw_namespace, _separator, _rest = remainder.partition("/")
    return _normalize_optional_text(unquote(raw_namespace))


def _scope_ref_from_longterm_origin_uri(uri: Any, *, namespace: Optional[str]) -> Optional[str]:
    normalized_uri = _normalize_optional_text(uri)
    if normalized_uri is None:
        return None
    prefix = "twinr://longterm/"
    if not normalized_uri.startswith(prefix):
        return None
    remainder = normalized_uri[len(prefix) :]
    raw_namespace, separator, raw_segment = remainder.partition("/")
    if not separator:
        return None
    uri_namespace = _normalize_optional_text(unquote(raw_namespace))
    if namespace is not None and uri_namespace is not None and uri_namespace != namespace:
        return None
    first_segment = _normalize_optional_text(unquote(raw_segment.split("/", 1)[0]))
    return _scope_ref_for_snapshot_kind(first_segment)


def _extract_scope_cache_namespace_from_item(item: Mapping[str, Any]) -> Optional[str]:
    payload = item.get("payload")
    payload_mapping = payload if isinstance(payload, Mapping) else {}
    metadata = item.get("metadata")
    metadata_mapping = metadata if isinstance(metadata, Mapping) else {}
    for candidate in (
        item.get("namespace"),
        payload_mapping.get("namespace"),
        metadata_mapping.get("twinr_namespace"),
        item.get("uri"),
        metadata_mapping.get("origin_uri"),
        payload_mapping.get("origin_uri"),
    ):
        normalized_namespace = _normalize_optional_text(candidate)
        if normalized_namespace is not None and "://" not in normalized_namespace:
            return normalized_namespace
        uri_namespace = _namespace_from_longterm_origin_uri(candidate)
        if uri_namespace is not None:
            return uri_namespace
    return None


def _extract_scope_cache_scope_refs_from_item(
    item: Mapping[str, Any],
    *,
    namespace: Optional[str],
) -> tuple[List[str], bool]:
    payload = item.get("payload")
    payload_mapping = payload if isinstance(payload, Mapping) else {}
    metadata = item.get("metadata")
    metadata_mapping = metadata if isinstance(metadata, Mapping) else {}
    body = payload_mapping.get("body")
    body_mapping = body if isinstance(body, Mapping) else {}
    resolved_scope_refs: List[str] = []
    recognized_twinr_signal = False

    for candidate in (
        item.get("snapshot_kind"),
        payload_mapping.get("snapshot_kind"),
        metadata_mapping.get("twinr_snapshot_kind"),
    ):
        normalized_candidate = _normalize_optional_text(candidate)
        if normalized_candidate is None:
            continue
        recognized_twinr_signal = True
        scope_ref = _scope_ref_for_snapshot_kind(normalized_candidate)
        if scope_ref is not None:
            resolved_scope_refs.append(scope_ref)

    for candidate in (
        payload_mapping.get("schema"),
        body_mapping.get("schema"),
        metadata_mapping.get("twinr_snapshot_schema"),
    ):
        scope_ref = _scope_ref_for_catalog_schema(candidate)
        if scope_ref is None:
            continue
        recognized_twinr_signal = True
        resolved_scope_refs.append(scope_ref)

    for candidate in (
        item.get("uri"),
        payload_mapping.get("uri"),
        payload_mapping.get("origin_uri"),
        metadata_mapping.get("origin_uri"),
    ):
        normalized_candidate = _normalize_optional_text(candidate)
        if normalized_candidate is None:
            continue
        if normalized_candidate.startswith("twinr://longterm/"):
            recognized_twinr_signal = True
        scope_ref = _scope_ref_from_longterm_origin_uri(
            normalized_candidate,
            namespace=namespace,
        )
        if scope_ref is not None:
            resolved_scope_refs.append(scope_ref)

    return _dedupe_doc_ids(resolved_scope_refs), recognized_twinr_signal


def _extract_scope_cache_invalidation_targets_from_items(
    items: List[Dict[str, Any]],
) -> Optional[Dict[str, Optional[List[str]]]]:
    invalidation_targets: Dict[str, Optional[set[str]]] = {}
    saw_global_unknown = False
    for item in items:
        if not isinstance(item, Mapping):
            continue
        _unknown_scope_refs, recognized_without_namespace = (
            _extract_scope_cache_scope_refs_from_item(
                item,
                namespace=None,
            )
        )
        normalized_namespace = _extract_scope_cache_namespace_from_item(item)
        if normalized_namespace is None:
            if recognized_without_namespace:
                saw_global_unknown = True
            continue
        scope_refs, recognized_twinr_signal = _extract_scope_cache_scope_refs_from_item(
            item,
            namespace=normalized_namespace,
        )
        if scope_refs:
            existing = invalidation_targets.get(normalized_namespace)
            if existing is None and normalized_namespace in invalidation_targets:
                continue
            if existing is None:
                invalidation_targets[normalized_namespace] = set(scope_refs)
            else:
                existing.update(scope_refs)
            continue
        if recognized_twinr_signal:
            continue
        invalidation_targets[normalized_namespace] = None
    if saw_global_unknown:
        return None
    normalized_targets: Dict[str, Optional[List[str]]] = {}
    for namespace_key, scope_refs in invalidation_targets.items():
        if scope_refs is None:
            normalized_targets[namespace_key] = None
            continue
        normalized_targets[namespace_key] = sorted(scope_refs)
    return normalized_targets


def _invalidate_scope_allowed_doc_ids_cache(
    *,
    invalidation_targets: Optional[Dict[str, Optional[List[str]]]] = None,
) -> None:
    with _SCOPE_ALLOWED_DOC_IDS_CACHE_LOCK:
        if invalidation_targets is None:
            removed_count = len(_SCOPE_ALLOWED_DOC_IDS_CACHE)
            _SCOPE_ALLOWED_DOC_IDS_CACHE.clear()
            if removed_count:
                _LOGGER.warning(
                    "scope_cache_invalidation mode=global_unknown removed=%s",
                    removed_count,
                )
            return
        removed_cache_keys: List[str] = []
        for cache_key in list(_SCOPE_ALLOWED_DOC_IDS_CACHE):
            cache_namespace, cache_scope_ref = cache_key
            if cache_namespace not in invalidation_targets:
                continue
            target_scope_refs = invalidation_targets[cache_namespace]
            if target_scope_refs is None or cache_scope_ref in set(target_scope_refs):
                _SCOPE_ALLOWED_DOC_IDS_CACHE.pop(cache_key, None)
                removed_cache_keys.append(f"{cache_namespace}|{cache_scope_ref}")
        if removed_cache_keys:
            _LOGGER.warning(
                "scope_cache_invalidation mode=targeted targets=%s removed=%s keys=%s",
                invalidation_targets,
                len(removed_cache_keys),
                removed_cache_keys,
            )


def _encode_uri_segment(value: str) -> str:
    return quote(str(value), safe="")


def _snapshot_origin_uri(*, namespace: str, snapshot_kind: str) -> str:
    return f"twinr://longterm/{_encode_uri_segment(namespace)}/{_encode_uri_segment(snapshot_kind)}"


def _pointer_snapshot_kind(snapshot_kind: str) -> str:
    return f"{_TWINR_SNAPSHOT_POINTER_PREFIX}{snapshot_kind}"


def _pointer_origin_uri(*, namespace: str, snapshot_kind: str) -> str:
    return _snapshot_origin_uri(
        namespace=namespace,
        snapshot_kind=_pointer_snapshot_kind(snapshot_kind),
    )


def _segment_origin_uri(*, namespace: str, uri_segment: str, segment_index: int) -> str:
    return (
        f"twinr://longterm/{_encode_uri_segment(namespace)}/"
        f"{_encode_uri_segment(uri_segment)}/catalog/segment/{int(segment_index):04d}"
    )


def _bounded_scope_phase_content_chars(
    *,
    scope_phase: Optional[str],
    requested_max_content_chars: int,
) -> int:
    requested = max(100, min(int(requested_max_content_chars), 2_000_000))
    phase = str(scope_phase or "").strip()
    env_name = _TWINR_SCOPE_PHASE_CONTENT_CAP_ENVS.get(phase)
    default_cap = int(_TWINR_SCOPE_PHASE_CONTENT_CAP_DEFAULTS.get(phase, requested))
    configured_cap = default_cap
    if env_name:
        try:
            configured_cap = int(os.getenv(env_name, str(default_cap)) or str(default_cap))
        except (TypeError, ValueError):
            configured_cap = default_cap
        configured_cap = max(100, min(int(configured_cap), 2_000_000))
    return min(requested, configured_cap)


def _scope_document_slow_threshold_s() -> float:
    try:
        threshold = float(
            os.getenv("CHONKY_SCOPE_DOCUMENT_TELEMETRY_SLOW_S", "0.25") or "0.25"
        )
    except (TypeError, ValueError):
        threshold = 0.25
    return max(0.0, threshold)


def _parse_twinr_scope_origin_uri(origin_uri: Optional[str]) -> Optional[Dict[str, str]]:
    normalized = _normalize_optional_text(origin_uri)
    if normalized is None or not normalized.startswith("twinr://longterm/"):
        return None
    suffix = normalized[len("twinr://longterm/") :]
    raw_parts = suffix.split("/")
    if len(raw_parts) == 2:
        namespace = unquote(raw_parts[0])
        snapshot_token = unquote(raw_parts[1])
        if snapshot_token.startswith(_TWINR_SNAPSHOT_POINTER_PREFIX):
            snapshot_kind = snapshot_token[len(_TWINR_SNAPSHOT_POINTER_PREFIX) :]
            if snapshot_kind:
                return {
                    "scope_phase": _TWINR_SCOPE_POINTER_PHASE,
                    "namespace": namespace,
                    "snapshot_kind": snapshot_kind,
                }
        if snapshot_token:
            return {
                "scope_phase": _TWINR_SCOPE_SNAPSHOT_PHASE,
                "namespace": namespace,
                "snapshot_kind": snapshot_token,
            }
        return None
    if (
        len(raw_parts) == 5
        and raw_parts[2] == "catalog"
        and raw_parts[3] == "segment"
    ):
        namespace = unquote(raw_parts[0])
        uri_segment = unquote(raw_parts[1])
        return {
            "scope_phase": _TWINR_SCOPE_SEGMENT_PHASE,
            "namespace": namespace,
            "snapshot_kind": uri_segment,
        }
    return None


def _annotate_scope_document_request_context(
    *,
    scope_phase: Optional[str],
    namespace: Optional[str],
    snapshot_kind: Optional[str],
    scope_ref: Optional[str],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    normalized_phase = _normalize_optional_text(scope_phase)
    normalized_namespace = _normalize_optional_text(namespace)
    normalized_snapshot_kind = _normalize_optional_text(snapshot_kind)
    normalized_scope_ref = _normalize_optional_text(scope_ref)
    if normalized_phase is not None:
        payload["full_document_scope_phase"] = normalized_phase
    if normalized_namespace is not None:
        payload["full_document_scope_namespace"] = normalized_namespace
    if normalized_snapshot_kind is not None:
        payload["full_document_snapshot_kind"] = normalized_snapshot_kind
    if normalized_scope_ref is not None:
        payload["full_document_scope_ref"] = normalized_scope_ref
    if not payload:
        return {}
    return update_request_context(**payload)


def _payload_core_read_last_event(router: Any) -> Dict[str, Any]:
    for attr_name in ("get_core_for_read", "get_core"):
        get_core_fn = getattr(router, attr_name, None)
        if not callable(get_core_fn):
            continue
        try:
            core = get_core_fn(FileType.PAYLOAD)
        except Exception:
            continue
        state = getattr(core, "_core_read_telemetry_state_cache", None)
        if not isinstance(state, Mapping):
            continue
        last_event = state.get("last_event")
        if isinstance(last_event, Mapping):
            return dict(last_event)
    return {}


def _payload_uuid_map_for_doc_ids(dm: Any, doc_ids: List[int]) -> Dict[int, Optional[str]]:
    normalized_doc_ids: List[int] = []
    seen_doc_ids: set[int] = set()
    for raw_doc_id in list(doc_ids or []):
        try:
            doc_id_int = int(raw_doc_id)
        except (TypeError, ValueError):
            continue
        if doc_id_int < 0 or doc_id_int in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id_int)
        normalized_doc_ids.append(doc_id_int)
    if not normalized_doc_ids:
        return {}

    lookup_batch = getattr(dm, "get_uuid_for_doc_ids_batch", None)
    if callable(lookup_batch):
        try:
            raw_mapping = lookup_batch(list(normalized_doc_ids))
        except Exception:
            raw_mapping = None
        if isinstance(raw_mapping, Mapping):
            out: Dict[int, Optional[str]] = {}
            for doc_id_int in normalized_doc_ids:
                value = _normalize_optional_text(raw_mapping.get(int(doc_id_int)))
                out[int(doc_id_int)] = value
            return out

    lookup_single = getattr(dm, "get_uuid_for_doc_id", None)
    out = {}
    for doc_id_int in normalized_doc_ids:
        value_text: Optional[str] = None
        if callable(lookup_single):
            try:
                value_text = _normalize_optional_text(lookup_single(int(doc_id_int)))
            except Exception:
                value_text = None
        out[int(doc_id_int)] = value_text
    return out


def _rewrite_origin_lookup_payload_doc_ids(
    router: Any,
    *,
    lookup_key: str,
    lookup_value: str,
    payload: Mapping[str, Any],
    live_doc_ids: List[int],
) -> bool:
    if not live_doc_ids:
        return False
    pack_payload = getattr(router, "_pack_chonkbin", None)
    store_payload = getattr(router, "store_payload", None)
    if not callable(pack_payload) or not callable(store_payload):
        return False

    normalized_live_doc_ids: List[int] = []
    seen_doc_ids: set[int] = set()
    for raw_doc_id in list(live_doc_ids):
        try:
            doc_id_int = int(raw_doc_id)
        except (TypeError, ValueError):
            continue
        if doc_id_int < 0 or doc_id_int in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id_int)
        normalized_live_doc_ids.append(doc_id_int)
    if not normalized_live_doc_ids:
        return False

    payload_obj = dict(payload)
    payload_obj["lookup_value"] = str(payload.get("lookup_value") or lookup_value)
    payload_obj["doc_ids"] = list(sorted(normalized_live_doc_ids))
    payload_obj["doc_count"] = int(len(normalized_live_doc_ids))
    try:
        blob = pack_payload(payload_obj)
        store_payload(str(lookup_key), blob, DataType.METADATA, None)
    except Exception:
        return False
    return True


def _resolve_latest_live_origin_lookup_document_id(
    service: Any,
    *,
    origin_uri: str,
    scope_phase: str,
    namespace: str,
    snapshot_kind: str,
) -> Dict[str, Any]:
    resolution: Dict[str, Any] = {
        "available": False,
        "document_id": None,
    }
    ensure_api = getattr(service, "_ensure_api_server_main_api_for_request", None)
    ensure_ready = getattr(service, "_ensure_engine_docid_router_ready", None)
    decode_payload = getattr(service, "_decode_payload_object", None)
    if not all(callable(candidate) for candidate in (ensure_api, ensure_ready, decode_payload)):
        return resolution

    resolution["available"] = True
    _annotate_scope_document_request_context(
        scope_phase=scope_phase,
        namespace=namespace,
        snapshot_kind=snapshot_kind,
        scope_ref=None,
    )

    try:
        api = ensure_api(timeout_seconds=5.0)
        _engine, dm, router = ensure_ready(api, require_router=True)
    except Exception as exc:
        _LOGGER.warning(
            "scope_head_origin_lookup_unavailable phase=%s namespace=%s snapshot_kind=%s "
            "origin_uri=%s error_type=%s",
            scope_phase,
            namespace,
            snapshot_kind,
            origin_uri,
            type(exc).__name__,
        )
        return resolution

    key_builder = getattr(router, "_origin_lookup_payload_key", None)
    read_payload = getattr(router, "read_payload", None)
    if not callable(key_builder) or not callable(read_payload):
        return resolution

    try:
        lookup_key = str(key_builder(origin_uri))
    except Exception:
        return resolution

    read_mode = "allow_header_reload_on_miss_false"

    payload_core_before = _payload_core_read_last_event(router)
    raw_lookup: Any = None
    read_started_at = time.perf_counter()
    try:
        raw_lookup = read_payload(
            key=lookup_key,
            allow_header_reload_on_miss=False,
        )
    except KeyError:
        raw_lookup = None
    except TypeError as exc:
        raise _contract_http_exception(
            status_code=503,
            error_code="origin_lookup_read_payload_contract_mismatch",
            error="router.read_payload must accept allow_header_reload_on_miss for origin_lookup reads",
            contract="router.read_payload(key=..., allow_header_reload_on_miss=False)",
            original_exception=exc,
            context={
                "scope_phase": scope_phase,
                "namespace": namespace,
                "snapshot_kind": snapshot_kind,
                "origin_uri": origin_uri,
                "lookup_key": lookup_key,
            },
        ) from exc
    except Exception as exc:
        payload_core_after = _payload_core_read_last_event(router)
        elapsed_s = max(0.0, time.perf_counter() - read_started_at)
        payload_event_changed = payload_core_after != payload_core_before
        payload_lock_wait_s = (
            float(payload_core_after.get("lock_wait_s") or 0.0)
            if payload_event_changed
            else 0.0
        )
        payload_total_s = (
            float(payload_core_after.get("total_s") or 0.0)
            if payload_event_changed
            else 0.0
        )
        _LOGGER.warning(
            "scope_head_origin_lookup_error phase=%s namespace=%s snapshot_kind=%s "
            "origin_uri=%s lookup_key=%s read_mode=%s elapsed_s=%.6f "
            "payload_lock_wait_s=%.6f payload_total_s=%.6f error_type=%s",
            scope_phase,
            namespace,
            snapshot_kind,
            origin_uri,
            lookup_key,
            read_mode,
            elapsed_s,
            payload_lock_wait_s,
            payload_total_s,
            type(exc).__name__,
        )
        return resolution

    elapsed_s = max(0.0, time.perf_counter() - read_started_at)
    payload_core_after = _payload_core_read_last_event(router)
    payload_event_changed = payload_core_after != payload_core_before
    payload_lock_wait_s = (
        float(payload_core_after.get("lock_wait_s") or 0.0)
        if payload_event_changed
        else 0.0
    )
    payload_total_s = (
        float(payload_core_after.get("total_s") or 0.0)
        if payload_event_changed
        else 0.0
    )

    if raw_lookup is None:
        if elapsed_s >= _scope_document_slow_threshold_s() or payload_lock_wait_s > 0.0:
            _LOGGER.warning(
                "scope_head_origin_lookup_miss phase=%s namespace=%s snapshot_kind=%s "
                "origin_uri=%s lookup_key=%s read_mode=%s elapsed_s=%.6f "
                "payload_lock_wait_s=%.6f payload_total_s=%.6f",
                scope_phase,
                namespace,
                snapshot_kind,
                origin_uri,
                lookup_key,
                read_mode,
                elapsed_s,
                payload_lock_wait_s,
                payload_total_s,
            )
        return resolution

    payload = decode_payload(raw_lookup)
    if not isinstance(payload, Mapping):
        _LOGGER.warning(
            "scope_head_origin_lookup_decode_failed phase=%s namespace=%s snapshot_kind=%s "
            "origin_uri=%s lookup_key=%s payload_type=%s",
            scope_phase,
            namespace,
            snapshot_kind,
            origin_uri,
            lookup_key,
            type(payload).__name__,
        )
        return resolution

    raw_doc_ids = payload.get("doc_ids")
    if not isinstance(raw_doc_ids, (list, tuple, set)):
        _LOGGER.warning(
            "scope_head_origin_lookup_malformed phase=%s namespace=%s snapshot_kind=%s "
            "origin_uri=%s lookup_key=%s doc_ids_type=%s",
            scope_phase,
            namespace,
            snapshot_kind,
            origin_uri,
            lookup_key,
            type(raw_doc_ids).__name__,
        )
        return resolution

    deduped_doc_ids: List[int] = []
    seen_doc_ids: set[int] = set()
    for raw_doc_id in list(raw_doc_ids):
        try:
            doc_id_int = int(raw_doc_id)
        except (TypeError, ValueError):
            continue
        if doc_id_int < 0 or doc_id_int in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id_int)
        deduped_doc_ids.append(doc_id_int)

    ordered_doc_ids = list(sorted(deduped_doc_ids, reverse=True))
    uuid_by_doc_id = _payload_uuid_map_for_doc_ids(dm, ordered_doc_ids)
    live_doc_ids = [
        int(doc_id_int)
        for doc_id_int in ordered_doc_ids
        if _normalize_optional_text(uuid_by_doc_id.get(int(doc_id_int))) is not None
    ]
    stale_doc_ids = max(0, len(ordered_doc_ids) - len(live_doc_ids))
    if live_doc_ids:
        selected_doc_id_int = int(live_doc_ids[0])
        document_id = _normalize_optional_text(uuid_by_doc_id.get(int(selected_doc_id_int)))
        if document_id is None:
            return resolution
        resolution["document_id"] = document_id
        pruned_stale_doc_ids = False
        if stale_doc_ids > 0:
            pruned_stale_doc_ids = _rewrite_origin_lookup_payload_doc_ids(
                router,
                lookup_key=lookup_key,
                lookup_value=origin_uri,
                payload=payload,
                live_doc_ids=list(live_doc_ids),
            )
        _LOGGER.warning(
            "scope_head_origin_lookup_resolved phase=%s namespace=%s snapshot_kind=%s "
            "origin_uri=%s lookup_key=%s selected_document_id=%s doc_ids_seen=%s stale_doc_ids=%s "
            "read_mode=%s elapsed_s=%.6f payload_lock_wait_s=%.6f payload_total_s=%.6f "
            "uuid_lookup_mode=%s pruned_stale_doc_ids=%s",
            scope_phase,
            namespace,
            snapshot_kind,
            origin_uri,
            lookup_key,
            document_id,
            len(deduped_doc_ids),
            stale_doc_ids,
            read_mode,
            elapsed_s,
            payload_lock_wait_s,
            payload_total_s,
            "batch",
            pruned_stale_doc_ids,
        )
        return resolution

    if stale_doc_ids > 0 or elapsed_s >= _scope_document_slow_threshold_s():
        _LOGGER.warning(
            "scope_head_origin_lookup_no_live_doc phase=%s namespace=%s snapshot_kind=%s "
            "origin_uri=%s lookup_key=%s doc_ids_seen=%s stale_doc_ids=%s "
            "read_mode=%s elapsed_s=%.6f payload_lock_wait_s=%.6f payload_total_s=%.6f "
            "uuid_lookup_mode=%s",
            scope_phase,
            namespace,
            snapshot_kind,
            origin_uri,
            lookup_key,
            len(deduped_doc_ids),
            stale_doc_ids,
            read_mode,
            elapsed_s,
            payload_lock_wait_s,
            payload_total_s,
            "batch",
        )
    return resolution


def _clear_twinr_current_document_cache() -> None:
    with _TWINR_CURRENT_DOCUMENT_CACHE_LOCK:
        _TWINR_CURRENT_DOCUMENT_CACHE_BY_ORIGIN_URI.clear()
        _TWINR_CURRENT_DOCUMENT_CACHE_BY_DOCUMENT_ID.clear()


def _is_cacheable_twinr_document_payload(payload: Mapping[str, Any]) -> bool:
    schema = _normalize_optional_text(payload.get("schema"))
    if schema == _TWINR_REMOTE_SNAPSHOT_SCHEMA or schema == _TWINR_SNAPSHOT_POINTER_SCHEMA:
        return True
    return any(
        schema == definition["segment_schema"]
        for definition in _TWINR_LONGTERM_SCOPE_DEFINITIONS.values()
    )


def _request_item_payload_mapping(item: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    payload = item.get("payload")
    if isinstance(payload, Mapping):
        return dict(payload)
    return _parse_json_mapping_text(item.get("content"))


def _build_cached_twinr_document(
    *,
    request_item: Mapping[str, Any],
    response_item: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    origin_uri = _normalize_optional_text(
        request_item.get("uri") or request_item.get("origin_uri")
    )
    if origin_uri is None or not origin_uri.startswith("twinr://longterm/"):
        return None
    payload = _request_item_payload_mapping(request_item)
    if payload is None or not _is_cacheable_twinr_document_payload(payload):
        return None
    content = _normalize_optional_text(request_item.get("content"))
    if content is None:
        try:
            content = json.dumps(payload, ensure_ascii=False)
        except (TypeError, ValueError):
            return None
    cached_document: Dict[str, Any] = {
        "success": True,
        "origin_uri": origin_uri,
        "content": content,
    }
    metadata = request_item.get("metadata")
    if isinstance(metadata, Mapping):
        cached_document["metadata"] = dict(metadata)
    document_id = _normalize_optional_text(
        None if response_item is None else response_item.get("document_id")
    )
    if document_id is None:
        document_id = _normalize_optional_text(
            request_item.get("document_id")
            or request_item.get("payload_id")
            or request_item.get("chonky_id")
        )
    if document_id is not None:
        cached_document["document_id"] = document_id
    return cached_document


def _build_cached_twinr_document_from_result(
    document: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    def _cache_candidate(
        *,
        origin_uri_value: Any,
        content_value: Any,
        document_id_value: Any,
        metadata_value: Any,
    ) -> Optional[tuple[tuple[int, float], Dict[str, Any]]]:
        origin_uri = _normalize_optional_text(origin_uri_value)
        content = _normalize_optional_text(content_value)
        if origin_uri is None or content is None or not origin_uri.startswith(
            "twinr://longterm/"
        ):
            return None
        payload = _parse_json_mapping_text(content)
        if payload is None or not _is_cacheable_twinr_document_payload(payload):
            return None
        cached_document: Dict[str, Any] = {
            "success": True,
            "origin_uri": origin_uri,
            "content": content,
        }
        if isinstance(metadata_value, Mapping):
            cached_document["metadata"] = dict(metadata_value)
        document_id = _normalize_optional_text(document_id_value)
        if document_id is not None:
            cached_document["document_id"] = document_id
        return _snapshot_updated_at_sort_key(payload.get("updated_at")), cached_document

    direct_candidate = _cache_candidate(
        origin_uri_value=document.get("origin_uri"),
        content_value=document.get("content"),
        document_id_value=document.get("document_id"),
        metadata_value=document.get("metadata"),
    )
    if direct_candidate is not None:
        return direct_candidate[1]

    best_candidate: Optional[tuple[tuple[int, float], Dict[str, Any]]] = None
    chunks = document.get("chunks")
    if not isinstance(chunks, list):
        return None
    for chunk in chunks:
        if not isinstance(chunk, Mapping):
            continue
        metadata = chunk.get("metadata")
        metadata_mapping = metadata if isinstance(metadata, Mapping) else None
        candidate = _cache_candidate(
            origin_uri_value=(
                chunk.get("origin_uri")
                or (
                    metadata_mapping.get("origin_uri")
                    if metadata_mapping is not None
                    else None
                )
                or (
                    metadata_mapping.get("uri")
                    if metadata_mapping is not None
                    else None
                )
                or document.get("origin_uri")
            ),
            content_value=chunk.get("content"),
            document_id_value=(
                chunk.get("document_id")
                or chunk.get("payload_id")
                or chunk.get("chonky_id")
            ),
            metadata_value=metadata_mapping,
        )
        if candidate is None:
            continue
        if best_candidate is None or candidate[0] > best_candidate[0]:
            best_candidate = candidate
    return None if best_candidate is None else best_candidate[1]


def _store_cached_twinr_document(document: Mapping[str, Any]) -> None:
    origin_uri = _normalize_optional_text(document.get("origin_uri"))
    document_id = _normalize_optional_text(document.get("document_id"))
    if origin_uri is None and document_id is None:
        return
    normalized_document = dict(document)
    with _TWINR_CURRENT_DOCUMENT_CACHE_LOCK:
        previous_document_id: Optional[str] = None
        if origin_uri is not None:
            previous = _TWINR_CURRENT_DOCUMENT_CACHE_BY_ORIGIN_URI.get(origin_uri)
            if isinstance(previous, Mapping):
                previous_document_id = _normalize_optional_text(
                    previous.get("document_id")
                )
            _TWINR_CURRENT_DOCUMENT_CACHE_BY_ORIGIN_URI[origin_uri] = normalized_document
        if previous_document_id is not None and previous_document_id != document_id:
            _TWINR_CURRENT_DOCUMENT_CACHE_BY_DOCUMENT_ID.pop(previous_document_id, None)
        if document_id is not None:
            _TWINR_CURRENT_DOCUMENT_CACHE_BY_DOCUMENT_ID[document_id] = normalized_document


def _prime_twinr_current_document_cache_from_write_result(
    *,
    request_items: List[Mapping[str, Any]],
    response_items: Optional[List[Mapping[str, Any]]] = None,
) -> None:
    normalized_response_items = response_items or []
    for index, request_item in enumerate(request_items):
        response_item = None
        if index < len(normalized_response_items):
            candidate = normalized_response_items[index]
            if isinstance(candidate, Mapping):
                response_item = candidate
        cached_document = _build_cached_twinr_document(
            request_item=request_item,
            response_item=response_item,
        )
        if cached_document is not None:
            _store_cached_twinr_document(cached_document)


def _load_cached_twinr_document(
    *,
    document_id: Optional[str],
    origin_uri: Optional[str],
    include_content: bool,
    max_content_chars: int,
) -> Optional[Dict[str, Any]]:
    normalized_origin_uri = _normalize_optional_text(origin_uri)
    normalized_document_id = _normalize_optional_text(document_id)
    cached_document: Optional[Dict[str, Any]] = None
    with _TWINR_CURRENT_DOCUMENT_CACHE_LOCK:
        if normalized_origin_uri is not None:
            by_origin = _TWINR_CURRENT_DOCUMENT_CACHE_BY_ORIGIN_URI.get(
                normalized_origin_uri
            )
            if isinstance(by_origin, Mapping):
                cached_document = dict(by_origin)
        if cached_document is None and normalized_document_id is not None:
            by_document_id = _TWINR_CURRENT_DOCUMENT_CACHE_BY_DOCUMENT_ID.get(
                normalized_document_id
            )
            if isinstance(by_document_id, Mapping):
                cached_document = dict(by_document_id)
    if cached_document is None:
        return None
    content = cached_document.get("content")
    if not include_content:
        cached_document.pop("content", None)
    elif isinstance(content, str) and len(content) > max_content_chars:
        cached_document["content"] = content[:max_content_chars]
    return cached_document


def _iter_json_mapping_candidates(payload: Mapping[str, Any]):
    yield payload
    direct = payload.get("payload")
    if isinstance(direct, Mapping):
        yield direct
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        yield metadata
        nested_payload = metadata.get("twinr_payload")
        if isinstance(nested_payload, Mapping):
            yield nested_payload
    record = payload.get("record")
    if isinstance(record, Mapping):
        yield from _iter_json_mapping_candidates(record)
    document = payload.get("document")
    if isinstance(document, Mapping):
        yield from _iter_json_mapping_candidates(document)
    for field_name in ("content", "content_summary"):
        parsed = _parse_json_mapping_text(payload.get(field_name))
        if parsed is not None:
            yield parsed
    chunks = payload.get("chunks")
    if isinstance(chunks, list):
        for chunk in chunks:
            if not isinstance(chunk, Mapping):
                continue
            yield from _iter_json_mapping_candidates(chunk)


def _is_segmented_catalog_payload(payload: Mapping[str, Any], *, definition: Dict[str, str]) -> bool:
    return (
        payload.get("schema") == definition["catalog_schema"]
        and payload.get("version") == _TWINR_REMOTE_CATALOG_VERSION
        and isinstance(payload.get("segments"), list)
    )


def _is_legacy_catalog_payload(payload: Mapping[str, Any], *, definition: Dict[str, str]) -> bool:
    return (
        payload.get("schema") == definition["legacy_catalog_schema"]
        and payload.get("version") == _TWINR_REMOTE_LEGACY_CATALOG_VERSION
        and isinstance(payload.get("items"), list)
    )


def _extract_catalog_payload(
    payload: Mapping[str, Any],
    *,
    namespace: str,
    definition: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    snapshot_kind = definition["snapshot_kind"]
    latest_snapshot_catalog_body: Optional[Dict[str, Any]] = None
    latest_snapshot_sort_key: Optional[tuple[int, float, int]] = None
    for ordinal, candidate in enumerate(_iter_json_mapping_candidates(payload)):
        if candidate.get("schema") != _TWINR_REMOTE_SNAPSHOT_SCHEMA:
            continue
        if _normalize_optional_text(candidate.get("namespace")) != namespace:
            continue
        if _normalize_optional_text(candidate.get("snapshot_kind")) != snapshot_kind:
            continue
        body = candidate.get("body")
        if not isinstance(body, Mapping):
            continue
        if not (
            _is_segmented_catalog_payload(body, definition=definition)
            or _is_legacy_catalog_payload(body, definition=definition)
        ):
            continue
        sort_key = (*_snapshot_updated_at_sort_key(candidate.get("updated_at")), ordinal)
        if latest_snapshot_sort_key is None or sort_key >= latest_snapshot_sort_key:
            latest_snapshot_sort_key = sort_key
            latest_snapshot_catalog_body = dict(body)
    if latest_snapshot_catalog_body is not None:
        return latest_snapshot_catalog_body
    latest_direct_catalog: Optional[Dict[str, Any]] = None
    for candidate in _iter_json_mapping_candidates(payload):
        if _is_segmented_catalog_payload(candidate, definition=definition) or _is_legacy_catalog_payload(
            candidate,
            definition=definition,
        ):
            latest_direct_catalog = dict(candidate)
    return latest_direct_catalog


def _snapshot_updated_at_sort_key(value: Any) -> tuple[int, float]:
    normalized = _normalize_optional_text(value)
    if normalized is None:
        return (0, 0.0)
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        return (0, 0.0)
    aware = parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    return (1, aware.astimezone(timezone.utc).timestamp())


def _extract_segment_doc_ids(payload: Mapping[str, Any], *, definition: Dict[str, str]) -> List[str]:
    snapshot_kind = definition["snapshot_kind"]
    for candidate in _iter_json_mapping_candidates(payload):
        if candidate.get("schema") != definition["segment_schema"]:
            continue
        if candidate.get("version") != _TWINR_REMOTE_SEGMENT_VERSION:
            continue
        if _normalize_optional_text(candidate.get("snapshot_kind")) != snapshot_kind:
            continue
        raw_items = candidate.get("items")
        if not isinstance(raw_items, list):
            continue
        doc_ids: List[str] = []
        for raw_item in raw_items:
            if not isinstance(raw_item, Mapping):
                continue
            document_id = _normalize_optional_text(raw_item.get("document_id"))
            if document_id is not None:
                doc_ids.append(document_id)
        return doc_ids
    return []


def _dedupe_doc_ids(values: List[str]) -> List[str]:
    deduped: List[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize_optional_text(value)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _extract_pointer_document_id(
    payload: Mapping[str, Any],
    *,
    snapshot_kind: str,
) -> Optional[str]:
    latest_document_id: Optional[str] = None
    latest_sort_key: Optional[tuple[int, float, int]] = None
    for ordinal, candidate in enumerate(_iter_json_mapping_candidates(payload)):
        pointer_payload = candidate
        updated_at = candidate.get("updated_at")
        if candidate.get("schema") == _TWINR_REMOTE_SNAPSHOT_SCHEMA:
            body = candidate.get("body")
            if isinstance(body, Mapping):
                pointer_payload = body
        if pointer_payload.get("schema") != _TWINR_SNAPSHOT_POINTER_SCHEMA:
            continue
        if pointer_payload.get("version") != _TWINR_SNAPSHOT_POINTER_VERSION:
            continue
        if _normalize_optional_text(pointer_payload.get("snapshot_kind")) != snapshot_kind:
            continue
        document_id = _normalize_optional_text(pointer_payload.get("document_id"))
        if document_id is None:
            continue
        sort_key = (*_snapshot_updated_at_sort_key(updated_at), ordinal)
        if latest_sort_key is None or sort_key >= latest_sort_key:
            latest_sort_key = sort_key
            latest_document_id = document_id
    if latest_document_id is not None:
        return latest_document_id
    for candidate in _iter_json_mapping_candidates(payload):
        pointer_payload = candidate
        if candidate.get("schema") == _TWINR_REMOTE_SNAPSHOT_SCHEMA:
            body = candidate.get("body")
            if isinstance(body, Mapping):
                pointer_payload = body
        if pointer_payload.get("schema") != _TWINR_SNAPSHOT_POINTER_SCHEMA:
            continue
        if pointer_payload.get("version") != _TWINR_SNAPSHOT_POINTER_VERSION:
            continue
        if _normalize_optional_text(pointer_payload.get("snapshot_kind")) != snapshot_kind:
            continue
        document_id = _normalize_optional_text(pointer_payload.get("document_id"))
        if document_id is not None:
            return document_id
    return None


def _normalize_payload_blob_mapping(decoded: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(decoded, Mapping):
        return None
    nested_payload = decoded.get("payload")
    if isinstance(nested_payload, Mapping):
        return dict(nested_payload)
    return dict(decoded)


def _cacheable_twinr_payload_from_content_text(
    content_text: Any,
) -> Optional[Dict[str, Any]]:
    normalized_content = _normalize_optional_text(content_text)
    if normalized_content is None:
        return None
    parsed_payload = _parse_json_mapping_text(normalized_content)
    if parsed_payload is None or not _is_cacheable_twinr_document_payload(parsed_payload):
        return None
    return parsed_payload


def _origin_uri_from_twinr_payload(
    payload: Mapping[str, Any],
    *,
    origin_uri_hint: Optional[str],
) -> Optional[str]:
    for candidate in (
        origin_uri_hint,
        payload.get("origin_uri"),
        payload.get("uri"),
        payload.get("source_uri"),
        payload.get("document_uri"),
    ):
        normalized = _normalize_optional_text(candidate)
        if normalized is not None:
            return normalized

    schema = _normalize_optional_text(payload.get("schema"))
    namespace = _normalize_optional_text(payload.get("namespace"))
    snapshot_kind = _normalize_optional_text(payload.get("snapshot_kind"))
    if schema == _TWINR_REMOTE_SNAPSHOT_SCHEMA and namespace is not None and snapshot_kind is not None:
        return _snapshot_origin_uri(namespace=namespace, snapshot_kind=snapshot_kind)
    if schema == _TWINR_SNAPSHOT_POINTER_SCHEMA and origin_uri_hint is not None:
        return origin_uri_hint
    if namespace is None or snapshot_kind is None:
        return origin_uri_hint
    definition = _TWINR_LONGTERM_SCOPE_DEFINITIONS.get(snapshot_kind)
    if definition is None:
        return origin_uri_hint
    if schema == definition["segment_schema"]:
        try:
            segment_index = int(payload.get("segment_index"))
        except (TypeError, ValueError):
            return origin_uri_hint
        return _segment_origin_uri(
            namespace=namespace,
            uri_segment=definition["uri_segment"],
            segment_index=segment_index,
        )
    return origin_uri_hint


def _load_scope_document_payload_blob_fastpath(
    service: Any,
    *,
    document_id: str,
    origin_uri: Optional[str],
    max_content_chars: int,
    scope_phase: Optional[str],
    scope_ref: Optional[str],
    namespace: Optional[str],
    snapshot_kind: Optional[str],
) -> Optional[Dict[str, Any]]:
    ensure_api = getattr(service, "_ensure_api_server_main_api_for_request", None)
    ensure_ready = getattr(service, "_ensure_engine_docid_router_ready", None)
    resolve_doc_int = getattr(service, "_resolve_doc_int_for_payload_ref", None)
    decode_payload = getattr(service, "_decode_payload_object", None)
    read_payload_components = getattr(service, "_read_payload_components_fast", None)
    if not all(callable(candidate) for candidate in (ensure_api, ensure_ready, resolve_doc_int)):
        return None

    started_at = time.perf_counter()
    try:
        api = ensure_api(timeout_seconds=5.0)
        _engine, dm, router = ensure_ready(api, require_router=True)
    except Exception as exc:
        _LOGGER.warning(
            "scope_document_payload_blob_fastpath_unavailable phase=%s scope_ref=%s namespace=%s snapshot_kind=%s "
            "document_id=%s origin_uri=%s error_type=%s",
            scope_phase or "",
            scope_ref or "",
            namespace or "",
            snapshot_kind or "",
            document_id,
            origin_uri or "",
            type(exc).__name__,
        )
        return None

    read_document_data = getattr(dm, "read_document_data", None)
    if not callable(read_document_data):
        return None

    try:
        doc_int = resolve_doc_int(dm, payload_id=document_id, chonky_id=document_id)
    except Exception:
        doc_int = None
    if doc_int is None:
        elapsed_s = max(0.0, time.perf_counter() - started_at)
        _LOGGER.info(
            "scope_document_payload_blob_fastpath_miss phase=%s scope_ref=%s namespace=%s snapshot_kind=%s "
            "document_id=%s origin_uri=%s reason=%s elapsed_s=%.6f",
            scope_phase or "",
            scope_ref or "",
            namespace or "",
            snapshot_kind or "",
            document_id,
            origin_uri or "",
            "doc_id_unresolved",
            elapsed_s,
        )
        return None

    payload_mapping: Optional[Dict[str, Any]] = None
    metadata_value: Optional[Mapping[str, Any]] = None
    content: Optional[str] = None
    source = ""
    miss_reason = "component_text_missing"

    if callable(read_payload_components):
        try:
            metadata_candidate, content_candidate = read_payload_components(
                dm=dm,
                router=router,
                doc_int=int(doc_int),
                include_content=True,
                content_mode="full",
                content_char_limit=None,
            )
        except Exception:
            metadata_candidate, content_candidate = {}, ""
        component_payload_mapping = _cacheable_twinr_payload_from_content_text(
            content_candidate
        )
        if component_payload_mapping is not None:
            payload_mapping = component_payload_mapping
            metadata_value = (
                dict(metadata_candidate)
                if isinstance(metadata_candidate, Mapping)
                else None
            )
            content = str(content_candidate or "")
            source = "component_text"
        else:
            miss_reason = (
                "component_text_not_cacheable"
                if _normalize_optional_text(content_candidate) is not None
                else "component_text_missing"
            )

    if payload_mapping is None:
        if not callable(decode_payload):
            elapsed_s = max(0.0, time.perf_counter() - started_at)
            _LOGGER.info(
                "scope_document_payload_blob_fastpath_miss phase=%s scope_ref=%s namespace=%s snapshot_kind=%s "
                "document_id=%s origin_uri=%s doc_id_int=%s reason=%s elapsed_s=%.6f",
                scope_phase or "",
                scope_ref or "",
                namespace or "",
                snapshot_kind or "",
                document_id,
                origin_uri or "",
                int(doc_int),
                "payload_blob_decoder_missing",
                elapsed_s,
            )
            return None
        try:
            raw_payload = read_document_data(
                int(doc_int),
                "payload",
                raise_on_not_found=False,
                allow_header_reload_on_miss=False,
            )
        except TypeError as exc:
            raise _contract_http_exception(
                status_code=503,
                error_code="payload_blob_read_contract_mismatch",
                error="docid_mapping.read_document_data must accept allow_header_reload_on_miss for payload blob fastpaths",
                contract="docid_mapping.read_document_data(doc_id, 'payload', raise_on_not_found=False, allow_header_reload_on_miss=False)",
                original_exception=exc,
                context={
                    "scope_phase": scope_phase,
                    "scope_ref": scope_ref,
                    "namespace": namespace,
                    "snapshot_kind": snapshot_kind,
                    "document_id": document_id,
                    "origin_uri": origin_uri,
                    "doc_id_int": int(doc_int),
                },
            ) from exc
        except Exception:
            raw_payload = None
        if raw_payload is None:
            elapsed_s = max(0.0, time.perf_counter() - started_at)
            _LOGGER.info(
                "scope_document_payload_blob_fastpath_miss phase=%s scope_ref=%s namespace=%s snapshot_kind=%s "
                "document_id=%s origin_uri=%s doc_id_int=%s reason=%s elapsed_s=%.6f",
                scope_phase or "",
                scope_ref or "",
                namespace or "",
                snapshot_kind or "",
                document_id,
                origin_uri or "",
                int(doc_int),
                "payload_blob_missing",
                elapsed_s,
            )
            return None

        try:
            decoded = decode_payload(raw_payload)
        except Exception:
            elapsed_s = max(0.0, time.perf_counter() - started_at)
            _LOGGER.info(
                "scope_document_payload_blob_fastpath_miss phase=%s scope_ref=%s namespace=%s snapshot_kind=%s "
                "document_id=%s origin_uri=%s doc_id_int=%s reason=%s elapsed_s=%.6f",
                scope_phase or "",
                scope_ref or "",
                namespace or "",
                snapshot_kind or "",
                document_id,
                origin_uri or "",
                int(doc_int),
                "payload_blob_decode_error",
                elapsed_s,
            )
            return None
        payload_mapping = _normalize_payload_blob_mapping(decoded)
        if payload_mapping is None or not _is_cacheable_twinr_document_payload(payload_mapping):
            miss_reason = "payload_blob_not_cacheable"
            elapsed_s = max(0.0, time.perf_counter() - started_at)
            _LOGGER.info(
                "scope_document_payload_blob_fastpath_miss phase=%s scope_ref=%s namespace=%s snapshot_kind=%s "
                "document_id=%s origin_uri=%s doc_id_int=%s reason=%s elapsed_s=%.6f",
                scope_phase or "",
                scope_ref or "",
                namespace or "",
                snapshot_kind or "",
                document_id,
                origin_uri or "",
                int(doc_int),
                miss_reason,
                elapsed_s,
            )
            return None
        try:
            content = json.dumps(payload_mapping, ensure_ascii=False)
        except (TypeError, ValueError):
            elapsed_s = max(0.0, time.perf_counter() - started_at)
            _LOGGER.info(
                "scope_document_payload_blob_fastpath_miss phase=%s scope_ref=%s namespace=%s snapshot_kind=%s "
                "document_id=%s origin_uri=%s doc_id_int=%s reason=%s elapsed_s=%.6f",
                scope_phase or "",
                scope_ref or "",
                namespace or "",
                snapshot_kind or "",
                document_id,
                origin_uri or "",
                int(doc_int),
                "payload_blob_serialize_error",
                elapsed_s,
            )
            return None
        source = "payload_blob"

    if metadata_value is None:
        payload_metadata = payload_mapping.get("metadata")
        if isinstance(payload_metadata, Mapping):
            metadata_value = dict(payload_metadata)
    if content is None:
        try:
            content = json.dumps(payload_mapping, ensure_ascii=False)
        except (TypeError, ValueError):
            return None
    if len(content) > int(max_content_chars):
        content = content[: int(max_content_chars)]

    resolved_origin_uri = _origin_uri_from_twinr_payload(
        payload_mapping,
        origin_uri_hint=origin_uri,
    )
    document: Dict[str, Any] = {
        "success": True,
        "document_id": str(document_id),
        "content": str(content),
    }
    if resolved_origin_uri is not None:
        document["origin_uri"] = resolved_origin_uri
    if isinstance(metadata_value, Mapping):
        document["metadata"] = dict(metadata_value)

    elapsed_s = max(0.0, time.perf_counter() - started_at)
    if elapsed_s >= _scope_document_slow_threshold_s():
        _LOGGER.warning(
            "scope_document_payload_blob_fastpath_slow phase=%s scope_ref=%s namespace=%s snapshot_kind=%s "
            "document_id=%s origin_uri=%s doc_id_int=%s source=%s elapsed_s=%.6f",
            scope_phase or "",
            scope_ref or "",
            namespace or "",
            snapshot_kind or "",
            document_id,
            origin_uri or "",
            int(doc_int),
            source,
            elapsed_s,
        )
    else:
        _LOGGER.info(
            "scope_document_payload_blob_fastpath_hit phase=%s scope_ref=%s namespace=%s snapshot_kind=%s "
            "document_id=%s origin_uri=%s doc_id_int=%s source=%s elapsed_s=%.6f",
            scope_phase or "",
            scope_ref or "",
            namespace or "",
            snapshot_kind or "",
            document_id,
            origin_uri or "",
            int(doc_int),
            source,
            elapsed_s,
        )
    return document


async def _load_scope_document(
    service: Any,
    *,
    document_id: Optional[str] = None,
    origin_uri: Optional[str] = None,
    include_content: bool = True,
    max_content_chars: int = 2_000_000,
    scope_phase: Optional[str] = None,
    scope_ref: Optional[str] = None,
    namespace: Optional[str] = None,
    snapshot_kind: Optional[str] = None,
) -> Dict[str, Any]:
    requested_max_content_chars = max(100, min(int(max_content_chars), 2_000_000))
    effective_max_content_chars = _bounded_scope_phase_content_chars(
        scope_phase=scope_phase,
        requested_max_content_chars=requested_max_content_chars,
    )
    _annotate_scope_document_request_context(
        scope_phase=scope_phase,
        namespace=namespace,
        snapshot_kind=snapshot_kind,
        scope_ref=scope_ref,
    )
    normalized_document_id = _normalize_optional_text(document_id)
    if normalized_document_id is not None:
        fastpath_document = _load_scope_document_payload_blob_fastpath(
            service,
            document_id=normalized_document_id,
            origin_uri=_normalize_optional_text(origin_uri),
            max_content_chars=effective_max_content_chars,
            scope_phase=scope_phase,
            scope_ref=scope_ref,
            namespace=namespace,
            snapshot_kind=snapshot_kind,
        )
        if fastpath_document is not None:
            return fastpath_document
    start = time.perf_counter()
    try:
        result = await service.get_full_document(
            document_id=document_id,
            origin_uri=origin_uri,
            include_content=include_content,
            max_content_chars=effective_max_content_chars,
        )
        document = _raise_for_failed_result(result, default_status=503)
    except HTTPException as exc:
        _LOGGER.warning(
            "scope_document_read_failed phase=%s scope_ref=%s namespace=%s snapshot_kind=%s "
            "document_id=%s origin_uri=%s requested_max_content_chars=%s effective_max_content_chars=%s "
            "status=%s elapsed_s=%.6f detail=%s",
            scope_phase or "",
            scope_ref or "",
            namespace or "",
            snapshot_kind or "",
            document_id or "",
            origin_uri or "",
            requested_max_content_chars,
            effective_max_content_chars,
            exc.status_code,
            time.perf_counter() - start,
            exc.detail,
        )
        raise
    elapsed_s = time.perf_counter() - start
    if elapsed_s >= _scope_document_slow_threshold_s():
        _LOGGER.warning(
            "scope_document_read_slow phase=%s scope_ref=%s namespace=%s snapshot_kind=%s "
            "document_id=%s origin_uri=%s requested_max_content_chars=%s effective_max_content_chars=%s "
            "elapsed_s=%.6f",
            scope_phase or "",
            scope_ref or "",
            namespace or "",
            snapshot_kind or "",
            document_id or "",
            origin_uri or "",
            requested_max_content_chars,
            effective_max_content_chars,
            elapsed_s,
        )
    return document


class _TwinrSnapshotWarmupRetryableError(Exception):
    def __init__(
        self,
        *,
        http_exc: HTTPException,
        origin_uri: str,
        pointer_head_available: bool,
        snapshot_head_available: bool,
    ) -> None:
        super().__init__(str(origin_uri))
        self.http_exc = http_exc
        self.origin_uri = str(origin_uri)
        self.pointer_head_available = bool(pointer_head_available)
        self.snapshot_head_available = bool(snapshot_head_available)


def _scope_warmup_payload(request: Optional[Request]) -> Dict[str, Any]:
    if request is None:
        return {
            "pending": False,
            "pending_count": 0,
            "pending_tasks": [],
            "blocking_pending": False,
            "blocking_count": 0,
            "blocking_tasks": [],
            "readiness_scope": _TWINR_SCOPE_READINESS_SCOPE,
        }
    try:
        warmup = health_router._warmup_pending_payload(
            request,
            readiness_scope=_TWINR_SCOPE_READINESS_SCOPE,
        )
    except Exception:
        warmup = {}
    if isinstance(warmup, Mapping):
        return dict(warmup)
    return {
        "pending": False,
        "pending_count": 0,
        "pending_tasks": [],
        "blocking_pending": False,
        "blocking_count": 0,
        "blocking_tasks": [],
        "readiness_scope": _TWINR_SCOPE_READINESS_SCOPE,
    }


def _is_retryable_twinr_snapshot_not_found(exc: HTTPException) -> bool:
    if int(getattr(exc, "status_code", 500)) != 404:
        return False
    detail = getattr(exc, "detail", None)
    if not isinstance(detail, Mapping):
        return False
    error = str(detail.get("error") or "").strip()
    error_type = str(detail.get("error_type") or "").strip()
    return error == "document_not_found" and error_type != "warmup_pending"


def _scope_warmup_pending_http_exception(
    *,
    request: Optional[Request],
    namespace: str,
    snapshot_kind: str,
    origin_uri: str,
    attempts: int,
    last_exc: HTTPException,
    pointer_head_available: bool,
    snapshot_head_available: bool,
) -> HTTPException:
    warmup = _scope_warmup_payload(request)
    detail = getattr(last_exc, "detail", None)
    retryable_error = (
        str(detail.get("error") or "").strip()
        if isinstance(detail, Mapping)
        else ""
    )
    payload: Dict[str, Any] = {
        "success": False,
        "error": "warmup_pending",
        "error_type": "warmup_pending",
        "detail": "twinr_snapshot_scope_warmup_pending",
        "message": "twinr_snapshot_scope_warmup_pending",
        "namespace": str(namespace),
        "snapshot_kind": str(snapshot_kind),
        "origin_uri": str(origin_uri),
        "attempts": int(attempts),
        "retryable_error": retryable_error,
        "pointer_head_available": bool(pointer_head_available),
        "snapshot_head_available": bool(snapshot_head_available),
    }
    payload = health_router._with_warmup_status(payload, warmup=warmup)
    return HTTPException(status_code=503, detail=to_json_safe(payload))


async def _load_current_twinr_snapshot_document_once(
    service: Any,
    *,
    namespace: str,
    snapshot_kind: str,
    include_content: bool,
    max_content_chars: int,
) -> Dict[str, Any]:
    pointer_origin_uri = _pointer_origin_uri(
        namespace=namespace,
        snapshot_kind=snapshot_kind,
    )
    snapshot_origin_uri = _snapshot_origin_uri(
        namespace=namespace,
        snapshot_kind=snapshot_kind,
    )
    pointer_payload: Dict[str, Any] = {}
    pointer_head = _resolve_latest_live_origin_lookup_document_id(
        service,
        origin_uri=pointer_origin_uri,
        scope_phase=_TWINR_SCOPE_POINTER_PHASE,
        namespace=namespace,
        snapshot_kind=snapshot_kind,
    )
    pointer_head_document_id = _normalize_optional_text(pointer_head.get("document_id"))
    if pointer_head_document_id is not None:
        try:
            pointer_payload = await _load_scope_document(
                service,
                document_id=pointer_head_document_id,
                origin_uri=pointer_origin_uri,
                include_content=True,
                max_content_chars=32_768,
                scope_phase=_TWINR_SCOPE_POINTER_PHASE,
                scope_ref=None,
                namespace=namespace,
                snapshot_kind=snapshot_kind,
            )
        except HTTPException as exc:
            _reraise_contract_http_exception(exc)
            pointer_payload = {}
    try:
        if not pointer_payload and not bool(pointer_head.get("available")):
            pointer_payload = await _load_scope_document(
                service,
                origin_uri=pointer_origin_uri,
                include_content=True,
                max_content_chars=32_768,
                scope_phase=_TWINR_SCOPE_POINTER_PHASE,
                scope_ref=None,
                namespace=namespace,
                snapshot_kind=snapshot_kind,
            )
    except HTTPException:
        pointer_payload = {}
    pointer_document_id = _extract_pointer_document_id(
        pointer_payload,
        snapshot_kind=snapshot_kind,
    )
    if pointer_document_id is not None:
        try:
            return await _load_scope_document(
                service,
                document_id=pointer_document_id,
                origin_uri=snapshot_origin_uri,
                include_content=include_content,
                max_content_chars=max_content_chars,
                scope_phase=_TWINR_SCOPE_SNAPSHOT_PHASE,
                scope_ref=None,
                namespace=namespace,
                snapshot_kind=snapshot_kind,
            )
        except HTTPException as exc:
            _reraise_contract_http_exception(exc)
            pass
    snapshot_head = _resolve_latest_live_origin_lookup_document_id(
        service,
        origin_uri=snapshot_origin_uri,
        scope_phase=_TWINR_SCOPE_SNAPSHOT_PHASE,
        namespace=namespace,
        snapshot_kind=snapshot_kind,
    )
    snapshot_head_document_id = _normalize_optional_text(snapshot_head.get("document_id"))
    if snapshot_head_document_id is not None:
        try:
            return await _load_scope_document(
                service,
                document_id=snapshot_head_document_id,
                origin_uri=snapshot_origin_uri,
                include_content=include_content,
                max_content_chars=max_content_chars,
                scope_phase=_TWINR_SCOPE_SNAPSHOT_PHASE,
                scope_ref=None,
                namespace=namespace,
                snapshot_kind=snapshot_kind,
            )
        except HTTPException as exc:
            _reraise_contract_http_exception(exc)
            pass
    try:
        return await _load_scope_document(
            service,
            origin_uri=snapshot_origin_uri,
            include_content=include_content,
            max_content_chars=max_content_chars,
            scope_phase=_TWINR_SCOPE_SNAPSHOT_PHASE,
            scope_ref=None,
            namespace=namespace,
            snapshot_kind=snapshot_kind,
        )
    except HTTPException as exc:
        if _is_retryable_twinr_snapshot_not_found(exc):
            raise _TwinrSnapshotWarmupRetryableError(
                http_exc=exc,
                origin_uri=snapshot_origin_uri,
                pointer_head_available=bool(pointer_head.get("available")),
                snapshot_head_available=bool(snapshot_head.get("available")),
            ) from exc
        raise


async def _load_current_twinr_snapshot_document(
    service: Any,
    *,
    namespace: str,
    snapshot_kind: str,
    include_content: bool,
    max_content_chars: int,
    request: Optional[Request] = None,
) -> Dict[str, Any]:
    last_retryable_exc: Optional[_TwinrSnapshotWarmupRetryableError] = None
    for attempt in range(1, _TWINR_SCOPE_WARMUP_RETRY_ATTEMPTS + 1):
        try:
            return await _load_current_twinr_snapshot_document_once(
                service,
                namespace=namespace,
                snapshot_kind=snapshot_kind,
                include_content=include_content,
                max_content_chars=max_content_chars,
            )
        except _TwinrSnapshotWarmupRetryableError as exc:
            last_retryable_exc = exc
            warmup = _scope_warmup_payload(request)
            if not bool(warmup.get("pending")):
                raise exc.http_exc
            if attempt < _TWINR_SCOPE_WARMUP_RETRY_ATTEMPTS:
                _LOGGER.warning(
                    "scope_snapshot_read_retry namespace=%s snapshot_kind=%s origin_uri=%s "
                    "attempt=%s/%s delay_s=%.3f pending_tasks=%s pointer_head_available=%s "
                    "snapshot_head_available=%s",
                    namespace,
                    snapshot_kind,
                    exc.origin_uri,
                    attempt,
                    _TWINR_SCOPE_WARMUP_RETRY_ATTEMPTS,
                    _TWINR_SCOPE_WARMUP_RETRY_DELAY_S,
                    list(warmup.get("pending_tasks") or [])[:8],
                    exc.pointer_head_available,
                    exc.snapshot_head_available,
                )
                await asyncio.sleep(_TWINR_SCOPE_WARMUP_RETRY_DELAY_S)
                continue
            raise _scope_warmup_pending_http_exception(
                request=request,
                namespace=namespace,
                snapshot_kind=snapshot_kind,
                origin_uri=exc.origin_uri,
                attempts=attempt,
                last_exc=exc.http_exc,
                pointer_head_available=exc.pointer_head_available,
                snapshot_head_available=exc.snapshot_head_available,
            )
    if last_retryable_exc is not None:
        raise last_retryable_exc.http_exc
    raise RuntimeError("unreachable twinr snapshot loader state")


async def _load_current_scope_snapshot(
    service: Any,
    *,
    namespace: str,
    scope_ref: str,
    definition: Dict[str, str],
) -> Dict[str, Any]:
    snapshot_kind = definition["snapshot_kind"]
    snapshot_payload = await _load_current_twinr_snapshot_document(
        service,
        namespace=namespace,
        snapshot_kind=snapshot_kind,
        include_content=True,
        max_content_chars=512_000,
    )
    _annotate_scope_document_request_context(
        scope_phase=_TWINR_SCOPE_SNAPSHOT_PHASE,
        namespace=namespace,
        snapshot_kind=snapshot_kind,
        scope_ref=scope_ref,
    )
    return snapshot_payload


async def _resolve_scope_allowed_doc_ids(
    service: Any,
    *,
    namespace: str,
    scope_ref: str,
    use_cache: bool = True,
) -> tuple[List[str], bool]:
    normalized_scope_ref, definition = _scope_definition_for_ref(scope_ref)
    if normalized_scope_ref is None or definition is None:
        raise HTTPException(
            status_code=400,
            detail={"success": False, "error": f"unsupported scope_ref: {scope_ref}"},
        )
    normalized_namespace = _normalize_optional_text(namespace)
    if normalized_namespace is None:
        raise HTTPException(
            status_code=400,
            detail={"success": False, "error": "namespace required when scope_ref is used without allowed_doc_ids"},
        )

    if use_cache:
        cached_doc_ids = _load_cached_scope_allowed_doc_ids(
            namespace=normalized_namespace,
            scope_ref=normalized_scope_ref,
        )
        if cached_doc_ids is not None:
            return cached_doc_ids, True
        inflight_refresh = _get_scope_refresh_future(
            namespace=normalized_namespace,
            scope_ref=normalized_scope_ref,
        )
        if inflight_refresh is not None:
            refreshed_doc_ids = await inflight_refresh
            return list(refreshed_doc_ids), True

    snapshot_payload = await _load_current_scope_snapshot(
        service,
        namespace=normalized_namespace,
        scope_ref=normalized_scope_ref,
        definition=definition,
    )
    catalog_payload = _extract_catalog_payload(
        snapshot_payload,
        namespace=normalized_namespace,
        definition=definition,
    )
    if catalog_payload is None:
        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "error": f"scope_ref could not resolve a current catalog for namespace={normalized_namespace!r}",
                "scope_ref": normalized_scope_ref,
                "namespace": normalized_namespace,
            },
        )

    if _is_legacy_catalog_payload(catalog_payload, definition=definition):
        raw_items = catalog_payload.get("items")
        assert isinstance(raw_items, list)
        resolved_doc_ids = _dedupe_doc_ids(
            [
                str(raw_item.get("document_id") or "")
                for raw_item in raw_items
                if isinstance(raw_item, Mapping)
            ]
        )
        _store_cached_scope_allowed_doc_ids(
            namespace=normalized_namespace,
            scope_ref=normalized_scope_ref,
            doc_ids=resolved_doc_ids,
        )
        return resolved_doc_ids, False

    if not _is_segmented_catalog_payload(catalog_payload, definition=definition):
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "error": f"scope_ref resolved an unsupported catalog payload for {normalized_scope_ref}",
                "scope_ref": normalized_scope_ref,
                "namespace": normalized_namespace,
            },
        )

    raw_segments = catalog_payload.get("segments")
    assert isinstance(raw_segments, list)
    resolved_doc_ids: List[str] = []
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, Mapping):
            continue
        try:
            segment_index = int(raw_segment.get("segment_index"))
        except (TypeError, ValueError):
            segment_index = 0
        segment_document_id = _normalize_optional_text(raw_segment.get("document_id"))
        segment_uri = _normalize_optional_text(raw_segment.get("uri")) or _segment_origin_uri(
            namespace=normalized_namespace,
            uri_segment=definition["uri_segment"],
            segment_index=segment_index,
        )
        try:
            segment_payload = await _load_scope_document(
                service,
                document_id=segment_document_id,
                origin_uri=None if segment_document_id else segment_uri,
                scope_phase=_TWINR_SCOPE_SEGMENT_PHASE,
                scope_ref=normalized_scope_ref,
                namespace=normalized_namespace,
                snapshot_kind=definition["snapshot_kind"],
            )
        except HTTPException:
            if segment_document_id is None:
                raise
            segment_payload = await _load_scope_document(
                service,
                origin_uri=segment_uri,
                scope_phase=_TWINR_SCOPE_SEGMENT_PHASE,
                scope_ref=normalized_scope_ref,
                namespace=normalized_namespace,
                snapshot_kind=definition["snapshot_kind"],
            )
        resolved_doc_ids.extend(_extract_segment_doc_ids(segment_payload, definition=definition))
    deduped_doc_ids = _dedupe_doc_ids(resolved_doc_ids)
    _store_cached_scope_allowed_doc_ids(
        namespace=normalized_namespace,
        scope_ref=normalized_scope_ref,
        doc_ids=deduped_doc_ids,
    )
    return deduped_doc_ids, False


async def _refresh_scope_allowed_doc_ids_cache(
    service: Any,
    *,
    invalidation_targets: Optional[Dict[str, Optional[List[str]]]],
) -> None:
    if not invalidation_targets:
        return

    refresh_jobs: List[tuple[str, str]] = []
    for namespace, scope_refs in invalidation_targets.items():
        if not scope_refs:
            continue
        for scope_ref in scope_refs:
            normalized_scope_ref = _normalize_optional_text(scope_ref)
            if normalized_scope_ref is not None:
                refresh_jobs.append((namespace, normalized_scope_ref))

    async def _refresh_one(namespace: str, scope_ref: str) -> None:
        refresh_future, is_owner = _begin_scope_refresh_future(
            namespace=namespace,
            scope_ref=scope_ref,
        )
        if not is_owner:
            await refresh_future
            return
        try:
            resolved_doc_ids, _cache_hit = await _resolve_scope_allowed_doc_ids(
                service,
                namespace=namespace,
                scope_ref=scope_ref,
                use_cache=False,
            )
            _finish_scope_refresh_future(
                namespace=namespace,
                scope_ref=scope_ref,
                future=refresh_future,
                result=resolved_doc_ids,
            )
            _LOGGER.warning(
                "scope_cache_refresh namespace=%s scope_ref=%s doc_ids=%s",
                namespace,
                scope_ref,
                len(resolved_doc_ids),
            )
        except Exception as exc:
            _finish_scope_refresh_future(
                namespace=namespace,
                scope_ref=scope_ref,
                future=refresh_future,
                exc=exc,
            )
            _LOGGER.warning(
                "scope_cache_refresh_failed namespace=%s scope_ref=%s error=%s",
                namespace,
                scope_ref,
                exc,
            )

    await asyncio.gather(
        *[_refresh_one(namespace, scope_ref) for namespace, scope_ref in refresh_jobs]
    )


def _extract_topk_payload(hit: Mapping[str, Any]) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    metadata = _copy_optional_mapping(hit.get("metadata"))
    if isinstance(metadata, dict):
        nested = metadata.get("twinr_payload")
        if isinstance(nested, Mapping):
            return dict(nested), "metadata.twinr_payload"
    direct_payload = hit.get("payload")
    if isinstance(direct_payload, Mapping):
        return dict(direct_payload), "payload"
    record = hit.get("record")
    if isinstance(record, Mapping):
        nested_payload = record.get("payload")
        if isinstance(nested_payload, Mapping):
            return dict(nested_payload), "record.payload"
    document = hit.get("document")
    if isinstance(document, Mapping):
        nested_payload = document.get("payload")
        if isinstance(nested_payload, Mapping):
            return dict(nested_payload), "document.payload"
    for field_name in ("content", "content_summary"):
        parsed = _parse_json_mapping_text(hit.get(field_name))
        if parsed is not None:
            return parsed, field_name
    return None, None


def _extract_topk_identifier(hit: Mapping[str, Any]) -> Optional[str]:
    value = str(hit.get("payload_id") or hit.get("document_id") or "").strip()
    return value or None


def _build_topk_payload_blob_loader(
    service: Any,
) -> Optional[Callable[[Mapping[str, Any]], Optional[Dict[str, Any]]]]:
    ensure_api = getattr(service, "_ensure_api_server_main_api_for_request", None)
    ensure_ready = getattr(service, "_ensure_engine_docid_router_ready", None)
    resolve_doc_int = getattr(service, "_resolve_doc_int_for_payload_ref", None)
    decode_payload = getattr(service, "_decode_payload_object", None)
    if not all(callable(candidate) for candidate in (ensure_api, ensure_ready, resolve_doc_int, decode_payload)):
        return None

    try:
        api = ensure_api(timeout_seconds=5.0)
        _engine, dm, _router = ensure_ready(api, require_router=True)
    except Exception:
        return None

    read_document_data = getattr(dm, "read_document_data", None)
    if not callable(read_document_data):
        return None

    def _load(hit: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        identifier = _extract_topk_identifier(hit)
        if not identifier:
            return None
        try:
            doc_int = resolve_doc_int(dm, payload_id=identifier, chonky_id=identifier)
        except Exception:
            return None
        if doc_int is None:
            return None
        try:
            raw = read_document_data(
                int(doc_int),
                "payload",
                raise_on_not_found=False,
                allow_header_reload_on_miss=False,
            )
        except TypeError as exc:
            raise _contract_http_exception(
                status_code=503,
                error_code="topk_payload_blob_read_contract_mismatch",
                error="docid_mapping.read_document_data must accept allow_header_reload_on_miss for top-k payload blob loads",
                contract="docid_mapping.read_document_data(doc_id, 'payload', raise_on_not_found=False, allow_header_reload_on_miss=False)",
                original_exception=exc,
                context={
                    "payload_id": identifier,
                    "document_id": identifier,
                    "doc_id_int": int(doc_int),
                },
            ) from exc
        except Exception:
            return None
        try:
            decoded = decode_payload(raw)
        except Exception:
            return None
        if not isinstance(decoded, Mapping):
            return None
        nested_payload = decoded.get("payload")
        if isinstance(nested_payload, Mapping):
            return dict(nested_payload)
        return dict(decoded)

    return _load


async def _materialize_topk_results(
    service: Any,
    *,
    hits: List[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    payload_loader = _build_topk_payload_blob_loader(service)
    if payload_loader is None:
        return [_materialize_topk_result(hit) for hit in hits]

    try:
        max_concurrency = int(
            os.getenv("CHONKY_API_TOPK_PAYLOAD_BLOB_MAX_CONCURRENCY", "8") or "8"
        )
    except (TypeError, ValueError):
        max_concurrency = 8
    max_concurrency = max(1, min(32, int(max_concurrency)))
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _one(hit: Mapping[str, Any]) -> Dict[str, Any]:
        payload, payload_source = _extract_topk_payload(hit)
        if payload is None:
            async with semaphore:
                loaded = await asyncio.to_thread(payload_loader, hit)
            if isinstance(loaded, Mapping):
                payload = dict(loaded)
                payload_source = "service.payload_blob"
        return _materialize_topk_result(
            hit,
            payload_override=payload,
            payload_source_override=payload_source,
        )

    return list(await asyncio.gather(*(_one(hit) for hit in hits)))


def _materialize_topk_result(
    hit: Mapping[str, Any],
    *,
    payload_override: Optional[Dict[str, Any]] = None,
    payload_source_override: Optional[str] = None,
) -> Dict[str, Any]:
    payload, payload_source = _extract_topk_payload(hit)
    if payload_override is not None:
        payload = dict(payload_override)
        payload_source = payload_source_override or payload_source or "override"
    result = _drop_none_top_level(
        {
            "payload_id": str(hit.get("payload_id") or hit.get("document_id") or "").strip() or None,
            "document_id": str(hit.get("document_id") or hit.get("payload_id") or "").strip() or None,
            "score": hit.get("score"),
            "relevance_score": hit.get("relevance_score"),
            "source_index": hit.get("source_index"),
            "candidate_origin": hit.get("candidate_origin"),
            "payload_source": payload_source,
            "payload": payload,
            "metadata": _copy_optional_mapping(hit.get("metadata")),
            "score_breakdown": _copy_optional_mapping(hit.get("score_breakdown")),
        }
    )
    return result


def _topk_query_plan(query_result: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    debug = _copy_optional_mapping(query_result.get("debug"))
    latency_ms = None if debug is None else _copy_optional_mapping(debug.get("latency_breakdown_ms"))
    query_plan = _drop_none_top_level(
        {
            "latency_ms": latency_ms,
            "indexes_used": list(query_result.get("indexes_used") or []),
        }
    )
    return query_plan or None


@router.get(
    "/instance",
    response_model=ExternalInstanceResponse,
    summary="External memory instance overview",
)
async def instance_overview(request: Request):
    service = _get_svc(request)
    component_timeout_s = _instance_component_timeout_s()
    stats, indexes, metrics = await asyncio.gather(
        _await_instance_component(
            "system_stats",
            admin_router._cached_system_stats_payload(
                service,
                timeout_seconds=component_timeout_s,
            ),
            timeout_seconds=component_timeout_s,
        ),
        _await_instance_component(
            "indexes",
            _call_service(service, "list_available_indexes"),
            timeout_seconds=component_timeout_s,
        ),
        _await_instance_component(
            "basic_metrics",
            _call_service(service, "get_basic_metrics"),
            timeout_seconds=component_timeout_s,
        ),
    )

    ready_event = getattr(
        getattr(request.app, "state", None), "_warmup_ready_event", None
    )
    ready = bool(getattr(ready_event, "is_set", lambda: True)())
    gate_enabled = bool(
        getattr(getattr(request.app, "state", None), "_warmup_gate_enabled", False)
    )
    degraded_components = [
        name
        for name, payload in (
            ("system_stats", stats),
            ("indexes", indexes),
            ("basic_metrics", metrics),
        )
        if isinstance(payload, Mapping) and payload.get("success") is False
    ]
    if degraded_components:
        _LOGGER.warning(
            "instance_overview degraded ready=%s gate_enabled=%s components=%s",
            bool(ready),
            bool(gate_enabled),
            ",".join(degraded_components),
        )

    return _drop_none_top_level(
        {
            "success": True,
            "service": "ccodex_memory",
            "ready": bool(ready and not gate_enabled),
            "auth_enabled": bool(str(os.getenv("CHONKDB_API_KEY", "")).strip()),
            "degraded": bool(degraded_components),
            "degraded_components": degraded_components or None,
            "component_timeout_seconds": float(component_timeout_s),
            "system_stats": stats,
            "indexes": indexes,
            "basic_metrics": metrics,
        }
    )


@router.get(
    "/admin/auth",
    summary="External admin auth configuration overview",
)
async def admin_auth_config():
    return _auth_config_payload()


@router.get(
    "/admin/docs",
    summary="External admin API documentation index",
)
async def admin_docs_index(request: Request):
    base_url = _public_base_url(request)
    return {
        "success": True,
        "auth": _auth_config_payload(),
        "documentation_file": _API_DOCS_PATH.name,
        "public_base_url": base_url or None,
        "service_api_root": _join_public_url(base_url, "/v1"),
        "swagger_ui_url": _join_public_url(base_url, "/docs"),
        "redoc_url": _join_public_url(base_url, "/redoc"),
        "openapi_url": _join_public_url(base_url, "/openapi.json"),
        "guide_url": _join_public_url(base_url, "/v1/external/admin/docs/guide"),
        "openapi_alias_url": _join_public_url(
            base_url, "/v1/external/admin/docs/openapi.json"
        ),
        "guide_endpoint": "/v1/external/admin/docs/guide",
        "openapi_alias_endpoint": "/v1/external/admin/docs/openapi.json",
    }


@router.get(
    "/admin/docs/guide",
    response_class=PlainTextResponse,
    summary="External admin detailed API guide",
)
async def admin_docs_guide():
    return PlainTextResponse(
        _load_api_documentation_markdown(),
        media_type="text/markdown; charset=utf-8",
    )


@router.get(
    "/admin/docs/openapi.json",
    summary="External admin OpenAPI schema alias",
)
async def admin_docs_openapi(request: Request):
    return JSONResponse(content=to_json_safe(request.app.openapi()))


@router.get(
    "/admin/health",
    summary="External admin health probe",
)
async def admin_health(
    request: Request,
    scope: Optional[str] = Query(
        default=None,
        description="Optional readiness scope for warmup status shaping. Supported: full, token_fast.",
    ),
):
    service = _get_svc(request)
    payload = _health_payload_or_http_error(
        await _call_service(service, "health_check")
    )
    readiness_scope = health_router._normalize_ready_scope(scope)
    warmup = health_router._warmup_pending_payload(
        request,
        readiness_scope=readiness_scope,
    )
    payload = health_router._with_warmup_status(payload, warmup=warmup)
    payload["auth"] = _auth_config_payload()
    return payload


@router.get(
    "/admin/ready",
    summary="External admin readiness probe",
)
async def admin_ready(
    request: Request,
    scope: Optional[str] = Query(
        default=None,
        description="Optional readiness scope. Supported: full, token_fast.",
    ),
):
    service = _get_svc(request)
    raw_payload = _health_payload_or_http_error(
        await _call_service(service, "health_check")
    )
    readiness_scope = health_router._normalize_ready_scope(scope)
    warmup = health_router._warmup_pending_payload(
        request,
        readiness_scope=readiness_scope,
    )
    payload = health_router._with_warmup_status(raw_payload, warmup=warmup)
    ready_require_warmup = health_router._ready_require_warmup()
    if health_router._ready_payload(payload) or (
        not ready_require_warmup and health_router._ready_payload(raw_payload)
    ):
        return payload
    raise HTTPException(status_code=503, detail=to_json_safe(payload))


@router.get(
    "/admin/stats",
    summary="External admin system statistics",
)
async def admin_stats(request: Request):
    service = _get_svc(request)
    return await admin_router._cached_system_stats_payload(service)


@router.get(
    "/admin/metrics/basic",
    summary="External admin basic metrics",
)
async def admin_basic_metrics(request: Request):
    service = _get_svc(request)
    return _raise_for_failed_result(
        await _call_service(service, "get_basic_metrics"), default_status=503
    )


@router.get(
    "/admin/indexes",
    summary="External admin index inventory",
)
async def admin_indexes(request: Request):
    service = _get_svc(request)
    return _raise_for_failed_result(
        await _call_service(service, "list_available_indexes"), default_status=503
    )


@router.get(
    "/admin/storage",
    summary="External admin storage and data-dir footprint",
)
async def admin_storage(request: Request):
    service = _get_svc(request)
    stats = await admin_router._cached_system_stats_payload(service)
    return _storage_payload_from_stats(stats)


@router.post(
    "/admin/provision",
    summary="Idempotently provision or open the external memory instance",
)
async def admin_provision(request: Request, req: ExternalAdminProvisionRequest):
    service = _get_svc(request)
    stats = await admin_router._cached_system_stats_payload(service)
    indexes = _raise_for_failed_result(
        await _call_service(service, "list_available_indexes"), default_status=503
    )
    out: Dict[str, Any] = {
        "success": True,
        "provisioned": True,
        "auth": _auth_config_payload(),
        "stats": stats,
        "indexes": indexes,
        "storage": _storage_payload_from_stats(stats),
    }

    if bool(req.warm_start):
        warm_payload = _drop_none_top_level(
            {
                "components": req.components,
                "load_embeddings": req.load_embeddings,
                "load_information_extraction": req.load_information_extraction,
                "timeout_seconds": req.timeout_seconds,
            }
        )
        out["warm_start"] = _raise_for_failed_result(
            await _call_service(service, "warm_start_models", **warm_payload),
            default_status=503,
        )

    if bool(req.flush_after):
        out["flush"] = _raise_for_failed_result(
            await _call_service(
                service,
                "flush_all",
                strongest=bool(req.strongest_flush),
                timeout_seconds=req.timeout_seconds,
            ),
            default_status=503,
        )

    if bool(req.include_health):
        out["health"] = _health_payload_or_http_error(
            await _call_service(service, "health_check")
        )

    return out


@router.post(
    "/admin/warm-start",
    summary="External admin warm-start of heavy models and caches",
)
async def admin_warm_start(request: Request, req: WarmStartRequest):
    service = _get_svc(request)
    payload = _drop_none_top_level(_model_dump(req))
    return _raise_for_failed_result(
        await _call_service(service, "warm_start_models", **payload),
        default_status=503,
    )


@router.post(
    "/admin/optimize",
    summary="External admin optimize and maintenance run",
)
async def admin_optimize(request: Request, req: OptimizeIndexesRequest):
    service = _get_svc(request)
    payload = _drop_none_top_level(_model_dump(req))
    return _raise_for_failed_result(
        await _call_service(service, "optimize_indexes", **payload),
        default_status=503,
    )


@router.post(
    "/admin/reload",
    summary="External admin reload from disk",
)
async def admin_reload(request: Request):
    service = _get_svc(request)
    return _raise_for_failed_result(
        await _call_service(service, "reload_from_disk"), default_status=501
    )


@router.post(
    "/admin/flush",
    summary="External admin durable flush",
)
async def admin_flush(
    request: Request,
    strongest: bool = Query(default=True),
    timeout_seconds: Optional[float] = Query(default=None, ge=0.001, le=3600),
):
    service = _get_svc(request)
    return _raise_for_failed_result(
        await _call_service(
            service,
            "flush_all",
            strongest=strongest,
            timeout_seconds=timeout_seconds,
        ),
        default_status=503,
    )


@router.delete(
    "/admin/data",
    summary="External admin clear-all operation",
)
async def admin_clear_all(
    request: Request,
    confirm: bool = Query(default=False),
):
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail={"success": False, "error": "confirm=true required"},
        )
    service = _get_svc(request)
    result = _raise_for_failed_result(
        await _call_service(service, "clear_all", confirm=True),
        default_status=503,
    )
    _clear_twinr_current_document_cache()
    _invalidate_scope_allowed_doc_ids_cache(invalidation_targets=None)
    return result


@router.post(
    "/records",
    summary="Ingest one record through the external memory facade",
)
async def ingest_record(request: Request, req: ExternalRecordRequest):
    service = _get_svc(request)
    operation, execution_mode, client_request_id, payload = _build_record_payload(req)
    invalidation_targets = _extract_scope_cache_invalidation_targets_from_items(
        [payload]
    )

    if operation == "ingest_document":
        if execution_mode == "async":
            res = await _start_custom_job(
                name="external_ingest_document",
                coro_factory=lambda: service.ingest_document(**payload),
            )
            res["operation"] = operation
            res["execution_mode"] = execution_mode
            if client_request_id:
                res["client_request_id"] = client_request_id
            return JSONResponse(status_code=202, content=to_json_safe(res))
        res = _raise_for_failed_result(await service.ingest_document(**payload))
        res["operation"] = operation
        res["execution_mode"] = execution_mode
        if client_request_id:
            res["client_request_id"] = client_request_id
        return res

    if execution_mode == "async":
        res = _raise_for_failed_result(await service.start_ingest_job(**payload))
        _prime_twinr_current_document_cache_from_write_result(
            request_items=[payload],
        )
        res["operation"] = operation
        res["execution_mode"] = execution_mode
        if client_request_id:
            res["client_request_id"] = client_request_id
        return JSONResponse(status_code=202, content=to_json_safe(res))

    timeout_seconds = payload.pop("timeout_seconds", None)
    res = _single_item_bulk_result_or_http_error(
        await service.store_payloads_sync_bulk(
            items=[payload],
            timeout_seconds=timeout_seconds,
            client_request_id=client_request_id,
        ),
        default_status=503,
    )
    _prime_twinr_current_document_cache_from_write_result(
        request_items=[payload],
        response_items=[res],
    )
    _invalidate_scope_allowed_doc_ids_cache(invalidation_targets=invalidation_targets)
    await _refresh_scope_allowed_doc_ids_cache(
        service,
        invalidation_targets=invalidation_targets,
    )
    res["operation"] = operation
    res["execution_mode"] = execution_mode
    if client_request_id:
        res["client_request_id"] = client_request_id
    return res


@router.post(
    "/records/bulk",
    summary="Bulk ingest through the external memory facade",
)
async def ingest_records_bulk(request: Request, req: ExternalRecordBulkRequest):
    service = _get_svc(request)
    (
        operation,
        execution_mode,
        client_request_id,
        timeout_seconds,
        finalize_vector_segments,
        items,
    ) = _build_bulk_payload(req)
    invalidation_targets = _extract_scope_cache_invalidation_targets_from_items(items)

    if operation == "ingest_document":
        if execution_mode == "async":
            res = await _start_custom_job(
                name="external_ingest_document_bulk",
                coro_factory=lambda: _execute_bulk_document_ingest(
                    service,
                    items=items,
                ),
            )
            res["operation"] = operation
            res["execution_mode"] = execution_mode
            res["items"] = int(len(items))
            if client_request_id:
                res["client_request_id"] = client_request_id
            return JSONResponse(status_code=202, content=to_json_safe(res))

        res = await _execute_bulk_document_ingest(service, items=items)
        res["operation"] = operation
        res["execution_mode"] = execution_mode
        if client_request_id:
            res["client_request_id"] = client_request_id
        return res

    if execution_mode == "async":
        res = _raise_for_failed_result(
            await service.start_ingest_job_bulk(
                items,
                timeout_seconds=timeout_seconds,
                finalize_vector_segments=finalize_vector_segments,
            )
        )
        _prime_twinr_current_document_cache_from_write_result(
            request_items=[item for item in items if isinstance(item, Mapping)],
        )
        res["operation"] = operation
        res["execution_mode"] = execution_mode
        if client_request_id:
            res["client_request_id"] = client_request_id
        return JSONResponse(status_code=202, content=to_json_safe(res))

    res = _raise_for_failed_result(
        await service.store_payloads_sync_bulk(
            items=items,
            timeout_seconds=timeout_seconds,
            client_request_id=client_request_id,
        )
    )
    _prime_twinr_current_document_cache_from_write_result(
        request_items=[item for item in items if isinstance(item, Mapping)],
        response_items=list(res.get("items") or []),
    )
    _invalidate_scope_allowed_doc_ids_cache(invalidation_targets=invalidation_targets)
    await _refresh_scope_allowed_doc_ids_cache(
        service,
        invalidation_targets=invalidation_targets,
    )
    res["operation"] = operation
    res["execution_mode"] = execution_mode
    return res


@router.post(
    "/retrieve",
    summary="Unified retrieval endpoint for basic, advanced, and graph search",
)
async def retrieve(request: Request, req: ExternalRetrieveRequest):
    service = _get_svc(request)
    mode = str(req.mode or "advanced").strip().lower()

    if mode == "graph":
        res = await service.graph_semantic_filter(**_graph_query_payload(req))
        out = _query_response_or_http_error(res)
        out["mode"] = mode
        return out

    if not str(req.query_text or "").strip():
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": "query_text required for basic or advanced mode",
            },
        )

    if mode == "basic":
        payload = _advanced_query_payload(req)
        res = await service.query_payloads(**payload)
        out = _query_response_or_http_error(res)
        out["mode"] = mode
        return out

    if mode != "advanced":
        raise HTTPException(
            status_code=400,
            detail={"success": False, "error": f"unsupported mode: {mode}"},
        )

    payload = _advanced_query_payload(req)
    res = await service.query_payloads_advanced(**payload)
    out = _query_response_or_http_error(res)
    out["mode"] = mode
    return out


@router.post(
    "/retrieve/topk_records",
    summary="Advanced retrieval endpoint that returns structured top-k payloads",
)
async def retrieve_topk_records(request: Request, req: ExternalTopKRecordsRequest):
    service = _get_svc(request)
    mode = str(req.mode or "advanced").strip().lower()
    if mode != "advanced":
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": f"unsupported mode for topk_records: {mode}",
            },
        )
    if not str(req.query_text or "").strip():
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": "query_text required for topk_records",
            },
        )
    normalized_scope_ref = _normalize_optional_text(req.scope_ref)
    resolved_scope_doc_ids: Optional[List[str]] = None
    scope_resolve_latency_ms: Optional[float] = None
    scope_cache_hit = False
    if normalized_scope_ref and not list(req.allowed_doc_ids or []):
        normalized_namespace = _normalize_optional_text(req.namespace)
        if normalized_namespace is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "error": "namespace required when scope_ref is used without allowed_doc_ids",
                },
            )
        started_scope_resolve = time.perf_counter()
        resolved_scope_doc_ids, scope_cache_hit = await _resolve_scope_allowed_doc_ids(
            service,
            namespace=normalized_namespace,
            scope_ref=normalized_scope_ref,
        )
        scope_resolve_latency_ms = round(
            max(0.0, (time.perf_counter() - started_scope_resolve) * 1000.0),
            3,
        )

    payload = _advanced_query_payload(req)
    if resolved_scope_doc_ids is not None:
        if not resolved_scope_doc_ids:
            empty_query_plan: Optional[Dict[str, Any]] = {
                "scope_cache_hit": scope_cache_hit
            }
            if scope_resolve_latency_ms is not None:
                empty_query_plan["latency_ms"] = {
                    "scope_resolve": scope_resolve_latency_ms
                }
            return to_json_safe(
                _drop_none_top_level(
                    {
                        "success": True,
                        "mode": "advanced",
                        "results": [],
                        "indexes_used": [],
                        "scope_ref": normalized_scope_ref,
                        "query_plan": empty_query_plan,
                    }
                )
            )
        payload["allowed_doc_ids"] = resolved_scope_doc_ids
    res = await service.query_payloads_advanced(**payload)
    out = _query_response_or_http_error(res)
    raw_results = out.get("results")
    hits = (
        [item for item in raw_results if isinstance(item, Mapping)]
        if isinstance(raw_results, list)
        else []
    )
    materialized_hits = await _materialize_topk_results(service, hits=hits)
    query_plan = _topk_query_plan(out)
    if scope_resolve_latency_ms is not None:
        query_plan = dict(query_plan or {})
        latency_ms = _copy_optional_mapping(query_plan.get("latency_ms")) or {}
        latency_ms["scope_resolve"] = scope_resolve_latency_ms
        query_plan["latency_ms"] = latency_ms
    query_plan = dict(query_plan or {})
    query_plan["scope_cache_hit"] = scope_cache_hit
    response = {
        "success": True,
        "mode": "advanced",
        "results": materialized_hits,
        "indexes_used": list(out.get("indexes_used") or []),
        "scope_ref": normalized_scope_ref,
        "query_plan": query_plan,
    }
    return to_json_safe(_drop_none_top_level(response))


@router.post(
    "/graph/edges",
    summary="External graph edge creation by node ids",
)
async def external_graph_add_edge(request: Request, req: AddEdgeRequest):
    service = _get_svc(request)
    return _graph_response_or_http_error(
        await _call_service(service, "add_graph_edge", **_model_dump(req))
    )


@router.post(
    "/graph/edges/smart",
    summary="External graph edge creation by labels or ids",
)
async def external_graph_add_edge_smart(request: Request, req: AddEdgeSmartRequest):
    service = _get_svc(request)
    return _graph_response_or_http_error(
        await _call_service(service, "add_graph_edge_smart", **_model_dump(req))
    )


@router.post(
    "/graph/neighbors",
    summary="External graph neighbors lookup",
)
async def external_graph_neighbors(
    request: Request, req: ExternalGraphNeighborsRequest
):
    service = _get_svc(request)
    label_or_id = str(req.label_or_id or req.label or "").strip()
    if not label_or_id:
        raise HTTPException(
            status_code=400,
            detail={"success": False, "error": "label_or_id required"},
        )
    payload: Dict[str, Any] = {
        "index_name": graph_router._normalize_graph_index(req.index_name),
        "label": label_or_id,
        "edge_types": req.edge_types,
        "direction": str(req.direction or "both"),
        "return_ids": bool(req.return_ids),
        "with_edges": bool(req.with_edges),
        "limit": req.limit,
        "timeout_seconds": req.timeout_seconds,
    }
    return _graph_response_or_http_error(
        await _call_service(service, "graph_neighbors", **_drop_none_top_level(payload))
    )


@router.post(
    "/graph/path",
    summary="External shortest-path lookup between two graph labels",
)
async def external_graph_path(request: Request, req: ExternalGraphPathRequest):
    service = _get_svc(request)
    source_label = str(req.source_label or req.source or "").strip()
    target_label = str(req.target_label or req.target or "").strip()
    if not source_label or not target_label:
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": "source_label and target_label required",
            },
        )
    payload = {
        "index_name": graph_router._normalize_graph_index(req.index_name),
        "source_label": source_label,
        "target_label": target_label,
        "edge_types": req.edge_types,
        "return_ids": bool(req.return_ids),
    }
    return _graph_response_or_http_error(
        await _call_service(service, "graph_path", **_drop_none_top_level(payload))
    )


@router.post(
    "/graph/patterns",
    summary="External graph pattern query",
)
async def external_graph_patterns(request: Request, req: ExternalGraphPatternsRequest):
    service = _get_svc(request)
    payload = {
        "index_name": graph_router._normalize_graph_index(req.index_name),
        "patterns": list(req.patterns or []),
        "limit": int(req.limit),
        "max_depth": int(req.max_depth),
        "include_content": bool(req.include_content),
    }
    return _graph_response_or_http_error(
        await _call_service(service, "query_graph_patterns", **payload)
    )


@router.get(
    "/records",
    summary="List stored records through the external facade",
)
async def list_records(
    request: Request,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
    include_metadata: bool = Query(default=True),
):
    service = _get_svc(request)
    return _raise_for_failed_result(
        await service.list_all_payloads(
            offset=offset,
            limit=limit,
            include_metadata=include_metadata,
        )
    )


@router.get(
    "/records/{record_id}",
    summary="Fetch one record and optional full document materialization",
)
async def get_record(
    request: Request,
    record_id: str,
    by: str = Query(default="payload", pattern="^(payload|chonky)$"),
    include_chunks: bool = Query(default=False),
    include_relationships: bool = Query(default=False),
    include_document: bool = Query(default=False),
    include_content: bool = Query(default=True),
    max_content_chars: int = Query(default=4000, ge=100, le=2_000_000),
):
    service = _get_svc(request)
    info = _raise_for_failed_result(
        await service.get_payload_info(
            payload_id=record_id if by == "payload" else None,
            chonky_id=record_id if by == "chonky" else None,
            include_chunks=include_chunks,
            include_relationships=include_relationships,
        ),
        default_status=404,
    )
    if not include_document:
        return info

    document_id = str(info.get("chonky_id") or "").strip() or None
    origin_uri = None
    metadata = info.get("metadata")
    if isinstance(metadata, Mapping):
        origin_uri = (
            str(
                metadata.get("origin_uri")
                or metadata.get("uri")
                or metadata.get("source_uri")
                or ""
            ).strip()
            or None
        )
    document = _raise_for_failed_result(
        await service.get_full_document(
            document_id=document_id,
            origin_uri=origin_uri,
            include_content=include_content,
            max_content_chars=max_content_chars,
        ),
        default_status=404,
    )
    out = dict(info)
    out["document"] = document
    return out


@router.get(
    "/documents/full",
    summary="Fetch a full document by internal id or origin URI",
)
async def get_full_document(
    request: Request,
    document_id: Optional[str] = Query(default=None),
    origin_uri: Optional[str] = Query(default=None),
    include_content: bool = Query(default=True),
    max_content_chars: int = Query(default=4000, ge=100, le=2_000_000),
):
    if not document_id and not origin_uri:
        raise HTTPException(
            status_code=400,
            detail={"success": False, "error": "document_id or origin_uri required"},
        )
    scope_origin_details = _parse_twinr_scope_origin_uri(origin_uri)
    effective_max_content_chars = max_content_chars
    if scope_origin_details is not None:
        effective_max_content_chars = _bounded_scope_phase_content_chars(
            scope_phase=scope_origin_details.get("scope_phase"),
            requested_max_content_chars=max_content_chars,
        )
        _annotate_scope_document_request_context(
            scope_phase=scope_origin_details.get("scope_phase"),
            namespace=scope_origin_details.get("namespace"),
            snapshot_kind=scope_origin_details.get("snapshot_kind"),
            scope_ref=None,
        )
    cached_document = _load_cached_twinr_document(
        document_id=document_id,
        origin_uri=origin_uri,
        include_content=include_content,
        max_content_chars=effective_max_content_chars,
    )
    if cached_document is not None:
        return cached_document
    service = _get_svc(request)
    if (
        document_id is None
        and scope_origin_details is not None
        and scope_origin_details.get("scope_phase") == _TWINR_SCOPE_SNAPSHOT_PHASE
    ):
        document = await _load_current_twinr_snapshot_document(
            service,
            namespace=str(scope_origin_details.get("namespace") or ""),
            snapshot_kind=str(scope_origin_details.get("snapshot_kind") or ""),
            include_content=include_content,
            max_content_chars=effective_max_content_chars,
            request=request,
        )
    else:
        document = _raise_for_failed_result(
            await service.get_full_document(
                document_id=document_id,
                origin_uri=origin_uri,
                include_content=include_content,
                max_content_chars=effective_max_content_chars,
            ),
            default_status=404,
        )
    cached_result = _build_cached_twinr_document_from_result(document)
    if cached_result is not None:
        _store_cached_twinr_document(cached_result)
    return document


@router.delete(
    "/records/{record_id}",
    summary="Delete a stored record through the external facade",
)
async def delete_record(
    request: Request,
    record_id: str,
    by: str = Query(default="payload", pattern="^(payload|chonky)$"),
):
    service = _get_svc(request)
    result = _raise_for_failed_result(
        await service.delete_payload(
            payload_id=record_id if by == "payload" else None,
            chonky_id=record_id if by == "chonky" else None,
        )
    )
    _clear_twinr_current_document_cache()
    _invalidate_scope_allowed_doc_ids_cache(invalidation_targets=None)
    return result


@router.get(
    "/jobs/{job_id}",
    response_model=ExternalJobResponse,
    summary="Get external ingest/retrieval job status",
)
async def get_job(request: Request, job_id: str):
    service = _get_svc(request)
    return _raise_for_failed_result(await service.job_status(str(job_id)))


@router.post(
    "/jobs/{job_id}/cancel",
    response_model=ExternalJobResponse,
    summary="Cancel a running external job",
)
async def cancel_job(request: Request, job_id: str):
    service = _get_svc(request)
    return _raise_for_failed_result(await service.cancel_job(str(job_id)))
