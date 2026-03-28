"""Serve Twinr's local FastAPI control surface.

##REFACTOR: 2026-03-27##

This module preserves the stable `twinr.web.app` import surface while the real
implementation now lives under `twinr.web.app_impl/`.
"""

# ruff: noqa: F401

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI

from twinr.agent.self_coding.retest import run_self_coding_skill_retest
from twinr.channels.whatsapp.history_import import WhatsAppHistoryImportQueue
from twinr.integrations.email.connectivity import run_email_connectivity_test
from twinr.memory.longterm.retrieval.operator_search import run_long_term_operator_search
from twinr.ops import TwinrSelfTestRunner, collect_device_overview
from twinr.web.conversation_lab import (
    create_conversation_lab_session,
    load_conversation_lab_state,
    run_conversation_lab_turn,
)
from twinr.web.presenters import _capture_voice_profile_sample, _recent_named_files
from twinr.web.support.whatsapp import WhatsAppPairingCoordinator, probe_whatsapp_runtime

from .app_impl.compat import (
    _DEFAULT_ALLOWED_HOSTS,
    _DEFAULT_MAX_FORM_BYTES,
    _MAX_FORM_BYTES_CAP,
    _apply_auth_context,
    _auth_login_location,
    _call_sync,
    _clear_managed_auth_cookie,
    _conversation_lab_href,
    _env_bool,
    _env_int,
    _error_response,
    _forwarded_header_value,
    _has_trusted_same_origin,
    _has_valid_basic_auth,
    _is_allowed_host,
    _is_default_origin_port,
    _is_loopback_host,
    _is_same_origin_url,
    _normalize_host_header,
    _parse_allowed_hosts,
    _parse_bounded_form,
    _public_error_message,
    _redirect_location,
    _redirect_saved,
    _redirect_with_error,
    _reminder_sort_key,
    _request_origin,
    _request_target_path,
    _require_non_empty,
    _require_positive_int,
    _resolve_downloadable_file,
    _safe_file_in_dir,
    _safe_next_path,
    _safe_project_subpath,
    _secure_response,
    _set_managed_auth_cookie,
    logger,
)
from .app_impl.main import create_app as _create_app

__all__ = ["create_app"]


def create_app(env_file: str | Path = ".env") -> FastAPI:
    """Create the FastAPI app for Twinr's local control surface."""

    return _create_app(env_file, surface_module=sys.modules[__name__])
