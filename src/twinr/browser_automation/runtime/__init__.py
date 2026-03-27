"""Tracked runtime helpers for Twinr browser-automation orchestration.

The repo-root ``browser_automation/`` workspace stays the optional executor
surface. Shared typed-result and auth-state helpers live here so both the
tracked benchmark adapters and the local executor bridge can agree on one
stable shape without growing more logic inside the ignored workspace.
"""

from twinr.browser_automation.runtime.typed_results import (
    BrowserAuthStateSummary,
    BrowserTypedResult,
    build_auth_state_summary,
    build_typed_result,
    normalize_typed_result,
    results_schema_expects_null,
)
from twinr.browser_automation.runtime.document_reader import (
    DocumentReadResult,
    PdfDocumentReader,
    is_probable_pdf_url,
)
from twinr.browser_automation.runtime.provider_retry import (
    call_openai_with_retry,
    is_retryable_openai_error,
)

__all__ = [
    "BrowserAuthStateSummary",
    "BrowserTypedResult",
    "DocumentReadResult",
    "PdfDocumentReader",
    "build_auth_state_summary",
    "build_typed_result",
    "call_openai_with_retry",
    "is_probable_pdf_url",
    "is_retryable_openai_error",
    "normalize_typed_result",
    "results_schema_expects_null",
]
