"""Read downloadable PDF documents for browser-automation evidence.

This module keeps document-specific runtime logic out of the ignored
``browser_automation/`` workspace. The current baseline focuses on PDFs because
they are common on public-service websites such as schedules, forms, and
notices. It downloads the document with optional browser-auth cookies, extracts
text via ``pdftotext``, and returns verifier-friendly evidence packets.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import hashlib
import ipaddress
import json
import subprocess

from twinr.browser_automation import BrowserAutomationArtifact

_MAX_TEXT_EXCERPT_CHARS = 20_000
_MAX_MATCH_SNIPPET_CHARS = 500
_MAX_MATCH_PACKETS = 8


def _normalize_text(value: object) -> str:
    """Return one whitespace-normalized text fragment."""

    return " ".join(str(value or "").strip().split())


def _truncate(text: str, *, limit: int) -> str:
    """Return one compact string suitable for evidence packets."""

    normalized = str(text or "").strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 1)].rstrip() + "…"


def _normalize_host(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if not host:
        raise ValueError("URL host is required")
    return host


def _host_is_private_or_local(host: str) -> bool:
    if host in {"localhost", "localhost.localdomain"}:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast)


def _host_allowed(host: str, allowed_domains: tuple[str, ...]) -> bool:
    if not allowed_domains:
        return not _host_is_private_or_local(host)
    for allowed_domain in allowed_domains:
        normalized = allowed_domain.strip().lower().rstrip(".")
        if host == normalized or host.endswith(f".{normalized}"):
            return True
    return False


def _validate_url(url: str, *, allowed_domains: tuple[str, ...]) -> str:
    parsed = urlparse(str(url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http/https URLs are supported")
    if not parsed.netloc:
        raise ValueError("A full URL with host is required")
    host = _normalize_host(url)
    if not _host_allowed(host, allowed_domains):
        raise ValueError(f"Host {host!r} is outside the allowed browser domain policy")
    return parsed.geturl()


def is_probable_pdf_url(url: str, *, content_type: str | None = None) -> bool:
    """Return whether one URL or response metadata points to a PDF document."""

    if str(content_type or "").strip().lower().startswith("application/pdf"):
        return True
    path = urlparse(str(url or "").strip()).path.lower()
    return path.endswith(".pdf")


def _head_tail_excerpt(text: str, *, limit: int) -> str:
    """Keep both early and late document context within one bounded excerpt."""

    normalized = str(text or "").strip()
    if len(normalized) <= limit:
        return normalized
    half = max(256, (limit - 16) // 2)
    return f"{normalized[:half].rstrip()}\n\n…\n\n{normalized[-half:].lstrip()}"


def _string_headers(extra_http_headers: Mapping[str, Any] | None) -> dict[str, str]:
    """Return safe string headers for an outbound document fetch."""

    headers: dict[str, str] = {}
    for key, value in dict(extra_http_headers or {}).items():
        cleaned_key = str(key or "").strip()
        cleaned_value = str(value or "").strip()
        if cleaned_key and cleaned_value:
            headers[cleaned_key] = cleaned_value
    return headers


def _cookie_header_from_storage_state(*, storage_state_path: str | None, url: str) -> str | None:
    """Build one Cookie header for the requested URL from Playwright storage state."""

    candidate = Path(str(storage_state_path or "")).expanduser().resolve()
    if not candidate.is_file():
        return None
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return None
    cookies = list(payload.get("cookies") or [])
    parsed = urlparse(url)
    host = (parsed.hostname or "").strip().lower()
    path = parsed.path or "/"
    secure = parsed.scheme == "https"
    pairs: list[str] = []
    for raw_cookie in cookies:
        if not isinstance(raw_cookie, Mapping):
            continue
        domain = str(raw_cookie.get("domain") or "").strip().lstrip(".").lower()
        cookie_path = str(raw_cookie.get("path") or "/").strip() or "/"
        if domain and host != domain and not host.endswith(f".{domain}"):
            continue
        if not path.startswith(cookie_path):
            continue
        if bool(raw_cookie.get("secure")) and not secure:
            continue
        name = str(raw_cookie.get("name") or "").strip()
        value = str(raw_cookie.get("value") or "").strip()
        if name:
            pairs.append(f"{name}={value}")
    if not pairs:
        return None
    return "; ".join(pairs)


def _parse_pdf_page_count(pdf_path: Path) -> int | None:
    """Return the number of pages when ``pdfinfo`` is available."""

    try:
        result = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    for raw_line in result.stdout.splitlines():
        label, _, value = raw_line.partition(":")
        if label.strip().lower() != "pages":
            continue
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _extract_pdf_text(pdf_path: Path) -> str:
    """Extract readable text from one PDF using ``pdftotext``."""

    try:
        result = subprocess.run(
            ["pdftotext", str(pdf_path), "-"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30.0,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("pdftotext is required for PDF browser automation") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"pdftotext timed out for {pdf_path.name}") from exc
    text = str(result.stdout or "").replace("\x0c", "\n")
    if text.strip():
        return text
    stderr = _truncate(str(result.stderr or "").strip(), limit=400)
    raise RuntimeError(f"pdftotext produced no text for {pdf_path.name}: {stderr}")


def _match_packets(*, text: str, query_terms: Sequence[str], url: str, source_label: str | None) -> tuple[dict[str, Any], ...]:
    """Return small document snippets for query-relevant lines."""

    lines = [str(line or "").strip() for line in str(text or "").splitlines() if str(line or "").strip()]
    unique_terms: list[str] = []
    seen_terms: set[str] = set()
    for raw_term in list(query_terms or ()):
        term = _normalize_text(raw_term)
        if len(term) < 2:
            continue
        lowered = term.lower()
        if lowered in seen_terms:
            continue
        seen_terms.add(lowered)
        unique_terms.append(term)
    packets: list[dict[str, Any]] = []
    seen_snippets: set[tuple[str, str]] = set()
    for term in unique_terms:
        lowered = term.lower()
        for index, line in enumerate(lines):
            if lowered not in line.lower():
                continue
            start = max(0, index - 1)
            end = min(len(lines), index + 2)
            snippet = "\n".join(lines[start:end]).strip()
            key = (term, snippet)
            if key in seen_snippets:
                continue
            seen_snippets.add(key)
            packets.append(
                {
                    "kind": "document_match",
                    "url": url,
                    "source_label": str(source_label or "").strip(),
                    "query": term,
                    "text": _truncate(snippet, limit=_MAX_MATCH_SNIPPET_CHARS),
                }
            )
            if len(packets) >= _MAX_MATCH_PACKETS:
                return tuple(packets)
    return tuple(packets)


@dataclass(frozen=True, slots=True)
class DocumentReadResult:
    """Describe one downloaded PDF plus verifier-friendly evidence packets."""

    url: str
    content_type: str | None
    local_path: str
    text: str
    text_excerpt: str
    page_count: int | None
    artifacts: tuple[BrowserAutomationArtifact, ...]

    def to_evidence_packets(
        self,
        *,
        query_terms: Sequence[str],
        source_label: str | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Render document evidence packets for the dense reader/verifier."""

        packets: list[dict[str, Any]] = [
            {
                "kind": "document_snapshot",
                "url": self.url,
                "source_label": str(source_label or "").strip(),
                "content_type": str(self.content_type or "").strip(),
                "local_path": self.local_path,
                "page_count": self.page_count,
                "text_excerpt": self.text_excerpt,
            }
        ]
        packets.extend(
            _match_packets(
                text=self.text,
                query_terms=query_terms,
                url=self.url,
                source_label=source_label,
            )
        )
        return tuple(packets)


class PdfDocumentReader:
    """Download PDF documents and turn them into evidence packets."""

    def __init__(self, *, artifacts_root: Path, allowed_domains: tuple[str, ...]) -> None:
        self._artifacts_root = Path(artifacts_root).resolve()
        self._artifacts_root.mkdir(parents=True, exist_ok=True)
        self._allowed_domains = tuple(allowed_domains)

    def read_pdf(
        self,
        *,
        url: str,
        task_token: str,
        query_terms: Sequence[str],
        storage_state_path: str | None = None,
        extra_http_headers: Mapping[str, Any] | None = None,
        timeout_s: float = 30.0,
    ) -> DocumentReadResult:
        """Download one PDF URL, extract its text, and persist bounded artifacts."""

        resolved_url = _validate_url(url, allowed_domains=self._allowed_domains)
        digest = hashlib.sha256(resolved_url.encode("utf-8")).hexdigest()[:12]
        pdf_path = self._artifacts_root / f"{task_token}-{digest}.pdf"
        request_headers = {
            "User-Agent": "TwinrBrowserAutomation/1.0",
            **_string_headers(extra_http_headers),
        }
        cookie_header = _cookie_header_from_storage_state(
            storage_state_path=storage_state_path,
            url=resolved_url,
        )
        if cookie_header:
            request_headers["Cookie"] = cookie_header
        request = Request(resolved_url, headers=request_headers)
        with urlopen(request, timeout=max(5.0, float(timeout_s))) as response:  # noqa: S310
            content_type = str(response.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower() or None
            payload = response.read()
        if not is_probable_pdf_url(resolved_url, content_type=content_type):
            raise RuntimeError(
                f"Document URL did not resolve to a PDF: {resolved_url} ({content_type or 'unknown content-type'})"
            )
        pdf_path.write_bytes(payload)
        text = _extract_pdf_text(pdf_path)
        text_excerpt = _head_tail_excerpt(text, limit=_MAX_TEXT_EXCERPT_CHARS)
        text_path = self._artifacts_root / f"{task_token}-{digest}.txt"
        text_path.write_text(text, encoding="utf-8")
        page_count = _parse_pdf_page_count(pdf_path)
        artifacts = (
            BrowserAutomationArtifact(
                kind="document",
                path=str(pdf_path),
                content_type="application/pdf",
                description="Downloaded PDF document for browser-automation evidence.",
            ),
            BrowserAutomationArtifact(
                kind="document_text",
                path=str(text_path),
                content_type="text/plain",
                description="Extracted text for one downloaded PDF document.",
            ),
        )
        return DocumentReadResult(
            url=resolved_url,
            content_type=content_type,
            local_path=str(pdf_path),
            text=text,
            text_excerpt=text_excerpt,
            page_count=page_count,
            artifacts=artifacts,
        )


__all__ = [
    "DocumentReadResult",
    "PdfDocumentReader",
    "is_probable_pdf_url",
]
