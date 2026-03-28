#!/usr/bin/env python3
# CHANGELOG: 2026-03-27
# BUG-1: Existing files are no longer trusted on existence alone; they are verified against a local lockfile and optional pinned digests.
# BUG-2: Downloads are now streamed into temporary files and atomically replaced, preventing persistent partial `.task` files after interruption.
# BUG-3: The default output directory is now resolved from the repository root instead of the caller's current working directory.
# SEC-1: Added integrity validation using Content-Length, GCS `x-goog-hash` MD5 (when present), local SHA-256 lock metadata, and optional manifest pins.
# SEC-2: Refused to overwrite symlinked managed files, enforced bounded downloads, and used an explicit secure TLS context.
# IMP-1: Added a persisted lockfile with provenance metadata (`sha256`, `md5`, `etag`, `generation`, `resolved_url`, timestamps).
# IMP-2: Added verify-only mode, retry/backoff, model subset selection, pose variant selection, upstream freshness checks, and manifest-based overrides.
# /// script
# requires-python = ">=3.11"
# ///
"""Download and verify the MediaPipe task models needed by Twinr's Pi camera path.

This helper is intentionally separate from the runtime. It fetches the
MediaPipe task bundles into a bounded local directory so the runtime does not
perform hidden downloads during normal camera inference.

What changed
------------
This version turns the helper into a reproducible asset fetcher:

* verifies existing files instead of trusting mere presence,
* downloads with retries, streaming, and atomic replace,
* persists a lockfile with provenance metadata,
* supports verify-only runs for offline acceptance checks,
* can optionally check whether upstream official `latest` assets changed,
* allows manifest-driven overrides for custom `.task` bundles.

Usage
-----
Command-line invocation examples::

    python3 hardware/piaicam/fetch_mediapipe_models.py
    python3 hardware/piaicam/fetch_mediapipe_models.py --output-dir /twinr/state/mediapipe/models
    python3 hardware/piaicam/fetch_mediapipe_models.py --force
    python3 hardware/piaicam/fetch_mediapipe_models.py --verify-only
    python3 hardware/piaicam/fetch_mediapipe_models.py --pose-variant lite
    python3 hardware/piaicam/fetch_mediapipe_models.py --check-upstream
    python3 hardware/piaicam/fetch_mediapipe_models.py --manifest path/to/models.json

Manifest schema
---------------
The optional manifest can either be a JSON list of model objects or a JSON
object with a top-level `models` list. Each model object uses this schema::

    {
      "name": "gesture_recognizer",
      "family": "gesture",
      "url": "https://.../gesture_recognizer.task",
      "filename": "gesture_recognizer.task",
      "expected_sha256": "...optional 64-char lowercase hex...",
      "expected_size_bytes": 12345678
    }

Outputs
-------
* Writes verified `.task` model files under the selected output directory.
* Writes a persistent lockfile with local integrity and remote provenance.
* Prints a JSON summary describing paths, sizes, digests, and fetch status.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import ssl
import sys
import tempfile
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request


def _repo_root() -> Path:
    resolved = Path(__file__).resolve()
    try:
        return resolved.parents[2]
    except IndexError:
        return resolved.parent


# BREAKING: The default output directory is now repository-root relative, not
# caller-CWD relative. This avoids silent writes to the wrong directory when the
# helper is invoked from systemd, CI, Ansible, or any non-repo working directory.
DEFAULT_OUTPUT_DIR = _repo_root() / "state/mediapipe/models"
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_RETRY_COUNT = 3
DEFAULT_RETRY_BACKOFF_S = 1.0
DEFAULT_MAX_FILE_BYTES = 64 * 1024 * 1024
DEFAULT_LOCKFILE_NAME = "models.lock.json"
DEFAULT_CHUNK_SIZE = 1024 * 1024
DEFAULT_USER_AGENT = "TwinrMediaPipeModelFetcher/2026-03-27"
ALLOWED_URL_SCHEMES = {"https"}
RETRYABLE_HTTP_STATUS = {408, 425, 429, 500, 502, 503, 504}

POSE_MODEL_URLS = {
    "lite": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    ),
    "full": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    ),
    "heavy": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    ),
}
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
GESTURE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/latest/gesture_recognizer.task"
)


@dataclass(frozen=True, slots=True)
class DownloadSpec:
    """Describe one task asset to fetch."""

    name: str
    family: str
    url: str
    filename: str
    expected_sha256: str | None = None
    expected_size_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class RemoteMetadata:
    """Remote object metadata extracted from response headers."""

    requested_url: str
    resolved_url: str
    content_length: int | None
    content_type: str | None
    etag: str | None
    generation: str | None
    last_modified: str | None
    md5_base64: str | None
    crc32c_base64: str | None


@dataclass(frozen=True, slots=True)
class LocalMetadata:
    """Locally computed integrity metadata for one file."""

    size_bytes: int
    sha256: str
    md5_base64: str
    mtime_utc: str


class ModelFetchFailure(RuntimeError):
    """One model could not be verified or downloaded."""


class BatchFetchFailure(RuntimeError):
    """The batch run finished with at least one failure."""

    def __init__(self, summary: dict[str, Any]) -> None:
        self.summary = summary
        super().__init__("one or more MediaPipe model fetches failed")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the model fetch helper."""

    parser = argparse.ArgumentParser(
        description=(
            "Download and verify the MediaPipe task models used by Twinr's "
            "Pi-side camera pipeline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the task bundles will be written.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--retry-count",
        type=int,
        default=DEFAULT_RETRY_COUNT,
        help="Number of retries after the initial network attempt.",
    )
    parser.add_argument(
        "--retry-backoff-s",
        type=float,
        default=DEFAULT_RETRY_BACKOFF_S,
        help="Base exponential backoff in seconds between retries.",
    )
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=DEFAULT_MAX_FILE_BYTES,
        help="Hard upper bound per downloaded model file.",
    )
    parser.add_argument(
        "--lockfile",
        type=Path,
        default=None,
        help=(
            "Override the lockfile path. Defaults to <output-dir>/"
            f"{DEFAULT_LOCKFILE_NAME}."
        ),
    )
    parser.add_argument(
        "--models",
        type=str,
        default="pose,hand,gesture",
        help=(
            "Comma-separated selection of model families or names. Built-in "
            "families are pose, hand, and gesture."
        ),
    )
    parser.add_argument(
        "--pose-variant",
        choices=tuple(POSE_MODEL_URLS.keys()),
        default="full",
        help="Built-in pose landmarker variant to fetch when pose is selected.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Optional JSON manifest that replaces the built-in model set and "
            "can pin custom URLs, filenames, and SHA-256 digests."
        ),
    )
    parser.add_argument(
        "--check-upstream",
        action="store_true",
        help=(
            "For already trusted local files, issue HEAD requests and refresh "
            "them if upstream metadata changed."
        ),
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate trusted local files and lock metadata without downloading.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even when trusted local copies already exist.",
    )
    return parser


def _now_utc() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


def _new_ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context()


def _new_md5() -> Any:
    try:
        return hashlib.md5(usedforsecurity=False)
    except TypeError:
        return hashlib.md5()


def _parse_csv(value: str) -> set[str]:
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def _normalize_output_dir(path: Path) -> Path:
    path = path.expanduser()
    resolved = path.resolve(strict=False)
    if resolved.exists() and not resolved.is_dir():
        raise ModelFetchFailure(f"output path is not a directory: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _resolve_lockfile_path(output_dir: Path, lockfile: Path | None) -> Path:
    candidate = (lockfile or (output_dir / DEFAULT_LOCKFILE_NAME)).expanduser()
    if not candidate.is_absolute():
        candidate = (output_dir / candidate).resolve(strict=False)
    else:
        candidate = candidate.resolve(strict=False)
    if candidate.exists() and candidate.is_symlink():
        raise ModelFetchFailure(f"refusing to use symlinked lockfile: {candidate}")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def _validate_https_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ALLOWED_URL_SCHEMES:
        raise ModelFetchFailure(f"only HTTPS model URLs are allowed: {url}")
    if not parsed.netloc:
        raise ModelFetchFailure(f"model URL is missing a host: {url}")


def _validate_sha256_hex(digest: str) -> str:
    normalized = digest.strip().lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ModelFetchFailure(f"invalid SHA-256 digest: {digest}")
    return normalized


def _spec_from_dict(raw: dict[str, Any], *, index: int) -> DownloadSpec:
    try:
        name = str(raw["name"]).strip()
        url = str(raw["url"]).strip()
    except KeyError as exc:
        raise ModelFetchFailure(
            f"manifest model #{index} is missing required field {exc.args[0]!r}"
        ) from exc

    family = str(raw.get("family", name)).strip() or name
    filename = str(raw.get("filename") or Path(urllib.parse.urlparse(url).path).name).strip()
    expected_sha256_raw = raw.get("expected_sha256")
    expected_size_raw = raw.get("expected_size_bytes")

    if not name:
        raise ModelFetchFailure(f"manifest model #{index} has an empty name")
    if not filename.endswith(".task"):
        raise ModelFetchFailure(f"manifest model {name!r} must end with .task: {filename!r}")

    _validate_https_url(url)

    expected_sha256 = None
    if expected_sha256_raw is not None:
        expected_sha256 = _validate_sha256_hex(str(expected_sha256_raw))

    expected_size_bytes = None
    if expected_size_raw is not None:
        try:
            expected_size_bytes = int(expected_size_raw)
        except (TypeError, ValueError) as exc:
            raise ModelFetchFailure(
                f"manifest model {name!r} has invalid expected_size_bytes: {expected_size_raw!r}"
            ) from exc
        if expected_size_bytes <= 0:
            raise ModelFetchFailure(
                f"manifest model {name!r} has non-positive expected_size_bytes: {expected_size_bytes}"
            )

    return DownloadSpec(
        name=name,
        family=family,
        url=url,
        filename=filename,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
    )


def _load_manifest_specs(path: Path) -> tuple[DownloadSpec, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelFetchFailure(f"manifest file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ModelFetchFailure(f"manifest file is not valid JSON: {path}: {exc}") from exc

    raw_models: Any
    if isinstance(payload, dict):
        raw_models = payload.get("models")
    else:
        raw_models = payload

    if not isinstance(raw_models, list):
        raise ModelFetchFailure(
            "manifest must be a JSON list or an object with a top-level 'models' list"
        )
    if not raw_models:
        raise ModelFetchFailure("manifest does not define any models")

    specs_list: list[DownloadSpec] = []
    for index, item in enumerate(raw_models, start=1):
        if not isinstance(item, dict):
            raise ModelFetchFailure(
                f"manifest model #{index} must be a JSON object, got {type(item).__name__}"
            )
        specs_list.append(_spec_from_dict(item, index=index))
    specs = tuple(specs_list)

    seen_names: set[str] = set()
    seen_filenames: set[str] = set()
    for spec in specs:
        lowered_name = spec.name.lower()
        lowered_filename = spec.filename.lower()
        if lowered_name in seen_names:
            raise ModelFetchFailure(f"manifest contains duplicate model name: {spec.name!r}")
        if lowered_filename in seen_filenames:
            raise ModelFetchFailure(f"manifest contains duplicate filename: {spec.filename!r}")
        seen_names.add(lowered_name)
        seen_filenames.add(lowered_filename)
    return specs


def _builtin_specs(*, pose_variant: str) -> tuple[DownloadSpec, ...]:
    pose_variant = pose_variant.lower()
    if pose_variant not in POSE_MODEL_URLS:
        choices = ", ".join(sorted(POSE_MODEL_URLS))
        raise ModelFetchFailure(f"unsupported pose variant {pose_variant!r}; expected one of: {choices}")
    pose_url = POSE_MODEL_URLS[pose_variant]
    pose_name = f"pose_landmarker_{pose_variant}"
    pose_filename = f"pose_landmarker_{pose_variant}.task"
    return (
        DownloadSpec(
            name=pose_name,
            family="pose",
            url=pose_url,
            filename=pose_filename,
        ),
        DownloadSpec(
            name="hand_landmarker",
            family="hand",
            url=HAND_LANDMARKER_MODEL_URL,
            filename="hand_landmarker.task",
        ),
        DownloadSpec(
            name="gesture_recognizer",
            family="gesture",
            url=GESTURE_MODEL_URL,
            filename="gesture_recognizer.task",
        ),
    )


def _select_specs(specs: tuple[DownloadSpec, ...], requested: set[str]) -> tuple[DownloadSpec, ...]:
    if not requested:
        return specs
    selected = tuple(
        spec for spec in specs if spec.family.lower() in requested or spec.name.lower() in requested
    )
    if not selected:
        choices = ", ".join(sorted({spec.family for spec in specs} | {spec.name for spec in specs}))
        raise ModelFetchFailure(
            f"--models did not match any available model family or name. Available: {choices}"
        )
    return selected


def _load_lockfile(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    if path.is_symlink():
        raise ModelFetchFailure(f"refusing to use symlinked lockfile: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelFetchFailure(f"lockfile is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ModelFetchFailure(f"lockfile must contain a JSON object: {path}")
    models = payload.get("models", {})
    if not isinstance(models, dict):
        raise ModelFetchFailure(f"lockfile 'models' field must be an object: {path}")
    return {str(name): entry for name, entry in models.items() if isinstance(entry, dict)}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() and path.is_symlink():
        raise ModelFetchFailure(f"refusing to overwrite symlinked JSON file: {path}")

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write(os.linesep)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def _parse_x_goog_hash(value: str | None) -> dict[str, str]:
    hashes: dict[str, str] = {}
    if not value:
        return hashes
    for part in value.split(","):
        item = part.strip()
        if not item or "=" not in item:
            continue
        key, digest = item.split("=", 1)
        hashes[key.strip().lower()] = digest.strip()
    return hashes


def _content_type_from_headers(headers: Any) -> str | None:
    content_type = headers.get("Content-Type")
    if not content_type:
        return None
    return content_type.split(";", 1)[0].strip().lower()


def _response_metadata(*, requested_url: str, response: Any) -> RemoteMetadata:
    headers = response.headers
    raw_length = headers.get("Content-Length")
    content_length = None
    if raw_length is not None:
        try:
            content_length = int(raw_length)
        except ValueError:
            content_length = None
    hashes = _parse_x_goog_hash(headers.get("x-goog-hash"))
    resolved_url = getattr(response, "url", requested_url)
    return RemoteMetadata(
        requested_url=requested_url,
        resolved_url=str(resolved_url),
        content_length=content_length,
        content_type=_content_type_from_headers(headers),
        etag=headers.get("ETag"),
        generation=headers.get("x-goog-generation"),
        last_modified=headers.get("Last-Modified"),
        md5_base64=hashes.get("md5"),
        crc32c_base64=hashes.get("crc32c"),
    )


def _build_request(url: str, *, method: str = "GET") -> urllib.request.Request:
    return urllib.request.Request(
        url,
        method=method,
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "application/octet-stream, */*;q=0.1",
        },
    )


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in RETRYABLE_HTTP_STATUS
    return isinstance(exc, (urllib.error.URLError, TimeoutError, OSError))


def _open_with_retries(
    request: urllib.request.Request,
    *,
    timeout_s: float,
    retry_count: int,
    retry_backoff_s: float,
    ssl_context: ssl.SSLContext,
) -> Any:
    last_exc: BaseException | None = None
    for attempt in range(retry_count + 1):
        try:
            response = urllib.request.urlopen(request, timeout=timeout_s, context=ssl_context)
            resolved = urllib.parse.urlparse(str(getattr(response, "url", request.full_url)))
            if resolved.scheme not in ALLOWED_URL_SCHEMES:
                response.close()
                raise ModelFetchFailure(
                    f"redirected to a non-HTTPS URL, which is refused: {resolved.geturl()}"
                )
            return response
        except BaseException as exc:  # noqa: BLE001 - intentional retry gate
            last_exc = exc
            if attempt >= retry_count or not _is_retryable(exc):
                raise
            sleep_s = max(0.0, retry_backoff_s) * (2**attempt) + random.uniform(0.0, 0.25)
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


def _head_metadata(
    url: str,
    *,
    timeout_s: float,
    retry_count: int,
    retry_backoff_s: float,
    ssl_context: ssl.SSLContext,
) -> RemoteMetadata:
    with _open_with_retries(
        _build_request(url, method="HEAD"),
        timeout_s=timeout_s,
        retry_count=retry_count,
        retry_backoff_s=retry_backoff_s,
        ssl_context=ssl_context,
    ) as response:
        return _response_metadata(requested_url=url, response=response)


def _hash_file(path: Path) -> LocalMetadata:
    sha256 = hashlib.sha256()
    md5 = _new_md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(DEFAULT_CHUNK_SIZE)
            if not chunk:
                break
            sha256.update(chunk)
            md5.update(chunk)
    stat = path.stat()
    return LocalMetadata(
        size_bytes=stat.st_size,
        sha256=sha256.hexdigest(),
        md5_base64=base64.b64encode(md5.digest()).decode("ascii"),
        mtime_utc=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat(),
    )


def _assert_managed_target_is_safe(path: Path) -> None:
    if path.exists():
        if path.is_symlink():
            raise ModelFetchFailure(f"refusing to manage symlinked target file: {path}")
        if not path.is_file():
            raise ModelFetchFailure(f"managed target exists but is not a regular file: {path}")


def _expected_sha256_for(spec: DownloadSpec, lock_entry: dict[str, Any]) -> str | None:
    if spec.expected_sha256:
        return spec.expected_sha256
    candidate = lock_entry.get("sha256")
    if isinstance(candidate, str) and candidate:
        return candidate.strip().lower()
    return None


def _expected_size_for(spec: DownloadSpec, lock_entry: dict[str, Any]) -> int | None:
    if spec.expected_size_bytes is not None:
        return spec.expected_size_bytes
    candidate = lock_entry.get("size_bytes")
    if candidate is None:
        return None
    try:
        size = int(candidate)
    except (TypeError, ValueError):
        return None
    return size if size > 0 else None


def _verify_existing_file(
    *,
    path: Path,
    spec: DownloadSpec,
    lock_entry: dict[str, Any],
) -> LocalMetadata:
    _assert_managed_target_is_safe(path)
    local = _hash_file(path)

    expected_size = _expected_size_for(spec, lock_entry)
    if expected_size is not None and local.size_bytes != expected_size:
        raise ModelFetchFailure(
            f"size mismatch for {path.name}: expected {expected_size}, got {local.size_bytes}"
        )

    expected_sha256 = _expected_sha256_for(spec, lock_entry)
    if expected_sha256 is not None and local.sha256 != expected_sha256:
        raise ModelFetchFailure(
            f"sha256 mismatch for {path.name}: expected {expected_sha256}, got {local.sha256}"
        )

    expected_md5 = lock_entry.get("md5_base64")
    if isinstance(expected_md5, str) and expected_md5 and local.md5_base64 != expected_md5:
        raise ModelFetchFailure(
            f"md5 mismatch for {path.name}: expected {expected_md5}, got {local.md5_base64}"
        )

    if expected_size is None and expected_sha256 is None:
        raise ModelFetchFailure(
            f"{path.name} exists but is not trusted yet because no lock or manifest integrity metadata is available"
        )

    return local


def _upstream_changed(lock_entry: dict[str, Any], remote: RemoteMetadata) -> bool:
    checks: list[tuple[str, str | int | None, str | int | None]] = [
        ("generation", lock_entry.get("generation"), remote.generation),
        ("etag", lock_entry.get("etag"), remote.etag),
        ("md5_base64", lock_entry.get("md5_base64"), remote.md5_base64),
        ("size_bytes", lock_entry.get("size_bytes"), remote.content_length),
    ]
    comparable = False
    for _label, old_value, new_value in checks:
        if old_value is None or new_value is None:
            continue
        comparable = True
        if str(old_value) != str(new_value):
            return True
    return False if comparable else False


def _download_to_tempfile(
    *,
    spec: DownloadSpec,
    output_dir: Path,
    timeout_s: float,
    retry_count: int,
    retry_backoff_s: float,
    max_file_bytes: int,
    ssl_context: ssl.SSLContext,
) -> tuple[Path, LocalMetadata, RemoteMetadata]:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_dir,
            prefix=f".{spec.filename}.",
            suffix=".download",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            sha256 = hashlib.sha256()
            md5 = _new_md5()
            total_bytes = 0

            with _open_with_retries(
                _build_request(spec.url, method="GET"),
                timeout_s=timeout_s,
                retry_count=retry_count,
                retry_backoff_s=retry_backoff_s,
                ssl_context=ssl_context,
            ) as response:
                remote = _response_metadata(requested_url=spec.url, response=response)
                if remote.content_length is not None and remote.content_length > max_file_bytes:
                    raise ModelFetchFailure(
                        f"remote file {spec.name!r} exceeds max-file-bytes: "
                        f"{remote.content_length} > {max_file_bytes}"
                    )

                while True:
                    chunk = response.read(DEFAULT_CHUNK_SIZE)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > max_file_bytes:
                        raise ModelFetchFailure(
                            f"download for {spec.name!r} exceeded max-file-bytes: {total_bytes} > {max_file_bytes}"
                        )
                    handle.write(chunk)
                    sha256.update(chunk)
                    md5.update(chunk)

                handle.flush()
                os.fsync(handle.fileno())

            if remote.content_length is not None and total_bytes != remote.content_length:
                raise ModelFetchFailure(
                    f"content-length mismatch for {spec.name!r}: expected {remote.content_length}, got {total_bytes}"
                )

            sha256_hex = sha256.hexdigest()
            md5_base64 = base64.b64encode(md5.digest()).decode("ascii")

            if remote.md5_base64 and md5_base64 != remote.md5_base64:
                raise ModelFetchFailure(
                    f"remote MD5 mismatch for {spec.name!r}: expected {remote.md5_base64}, got {md5_base64}"
                )
            if spec.expected_sha256 and sha256_hex != spec.expected_sha256:
                raise ModelFetchFailure(
                    f"pinned SHA-256 mismatch for {spec.name!r}: expected {spec.expected_sha256}, got {sha256_hex}"
                )
            if spec.expected_size_bytes is not None and total_bytes != spec.expected_size_bytes:
                raise ModelFetchFailure(
                    f"pinned size mismatch for {spec.name!r}: expected {spec.expected_size_bytes}, got {total_bytes}"
                )

            local = LocalMetadata(
                size_bytes=total_bytes,
                sha256=sha256_hex,
                md5_base64=md5_base64,
                mtime_utc=_now_utc(),
            )
            return temp_path, local, remote
    except Exception:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise


def _install_downloaded_file(temp_path: Path, target_path: Path) -> None:
    if target_path.exists() and target_path.is_symlink():
        raise ModelFetchFailure(f"refusing to overwrite symlinked target file: {target_path}")
    os.replace(temp_path, target_path)


def _lock_entry_from_metadata(
    *,
    spec: DownloadSpec,
    target_path: Path,
    local: LocalMetadata,
    remote: RemoteMetadata | None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": spec.name,
        "family": spec.family,
        "filename": spec.filename,
        "path": str(target_path),
        "requested_url": spec.url,
        "sha256": local.sha256,
        "md5_base64": local.md5_base64,
        "size_bytes": local.size_bytes,
        "verified_at_utc": _now_utc(),
        "local_mtime_utc": local.mtime_utc,
    }
    if remote is not None:
        entry.update(
            {
                "resolved_url": remote.resolved_url,
                "etag": remote.etag,
                "generation": remote.generation,
                "last_modified": remote.last_modified,
                "content_type": remote.content_type,
                "remote_content_length": remote.content_length,
                "remote_md5_base64": remote.md5_base64,
                "remote_crc32c_base64": remote.crc32c_base64,
            }
        )
    if spec.expected_sha256 is not None:
        entry["pinned_sha256"] = spec.expected_sha256
    if spec.expected_size_bytes is not None:
        entry["pinned_size_bytes"] = spec.expected_size_bytes
    return entry


def _result_dict(
    *,
    spec: DownloadSpec,
    target_path: Path,
    status: str,
    local: LocalMetadata | None = None,
    remote: RemoteMetadata | None = None,
    verified: bool = False,
    error: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": spec.name,
        "family": spec.family,
        "filename": spec.filename,
        "path": str(target_path),
        "requested_url": spec.url,
        "status": status,
        "verified": verified,
    }
    if local is not None:
        result.update(
            {
                "size_bytes": local.size_bytes,
                "sha256": local.sha256,
                "md5_base64": local.md5_base64,
            }
        )
    if remote is not None:
        result.update(
            {
                "resolved_url": remote.resolved_url,
                "etag": remote.etag,
                "generation": remote.generation,
                "last_modified": remote.last_modified,
                "content_type": remote.content_type,
            }
        )
    if error is not None:
        result["error"] = error
    return result


def _process_one_model(
    *,
    spec: DownloadSpec,
    output_dir: Path,
    verify_only: bool,
    force: bool,
    check_upstream: bool,
    timeout_s: float,
    retry_count: int,
    retry_backoff_s: float,
    max_file_bytes: int,
    ssl_context: ssl.SSLContext,
    lock_entry: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    target_path = output_dir / spec.filename
    _assert_managed_target_is_safe(target_path)

    trusted_local: LocalMetadata | None = None
    remote_meta: RemoteMetadata | None = None

    if target_path.exists() and not force:
        try:
            trusted_local = _verify_existing_file(path=target_path, spec=spec, lock_entry=lock_entry)
        except ModelFetchFailure:
            if verify_only:
                raise
            trusted_local = None

    if trusted_local is not None:
        if check_upstream:
            remote_meta = _head_metadata(
                spec.url,
                timeout_s=timeout_s,
                retry_count=retry_count,
                retry_backoff_s=retry_backoff_s,
                ssl_context=ssl_context,
            )
            if _upstream_changed(lock_entry, remote_meta):
                trusted_local = None

    if trusted_local is not None:
        updated_lock = _lock_entry_from_metadata(
            spec=spec,
            target_path=target_path,
            local=trusted_local,
            remote=remote_meta,
        )
        status = "verified" if verify_only else "present"
        return (
            _result_dict(
                spec=spec,
                target_path=target_path,
                status=status,
                local=trusted_local,
                remote=remote_meta,
                verified=True,
            ),
            updated_lock,
        )

    if verify_only:
        raise ModelFetchFailure(
            f"{spec.name!r} is not trusted locally and verify-only mode forbids downloading"
        )

    temp_path, downloaded_local, remote_meta = _download_to_tempfile(
        spec=spec,
        output_dir=output_dir,
        timeout_s=timeout_s,
        retry_count=retry_count,
        retry_backoff_s=retry_backoff_s,
        max_file_bytes=max_file_bytes,
        ssl_context=ssl_context,
    )
    _install_downloaded_file(temp_path, target_path)
    updated_lock = _lock_entry_from_metadata(
        spec=spec,
        target_path=target_path,
        local=downloaded_local,
        remote=remote_meta,
    )
    return (
        _result_dict(
            spec=spec,
            target_path=target_path,
            status="downloaded",
            local=downloaded_local,
            remote=remote_meta,
            verified=True,
        ),
        updated_lock,
    )


def download_models(
    *,
    output_dir: Path,
    timeout_s: float,
    force: bool,
    verify_only: bool = False,
    retry_count: int = DEFAULT_RETRY_COUNT,
    retry_backoff_s: float = DEFAULT_RETRY_BACKOFF_S,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    pose_variant: str = "full",
    models: str = "pose,hand,gesture",
    manifest: Path | None = None,
    lockfile: Path | None = None,
    check_upstream: bool = False,
) -> dict[str, Any]:
    """Download or verify MediaPipe task bundles and return a JSON-friendly summary.

    This keeps the original public helper name for drop-in imports, but extends
    it with verification, retries, lockfile persistence, and optional manifest
    overrides.
    """

    if timeout_s <= 0:
        raise ModelFetchFailure(f"timeout must be > 0 seconds, got {timeout_s}")
    if retry_count < 0:
        raise ModelFetchFailure(f"retry_count must be >= 0, got {retry_count}")
    if retry_backoff_s < 0:
        raise ModelFetchFailure(f"retry_backoff_s must be >= 0, got {retry_backoff_s}")
    if max_file_bytes <= 0:
        raise ModelFetchFailure(f"max_file_bytes must be > 0, got {max_file_bytes}")
    if force and verify_only:
        raise ModelFetchFailure("force and verify_only are mutually exclusive")

    output_dir = _normalize_output_dir(output_dir)
    lockfile_path = _resolve_lockfile_path(output_dir, lockfile)
    ssl_context = _new_ssl_context()

    available_specs = (
        _load_manifest_specs(manifest)
        if manifest is not None
        else _builtin_specs(pose_variant=pose_variant)
    )
    selected_specs = _select_specs(available_specs, _parse_csv(models))

    existing_locks = _load_lockfile(lockfile_path)
    updated_locks = dict(existing_locks)
    results: list[dict[str, Any]] = []
    failures: list[str] = []

    for spec in selected_specs:
        target_path = output_dir / spec.filename
        try:
            result, lock_entry = _process_one_model(
                spec=spec,
                output_dir=output_dir,
                verify_only=verify_only,
                force=force,
                check_upstream=check_upstream,
                timeout_s=timeout_s,
                retry_count=retry_count,
                retry_backoff_s=retry_backoff_s,
                max_file_bytes=max_file_bytes,
                ssl_context=ssl_context,
                lock_entry=existing_locks.get(spec.name, {}),
            )
            updated_locks[spec.name] = lock_entry
            results.append(result)
        except Exception as exc:  # noqa: BLE001 - structured batch reporting is intentional
            failures.append(spec.name)
            results.append(
                _result_dict(
                    spec=spec,
                    target_path=target_path,
                    status="failed",
                    verified=False,
                    error=str(exc),
                )
            )

    lockfile_payload = {
        "schema_version": 1,
        "updated_at_utc": _now_utc(),
        "output_dir": str(output_dir),
        "models": updated_locks,
    }
    if not verify_only:
        _write_json_atomic(lockfile_path, lockfile_payload)

    summary = {
        "ok": not failures,
        "output_dir": str(output_dir),
        "lockfile": str(lockfile_path),
        "verify_only": verify_only,
        "force": force,
        "check_upstream": check_upstream,
        "selected_models": [spec.name for spec in selected_specs],
        "models": results,
    }
    if failures:
        summary["failed_models"] = failures
        raise BatchFetchFailure(summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    """Run the MediaPipe model fetch helper."""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        summary = download_models(
            output_dir=args.output_dir,
            timeout_s=float(args.timeout_s),
            force=bool(args.force),
            verify_only=bool(args.verify_only),
            retry_count=int(args.retry_count),
            retry_backoff_s=float(args.retry_backoff_s),
            max_file_bytes=int(args.max_file_bytes),
            pose_variant=str(args.pose_variant),
            models=str(args.models),
            manifest=args.manifest,
            lockfile=args.lockfile,
            check_upstream=bool(args.check_upstream),
        )
    except BatchFetchFailure as exc:
        json.dump(exc.summary, sys.stdout, indent=2)
        sys.stdout.write(os.linesep)
        return 2
    except Exception as exc:  # noqa: BLE001 - final structured CLI error is intentional
        json.dump(
            {
                "ok": False,
                "error": str(exc),
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write(os.linesep)
        return 2

    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write(os.linesep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())