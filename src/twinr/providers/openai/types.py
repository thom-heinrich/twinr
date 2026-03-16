from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypeAlias
import errno
import mimetypes
import os
import stat

from twinr.ops.usage import TokenUsage

_ENV_MAX_IMAGE_BYTES = "TWINR_OPENAI_IMAGE_MAX_BYTES"
_DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024


def _configured_max_image_bytes() -> int:
    # AUDIT-FIX(#2): cap image reads so a wrong or oversized file cannot exhaust RAM on the RPi.
    raw_value = os.getenv(_ENV_MAX_IMAGE_BYTES, "").strip()
    if not raw_value:
        return _DEFAULT_MAX_IMAGE_BYTES
    try:
        value = int(raw_value)
    except ValueError:
        return _DEFAULT_MAX_IMAGE_BYTES
    return value if value > 0 else _DEFAULT_MAX_IMAGE_BYTES


def _ensure_safe_filename(filename: str) -> str:
    # AUDIT-FIX(#6): reject unsafe filenames before they reach logs or multipart headers.
    if not isinstance(filename, str):
        raise TypeError("filename must be str")
    if not filename:
        raise ValueError("filename must not be empty")
    if "/" in filename or "\\" in filename or Path(filename).name != filename:
        raise ValueError("filename must be a basename, not a path")
    if any(ord(char) < 32 or ord(char) == 127 for char in filename):
        raise ValueError("filename must not contain control characters")
    return filename


def _ensure_image_content_type(content_type: str) -> str:
    # AUDIT-FIX(#6): enforce a real image MIME type for direct dataclass construction too.
    if not isinstance(content_type, str):
        raise TypeError("content_type must be str")
    media_type = content_type.split(";", 1)[0].strip().lower()
    if not media_type or "/" not in media_type:
        raise ValueError("content_type must be a valid MIME type")
    if not media_type.startswith("image/"):
        raise ValueError(f"unsupported image content_type: {content_type}")
    return media_type


def _sniff_image_content_type(data: bytes, filename: str) -> str:
    # AUDIT-FIX(#4): infer MIME from file bytes; extensions are advisory at best and lies at worst.
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"BM"):
        return "image/bmp"
    if data.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in {b"avif", b"avis"}:
            return "image/avif"
        if brand in {b"heic", b"heix", b"hevc", b"hevx"}:
            return "image/heic"
        if brand in {b"mif1", b"msf1"}:
            return "image/heif"
    if data.startswith(b"\x00\x00\x01\x00"):
        return "image/x-icon"

    stripped = data.lstrip()
    header = stripped[:1024].lower()
    if stripped.startswith(b"<svg") or (stripped.startswith(b"<?xml") and b"<svg" in header):
        return "image/svg+xml"

    guessed = mimetypes.guess_type(filename)[0]
    if guessed:
        raise ValueError(
            f"unsupported or unrecognized image type for file: {filename} "
            f"(filename suggests {guessed})"
        )
    raise ValueError(f"unsupported or unrecognized image type for file: {filename}")


def _resolve_input_path(path: str | Path, *, base_dir: str | Path | None = None) -> Path:
    candidate = Path(path).expanduser()

    try:
        candidate_stat = candidate.lstat()
    except FileNotFoundError:
        raise

    # AUDIT-FIX(#1): refuse symlink leaf paths so attacker-planted links cannot redirect reads.
    if stat.S_ISLNK(candidate_stat.st_mode):
        raise ValueError(f"refusing to follow symlink: {candidate}")

    resolved = candidate.resolve(strict=True)
    if base_dir is not None:
        allowed_root = Path(base_dir).expanduser().resolve(strict=True)
        try:
            resolved.relative_to(allowed_root)
        except ValueError as exc:
            raise ValueError(f"image path escapes base_dir: {candidate}") from exc

    return resolved


def _read_image_bytes(path: Path, *, max_bytes: int) -> bytes:
    if max_bytes <= 0:
        raise ValueError("max_bytes must be greater than zero")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)

    try:
        # AUDIT-FIX(#3): use a descriptor-based, non-blocking open so FIFOs/devices can be rejected safely.
        fd = os.open(path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(f"refusing to follow symlink: {path}") from exc
        raise

    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(f"image path is not a regular file: {path}")

        # AUDIT-FIX(#2): reject oversized images before reading to prevent process-level memory spikes.
        if file_stat.st_size > max_bytes:
            raise ValueError(f"image file exceeds max_bytes ({file_stat.st_size} > {max_bytes}): {path}")

        with os.fdopen(fd, "rb") as file_obj:
            fd = -1
            data = file_obj.read(max_bytes + 1)

        if len(data) > max_bytes:
            raise ValueError(f"image file exceeds max_bytes (> {max_bytes}): {path}")
        if not data:
            # AUDIT-FIX(#3): empty files must fail locally with a deterministic error.
            raise ValueError(f"image file is empty: {path}")

        return data
    finally:
        if fd != -1:
            os.close(fd)


@dataclass(frozen=True, slots=True)
class OpenAITextResponse:
    text: str
    response_id: str | None = None
    request_id: str | None = None
    model: str | None = None
    token_usage: TokenUsage | None = None
    used_web_search: bool = False


@dataclass(frozen=True, slots=True)
class OpenAISearchResult:
    answer: str
    sources: tuple[str, ...] = ()
    response_id: str | None = None
    request_id: str | None = None
    model: str | None = None
    token_usage: TokenUsage | None = None
    used_web_search: bool = False

    def __post_init__(self) -> None:
        # AUDIT-FIX(#5): normalize to tuple so the frozen dataclass is actually immutable in practice.
        if isinstance(self.sources, str):
            raise TypeError("sources must be an iterable of strings, not a single string")
        normalized_sources = tuple(self.sources)
        if any(not isinstance(source, str) for source in normalized_sources):
            raise TypeError("sources must contain only strings")
        object.__setattr__(self, "sources", normalized_sources)


@dataclass(frozen=True, slots=True)
class OpenAIImageInput:
    data: bytes
    content_type: str
    filename: str = "image"
    detail: str | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        # AUDIT-FIX(#6): validate direct construction too; not every caller will come through from_path().
        data = self.data
        if isinstance(data, (bytearray, memoryview)):
            data = bytes(data)
        elif not isinstance(data, bytes):
            raise TypeError("data must be bytes, bytearray, or memoryview")
        if not data:
            raise ValueError("data must not be empty")

        object.__setattr__(self, "data", data)
        object.__setattr__(self, "content_type", _ensure_image_content_type(self.content_type))
        object.__setattr__(self, "filename", _ensure_safe_filename(self.filename))

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        detail: str | None = None,
        label: str | None = None,
        base_dir: str | Path | None = None,
        max_bytes: int | None = None,
    ) -> "OpenAIImageInput":
        # AUDIT-FIX(#1): callers can now pin reads to a trusted ingest directory via base_dir.
        file_path = _resolve_input_path(path, base_dir=base_dir)
        data = _read_image_bytes(
            file_path,
            # AUDIT-FIX(#2): default to an env-backed size limit with optional per-call override.
            max_bytes=_configured_max_image_bytes() if max_bytes is None else max_bytes,
        )
        # AUDIT-FIX(#4): verify the payload is an actual image before constructing the request object.
        content_type = _sniff_image_content_type(data, file_path.name)
        return cls(
            data=data,
            content_type=content_type,
            filename=file_path.name,
            detail=detail,
            label=label,
        )


# AUDIT-FIX(#7): make the alias precise; the original union collapsed to Sequence[object] and lost static value.
ConversationMessage: TypeAlias = tuple[str, str]
ConversationLike: TypeAlias = Sequence[ConversationMessage]