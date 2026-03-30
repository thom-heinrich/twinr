# CHANGELOG: 2026-03-30
# BUG-1: Enforce the current OpenAI vision input allowlist (PNG/JPEG/WEBP/non-animated GIF); stop accepting
#        locally-valid but API-invalid formats like TIFF/BMP/SVG/ICO/HEIC as final payloads.
# BUG-2: Reject animated GIFs and invalid image detail hints locally instead of failing later at the API boundary.
# BUG-3: Interpret relative paths against base_dir and verify image payloads by decoding them when Pillow is available.
# SEC-1: Replace the check-then-open path flow with descriptor-relative path walking so symlink races and parent-link
#        escapes cannot redirect reads to unintended files.
# SEC-2: Normalize EXIF orientation and strip privacy-sensitive metadata on rewritten images by default.
# IMP-1: Add Pillow-backed image normalization/transcoding/resizing aligned with 2026 Responses API image limits.
# IMP-2: Add immutable image metadata (sha256/width/height/original_content_type/normalized) plus data-URL serialization.

"""Define shared OpenAI request and response value objects for Twinr.

This module contains the canonical dataclasses exchanged between OpenAI-backed
code paths plus the filesystem and image-validation helpers used to construct
safe image inputs for the OpenAI Responses API.

Frontier notes:
- Final image payloads are constrained to the current OpenAI vision allowlist:
  PNG, JPEG, WEBP, and non-animated GIF.
- When Pillow is installed, the module verifies images by decoding them,
  optionally transcodes common camera/container formats to supported outputs,
  normalizes EXIF orientation, strips privacy-sensitive metadata on rewritten
  images, and resizes to detail-aware bounds before upload.
- On Raspberry Pi / Linux deployments, path reads are performed via descriptor-
  relative path walking to block symlink races and path traversal escapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, TypeAlias
from urllib.parse import urlparse, urlunparse
import base64
import errno
import hashlib
import importlib
import io
import mimetypes
import os
import stat
import struct

from twinr.ops.usage import TokenUsage

try:  # Optional but strongly recommended for deep image validation/transcoding.
    from PIL import Image, ImageOps, UnidentifiedImageError
except Exception:  # pragma: no cover - optional dependency fallback
    Image = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]

    class UnidentifiedImageError(Exception):
        """Fallback placeholder used when Pillow is unavailable."""


_ENV_MAX_IMAGE_BYTES = "TWINR_OPENAI_IMAGE_MAX_BYTES"
_ENV_MAX_IMAGE_PIXELS = "TWINR_OPENAI_IMAGE_MAX_PIXELS"
_ENV_MAX_IMAGE_DIMENSION = "TWINR_OPENAI_IMAGE_MAX_DIMENSION"

_DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_HARD_MAX_IMAGE_BYTES = 64 * 1024 * 1024

# 6000px matches the current highest useful "detail=original" bound in the
# OpenAI vision docs for GPT-5.4 and future models.
_DEFAULT_MAX_IMAGE_DIMENSION = 6000
_HARD_MAX_IMAGE_DIMENSION = 6000

# Cap decompressed pixel count to a Pi-friendly ceiling that still covers the
# current "detail=original" 6000x6000 bound.
_DEFAULT_MAX_IMAGE_PIXELS = _DEFAULT_MAX_IMAGE_DIMENSION * _DEFAULT_MAX_IMAGE_DIMENSION
_HARD_MAX_IMAGE_PIXELS = 64_000_000

_SUPPORTED_IMAGE_CONTENT_TYPES = frozenset(
    {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/gif",
    }
)
_SUPPORTED_DETAIL_HINTS = frozenset({"low", "high", "original", "auto"})
_IMAGE_DETAIL_MAX_DIMENSION = {
    None: _DEFAULT_MAX_IMAGE_DIMENSION,
    "auto": _DEFAULT_MAX_IMAGE_DIMENSION,
    "original": _DEFAULT_MAX_IMAGE_DIMENSION,
    "high": 2048,
    "low": 512,
}
_PILLOW_FORMAT_TO_CONTENT_TYPE = {
    "JPEG": "image/jpeg",
    "MPO": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "GIF": "image/gif",
    "BMP": "image/bmp",
    "DIB": "image/bmp",
    "TIFF": "image/tiff",
    "HEIF": "image/heif",
    "HEIC": "image/heic",
    "AVIF": "image/avif",
}
_CONTENT_TYPE_TO_EXTENSION = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
_CONTROL_CHAR_ORDS = frozenset({127, *range(0, 32)})

try:
    _HAS_WEBP_SAVE = bool(Image is not None and getattr(Image, "SAVE", None) and "WEBP" in Image.SAVE)
except Exception:  # pragma: no cover - extremely defensive
    _HAS_WEBP_SAVE = False


def _configured_positive_int(
    env_name: str,
    default: int,
    *,
    hard_max: int,
) -> int:
    """Read a positive integer environment variable with a hard ceiling."""

    raw_value = os.getenv(env_name, "").strip()
    if not raw_value:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        return default
    if value <= 0:
        return default
    return min(value, hard_max)


def _configured_max_image_bytes() -> int:
    """Return the configured upper bound for image payload reads."""

    return _configured_positive_int(
        _ENV_MAX_IMAGE_BYTES,
        _DEFAULT_MAX_IMAGE_BYTES,
        hard_max=_HARD_MAX_IMAGE_BYTES,
    )


def _configured_max_image_dimension() -> int:
    """Return the configured maximum output dimension for normalized images."""

    return _configured_positive_int(
        _ENV_MAX_IMAGE_DIMENSION,
        _DEFAULT_MAX_IMAGE_DIMENSION,
        hard_max=_HARD_MAX_IMAGE_DIMENSION,
    )


def _configured_max_image_pixels() -> int:
    """Return the configured maximum decoded pixel count."""

    configured = _configured_positive_int(
        _ENV_MAX_IMAGE_PIXELS,
        _DEFAULT_MAX_IMAGE_PIXELS,
        hard_max=_HARD_MAX_IMAGE_PIXELS,
    )
    dimension_cap = _configured_max_image_dimension()
    return min(configured, dimension_cap * dimension_cap)


def _normalize_optional_string(
    value: str | None,
    *,
    field_name: str,
    lowercase: bool = False,
    max_length: int | None = None,
) -> str | None:
    """Normalize optional string metadata."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str or None")
    normalized = value.strip()
    if not normalized:
        return None
    if any(ord(char) in _CONTROL_CHAR_ORDS for char in normalized):
        raise ValueError(f"{field_name} must not contain control characters")
    if lowercase:
        normalized = normalized.lower()
    if max_length is not None and len(normalized) > max_length:
        raise ValueError(f"{field_name} exceeds max length ({max_length})")
    return normalized


def _ensure_safe_filename(filename: str) -> str:
    """Validate that a filename is a safe basename for outbound uploads."""

    if not isinstance(filename, str):
        raise TypeError("filename must be str")
    normalized = filename.strip()
    if not normalized:
        raise ValueError("filename must not be empty")
    if len(normalized) > 255:
        raise ValueError("filename must be 255 characters or fewer")
    if normalized in {".", ".."}:
        raise ValueError("filename must not be '.' or '..'")
    if "/" in normalized or "\\" in normalized or Path(normalized).name != normalized:
        raise ValueError("filename must be a basename, not a path")
    if any(ord(char) in _CONTROL_CHAR_ORDS for char in normalized):
        raise ValueError("filename must not contain control characters")
    return normalized


def _normalize_image_detail(detail: str | None) -> ImageDetail | None:
    """Validate and normalize an image detail hint."""

    if detail is None:
        return None
    if not isinstance(detail, str):
        raise TypeError("detail must be str or None")
    normalized = detail.strip().lower()
    if not normalized:
        return None
    if normalized not in _SUPPORTED_DETAIL_HINTS:
        raise ValueError(
            f"detail must be one of {sorted(_SUPPORTED_DETAIL_HINTS)} or None"
        )
    return normalized  # type: ignore[return-value]


def _ensure_image_content_type(content_type: str) -> str:
    """Validate and normalize a supported OpenAI image MIME type."""

    if not isinstance(content_type, str):
        raise TypeError("content_type must be str")
    media_type = content_type.split(";", 1)[0].strip().lower()
    if not media_type or "/" not in media_type:
        raise ValueError("content_type must be a valid MIME type")
    if media_type not in _SUPPORTED_IMAGE_CONTENT_TYPES:
        raise ValueError(
            f"unsupported image content_type: {content_type} "
            f"(supported: {sorted(_SUPPORTED_IMAGE_CONTENT_TYPES)})"
        )
    return media_type


def _replace_extension(filename: str, content_type: str) -> str:
    """Return a safe filename with an extension consistent with content_type."""

    safe_filename = _ensure_safe_filename(filename)
    extension = _CONTENT_TYPE_TO_EXTENSION[content_type]
    stem = Path(safe_filename).stem or "image"
    candidate = f"{stem}{extension}"
    return _ensure_safe_filename(candidate)


def _normalize_http_url(value: str, *, field_name: str) -> str:
    """Validate and normalize an HTTP(S) URL intended for UI rendering."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if any(ord(char) in _CONTROL_CHAR_ORDS for char in normalized):
        raise ValueError(f"{field_name} must not contain control characters")

    parsed = urlparse(normalized)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError(f"{field_name} must use http or https")
    if not parsed.netloc:
        raise ValueError(f"{field_name} must include a hostname")

    rebuilt = urlunparse(
        parsed._replace(
            scheme=parsed.scheme.lower(),
            netloc=parsed.netloc.lower(),
        )
    )
    return rebuilt


def _normalize_domain(value: str | None, *, field_name: str) -> str | None:
    """Normalize a hostname/domain-like string."""

    if value is None:
        return None
    normalized = _normalize_optional_string(value, field_name=field_name, lowercase=True)
    if normalized is None:
        return None
    if "/" in normalized or "\\" in normalized or ":" in normalized:
        raise ValueError(f"{field_name} must be a bare host/domain, not a URL")
    return normalized


def _normalize_url_tuple(urls: Sequence[str]) -> tuple[str, ...]:
    """Normalize, validate, and deduplicate URL tuples while preserving order."""

    normalized_urls: list[str] = []
    seen: set[str] = set()
    for raw_url in urls:
        normalized = _normalize_http_url(raw_url, field_name="sources[]")
        if normalized not in seen:
            normalized_urls.append(normalized)
            seen.add(normalized)
    return tuple(normalized_urls)


def _sniff_supported_image_content_type(data: bytes, filename: str) -> str:
    """Infer a supported OpenAI image MIME type from file bytes."""

    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"

    guessed = mimetypes.guess_type(filename)[0]
    if guessed:
        if guessed.startswith("image/"):
            raise ValueError(
                f"unsupported or unrecognized final image type for file: {filename} "
                f"(filename suggests {guessed}; supported: {sorted(_SUPPORTED_IMAGE_CONTENT_TYPES)})"
            )
        raise ValueError(f"filename does not describe an image: {filename} ({guessed})")
    raise ValueError(f"unsupported or unrecognized final image type for file: {filename}")


def _skip_gif_subblocks(data: bytes, index: int) -> int:
    """Skip GIF data sub-blocks and return the next unread index."""

    while True:
        if index >= len(data):
            raise ValueError("truncated GIF data")
        block_size = data[index]
        index += 1
        if block_size == 0:
            return index
        end = index + block_size
        if end > len(data):
            raise ValueError("truncated GIF data")
        index = end


def _gif_frame_count(data: bytes, *, stop_after: int = 2) -> int:
    """Return the GIF frame count up to stop_after, rejecting corrupt payloads."""

    if not data.startswith((b"GIF87a", b"GIF89a")):
        return 0
    if len(data) < 13:
        raise ValueError("truncated GIF header")

    index = 6
    _width, _height, packed, _bg, _aspect = struct.unpack_from("<HHBBB", data, index)
    index += 7

    if packed & 0x80:
        global_color_table_entries = 1 << ((packed & 0x07) + 1)
        color_table_bytes = 3 * global_color_table_entries
        if index + color_table_bytes > len(data):
            raise ValueError("truncated GIF global color table")
        index += color_table_bytes

    frames = 0
    while index < len(data):
        introducer = data[index]
        index += 1

        if introducer == 0x3B:  # trailer
            return frames

        if introducer == 0x21:  # extension
            if index >= len(data):
                raise ValueError("truncated GIF extension")
            label = data[index]
            index += 1
            if label == 0xF9:  # Graphic Control Extension has a fixed 4-byte block.
                if index >= len(data):
                    raise ValueError("truncated GIF graphic control extension")
                block_size = data[index]
                index += 1
                end = index + block_size
                if end >= len(data):
                    raise ValueError("truncated GIF graphic control extension")
                index = end
                terminator = data[index]
                index += 1
                if terminator != 0x00:
                    raise ValueError("corrupt GIF graphic control extension")
            else:
                index = _skip_gif_subblocks(data, index)
            continue

        if introducer == 0x2C:  # image descriptor
            frames += 1
            if frames >= stop_after:
                return frames

            if index + 9 > len(data):
                raise ValueError("truncated GIF image descriptor")
            _left, _top, _width, _height, packed = struct.unpack_from("<HHHHB", data, index)
            index += 9

            if packed & 0x80:
                local_color_table_entries = 1 << ((packed & 0x07) + 1)
                color_table_bytes = 3 * local_color_table_entries
                if index + color_table_bytes > len(data):
                    raise ValueError("truncated GIF local color table")
                index += color_table_bytes

            if index >= len(data):
                raise ValueError("truncated GIF image data")
            index += 1  # LZW minimum code size
            index = _skip_gif_subblocks(data, index)
            continue

        raise ValueError("corrupt GIF stream")

    raise ValueError("truncated GIF stream")


def _is_animated_gif_bytes(data: bytes) -> bool:
    """Return True when GIF bytes contain more than one frame."""

    return _gif_frame_count(data, stop_after=2) > 1


def _image_has_alpha(image: "Image.Image") -> bool:
    """Return True when the Pillow image has an alpha channel or transparency."""

    if image.mode in {"RGBA", "LA"}:
        return True
    if image.mode == "P" and "transparency" in image.info:
        return True
    bands = getattr(image, "getbands", lambda: ())()
    return "A" in bands


def _image_has_sensitive_metadata(image: "Image.Image") -> bool:
    """Return True when the image appears to carry privacy-sensitive metadata."""

    try:
        exif = image.getexif()
        if exif:
            return True
    except Exception:
        pass

    for key in ("exif", "xmp", "XML:com.adobe.xmp", "iptc", "photoshop", "comment"):
        if image.info.get(key):
            return True
    return False


def _pillow_source_content_type(image: "Image.Image", filename: str) -> str:
    """Map a Pillow image format to a MIME type."""

    format_name = (getattr(image, "format", None) or "").upper()
    if format_name in _PILLOW_FORMAT_TO_CONTENT_TYPE:
        return _PILLOW_FORMAT_TO_CONTENT_TYPE[format_name]

    guessed = mimetypes.guess_type(filename)[0]
    if guessed and guessed.startswith("image/"):
        return guessed.lower()
    raise ValueError(f"unsupported or unrecognized image type for file: {filename}")


def _effective_image_limits(
    detail: ImageDetail | None,
    *,
    max_pixels: int | None,
    max_dimension: int | None,
) -> tuple[int, int]:
    """Return decoded-safety and output-dimension limits for the requested detail."""

    configured_dimension = _configured_max_image_dimension() if max_dimension is None else max_dimension
    configured_dimension = min(int(configured_dimension), _HARD_MAX_IMAGE_DIMENSION)
    if configured_dimension <= 0:
        raise ValueError("max_dimension must be greater than zero")

    detail_dimension = _IMAGE_DETAIL_MAX_DIMENSION[detail]
    effective_dimension = min(configured_dimension, detail_dimension)

    configured_pixels = _configured_max_image_pixels() if max_pixels is None else int(max_pixels)
    if configured_pixels <= 0:
        raise ValueError("max_pixels must be greater than zero")

    # max_pixels is a decoded-image safety ceiling; detail-specific downsizing happens afterwards.
    decoded_safety_pixels = min(configured_pixels, configured_dimension * configured_dimension)
    return decoded_safety_pixels, effective_dimension


def _thumbnail_copy(image: "Image.Image", max_dimension: int) -> "Image.Image":
    """Return a resized copy bounded by max_dimension while preserving aspect ratio."""

    if max_dimension <= 0:
        raise ValueError("max_dimension must be greater than zero")
    if max(image.size) <= max_dimension:
        return image.copy()

    working = image.copy()
    resampling = getattr(getattr(Image, "Resampling", None), "LANCZOS", getattr(Image, "LANCZOS", 1))
    working.thumbnail((max_dimension, max_dimension), resampling)
    return working


def _encode_candidate_image(
    image: "Image.Image",
    *,
    content_type: str,
    preserve_icc_profile: bytes | None,
    max_bytes: int,
) -> bytes:
    """Encode a Pillow image to one supported content type."""

    buffer = io.BytesIO()
    save_kwargs: dict[str, object] = {}

    if preserve_icc_profile:
        save_kwargs["icc_profile"] = preserve_icc_profile

    if content_type == "image/jpeg":
        working = image if image.mode in {"RGB", "L"} else image.convert("RGB")
        best: bytes | None = None
        for quality in (85, 75, 65, 55):
            buffer.seek(0)
            buffer.truncate(0)
            working.save(
                buffer,
                format="JPEG",
                quality=quality,
                optimize=True,
                progressive=True,
                **save_kwargs,
            )
            encoded = buffer.getvalue()
            if not encoded:
                continue
            best = encoded
            if len(encoded) <= max_bytes:
                return encoded
        if best is None:
            raise ValueError("failed to encode JPEG image")
        return best

    if content_type == "image/png":
        if image.mode not in {"1", "L", "LA", "P", "RGB", "RGBA"}:
            working = image.convert("RGBA" if _image_has_alpha(image) else "RGB")
        else:
            working = image
        working.save(
            buffer,
            format="PNG",
            optimize=True,
            compress_level=9,
            **save_kwargs,
        )
        return buffer.getvalue()

    if content_type == "image/webp":
        working = image
        if image.mode not in {"RGB", "RGBA", "L", "LA"}:
            working = image.convert("RGBA" if _image_has_alpha(image) else "RGB")
        best = None
        for quality in (82, 75, 65, 55):
            buffer.seek(0)
            buffer.truncate(0)
            working.save(
                buffer,
                format="WEBP",
                quality=quality,
                method=6,
                **save_kwargs,
            )
            encoded = buffer.getvalue()
            if not encoded:
                continue
            best = encoded
            if len(encoded) <= max_bytes:
                return encoded
        if best is None:
            raise ValueError("failed to encode WEBP image")
        return best

    if content_type == "image/gif":
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("animated GIF is not supported by OpenAI vision inputs")
        working = image.convert("P") if image.mode != "P" else image
        working.save(buffer, format="GIF")
        return buffer.getvalue()

    raise ValueError(f"unsupported target content_type: {content_type}")


def _preferred_output_content_types(
    image: "Image.Image",
    source_content_type: str,
) -> tuple[str, ...]:
    """Choose output encodings in a practical order for Pi deployments."""

    if _image_has_alpha(image):
        if _HAS_WEBP_SAVE:
            return ("image/png", "image/webp")
        return ("image/png",)

    if source_content_type in {"image/png", "image/gif"}:
        return ("image/png", "image/jpeg")

    if source_content_type == "image/webp" and _HAS_WEBP_SAVE:
        return ("image/webp", "image/jpeg", "image/png")

    return ("image/jpeg", "image/png")


@dataclass(frozen=True, slots=True)
class _PreparedImagePayload:
    """Internal container for a validated/normalized image payload."""

    data: bytes
    content_type: str
    filename: str
    width: int | None
    height: int | None
    normalized: bool
    original_content_type: str | None


def _prepare_image_bytes(
    data: bytes,
    *,
    filename: str,
    detail: ImageDetail | None,
    max_bytes: int,
    max_pixels: int | None,
    max_dimension: int | None,
) -> _PreparedImagePayload:
    """Validate and normalize arbitrary image bytes into a supported final payload."""

    if not data:
        raise ValueError("data must not be empty")
    if len(data) > max_bytes:
        raise ValueError(f"image data exceeds max_bytes ({len(data)} > {max_bytes})")

    safe_filename = _ensure_safe_filename(filename)

    if Image is None:
        content_type = _sniff_supported_image_content_type(data, safe_filename)
        if content_type == "image/gif" and _is_animated_gif_bytes(data):
            raise ValueError("animated GIF is not supported by OpenAI vision inputs")
        return _PreparedImagePayload(
            data=data,
            content_type=content_type,
            filename=_replace_extension(safe_filename, content_type),
            width=None,
            height=None,
            normalized=False,
            original_content_type=content_type,
        )

    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(data)) as image:
            source_content_type = _pillow_source_content_type(image, safe_filename)
            if source_content_type == "image/gif" and getattr(image, "n_frames", 1) != 1:
                raise ValueError("animated GIF is not supported by OpenAI vision inputs")

            decoded_safety_pixels, effective_max_dimension = _effective_image_limits(
                detail,
                max_pixels=max_pixels,
                max_dimension=max_dimension,
            )

            width, height = image.size
            if width <= 0 or height <= 0:
                raise ValueError(f"invalid image dimensions: {width}x{height}")
            if width * height > decoded_safety_pixels:
                raise ValueError(
                    f"image exceeds max_pixels ({width * height} > {decoded_safety_pixels}): {safe_filename}"
                )

            exif_orientation = None
            try:
                exif_orientation = image.getexif().get(274)
            except Exception:
                exif_orientation = None

            orientation_needs_fix = exif_orientation not in {None, 1}
            metadata_present = _image_has_sensitive_metadata(image)
            working = ImageOps.exif_transpose(image) if ImageOps is not None else image.copy()
            resized = max(working.size) > effective_max_dimension
            if resized:
                working = _thumbnail_copy(working, effective_max_dimension)

            final_width, final_height = working.size
            preserve_icc_profile = working.info.get("icc_profile") or image.info.get("icc_profile")

            keep_original_bytes = (
                source_content_type in _SUPPORTED_IMAGE_CONTENT_TYPES
                and not orientation_needs_fix
                and not metadata_present
                and not resized
            )
            if keep_original_bytes:
                return _PreparedImagePayload(
                    data=data,
                    content_type=source_content_type,
                    filename=_replace_extension(safe_filename, source_content_type),
                    width=final_width,
                    height=final_height,
                    normalized=False,
                    original_content_type=source_content_type,
                )

            preferred_content_types = _preferred_output_content_types(working, source_content_type)
            longest_side = max(working.size)
            scale_steps = (1.0, 0.92, 0.85, 0.75, 0.66, 0.5, 0.4, 0.33, 0.25)

            for content_type in preferred_content_types:
                for scale in scale_steps:
                    candidate = working if scale == 1.0 else _thumbnail_copy(
                        working,
                        max(1, int(round(longest_side * scale))),
                    )
                    encoded = _encode_candidate_image(
                        candidate,
                        content_type=content_type,
                        preserve_icc_profile=preserve_icc_profile,
                        max_bytes=max_bytes,
                    )
                    if len(encoded) <= max_bytes:
                        return _PreparedImagePayload(
                            data=encoded,
                            content_type=content_type,
                            filename=_replace_extension(safe_filename, content_type),
                            width=candidate.size[0],
                            height=candidate.size[1],
                            normalized=True,
                            original_content_type=source_content_type,
                        )

            raise ValueError(
                f"could not normalize image within max_bytes ({max_bytes}): {safe_filename}"
            )
    except UnidentifiedImageError as exc:
        raise ValueError(f"unsupported or unrecognized image type for file: {safe_filename}") from exc
    except OSError as exc:
        raise ValueError(f"image file is corrupt or unreadable: {safe_filename}") from exc


def _maybe_register_heif_opener() -> str | None:
    """Register an optional HEIF/AVIF Pillow opener when available."""

    if Image is None:
        return None

    for module_name in ("pi_heif", "pillow_heif"):
        try:
            module = importlib.import_module(module_name)
            register = getattr(module, "register_heif_opener", None)
            if callable(register):
                register()
                return module_name
        except Exception:
            continue
    return None


_REGISTERED_HEIF_OPENER = _maybe_register_heif_opener()


def _split_relative_path_parts(path: Path) -> tuple[str, ...]:
    """Split a path into safe relative components."""

    parts: list[str] = []
    for part in path.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            raise ValueError("path must not contain '..'")
        parts.append(part)
    if not parts:
        raise ValueError("path must not be empty")
    return tuple(parts)


def _open_directory_fd(path: Path) -> int:
    """Open a directory descriptor suitable for *at() path walking."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    return os.open(path, flags)


def _open_regular_file_descriptor(
    path: str | Path,
    *,
    base_dir: str | Path | None = None,
) -> tuple[int, Path]:
    """Open a regular file without following symlinks anywhere in the path chain."""

    candidate = Path(path).expanduser()
    # BREAKING: any symlink anywhere in the path chain is now rejected; the old code only rejected
    #            a symlink leaf and remained vulnerable to parent-link races/escapes.
    path_flags = getattr(os, "O_PATH", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)

    if base_dir is not None:
        allowed_root = Path(base_dir).expanduser().resolve(strict=True)
        root_fd = _open_directory_fd(allowed_root)
        # BREAKING: relative paths are now resolved against base_dir instead of the process CWD.
        if candidate.is_absolute():
            try:
                relative_candidate = candidate.relative_to(allowed_root)
            except ValueError as exc:
                os.close(root_fd)
                raise ValueError(f"image path escapes base_dir: {candidate}") from exc
        else:
            relative_candidate = candidate
        parts = _split_relative_path_parts(relative_candidate)
        display_path = allowed_root.joinpath(*parts)
    else:
        if candidate.is_absolute():
            anchor = Path(candidate.anchor or "/")
            root_fd = _open_directory_fd(anchor)
            parts = _split_relative_path_parts(Path(*candidate.parts[1:]))
            display_path = candidate
        else:
            root_fd = _open_directory_fd(Path.cwd())
            parts = _split_relative_path_parts(candidate)
            display_path = Path.cwd().joinpath(*parts)

    current_fd = root_fd
    try:
        for component in parts[:-1]:
            try:
                next_fd = os.open(component, path_flags, dir_fd=current_fd)
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise ValueError(
                        f"refusing to follow symlinked path component: {display_path}"
                    ) from exc
                raise

            component_stat = os.fstat(next_fd)
            if stat.S_ISLNK(component_stat.st_mode):
                os.close(next_fd)
                raise ValueError(f"refusing to follow symlinked path component: {display_path}")
            if not stat.S_ISDIR(component_stat.st_mode):
                os.close(next_fd)
                raise ValueError(f"path component is not a directory: {display_path}")

            if current_fd != root_fd:
                os.close(current_fd)
            current_fd = next_fd

        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            file_fd = os.open(parts[-1], flags, dir_fd=current_fd)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise ValueError(f"refusing to follow symlink: {display_path}") from exc
            raise
    finally:
        if current_fd != root_fd:
            os.close(current_fd)
        os.close(root_fd)

    file_stat = os.fstat(file_fd)
    if stat.S_ISLNK(file_stat.st_mode):
        os.close(file_fd)
        raise ValueError(f"refusing to follow symlink: {display_path}")
    if not stat.S_ISREG(file_stat.st_mode):
        os.close(file_fd)
        raise ValueError(f"image path is not a regular file: {display_path}")

    return file_fd, display_path


def _resolve_input_path(path: str | Path, *, base_dir: str | Path | None = None) -> Path:
    """Resolve an input path while rejecting symlinks and base_dir escapes."""

    fd, display_path = _open_regular_file_descriptor(path, base_dir=base_dir)
    os.close(fd)
    return display_path


def _read_image_bytes(path: Path, *, max_bytes: int) -> bytes:
    """Read a bounded regular-file image payload from disk."""

    if max_bytes <= 0:
        raise ValueError("max_bytes must be greater than zero")

    fd, display_path = _open_regular_file_descriptor(path)
    try:
        file_stat = os.fstat(fd)
        if file_stat.st_size > max_bytes:
            raise ValueError(f"image file exceeds max_bytes ({file_stat.st_size} > {max_bytes}): {display_path}")

        with os.fdopen(fd, "rb") as file_obj:
            fd = -1
            data = file_obj.read(max_bytes + 1)

        if len(data) > max_bytes:
            raise ValueError(f"image file exceeds max_bytes (> {max_bytes}): {display_path}")
        if not data:
            raise ValueError(f"image file is empty: {display_path}")
        return data
    finally:
        if fd != -1:
            os.close(fd)


@dataclass(frozen=True, slots=True)
class OpenAITextResponse:
    """Capture a plain-text OpenAI response plus optional request metadata."""

    text: str
    response_id: str | None = None
    request_id: str | None = None
    model: str | None = None
    token_usage: TokenUsage | None = None
    used_web_search: bool = False

    def __post_init__(self) -> None:
        """Normalize basic response metadata."""

        if not isinstance(self.text, str):
            raise TypeError("text must be str")
        object.__setattr__(self, "response_id", _normalize_optional_string(self.response_id, field_name="response_id"))
        object.__setattr__(self, "request_id", _normalize_optional_string(self.request_id, field_name="request_id"))
        object.__setattr__(self, "model", _normalize_optional_string(self.model, field_name="model"))
        object.__setattr__(self, "used_web_search", bool(self.used_web_search))


@dataclass(frozen=True, slots=True)
class OpenAISearchAttempt:
    """Capture one model attempt made while resolving a live web-search turn."""

    model: str
    api_path: str
    max_output_tokens: int | None = None
    outcome: str = "unknown"
    status: str | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        """Normalize structured search-attempt fields."""

        normalized_model = _normalize_optional_string(self.model, field_name="model")
        if normalized_model is None:
            raise ValueError("model must not be empty")
        object.__setattr__(self, "model", normalized_model)

        normalized_api_path = _normalize_optional_string(
            self.api_path,
            field_name="api_path",
            lowercase=True,
        )
        if normalized_api_path is None:
            raise ValueError("api_path must not be empty")
        object.__setattr__(self, "api_path", normalized_api_path)

        normalized_outcome = _normalize_optional_string(
            self.outcome,
            field_name="outcome",
            lowercase=True,
        ) or "unknown"
        object.__setattr__(self, "outcome", normalized_outcome)

        object.__setattr__(
            self,
            "status",
            _normalize_optional_string(self.status, field_name="status", lowercase=True),
        )
        object.__setattr__(
            self,
            "detail",
            _normalize_optional_string(self.detail, field_name="detail"),
        )

        max_output_tokens = self.max_output_tokens
        if max_output_tokens is None:
            return
        try:
            normalized_budget = int(max_output_tokens)
        except (TypeError, ValueError) as exc:
            raise TypeError("max_output_tokens must be int or None") from exc
        if normalized_budget <= 0:
            raise ValueError("max_output_tokens must be > 0")
        object.__setattr__(self, "max_output_tokens", normalized_budget)


@dataclass(frozen=True, slots=True)
class OpenAISearchResult:
    """Capture a search answer with its source URLs and response metadata."""

    answer: str
    sources: tuple[str, ...] = ()
    response_id: str | None = None
    request_id: str | None = None
    model: str | None = None
    token_usage: TokenUsage | None = None
    used_web_search: bool = False
    requested_model: str | None = None
    fallback_reason: str | None = None
    attempt_log: tuple[OpenAISearchAttempt, ...] = ()
    verification_status: VerificationStatus | None = None
    question_resolved: bool | None = None
    site_follow_up_recommended: bool = False
    site_follow_up_reason: str | None = None
    site_follow_up_url: str | None = None
    site_follow_up_domain: str | None = None

    def __post_init__(self) -> None:
        """Normalize OpenAI search result metadata."""

        if not isinstance(self.answer, str):
            raise TypeError("answer must be str")
        if isinstance(self.sources, str):
            raise TypeError("sources must be an iterable of strings, not a single string")

        object.__setattr__(self, "response_id", _normalize_optional_string(self.response_id, field_name="response_id"))
        object.__setattr__(self, "request_id", _normalize_optional_string(self.request_id, field_name="request_id"))
        object.__setattr__(self, "model", _normalize_optional_string(self.model, field_name="model"))
        object.__setattr__(self, "used_web_search", bool(self.used_web_search))
        object.__setattr__(
            self,
            "requested_model",
            _normalize_optional_string(self.requested_model, field_name="requested_model"),
        )
        object.__setattr__(
            self,
            "fallback_reason",
            _normalize_optional_string(self.fallback_reason, field_name="fallback_reason"),
        )

        normalized_sources = tuple(self.sources)
        if any(not isinstance(source, str) for source in normalized_sources):
            raise TypeError("sources must contain only strings")
        object.__setattr__(self, "sources", _normalize_url_tuple(normalized_sources))

        normalized_attempt_log = tuple(self.attempt_log)
        if any(not isinstance(item, OpenAISearchAttempt) for item in normalized_attempt_log):
            raise TypeError("attempt_log must contain only OpenAISearchAttempt items")
        object.__setattr__(self, "attempt_log", normalized_attempt_log)

        normalized_verification_status = _normalize_optional_string(
            self.verification_status,
            field_name="verification_status",
            lowercase=True,
        )
        if normalized_verification_status not in {None, "verified", "partial", "unverified"}:
            raise ValueError("verification_status must be verified, partial, unverified, or None")
        object.__setattr__(self, "verification_status", normalized_verification_status)

        normalized_question_resolved = self.question_resolved
        if normalized_question_resolved is not None and not isinstance(normalized_question_resolved, bool):
            raise TypeError("question_resolved must be bool or None")
        object.__setattr__(self, "question_resolved", normalized_question_resolved)

        site_follow_up_recommended = bool(self.site_follow_up_recommended)
        object.__setattr__(self, "site_follow_up_recommended", site_follow_up_recommended)

        normalized_follow_up_reason = _normalize_optional_string(
            self.site_follow_up_reason,
            field_name="site_follow_up_reason",
        )
        normalized_follow_up_url = (
            _normalize_http_url(self.site_follow_up_url, field_name="site_follow_up_url")
            if self.site_follow_up_url is not None
            else None
        )
        normalized_follow_up_domain = _normalize_domain(
            self.site_follow_up_domain,
            field_name="site_follow_up_domain",
        )
        if normalized_follow_up_url is not None and normalized_follow_up_domain is None:
            normalized_follow_up_domain = urlparse(normalized_follow_up_url).netloc.lower()

        if not site_follow_up_recommended:
            normalized_follow_up_reason = None
            normalized_follow_up_url = None
            normalized_follow_up_domain = None

        object.__setattr__(self, "site_follow_up_reason", normalized_follow_up_reason)
        object.__setattr__(self, "site_follow_up_url", normalized_follow_up_url)
        object.__setattr__(self, "site_follow_up_domain", normalized_follow_up_domain)


@dataclass(frozen=True, slots=True)
class OpenAIImageInput:
    """Represent one validated image payload for the OpenAI Responses API.

    Attributes:
        data: Raw image bytes to transmit.
        content_type: Canonical OpenAI-supported image MIME type.
        filename: Safe basename for logs and multipart metadata.
        detail: Optional Responses API detail hint.
        label: Optional text label associated with the image in higher-level code.
        width: Optional decoded width in pixels.
        height: Optional decoded height in pixels.
        sha256: SHA-256 hex digest of the final transmitted payload.
        original_content_type: MIME type detected before any normalization/transcoding.
        normalized: Whether the module rewrote/transcoded/resized the original input.
    """

    data: bytes
    content_type: str
    filename: str = "image"
    detail: ImageDetail | None = None
    label: str | None = None
    width: int | None = None
    height: int | None = None
    sha256: str | None = None
    original_content_type: str | None = None
    normalized: bool = False

    def __post_init__(self) -> None:
        """Normalize raw fields and reject malformed direct constructions."""

        data = self.data
        if isinstance(data, (bytearray, memoryview)):
            data = bytes(data)
        elif not isinstance(data, bytes):
            raise TypeError("data must be bytes, bytearray, or memoryview")
        if not data:
            raise ValueError("data must not be empty")

        max_bytes = _configured_max_image_bytes()
        if len(data) > max_bytes:
            raise ValueError(f"data exceeds max_bytes ({len(data)} > {max_bytes})")

        content_type = _ensure_image_content_type(self.content_type)
        filename = _replace_extension(self.filename, content_type)
        detail = _normalize_image_detail(self.detail)
        label = _normalize_optional_string(self.label, field_name="label", max_length=512)

        sniffed_content_type = _sniff_supported_image_content_type(data, filename)
        if sniffed_content_type != content_type:
            raise ValueError(
                f"content_type does not match image bytes ({content_type} != {sniffed_content_type})"
            )
        if content_type == "image/gif" and _is_animated_gif_bytes(data):
            raise ValueError("animated GIF is not supported by OpenAI vision inputs")

        width = self.width
        height = self.height
        if width is not None:
            width = int(width)
            if width <= 0:
                raise ValueError("width must be > 0")
        if height is not None:
            height = int(height)
            if height <= 0:
                raise ValueError("height must be > 0")

        sha256 = self.sha256
        if sha256 is None:
            sha256 = hashlib.sha256(data).hexdigest()
        else:
            sha256 = _normalize_optional_string(sha256, field_name="sha256", lowercase=True)
            if sha256 is None or len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
                raise ValueError("sha256 must be a 64-character lowercase hex digest")

        original_content_type = self.original_content_type
        if original_content_type is not None:
            normalized_original = original_content_type.split(";", 1)[0].strip().lower()
            if not normalized_original or "/" not in normalized_original:
                raise ValueError("original_content_type must be a valid MIME type or None")
            original_content_type = normalized_original

        object.__setattr__(self, "data", data)
        object.__setattr__(self, "content_type", content_type)
        object.__setattr__(self, "filename", filename)
        object.__setattr__(self, "detail", detail)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "sha256", sha256)
        object.__setattr__(self, "original_content_type", original_content_type or content_type)
        object.__setattr__(self, "normalized", bool(self.normalized))

    @property
    def size_bytes(self) -> int:
        """Return the number of bytes that will be transmitted."""

        return len(self.data)

    def to_data_url(self) -> str:
        """Return a data URL ready for Responses API ``input_image.image_url``."""

        encoded = base64.b64encode(self.data).decode("ascii")
        return f"data:{self.content_type};base64,{encoded}"

    def as_responses_input(self) -> dict[str, str]:
        """Return a Responses API ``input_image`` content item."""

        item: dict[str, str] = {
            "type": "input_image",
            "image_url": self.to_data_url(),
        }
        if self.detail is not None:
            item["detail"] = self.detail
        return item

    @classmethod
    def from_bytes(
        cls,
        data: bytes | bytearray | memoryview,
        *,
        filename: str = "image",
        detail: ImageDetail | None = None,
        label: str | None = None,
        max_bytes: int | None = None,
        max_pixels: int | None = None,
        max_dimension: int | None = None,
    ) -> "OpenAIImageInput":
        """Validate and normalize an image payload already loaded in memory."""

        raw_bytes = bytes(data) if isinstance(data, (bytearray, memoryview)) else data
        if not isinstance(raw_bytes, bytes):
            raise TypeError("data must be bytes, bytearray, or memoryview")

        detail = _normalize_image_detail(detail)
        effective_max_bytes = _configured_max_image_bytes() if max_bytes is None else int(max_bytes)
        if effective_max_bytes <= 0:
            raise ValueError("max_bytes must be greater than zero")

        prepared = _prepare_image_bytes(
            raw_bytes,
            filename=_ensure_safe_filename(filename),
            detail=detail,
            max_bytes=effective_max_bytes,
            max_pixels=max_pixels,
            max_dimension=max_dimension,
        )
        return cls(
            data=prepared.data,
            content_type=prepared.content_type,
            filename=prepared.filename,
            detail=detail,
            label=label,
            width=prepared.width,
            height=prepared.height,
            original_content_type=prepared.original_content_type,
            normalized=prepared.normalized,
        )

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        detail: ImageDetail | None = None,
        label: str | None = None,
        base_dir: str | Path | None = None,
        max_bytes: int | None = None,
        max_pixels: int | None = None,
        max_dimension: int | None = None,
    ) -> "OpenAIImageInput":
        """Load, validate, and normalize an image file from disk.

        Args:
            path: Filesystem path to the candidate image.
            detail: Optional OpenAI vision detail hint.
            label: Optional text label associated with the image.
            base_dir: Optional trusted root the image path must stay inside.
            max_bytes: Optional per-call byte limit. Defaults to the validated
                environment-backed module limit.
            max_pixels: Optional per-call decoded pixel ceiling.
            max_dimension: Optional per-call output max dimension.

        Returns:
            A fully validated ``OpenAIImageInput`` instance.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the file is unsafe, unsupported, empty, animated,
                corrupt, too large, or cannot be normalized to a supported
                OpenAI image format.
            OSError: If the file cannot be opened or read safely.
        """

        detail = _normalize_image_detail(detail)
        effective_max_bytes = _configured_max_image_bytes() if max_bytes is None else int(max_bytes)
        if effective_max_bytes <= 0:
            raise ValueError("max_bytes must be greater than zero")

        file_path = _resolve_input_path(path, base_dir=base_dir)
        data = _read_image_bytes(file_path, max_bytes=effective_max_bytes)
        return cls.from_bytes(
            data,
            filename=file_path.name,
            detail=detail,
            label=label,
            max_bytes=effective_max_bytes,
            max_pixels=max_pixels,
            max_dimension=max_dimension,
        )


ImageDetail: TypeAlias = Literal["low", "high", "original", "auto"]
VerificationStatus: TypeAlias = Literal["verified", "partial", "unverified"]

ConversationMessage: TypeAlias = tuple[str, str] | tuple[str, str, str | None]
ConversationLike: TypeAlias = Sequence[ConversationMessage]
