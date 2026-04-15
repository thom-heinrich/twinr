# CHANGELOG: 2026-03-27
# BUG-1: Parse image-size config values defensively instead of crashing on invalid config.
# BUG-2: Validate and normalize both camera and reference images before sending them downstream.
# BUG-3: Apply EXIF orientation so rotated phone photos do not silently produce wrong vision answers.
# SEC-1: Stop trusting only filename suffix/MIME guesses; verify image content and rewrite it to strip metadata.
# SEC-2: Bound decoded image dimensions/pixels to reduce practical DoS risk on Raspberry Pi deployments.
# IMP-1: Normalize images to a configurable realtime budget (resize/re-encode) to reduce latency and provider rejects.
# IMP-2: Cache the processed reference image across loop iterations to avoid repeated disk I/O and image decoding.

"""Reference-image and vision prompt helpers for realtime workflow loops."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import mimetypes
import os
from pathlib import Path
import stat
import threading
import time
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from twinr.providers.openai import OpenAIImageInput

# Pillow's default decompression-bomb threshold is intentionally global and quite high.
# We enforce tighter per-call bounds below based on runtime config, so we avoid mutating
# Image.MAX_IMAGE_PIXELS globally in a multi-threaded process.

_DEFAULT_MAX_DIMENSION = 2048
_DEFAULT_MAX_PIXELS = 24_000_000
_DEFAULT_JPEG_QUALITY = 90
_MIN_JPEG_QUALITY = 68
_MIN_IMAGE_DIMENSION = 256

_OUTPUT_EXTENSION_BY_CONTENT_TYPE = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
}

_REFERENCE_IMAGE_LABEL = (
    "Image 2: stored reference image of the main user. Use it only for person or identity comparison."
)
_CAPTURE_IMAGE_LABEL = "Image 1: live camera frame from the device."

_REFERENCE_IMAGE_CACHE_LOCK = threading.Lock()
_REFERENCE_IMAGE_CACHE: dict[tuple[Any, ...], "_PreparedImage"] = {}


@dataclass(frozen=True)
class _PreparedImage:
    data: bytes
    content_type: str
    filename: str
    width: int
    height: int


def normalize_reference_image_path(raw_path: str) -> Path:
    """Resolve a configured reference image path without following the final entry yet."""

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.parent.resolve(strict=True) / path.name


def validate_reference_image_base_dir(loop: Any, path: Path) -> bool:
    """Reject reference images outside the configured safe base directory."""

    base_dir_raw = (getattr(loop.config, "vision_reference_image_base_dir", "") or "").strip()
    if not base_dir_raw:
        return True
    try:
        base_dir = Path(base_dir_raw).expanduser().resolve(strict=True)
    except OSError:
        loop._try_emit("vision_reference_rejected=invalid_base_dir")
        return False
    try:
        path.relative_to(base_dir)
    except ValueError:
        loop._try_emit(f"vision_reference_rejected=outside_base_dir:{path.name}")
        return False
    return True


def _multimodal_evidence_enabled(loop: Any) -> bool:
    """Return whether this runtime lane should persist durable multimodal evidence."""

    return bool(getattr(loop, "_persist_multimodal_evidence", True))


def safe_read_reference_image_bytes(path: Path, *, max_bytes: int) -> bytes:
    """Read a reference image without following symlinks and with size checks."""

    flags = os.O_RDONLY
    nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
    if nofollow_flag:
        flags |= nofollow_flag
    cloexec_flag = getattr(os, "O_CLOEXEC", 0)
    if cloexec_flag:
        flags |= cloexec_flag

    fd = os.open(path, flags)
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise OSError("Reference image must be a regular file")
        if file_stat.st_size <= 0:
            raise OSError("Reference image is empty")
        if file_stat.st_size > max_bytes:
            raise OSError(f"Reference image exceeds {max_bytes} bytes")
        with os.fdopen(fd, "rb", closefd=True) as handle:
            data = handle.read(max_bytes + 1)
        fd = -1
    finally:
        if fd >= 0:
            os.close(fd)
    if len(data) > max_bytes:
        raise OSError(f"Reference image exceeds {max_bytes} bytes")
    return data


def build_image_input(data: bytes, *, path: Path, label: str) -> OpenAIImageInput:
    """Build one bounded image input payload for the OpenAI provider."""

    content_type = _guess_content_type_from_path(path)
    return OpenAIImageInput(
        data=data,
        content_type=content_type,
        filename=path.name,
        label=label,
    )


def _guess_content_type_from_path(path: Path) -> str:
    try:
        guessed = mimetypes.guess_file_type(path)[0]
    except AttributeError:
        guessed = mimetypes.guess_type(path.name)[0]
    return guessed or "application/octet-stream"


def _config_int(
    loop: Any,
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    raw_value = getattr(loop.config, name, default)
    if raw_value in (None, ""):
        return default
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        loop._try_emit(f"{name}_invalid={raw_value}")
        return default
    if value < minimum:
        loop._try_emit(f"{name}_clamped={value}->{minimum}")
        value = minimum
    if maximum is not None and value > maximum:
        loop._try_emit(f"{name}_clamped={value}->{maximum}")
        value = maximum
    return value


def _config_bool(loop: Any, name: str, default: bool) -> bool:
    raw_value = getattr(loop.config, name, default)
    if isinstance(raw_value, bool):
        return raw_value
    if raw_value is None:
        return default
    text = str(raw_value).strip().casefold()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    loop._try_emit(f"{name}_invalid={raw_value}")
    return default


def _image_processing_limits(loop: Any, *, default_max_bytes: int) -> tuple[int, int, int, bool]:
    max_bytes = _config_int(
        loop,
        "vision_image_max_bytes",
        _config_int(
            loop,
            "vision_reference_image_max_bytes",
            default_max_bytes,
            minimum=1024,
        ),
        minimum=1024,
    )
    max_dimension = _config_int(
        loop,
        "vision_image_max_dimension",
        _DEFAULT_MAX_DIMENSION,
        minimum=512,
        maximum=6000,
    )
    max_pixels = _config_int(
        loop,
        "vision_image_max_pixels",
        _DEFAULT_MAX_PIXELS,
        minimum=max_dimension * max_dimension,
        maximum=100_000_000,
    )
    prefer_lossless = _config_bool(loop, "vision_image_prefer_lossless", False)
    return max_bytes, max_dimension, max_pixels, prefer_lossless


def _jpeg_quality(loop: Any) -> int:
    return _config_int(
        loop,
        "vision_image_jpeg_quality",
        _DEFAULT_JPEG_QUALITY,
        minimum=_MIN_JPEG_QUALITY,
        maximum=95,
    )


def _reference_image_read_max_bytes(loop: Any, *, output_max_bytes: int) -> int:
    return _config_int(
        loop,
        "vision_reference_image_read_max_bytes",
        max(output_max_bytes * 4, 8 * 1024 * 1024),
        minimum=output_max_bytes,
        maximum=64 * 1024 * 1024,
    )


def _prepared_filename(original_name: str, content_type: str) -> str:
    suffix = _OUTPUT_EXTENSION_BY_CONTENT_TYPE.get(content_type, ".bin")
    stem = Path(original_name or "image").stem or "image"
    return f"{stem}{suffix}"


def _image_has_alpha(image: Image.Image) -> bool:
    if image.mode in {"RGBA", "LA"}:
        return True
    if image.mode == "P":
        return "transparency" in image.info
    return False


def _normalize_image_mode(image: Image.Image, *, keep_alpha: bool) -> Image.Image:
    if keep_alpha:
        if image.mode == "RGBA":
            return image
        return image.convert("RGBA")
    if image.mode == "RGB":
        return image
    return image.convert("RGB")


def _fit_size(width: int, height: int, max_dimension: int) -> tuple[int, int]:
    longest = max(width, height)
    if longest <= max_dimension:
        return width, height
    scale = max_dimension / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return new_width, new_height


def _downscale_size(width: int, height: int, *, factor: float = 0.85) -> tuple[int, int]:
    new_width = max(_MIN_IMAGE_DIMENSION, int(width * factor))
    new_height = max(_MIN_IMAGE_DIMENSION, int(height * factor))
    if new_width == width and width > _MIN_IMAGE_DIMENSION:
        new_width = width - 1
    if new_height == height and height > _MIN_IMAGE_DIMENSION:
        new_height = height - 1
    return max(1, new_width), max(1, new_height)


def _encode_image_bytes(
    image: Image.Image,
    *,
    content_type: str,
    jpeg_quality: int,
) -> bytes:
    output = BytesIO()
    if content_type == "image/png":
        image.save(output, format="PNG")
    elif content_type == "image/jpeg":
        image.save(
            output,
            format="JPEG",
            quality=jpeg_quality,
            optimize=False,
            progressive=False,
        )
    else:
        raise OSError(f"Unsupported normalized image content type: {content_type}")
    return output.getvalue()


def _load_validated_image(raw_bytes: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(raw_bytes)) as probe:
            probe.verify()
        image = Image.open(BytesIO(raw_bytes))
        image.load()
        return image
    except (UnidentifiedImageError, OSError) as exc:
        raise OSError("Image payload is invalid or unsupported") from exc


def _prepare_image_payload(
    raw_bytes: bytes,
    *,
    original_name: str,
    max_bytes: int,
    max_pixels: int,
    max_dimension: int,
    jpeg_quality: int,
    prefer_lossless: bool,
) -> _PreparedImage:
    if not raw_bytes:
        raise OSError("Image payload is empty")

    with _load_validated_image(raw_bytes) as decoded:
        width, height = decoded.size
        if width <= 0 or height <= 0:
            raise OSError("Image has invalid dimensions")
        if width * height > max_pixels:
            # BREAKING: oversized images are now rejected locally instead of being forwarded to the provider.
            raise OSError(f"Image exceeds {max_pixels} pixels")

        image = ImageOps.exif_transpose(decoded)
        if image.size != decoded.size:
            width, height = image.size

        keep_alpha = _image_has_alpha(image)
        # BREAKING: the helper now rewrites images to a sanitized JPEG/PNG payload, so callers
        # must not assume the original on-disk/camera encoding or filename extension is preserved.
        if prefer_lossless and not keep_alpha:
            normalized_content_type = "image/png"
        elif keep_alpha:
            normalized_content_type = "image/png"
        else:
            normalized_content_type = "image/jpeg"

        image = _normalize_image_mode(image, keep_alpha=normalized_content_type == "image/png")

        target_size = _fit_size(image.width, image.height, max_dimension)
        if image.size != target_size:
            image = image.resize(target_size, resample=Image.Resampling.LANCZOS, reducing_gap=2.0)

        current_quality = jpeg_quality
        for _attempt in range(12):
            encoded = _encode_image_bytes(
                image,
                content_type=normalized_content_type,
                jpeg_quality=current_quality,
            )
            if len(encoded) <= max_bytes:
                return _PreparedImage(
                    data=encoded,
                    content_type=normalized_content_type,
                    filename=_prepared_filename(original_name, normalized_content_type),
                    width=image.width,
                    height=image.height,
                )

            if normalized_content_type == "image/jpeg" and current_quality > _MIN_JPEG_QUALITY:
                current_quality = max(_MIN_JPEG_QUALITY, current_quality - 6)
                continue

            if image.width <= _MIN_IMAGE_DIMENSION and image.height <= _MIN_IMAGE_DIMENSION:
                break

            new_size = _downscale_size(image.width, image.height)
            if new_size == image.size:
                break
            image = image.resize(new_size, resample=Image.Resampling.LANCZOS, reducing_gap=2.0)

    raise OSError(f"Image exceeds {max_bytes} bytes after normalization")


def _emit_prepared_image_metrics(loop: Any, prefix: str, prepared: _PreparedImage) -> None:
    loop._try_emit(f"{prefix}_content_type={prepared.content_type}")
    loop._try_emit(f"{prefix}_size={prepared.width}x{prepared.height}")
    loop._try_emit(f"{prefix}_normalized_bytes={len(prepared.data)}")


def _reference_image_cache_key(
    path: Path,
    *,
    max_bytes: int,
    max_pixels: int,
    max_dimension: int,
    jpeg_quality: int,
    prefer_lossless: bool,
) -> tuple[Any, ...]:
    file_stat = os.stat(path, follow_symlinks=False)
    return (
        str(path),
        file_stat.st_dev,
        file_stat.st_ino,
        getattr(file_stat, "st_mtime_ns", int(file_stat.st_mtime * 1_000_000_000)),
        file_stat.st_size,
        max_bytes,
        max_pixels,
        max_dimension,
        jpeg_quality,
        prefer_lossless,
    )


def _load_reference_image_from_cache(
    path: Path,
    *,
    max_bytes: int,
    max_pixels: int,
    max_dimension: int,
    jpeg_quality: int,
    prefer_lossless: bool,
) -> tuple[_PreparedImage | None, tuple[Any, ...] | None]:
    try:
        cache_key = _reference_image_cache_key(
            path,
            max_bytes=max_bytes,
            max_pixels=max_pixels,
            max_dimension=max_dimension,
            jpeg_quality=jpeg_quality,
            prefer_lossless=prefer_lossless,
        )
    except OSError:
        return None, None

    with _REFERENCE_IMAGE_CACHE_LOCK:
        return _REFERENCE_IMAGE_CACHE.get(cache_key), cache_key


def _store_reference_image_in_cache(cache_key: tuple[Any, ...], prepared: _PreparedImage) -> None:
    with _REFERENCE_IMAGE_CACHE_LOCK:
        _REFERENCE_IMAGE_CACHE.clear()
        _REFERENCE_IMAGE_CACHE[cache_key] = prepared


def load_reference_image(
    loop: Any,
    *,
    allowed_suffixes: frozenset[str],
    default_max_bytes: int,
) -> OpenAIImageInput | None:
    """Load the configured reference image when it is safe and valid."""

    raw_path = (getattr(loop.config, "vision_reference_image_path", "") or "").strip()
    if not raw_path:
        return None

    try:
        path = normalize_reference_image_path(raw_path)
    except FileNotFoundError:
        loop._try_emit(f"vision_reference_missing={Path(raw_path).name}")
        return None
    except OSError as exc:
        loop._try_emit(f"vision_reference_error={loop._safe_error_text(exc)}")
        return None

    if not validate_reference_image_base_dir(loop, path):
        return None
    if path.suffix.casefold() not in allowed_suffixes:
        loop._try_emit(f"vision_reference_rejected=unsupported_file_type:{path.name}")
        return None

    max_bytes, max_dimension, max_pixels, prefer_lossless = _image_processing_limits(
        loop,
        default_max_bytes=default_max_bytes,
    )
    read_max_bytes = _reference_image_read_max_bytes(loop, output_max_bytes=max_bytes)
    jpeg_quality = _jpeg_quality(loop)

    prepared, cache_key = _load_reference_image_from_cache(
        path,
        max_bytes=max_bytes,
        max_pixels=max_pixels,
        max_dimension=max_dimension,
        jpeg_quality=jpeg_quality,
        prefer_lossless=prefer_lossless,
    )
    if prepared is not None:
        loop._try_emit(f"vision_reference_cache_hit={prepared.filename}")
        _emit_prepared_image_metrics(loop, "vision_reference", prepared)
        return OpenAIImageInput(
            data=prepared.data,
            content_type=prepared.content_type,
            filename=prepared.filename,
            label=_REFERENCE_IMAGE_LABEL,
        )

    try:
        raw_bytes = safe_read_reference_image_bytes(path, max_bytes=read_max_bytes)
        prepared = _prepare_image_payload(
            raw_bytes,
            original_name=path.name,
            max_bytes=max_bytes,
            max_pixels=max_pixels,
            max_dimension=max_dimension,
            jpeg_quality=jpeg_quality,
            prefer_lossless=prefer_lossless,
        )
    except FileNotFoundError:
        loop._try_emit(f"vision_reference_missing={path.name}")
        return None
    except OSError as exc:
        loop._try_emit(f"vision_reference_error={loop._safe_error_text(exc)}")
        return None

    if cache_key is not None:
        _store_reference_image_in_cache(cache_key, prepared)

    loop._try_emit(f"vision_reference_image={prepared.filename}")
    _emit_prepared_image_metrics(loop, "vision_reference", prepared)
    return OpenAIImageInput(
        data=prepared.data,
        content_type=prepared.content_type,
        filename=prepared.filename,
        label=_REFERENCE_IMAGE_LABEL,
    )


def _prepare_capture_image(
    loop: Any,
    capture: Any,
    *,
    default_filename: str,
    default_max_bytes: int,
    raw_bytes: bytes | None = None,
) -> _PreparedImage:
    max_bytes, max_dimension, max_pixels, prefer_lossless = _image_processing_limits(
        loop,
        default_max_bytes=default_max_bytes,
    )
    jpeg_quality = _jpeg_quality(loop)
    filename = Path(getattr(capture, "filename", default_filename) or default_filename).name
    raw_bytes = bytes(raw_bytes if raw_bytes is not None else (getattr(capture, "data", b"") or b""))
    prepared = _prepare_image_payload(
        raw_bytes,
        original_name=filename,
        max_bytes=max_bytes,
        max_pixels=max_pixels,
        max_dimension=max_dimension,
        jpeg_quality=jpeg_quality,
        prefer_lossless=prefer_lossless,
    )
    return prepared


def build_vision_images(
    loop: Any,
    *,
    allowed_suffixes: frozenset[str],
    default_max_bytes: int,
) -> list[OpenAIImageInput]:
    """Capture the live frame and append the optional stored reference image."""

    capture_filename = f"camera-capture-{time.time_ns()}.png"
    try:
        with loop._get_lock("_camera_lock"):
            capture = loop.camera.capture_photo(filename=capture_filename)
    except Exception as exc:
        loop._try_emit(f"camera_error={loop._safe_error_text(exc)}")
        raise RuntimeError("Camera capture failed") from exc

    source_device = getattr(capture, "source_device", "unknown")
    input_format = getattr(capture, "input_format", None) or "default"
    capture_bytes = bytes(getattr(capture, "data", b"") or b"")

    loop._try_emit(f"camera_device={source_device}")
    loop._try_emit(f"camera_input_format={input_format}")
    loop._try_emit(f"camera_capture_bytes={len(capture_bytes)}")

    if _multimodal_evidence_enabled(loop):
        try:
            loop.runtime.long_term_memory.enqueue_multimodal_evidence(
                event_name="camera_capture",
                modality="camera",
                source="camera_tool",
                message="Live camera frame captured for device interaction.",
                data={
                    "purpose": "vision_inspection",
                    "source_device": source_device,
                    "input_format": input_format,
                },
            )
        except Exception as exc:
            loop._try_emit(f"camera_memory_error={loop._safe_error_text(exc)}")
    else:
        loop._try_emit("camera_memory_persist=disabled")

    try:
        prepared_capture = _prepare_capture_image(
            loop,
            capture,
            default_filename=capture_filename,
            default_max_bytes=default_max_bytes,
            raw_bytes=capture_bytes,
        )
    except OSError as exc:
        loop._try_emit(f"camera_error={loop._safe_error_text(exc)}")
        raise RuntimeError("Camera capture failed") from exc

    _emit_prepared_image_metrics(loop, "camera_capture", prepared_capture)

    images = [
        OpenAIImageInput(
            data=prepared_capture.data,
            content_type=prepared_capture.content_type,
            filename=prepared_capture.filename,
            label=_CAPTURE_IMAGE_LABEL,
        )
    ]

    reference_image = load_reference_image(
        loop,
        allowed_suffixes=allowed_suffixes,
        default_max_bytes=default_max_bytes,
    )
    if reference_image is not None:
        images.append(reference_image)
    return images


def build_vision_prompt(question: str, *, include_reference: bool) -> str:
    """Build the user-facing vision prompt for the provider."""

    clean_question = str(question or "").strip() or "Describe what is visible."

    base_rules = (
        "This request includes camera input. "
        "Ground every claim in what is actually visible. "
        "Do not guess hidden details, unreadable text, or identity. "
        "If the view is blurry, dark, cropped, backlit, or too far away, say that plainly and tell the user exactly how to reposition themselves or the object."
    )

    if include_reference:
        return (
            f"{base_rules} "
            "Image 1 is the current live camera frame from the device. "
            "Image 2 is a stored reference image of the main user. "
            "Use Image 2 only when the user's request depends on whether Image 1 shows the same person. "
            "For identity-related requests, compare only visible face and stable appearance cues from the two images; "
            "do not infer identity from clothing, background, or prior context. "
            "If the face is not visible clearly enough for a reliable comparison, answer that identity is uncertain.\n\n"
            f"User request: {clean_question}"
        )

    return (
        f"{base_rules} "
        "Image 1 is the current live camera frame from the device.\n\n"
        f"User request: {clean_question}"
    )
