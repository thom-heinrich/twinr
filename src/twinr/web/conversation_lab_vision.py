"""Provide camera-backed helpers for Conversation Lab tool owners.

This module keeps ``conversation_lab.py`` focused on session persistence and
provider orchestration. The helpers here expose the camera and vision surface
expected by the realtime ``inspect_camera`` and portrait-identity tool
handlers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import mimetypes
import os
import stat
import time

from twinr.hardware.camera import V4L2StillCamera
from twinr.providers.openai.core.types import OpenAIImageInput


_ALLOWED_REFERENCE_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})
_DEFAULT_REFERENCE_IMAGE_MAX_BYTES = 10 * 1024 * 1024
_MAX_ERROR_TEXT_CHARS = 160


def owner_camera(owner: Any) -> V4L2StillCamera:
    """Return one lazily initialized still camera for the owner."""

    camera = getattr(owner, "_camera_instance", None)
    if camera is None:
        camera = V4L2StillCamera.from_config(getattr(owner, "config"))
        setattr(owner, "_camera_instance", camera)
    return camera


def build_vision_images(owner: Any) -> list[OpenAIImageInput]:
    """Capture the live camera frame plus an optional configured reference."""

    capture_filename = f"camera-capture-{time.time_ns()}.png"
    camera = owner_camera(owner)
    camera_lock = getattr(owner, "_camera_lock", None)
    try:
        if camera_lock is None:
            capture = camera.capture_photo(filename=capture_filename)
        else:
            with camera_lock:
                capture = camera.capture_photo(filename=capture_filename)
    except Exception as exc:
        _emit_safe(owner, f"camera_error={_safe_error_text(exc)}")
        raise RuntimeError("Camera capture failed") from exc

    _emit_safe(owner, f"camera_device={capture.source_device}")
    _emit_safe(owner, f"camera_input_format={capture.input_format or 'default'}")
    _emit_safe(owner, f"camera_capture_bytes={len(capture.data)}")
    capture_path = Path(getattr(capture, "filename", capture_filename)).name
    capture_content_type = capture.content_type or mimetypes.guess_type(capture_path)[0] or "application/octet-stream"
    images = [
        OpenAIImageInput(
            data=capture.data,
            content_type=capture_content_type,
            filename=capture_path,
            label="Image 1: live camera frame from the device.",
        )
    ]
    reference_image = _load_reference_image(owner)
    if reference_image is not None:
        images.append(reference_image)
    return images


def build_vision_prompt(question: str, *, include_reference: bool) -> str:
    """Build the operator-facing camera prompt for one live inspection call."""

    clean_question = question.strip()
    if include_reference:
        return (
            "This request includes camera input. "
            "Image 1 is the current live camera frame from the device. "
            "Image 2 is a stored reference image of the main user. "
            "Use the reference image only when the user's question depends on whether the live image shows that user. "
            "If identity is uncertain, say that clearly. "
            "If the camera view is too unclear, tell the user how to position themselves or the object.\n\n"
            f"User request: {clean_question}"
        )
    return (
        "This request includes camera input. "
        "Image 1 is the current live camera frame from the device. "
        "Answer from what is actually visible. "
        "If the view is too unclear, tell the user how to position themselves or the object in front of the camera.\n\n"
        f"User request: {clean_question}"
    )


def _emit_safe(owner: Any, payload: str) -> None:
    try:
        owner.emit(payload)
    except Exception:
        return


def _safe_error_text(exc: Exception) -> str:
    detail = " ".join(str(exc).split()).strip()
    if not detail:
        return type(exc).__name__
    if len(detail) > _MAX_ERROR_TEXT_CHARS:
        detail = detail[: _MAX_ERROR_TEXT_CHARS - 3].rstrip() + "..."
    return f"{type(exc).__name__}:{detail}"


def _load_reference_image(owner: Any) -> OpenAIImageInput | None:
    raw_path = (getattr(getattr(owner, "config", None), "vision_reference_image_path", "") or "").strip()
    if not raw_path:
        return None
    try:
        path = _normalize_reference_image_path(owner, raw_path)
    except FileNotFoundError:
        _emit_safe(owner, "vision_reference_missing=true")
        return None
    except OSError as exc:
        _emit_safe(owner, f"vision_reference_error={_safe_error_text(exc)}")
        return None
    if not _validate_reference_image_base_dir(owner, path):
        return None
    if path.suffix.casefold() not in _ALLOWED_REFERENCE_IMAGE_SUFFIXES:
        _emit_safe(owner, f"vision_reference_rejected=unsupported_file_type:{path.name}")
        return None
    max_bytes = max(
        1024,
        int(getattr(getattr(owner, "config", None), "vision_reference_image_max_bytes", _DEFAULT_REFERENCE_IMAGE_MAX_BYTES)),
    )
    try:
        data = _safe_read_reference_image_bytes(path, max_bytes=max_bytes)
    except FileNotFoundError:
        _emit_safe(owner, "vision_reference_missing=true")
        return None
    except OSError as exc:
        _emit_safe(owner, f"vision_reference_error={_safe_error_text(exc)}")
        return None
    _emit_safe(owner, f"vision_reference_image={path.name}")
    return OpenAIImageInput(
        data=data,
        content_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        filename=path.name,
        label="Image 2: stored reference image of the main user. Use it only for person or identity comparison.",
    )


def _normalize_reference_image_path(owner: Any, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        env_path = getattr(owner, "env_path", None)
        base_dir = Path(env_path).expanduser().resolve().parent if env_path is not None else Path.cwd()
        path = base_dir / path
    return path.parent.resolve(strict=True) / path.name


def _validate_reference_image_base_dir(owner: Any, path: Path) -> bool:
    base_dir_raw = (getattr(getattr(owner, "config", None), "vision_reference_image_base_dir", "") or "").strip()
    if not base_dir_raw:
        return True
    try:
        base_dir = Path(base_dir_raw).expanduser().resolve(strict=True)
    except OSError:
        _emit_safe(owner, "vision_reference_rejected=invalid_base_dir")
        return False
    try:
        path.relative_to(base_dir)
    except ValueError:
        _emit_safe(owner, f"vision_reference_rejected=outside_base_dir:{path.name}")
        return False
    return True


def _safe_read_reference_image_bytes(path: Path, *, max_bytes: int) -> bytes:
    flags = os.O_RDONLY
    nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
    if nofollow_flag:
        flags |= nofollow_flag
    fd = os.open(path, flags)
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise OSError("Reference image must be a regular file")
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
