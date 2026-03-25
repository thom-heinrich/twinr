"""Reference-image and vision prompt helpers for realtime workflow loops."""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path
import stat
import time
from typing import Any

from twinr.providers.openai import OpenAIImageInput


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


def safe_read_reference_image_bytes(path: Path, *, max_bytes: int) -> bytes:
    """Read a reference image without following symlinks and with size checks."""

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


def build_image_input(data: bytes, *, path: Path, label: str) -> OpenAIImageInput:
    """Build one bounded image input payload for the OpenAI provider."""

    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return OpenAIImageInput(
        data=data,
        content_type=content_type,
        filename=path.name,
        label=label,
    )


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
    max_bytes = max(
        1024,
        int(getattr(loop.config, "vision_reference_image_max_bytes", default_max_bytes)),
    )
    try:
        data = safe_read_reference_image_bytes(path, max_bytes=max_bytes)
    except FileNotFoundError:
        loop._try_emit(f"vision_reference_missing={path.name}")
        return None
    except OSError as exc:
        loop._try_emit(f"vision_reference_error={loop._safe_error_text(exc)}")
        return None
    loop._try_emit(f"vision_reference_image={path.name}")
    return build_image_input(
        data,
        path=path,
        label="Image 2: stored reference image of the main user. Use it only for person or identity comparison.",
    )


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
    loop._try_emit(f"camera_device={capture.source_device}")
    loop._try_emit(f"camera_input_format={capture.input_format or 'default'}")
    loop._try_emit(f"camera_capture_bytes={len(capture.data)}")
    try:
        loop.runtime.long_term_memory.enqueue_multimodal_evidence(
            event_name="camera_capture",
            modality="camera",
            source="camera_tool",
            message="Live camera frame captured for device interaction.",
            data={
                "purpose": "vision_inspection",
                "source_device": capture.source_device,
                "input_format": capture.input_format or "default",
            },
        )
    except Exception as exc:
        loop._try_emit(f"camera_memory_error={loop._safe_error_text(exc)}")
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
