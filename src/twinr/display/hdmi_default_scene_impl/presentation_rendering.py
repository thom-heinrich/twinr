# CHANGELOG: 2026-03-28
# BUG-1: Fixed portrait / EXIF-oriented images rendering sideways by applying EXIF transpose before resize.
# BUG-2: Fixed valid inline data-URL images never rendering in presentation image blocks; they now fall back after local-path load fails.
# BUG-3: Fixed transparency loss for PNG/WebP assets; alpha is preserved during paste instead of being flattened to black.
# BUG-4: Fixed frame-by-frame re-decode/re-resize of the same asset causing HDMI render-loop stalls; prepared thumbnails are now LRU-cached.
# SEC-1: Blocked practical image DoS vectors on Raspberry Pi by rejecting non-regular files, oversize files / data URLs, and over-budget pixel counts before decode.
# SEC-2: Switched inline base64 parsing to strict validation and MIME gating for data:image/* payloads.
# IMP-1: Added optional pyvips/libvips fast path (streaming, low-memory thumbnails) with Pillow fallback; this is the 2026 frontier path for ARM/Pi-class devices.
# IMP-2: Upgraded Pillow fallback to EXIF-aware, draft-assisted, LANCZOS resampling with reducing_gap for better quality / speed balance.
# IMP-3: Added optional presentation_media_roots allowlist support for deployments that want filesystem confinement of presentation assets.
"""Fullscreen presentation rendering helpers for the HDMI scene."""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
import threading
import warnings
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

from .typing_contracts import (
    HdmiImageSurface,
    HdmiPanelDrawSurface,
    HdmiPresentationNodeLike,
)


_PRESENTATION_CACHE_INIT_LOCK = threading.Lock()


class HdmiPresentationRenderingMixin:
    """Render fullscreen HDMI presentation overlays and their bounded media."""

    presentation_image_cache_limit = 24
    presentation_max_source_pixels = 32_000_000
    presentation_max_file_bytes = 64 * 1024 * 1024
    presentation_max_inline_bytes = 12 * 1024 * 1024
    presentation_media_roots: tuple[str | os.PathLike[str], ...] | None = None
    # BREAKING: when configured, assets outside presentation_media_roots are rejected.

    def _draw_presentation_surface(
        self,
        image: HdmiImageSurface,
        draw: HdmiPanelDrawSurface,
        *,
        presentation: HdmiPresentationNodeLike,
        queued_count: int,
    ) -> None:
        """Draw one expanding presentation surface over the default scene."""

        left, top, right, bottom = presentation.box
        compact = (right - left) < 360 or (bottom - top) < 240
        padding = 16 if compact else 24
        eyebrow_font = self.tools._font(14 if compact else 17, bold=True)
        title_font = self.tools._font(22 if compact else 40, bold=True)
        subtitle_font = self.tools._font(13 if compact else 18, bold=False)
        body_font = self.tools._font(14 if compact else 20, bold=False)
        label = "SHOWING" if presentation.kind == "image" else "FOCUS"
        label = label if queued_count <= 0 else f"{label} +{queued_count}"

        draw.rounded_rectangle(
            presentation.box,
            radius=32 if not compact else 20,
            fill=(2, 2, 2),
            outline=presentation.accent,
            width=3,
        )
        draw.rounded_rectangle(
            (left + 12, top + 12, left + 20, top + 20),
            radius=4,
            fill=presentation.accent,
        )
        text_left = left + padding
        text_top = top + padding
        inner_width = max(80, right - left - (padding * 2))
        draw.text((text_left + 18, text_top), label, fill=(176, 176, 176), font=eyebrow_font)
        text_top += self.tools._text_height(draw, font=eyebrow_font) + (10 if compact else 14)

        if presentation.title and presentation.chrome_progress >= 0.12:
            wrapped_title = self.tools._wrapped_lines(
                draw,
                (presentation.title,),
                max_width=inner_width,
                font=title_font,
                max_lines=2,
            )
            title_line_height = self.tools._text_height(draw, font=title_font) + 6
            for line in wrapped_title:
                draw.text((text_left, text_top), line, fill=(255, 255, 255), font=title_font)
                text_top += title_line_height

        if presentation.subtitle and presentation.content_progress > 0.0:
            subtitle_lines = self.tools._wrapped_lines(
                draw,
                (presentation.subtitle,),
                max_width=inner_width,
                font=subtitle_font,
                max_lines=2,
            )
            subtitle_line_height = self.tools._text_height(draw, font=subtitle_font) + 4
            for line in subtitle_lines:
                draw.text((text_left, text_top), line, fill=(214, 214, 214), font=subtitle_font)
                text_top += subtitle_line_height
            text_top += 4

        content_bottom = bottom - padding
        if presentation.kind == "image" and presentation.content_progress > 0.0:
            text_top = self._draw_presentation_image_block(
                image,
                draw,
                presentation=presentation,
                box=(text_left, text_top, right - padding, content_bottom),
                compact=compact,
            )

        remaining_top = min(text_top + (8 if compact else 14), content_bottom)
        remaining_box = (text_left, remaining_top, right - padding, content_bottom)
        self._draw_presentation_body(
            draw,
            body_lines=presentation.body_lines,
            box=remaining_box,
            font=body_font,
            compact=compact,
            reveal_progress=presentation.body_progress,
        )

    def _draw_presentation_image_block(
        self,
        image: HdmiImageSurface,
        draw: HdmiPanelDrawSurface,
        *,
        presentation: HdmiPresentationNodeLike,
        box: tuple[int, int, int, int],
        compact: bool,
    ) -> int:
        """Paste one local image into the presentation surface and return the next text Y."""

        left, top, right, bottom = box
        if right <= left or bottom <= top:
            return top

        available_width = max(60, right - left)
        available_height = max(60, bottom - top)
        image_height = int(available_height * (0.62 if not compact else 0.54))
        image_box = (left, top, right, min(bottom, top + image_height))
        if image_box[2] <= image_box[0] or image_box[3] <= image_box[1]:
            return top

        draw.rounded_rectangle(
            image_box,
            radius=18 if not compact else 12,
            fill=(8, 8, 8),
            outline=(84, 84, 84),
            width=2,
        )
        pasted = self._paste_presentation_image(
            image,
            box=image_box,
            image_path=getattr(presentation, "image_path", None),
            image_data_url=getattr(presentation, "image_data_url", None),
        )
        if not pasted:
            placeholder_font = self.tools._font(16 if compact else 24, bold=True)
            placeholder = "IMAGE UNAVAILABLE"
            placeholder_width = self.tools._text_width(draw, placeholder, font=placeholder_font)
            placeholder_height = self.tools._text_height(draw, font=placeholder_font)
            draw.text(
                (
                    left + max(0, (available_width - placeholder_width) // 2),
                    top + max(0, ((image_box[3] - image_box[1]) - placeholder_height) // 2),
                ),
                placeholder,
                fill=(170, 170, 170),
                font=placeholder_font,
            )
        return image_box[3]

    def _draw_presentation_body(
        self,
        draw: HdmiPanelDrawSurface,
        *,
        body_lines: tuple[str, ...],
        box: tuple[int, int, int, int],
        font: object,
        compact: bool,
        reveal_progress: float,
    ) -> None:
        """Draw the bounded body copy for a fullscreen presentation."""

        left, top, right, bottom = box
        if top >= bottom or right <= left or reveal_progress <= 0.0:
            return
        max_lines = 3 if compact else 4
        wrapped = self.tools._wrapped_lines(
            draw,
            body_lines,
            max_width=max(60, right - left),
            font=font,
            max_lines=max_lines,
        )
        line_height = self.tools._text_height(draw, font=font) + (4 if compact else 8)
        visible_lines = max(1, min(max_lines, int(round(reveal_progress * max_lines))))
        for index, line in enumerate(wrapped[:visible_lines]):
            y = top + (index * line_height)
            if y + line_height > bottom:
                break
            draw.text((left, y), line, fill=(236, 236, 236), font=font)

    def _paste_presentation_image(
        self,
        image: HdmiImageSurface,
        *,
        box: tuple[int, int, int, int],
        image_path: str | os.PathLike[str] | None,
        image_data_url: str | None,
    ) -> bool:
        """Paste the best available presentation image into the target box."""

        prepared = self._prepare_presentation_asset(
            box=box,
            image_path=image_path,
            image_data_url=image_data_url,
        )
        if prepared is None:
            return False

        target_left = box[0] + max(0, ((box[2] - box[0]) - prepared.width) // 2)
        target_top = box[1] + max(0, ((box[3] - box[1]) - prepared.height) // 2)
        if prepared.mode == "RGBA":
            image.paste(prepared, (target_left, target_top), prepared.getchannel("A"))
        else:
            image.paste(prepared, (target_left, target_top))
        return True

    def _prepare_presentation_asset(
        self,
        *,
        box: tuple[int, int, int, int],
        image_path: str | os.PathLike[str] | None,
        image_data_url: str | None,
    ):
        """Prepare a presentation asset once and cache the resized result."""

        target_size = (
            max(1, box[2] - box[0] - 12),
            max(1, box[3] - box[1] - 12),
        )
        if image_path:
            prepared = self._prepare_presentation_asset_from_path(image_path=image_path, target_size=target_size)
            if prepared is not None:
                return prepared
        if image_data_url:
            return self._prepare_presentation_asset_from_data_url(
                image_data_url=image_data_url,
                target_size=target_size,
            )
        return None

    def _prepare_presentation_asset_from_path(
        self,
        *,
        image_path: str | os.PathLike[str],
        target_size: tuple[int, int],
    ):
        """Prepare and cache a bounded asset from a local path."""

        resolved = self._resolve_presentation_image_path(image_path)
        if resolved is None:
            return None

        try:
            stat = resolved.stat()
        except OSError:
            return None

        file_budget = self._presentation_setting_int("presentation_max_file_bytes", 64 * 1024 * 1024, minimum=1)
        if stat.st_size <= 0 or stat.st_size > file_budget:
            # BREAKING: oversized assets are rejected to keep the HDMI render loop non-blocking on Pi-class hardware.
            return None

        cache_key = ("file", str(resolved), stat.st_mtime_ns, stat.st_size, target_size)
        cached = self._presentation_cache_get(cache_key)
        if cached is not None:
            return cached

        prepared = self._prepare_with_pyvips_from_path(resolved, target_size)
        if prepared is None:
            prepared = self._prepare_with_pillow_path(resolved, target_size)
        if prepared is None:
            return None

        self._presentation_cache_put(cache_key, prepared)
        return prepared

    def _prepare_presentation_asset_from_data_url(
        self,
        *,
        image_data_url: str,
        target_size: tuple[int, int],
    ):
        """Prepare and cache a bounded asset from an inline data URL."""

        header, separator, payload = str(image_data_url).partition(",")
        header_lower = header.lower().strip()
        payload = payload.strip()
        if separator != "," or not header_lower.startswith("data:image/") or ";base64" not in header_lower:
            return None

        max_inline_bytes = self._presentation_setting_int("presentation_max_inline_bytes", 12 * 1024 * 1024, minimum=1)
        max_payload_chars = ((max_inline_bytes + 2) // 3) * 4
        if not payload or len(payload) > (max_payload_chars + 8):
            return None

        payload_digest = hashlib.blake2b(payload.encode("ascii", "ignore"), digest_size=16).hexdigest()
        cache_key = ("data-url", payload_digest, target_size)
        cached = self._presentation_cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            decoded = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError):
            return None
        if not decoded or len(decoded) > max_inline_bytes:
            return None

        prepared = self._prepare_with_pyvips_from_bytes(decoded, target_size)
        if prepared is None:
            prepared = self._prepare_with_pillow_bytes(decoded, target_size)
        if prepared is None:
            return None

        self._presentation_cache_put(cache_key, prepared)
        return prepared

    def _prepare_with_pyvips_from_path(self, image_path: Path, target_size: tuple[int, int]):
        """Prepare a bounded asset with libvips/pyvips from a file path."""

        pyvips = self._get_pyvips()
        if pyvips is None:
            return None
        try:
            probe = pyvips.Image.new_from_file(str(image_path), access="sequential", fail=True)
            if not self._presentation_dimensions_allowed(probe.width, probe.height):
                return None
            thumbnail = pyvips.Image.thumbnail(
                str(image_path),
                target_size[0],
                height=target_size[1],
                size="both",
                no_rotate=False,
                fail_on="warning",
            )
            return self._pyvips_thumbnail_to_pil(thumbnail)
        except Exception:
            return None

    def _prepare_with_pyvips_from_bytes(self, image_bytes: bytes, target_size: tuple[int, int]):
        """Prepare a bounded asset with libvips/pyvips from encoded bytes."""

        pyvips = self._get_pyvips()
        if pyvips is None:
            return None
        try:
            probe = pyvips.Image.new_from_buffer(image_bytes, "", access="sequential", fail=True)
            if not self._presentation_dimensions_allowed(probe.width, probe.height):
                return None
            thumbnail = pyvips.Image.thumbnail_buffer(
                image_bytes,
                target_size[0],
                height=target_size[1],
                size="both",
                no_rotate=False,
                fail_on="warning",
            )
            return self._pyvips_thumbnail_to_pil(thumbnail)
        except Exception:
            return None

    def _pyvips_thumbnail_to_pil(self, thumbnail):
        """Convert a pyvips thumbnail to a detached PIL image."""

        try:
            from PIL import Image
        except ImportError:
            return None
        try:
            encoded = thumbnail.write_to_buffer(".png")
            with Image.open(BytesIO(encoded)) as opened:
                return opened.copy()
        except Exception:
            return None

    def _prepare_with_pillow_path(self, image_path: Path, target_size: tuple[int, int]):
        """Prepare a bounded asset with Pillow from a file path."""

        try:
            from PIL import Image
        except ImportError:
            return None
        decompression_warning = getattr(Image, "DecompressionBombWarning", Warning)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", decompression_warning)
                with Image.open(image_path) as opened:
                    return self._prepare_with_pillow_opened(opened, target_size)
        except Exception:
            return None

    def _prepare_with_pillow_bytes(self, image_bytes: bytes, target_size: tuple[int, int]):
        """Prepare a bounded asset with Pillow from encoded bytes."""

        try:
            from PIL import Image
        except ImportError:
            return None
        decompression_warning = getattr(Image, "DecompressionBombWarning", Warning)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", decompression_warning)
                with Image.open(BytesIO(image_bytes)) as opened:
                    return self._prepare_with_pillow_opened(opened, target_size)
        except Exception:
            return None

    def _prepare_with_pillow_opened(self, opened, target_size: tuple[int, int]):
        """Normalize, resize, and detach one Pillow image."""

        try:
            from PIL import Image, ImageOps
        except ImportError:
            return None

        if not self._presentation_dimensions_allowed(*opened.size):
            return None

        try:
            opened.draft("RGB", target_size)
        except Exception:
            pass

        transposed = ImageOps.exif_transpose(opened)
        preserve_alpha = self._pil_should_preserve_alpha(transposed)
        prepared = transposed.convert("RGBA" if preserve_alpha else "RGB")
        resample_lanczos = self._pil_lanczos(Image)

        if prepared.width > target_size[0] or prepared.height > target_size[1]:
            prepared.thumbnail(target_size, resample=resample_lanczos, reducing_gap=3.0)
        elif prepared.size != target_size:
            prepared = ImageOps.contain(prepared, target_size, method=resample_lanczos)

        return prepared.copy()

    def _pil_should_preserve_alpha(self, pil_image) -> bool:
        """Return whether the source image carries meaningful transparency."""

        if "A" in getattr(pil_image, "mode", ""):
            return True
        return getattr(pil_image, "mode", None) == "P" and "transparency" in getattr(pil_image, "info", {})

    def _pil_lanczos(self, ImageModule):
        """Return the best available LANCZOS resampling enum/value."""

        resampling = getattr(ImageModule, "Resampling", ImageModule)
        return getattr(resampling, "LANCZOS", getattr(ImageModule, "LANCZOS", 1))

    def _presentation_dimensions_allowed(self, width: int, height: int) -> bool:
        """Check source dimensions against Pi-friendly decode budgets."""

        try:
            width_int = int(width)
            height_int = int(height)
        except (TypeError, ValueError):
            return False
        if width_int <= 0 or height_int <= 0:
            return False
        max_pixels = self._presentation_setting_int("presentation_max_source_pixels", 32_000_000, minimum=1)
        return (width_int * height_int) <= max_pixels

    def _resolve_presentation_image_path(self, image_path: str | os.PathLike[str]) -> Path | None:
        """Resolve and validate a local presentation image path."""

        try:
            candidate = Path(os.fspath(image_path)).expanduser()
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError, TypeError, ValueError):
            return None

        try:
            if not resolved.is_file():
                # BREAKING: non-regular files (FIFO/device/socket) are refused to avoid blocking the HDMI render loop.
                return None
        except OSError:
            return None

        allowed_roots = self._presentation_media_root_paths()
        if allowed_roots and not any(self._path_within_root(resolved, root) for root in allowed_roots):
            return None
        return resolved

    def _presentation_media_root_paths(self) -> tuple[Path, ...]:
        """Return configured media roots as resolved paths."""

        configured = getattr(self, "presentation_media_roots", None)
        if not configured:
            return ()
        if isinstance(configured, (str, os.PathLike)):
            configured = (configured,)
        roots: list[Path] = []
        for root in configured:
            try:
                resolved = Path(os.fspath(root)).expanduser().resolve(strict=True)
            except (OSError, RuntimeError, TypeError, ValueError):
                continue
            roots.append(resolved)
        return tuple(roots)

    def _path_within_root(self, candidate: Path, root: Path) -> bool:
        """Return whether one resolved path is contained within another."""

        try:
            candidate.relative_to(root)
            return True
        except ValueError:
            return False

    def _presentation_setting_int(self, name: str, default: int, *, minimum: int) -> int:
        """Read one integer-like setting from the instance with bounds."""

        raw_value = getattr(self, name, default)
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            return default
        return max(minimum, value)

    def _presentation_cache_get(self, key):
        """Get one prepared asset from the local LRU cache."""

        cache, lock = self._presentation_cache_state()
        with lock:
            cached = cache.get(key)
            if cached is None:
                return None
            cache.move_to_end(key)
            return cached

    def _presentation_cache_put(self, key, prepared) -> None:
        """Store one prepared asset in the local LRU cache."""

        cache_limit = self._presentation_setting_int("presentation_image_cache_limit", 24, minimum=0)
        if cache_limit == 0:
            return

        cache, lock = self._presentation_cache_state()
        with lock:
            cache[key] = prepared
            cache.move_to_end(key)
            while len(cache) > cache_limit:
                cache.popitem(last=False)

    def _presentation_cache_state(self):
        """Create the shared cache and lock lazily on first use."""

        cache = getattr(self, "_presentation_asset_cache", None)
        lock = getattr(self, "_presentation_asset_cache_lock", None)
        if cache is not None and lock is not None:
            return cache, lock

        with _PRESENTATION_CACHE_INIT_LOCK:
            cache = getattr(self, "_presentation_asset_cache", None)
            lock = getattr(self, "_presentation_asset_cache_lock", None)
            if cache is None:
                cache = OrderedDict()
                setattr(self, "_presentation_asset_cache", cache)
            if lock is None:
                lock = threading.RLock()
                setattr(self, "_presentation_asset_cache_lock", lock)
        return cache, lock

    def _get_pyvips(self):
        """Import pyvips once and memoize the result."""

        sentinel = object()
        cached = getattr(self, "_presentation_pyvips_module", sentinel)
        if cached is sentinel:
            try:
                import pyvips
            except Exception:
                pyvips = None
            setattr(self, "_presentation_pyvips_module", pyvips)
            return pyvips
        return cached

    def _paste_local_image(
        self,
        image: HdmiImageSurface,
        *,
        box: tuple[int, int, int, int],
        image_path: str | os.PathLike[str] | None,
    ) -> bool:
        """Paste one bounded local image into the target box."""

        return self._paste_presentation_image(image, box=box, image_path=image_path, image_data_url=None)

    def _paste_inline_image(
        self,
        image: HdmiImageSurface,
        *,
        box: tuple[int, int, int, int],
        image_data_url: str | None,
    ) -> bool:
        """Paste one bounded inline data-URL image into the target box."""

        return self._paste_presentation_image(image, box=box, image_path=None, image_data_url=image_data_url)