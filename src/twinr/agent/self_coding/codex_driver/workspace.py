"""Build bounded temporary workspaces for local Codex compile jobs."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import secrets
import shutil
import tempfile
from typing import Iterator

from twinr.agent.self_coding.codex_driver.types import CodexCompileRequest, compile_output_schema
from twinr.agent.self_coding.contracts import CompileJobRecord, RequirementsDialogueSession

# AUDIT-FIX(#2): Bound workspace size to avoid SD-card exhaustion on small edge devices.
_DEFAULT_MAX_PROMPT_BYTES = 512 * 1024
# AUDIT-FIX(#2): Bound individual JSON artifacts independently from the prompt.
_DEFAULT_MAX_JSON_BYTES = 2 * 1024 * 1024
# AUDIT-FIX(#2): Bound total workspace size before any file is written.
_DEFAULT_MAX_TOTAL_BYTES = 8 * 1024 * 1024
_WORKSPACE_PREFIX = "twinr-self-coding-"
_JSON_WORKSPACE_FILES = (
    "skill_spec.json",
    "dialogue_session.json",
    "compile_job.json",
    "output_schema.json",
)


@dataclass(frozen=True, slots=True)
class CodexCompileWorkspace:
    """Describe one temporary compile workspace."""

    root: Path
    request_path: Path
    output_schema_path: Path


class CodexCompileWorkspaceBuilder:
    """Materialize compile context files under a bounded temporary root."""

    def __init__(
        self,
        *,
        temp_root: str | Path | None = None,
        max_prompt_bytes: int = _DEFAULT_MAX_PROMPT_BYTES,
        max_json_bytes: int = _DEFAULT_MAX_JSON_BYTES,
        max_total_bytes: int = _DEFAULT_MAX_TOTAL_BYTES,
    ) -> None:
        self.temp_root = None if temp_root is None else Path(temp_root).expanduser().resolve(strict=False)
        # AUDIT-FIX(#2): Validate limit configuration eagerly so bad values fail fast at startup.
        self.max_prompt_bytes = self._validate_positive_limit("max_prompt_bytes", max_prompt_bytes)
        # AUDIT-FIX(#2): Validate limit configuration eagerly so bad values fail fast at startup.
        self.max_json_bytes = self._validate_positive_limit("max_json_bytes", max_json_bytes)
        # AUDIT-FIX(#2): Validate limit configuration eagerly so bad values fail fast at startup.
        self.max_total_bytes = self._validate_positive_limit("max_total_bytes", max_total_bytes)

    @contextmanager
    def build(
        self,
        *,
        job: CompileJobRecord,
        session: RequirementsDialogueSession,
        prompt: str,
    ) -> Iterator[CodexCompileRequest]:
        try:
            # AUDIT-FIX(#5): Compute the schema once so the in-memory request and on-disk file stay identical.
            output_schema = compile_output_schema()
            payloads = self._build_workspace_payloads(
                job=job,
                session=session,
                prompt=prompt,
                output_schema=output_schema,
            )
            # AUDIT-FIX(#2): Enforce byte limits before any file-system mutation happens.
            self._validate_payload_sizes(payloads)
        except (TypeError, ValueError) as exc:
            # AUDIT-FIX(#3): Raise a phase-specific error so callers can classify serialization failures reliably.
            raise ValueError(f"Failed to serialize compile workspace payloads: {exc}") from exc

        root: Path | None = None
        request_path: Path | None = None
        output_schema_path: Path | None = None
        body_error: BaseException | None = None

        try:
            root = self._create_workspace_root()
            request_path = self._write_workspace_file(root, "REQUEST.md", payloads["REQUEST.md"])
            self._write_workspace_file(root, "skill_spec.json", payloads["skill_spec.json"])
            self._write_workspace_file(root, "dialogue_session.json", payloads["dialogue_session.json"])
            self._write_workspace_file(root, "compile_job.json", payloads["compile_job.json"])
            output_schema_path = self._write_workspace_file(root, "output_schema.json", payloads["output_schema.json"])
        except OSError as exc:
            if root is not None:
                self._cleanup_workspace(root, body_error=exc)
            # AUDIT-FIX(#3): Preserve the original cause while adding workspace-specific context for operators.
            raise OSError(
                f"Failed to build compile workspace in {self.temp_root or 'system temporary directory'}: {exc}"
            ) from exc

        try:
            yield CodexCompileRequest(
                job=job,
                session=session,
                prompt=prompt,
                output_schema=output_schema,
                workspace_root=str(root),
                request_path=str(request_path),
                output_schema_path=str(output_schema_path),
            )
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            self._cleanup_workspace(root, body_error=body_error)

    def _build_workspace_payloads(
        self,
        *,
        job: CompileJobRecord,
        session: RequirementsDialogueSession,
        prompt: str,
        output_schema: object,
    ) -> dict[str, bytes]:
        return {
            "REQUEST.md": self._encode_prompt(prompt),
            "skill_spec.json": self._encode_json_payload("skill_spec.json", session.to_skill_spec().to_payload()),
            "dialogue_session.json": self._encode_json_payload("dialogue_session.json", session.to_payload()),
            "compile_job.json": self._encode_json_payload("compile_job.json", job.to_payload()),
            "output_schema.json": self._encode_json_payload("output_schema.json", output_schema),
        }

    def _validate_payload_sizes(self, payloads: dict[str, bytes]) -> None:
        prompt_bytes = len(payloads["REQUEST.md"])
        if prompt_bytes > self.max_prompt_bytes:
            raise ValueError(
                f"Compile prompt is too large for a temporary workspace: "
                f"{prompt_bytes} bytes > {self.max_prompt_bytes} bytes"
            )

        for filename in _JSON_WORKSPACE_FILES:
            payload_bytes = len(payloads[filename])
            if payload_bytes > self.max_json_bytes:
                raise ValueError(
                    f"Workspace payload {filename!r} is too large: "
                    f"{payload_bytes} bytes > {self.max_json_bytes} bytes"
                )

        total_bytes = sum(len(payload) for payload in payloads.values())
        if total_bytes > self.max_total_bytes:
            raise ValueError(
                f"Compile workspace would exceed the temporary size budget: "
                f"{total_bytes} bytes > {self.max_total_bytes} bytes"
            )

    def _create_workspace_root(self) -> Path:
        if self.temp_root is None:
            return Path(tempfile.mkdtemp(prefix=_WORKSPACE_PREFIX))

        dir_fd = self._open_secure_directory(self.temp_root)
        try:
            # AUDIT-FIX(#1): Create the child workspace relative to an opened directory fd to block symlink and TOCTOU escapes.
            for _ in range(128):
                candidate_name = f"{_WORKSPACE_PREFIX}{secrets.token_hex(8)}"
                try:
                    os.mkdir(candidate_name, mode=0o700, dir_fd=dir_fd)
                    return self.temp_root / candidate_name
                except FileExistsError:
                    continue
        finally:
            os.close(dir_fd)

        raise OSError(f"Failed to allocate a unique compile workspace under {self.temp_root}")

    def _open_secure_directory(self, path: Path) -> int:
        if not path.is_absolute():
            raise OSError(f"Configured temp_root must resolve to an absolute path: {path}")

        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
        nofollow_flags = flags | getattr(os, "O_NOFOLLOW", 0)

        # AUDIT-FIX(#1): Walk the directory tree one component at a time via dir_fd to avoid symlink hops and path races.
        dir_fd = os.open(path.anchor, flags)
        try:
            for part in path.parts[1:]:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=dir_fd)
                except FileExistsError:
                    pass

                next_fd = os.open(part, nofollow_flags, dir_fd=dir_fd)
                os.close(dir_fd)
                dir_fd = next_fd
        except BaseException:
            os.close(dir_fd)
            raise

        return dir_fd

    def _write_workspace_file(self, root: Path, filename: str, payload: bytes) -> Path:
        path = root / filename
        with path.open("xb") as handle:
            handle.write(payload)
        return path

    def _cleanup_workspace(self, root: Path | None, *, body_error: BaseException | None) -> None:
        if root is None:
            return

        try:
            # AUDIT-FIX(#4): Preserve the primary failure and only annotate it if cleanup also goes sideways.
            shutil.rmtree(root)
        except FileNotFoundError:
            return
        except OSError as exc:
            if body_error is not None:
                body_error.add_note(f"Compile workspace cleanup also failed for {root}: {exc}")
                return
            raise OSError(f"Failed to clean up compile workspace {root}: {exc}") from exc

    def _encode_json_payload(self, filename: str, payload: object) -> bytes:
        encoded_chunks: list[bytes] = []
        payload_size = 1
        encoder = json.JSONEncoder(indent=2, sort_keys=True, ensure_ascii=False)

        # AUDIT-FIX(#2): Abort oversized JSON payloads during encoding instead of discovering them after the disk write.
        for chunk in encoder.iterencode(payload):
            encoded_chunk = chunk.encode("utf-8")
            payload_size += len(encoded_chunk)
            if payload_size > self.max_json_bytes:
                raise ValueError(
                    f"Workspace payload {filename!r} is too large: "
                    f"{payload_size} bytes > {self.max_json_bytes} bytes"
                )
            encoded_chunks.append(encoded_chunk)

        encoded_chunks.append(b"\n")
        return b"".join(encoded_chunks)

    def _encode_prompt(self, prompt: str) -> bytes:
        # AUDIT-FIX(#2): Reject obviously oversized prompts before allocating their UTF-8 byte representation.
        if len(prompt) > self.max_prompt_bytes:
            raise ValueError(
                f"Compile prompt is too large for a temporary workspace: "
                f"character length exceeds {self.max_prompt_bytes}"
            )

        prompt_bytes = prompt.encode("utf-8")
        if len(prompt_bytes) > self.max_prompt_bytes:
            raise ValueError(
                f"Compile prompt is too large for a temporary workspace: "
                f"{len(prompt_bytes)} bytes > {self.max_prompt_bytes} bytes"
            )
        return prompt_bytes

    @staticmethod
    def _validate_positive_limit(name: str, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
        return value