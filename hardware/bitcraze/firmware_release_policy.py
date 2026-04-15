"""Own fail-closed firmware lanes for Twinr's Crazyflie workflows.

This module centralizes the policy split between development firmware builds,
bench validation, operator promotion, and recovery. The Bitcraze helpers use
these attestation formats to prove what binary is being flashed, where it came
from, and whether the requested device role is allowed to receive it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1
APP_NAME = "twinr_on_device_failsafe"

LANE_DEV = "dev"
LANE_BENCH = "bench"
LANE_OPERATOR = "operator"
LANE_RECOVERY = "recovery"
LANES = (LANE_DEV, LANE_BENCH, LANE_OPERATOR, LANE_RECOVERY)

DEVICE_ROLE_DEV = "dev"
DEVICE_ROLE_BENCH = "bench"
DEVICE_ROLE_OPERATOR = "operator"
DEVICE_ROLES = (DEVICE_ROLE_DEV, DEVICE_ROLE_BENCH, DEVICE_ROLE_OPERATOR)

BUILD_ATTESTATION_KIND = "twinr_bitcraze_build_attestation"
RELEASE_ATTESTATION_KIND = "twinr_bitcraze_release_attestation"


class FirmwareReleasePolicyError(ValueError):
    """Raise when a firmware lane request violates the release policy."""


@dataclass(frozen=True)
class ArtifactAttestation:
    """Describe one concrete artifact fingerprint."""

    path: str
    sha256: str
    size_bytes: int
    source_kind: str | None


@dataclass(frozen=True)
class BuildSourceAttestation:
    """Capture the pinned source state used to produce one firmware artifact."""

    firmware_root: str
    firmware_git_commit: str
    firmware_git_tag: str | None
    expected_firmware_revision: str
    app_root: str
    app_tree_sha256: str
    toolchain: str
    firmware_patch_paths: tuple[str, ...] = ()
    firmware_patch_sha256s: tuple[str, ...] = ()


@dataclass(frozen=True)
class BuildAttestation:
    """Describe one dev-lane firmware build that came from pinned sources."""

    schema_version: int
    kind: str
    app_name: str
    lane: str
    created_at_utc: str
    artifact: ArtifactAttestation
    source: BuildSourceAttestation


@dataclass(frozen=True)
class PromotionApproval:
    """Record who approved one bench/operator firmware promotion."""

    approved_by: str
    reason: str
    validation_evidence: tuple[str, ...]


@dataclass(frozen=True)
class ReleaseAttestation:
    """Describe one promoted bench/operator firmware release."""

    schema_version: int
    kind: str
    app_name: str
    lane: str
    release_id: str
    created_at_utc: str
    artifact: ArtifactAttestation
    source: BuildSourceAttestation
    source_attestation_kind: str
    source_attestation_path: str
    source_attestation_sha256: str
    approval: PromotionApproval


@dataclass(frozen=True)
class FlashAuthorization:
    """Summarize the attested policy decision for one flash request."""

    lane: str
    device_role: str
    attestation_kind: str | None
    attestation_path: str | None
    release_id: str | None
    source_firmware_git_commit: str | None
    source_firmware_git_tag: str | None
    source_app_tree_sha256: str | None


Attestation = BuildAttestation | ReleaseAttestation


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_sha256(value: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise FirmwareReleasePolicyError(
            f"expected SHA-256 digest, got {value!r}"
        )
    return normalized


def _normalize_lane(lane: str) -> str:
    normalized = str(lane).strip().lower()
    if normalized not in LANES:
        raise FirmwareReleasePolicyError(f"unsupported firmware lane: {lane}")
    return normalized


def _normalize_device_role(device_role: str) -> str:
    normalized = str(device_role).strip().lower()
    if normalized not in DEVICE_ROLES:
        raise FirmwareReleasePolicyError(f"unsupported device role: {device_role}")
    return normalized


def _require_text(value: str, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise FirmwareReleasePolicyError(f"{field_name} is required")
    return normalized


def _read_json_mapping(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise FirmwareReleasePolicyError(f"attestation must be a JSON object: {path}")
    return payload


def _artifact_from_mapping(payload: Mapping[str, Any]) -> ArtifactAttestation:
    return ArtifactAttestation(
        path=_require_text(str(payload.get("path", "")), field_name="artifact.path"),
        sha256=_normalize_sha256(str(payload.get("sha256", ""))),
        size_bytes=int(payload.get("size_bytes", 0)),
        source_kind=str(payload.get("source_kind")).strip()
        if payload.get("source_kind") is not None
        else None,
    )


def _source_from_mapping(payload: Mapping[str, Any]) -> BuildSourceAttestation:
    patch_paths_raw = payload.get("firmware_patch_paths", [])
    if patch_paths_raw is None:
        patch_paths_raw = []
    if not isinstance(patch_paths_raw, list):
        raise FirmwareReleasePolicyError("source.firmware_patch_paths must be a list")
    patch_sha256s_raw = payload.get("firmware_patch_sha256s", [])
    if patch_sha256s_raw is None:
        patch_sha256s_raw = []
    if not isinstance(patch_sha256s_raw, list):
        raise FirmwareReleasePolicyError("source.firmware_patch_sha256s must be a list")
    patch_paths = tuple(_require_text(str(value), field_name="source.firmware_patch_paths[]") for value in patch_paths_raw)
    patch_sha256s = tuple(_normalize_sha256(str(value)) for value in patch_sha256s_raw)
    if len(patch_paths) != len(patch_sha256s):
        raise FirmwareReleasePolicyError(
            "source.firmware_patch_paths and source.firmware_patch_sha256s must have the same length"
        )
    return BuildSourceAttestation(
        firmware_root=_require_text(
            str(payload.get("firmware_root", "")),
            field_name="source.firmware_root",
        ),
        firmware_git_commit=_require_text(
            str(payload.get("firmware_git_commit", "")),
            field_name="source.firmware_git_commit",
        ),
        firmware_git_tag=str(payload.get("firmware_git_tag")).strip()
        if payload.get("firmware_git_tag") is not None
        else None,
        expected_firmware_revision=_require_text(
            str(payload.get("expected_firmware_revision", "")),
            field_name="source.expected_firmware_revision",
        ),
        app_root=_require_text(
            str(payload.get("app_root", "")),
            field_name="source.app_root",
        ),
        app_tree_sha256=_normalize_sha256(str(payload.get("app_tree_sha256", ""))),
        toolchain=_require_text(
            str(payload.get("toolchain", "")),
            field_name="source.toolchain",
        ),
        firmware_patch_paths=patch_paths,
        firmware_patch_sha256s=patch_sha256s,
    )


def _approval_from_mapping(payload: Mapping[str, Any]) -> PromotionApproval:
    evidence_raw = payload.get("validation_evidence")
    if not isinstance(evidence_raw, list) or not evidence_raw:
        raise FirmwareReleasePolicyError(
            "approval.validation_evidence must contain at least one entry"
        )
    evidence = tuple(_require_text(str(item), field_name="validation_evidence[]") for item in evidence_raw)
    return PromotionApproval(
        approved_by=_require_text(
            str(payload.get("approved_by", "")),
            field_name="approval.approved_by",
        ),
        reason=_require_text(
            str(payload.get("reason", "")),
            field_name="approval.reason",
        ),
        validation_evidence=evidence,
    )


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def compute_tree_sha256(
    root: Path,
    *,
    excluded_names: tuple[str, ...] = ("build", ".git", "__pycache__"),
) -> str:
    """Return a deterministic tree digest for one source directory."""

    root_path = Path(root).expanduser().resolve(strict=True)
    if not root_path.is_dir():
        raise FirmwareReleasePolicyError(f"app root is not a directory: {root_path}")
    files: list[Path] = []
    for path in root_path.rglob("*"):
        if path.is_dir() and path.name in excluded_names:
            continue
        if any(part in excluded_names for part in path.relative_to(root_path).parts):
            continue
        if path.is_file():
            files.append(path)
    if not files:
        raise FirmwareReleasePolicyError(f"app root has no tracked source files: {root_path}")
    digest = hashlib.sha256()
    for path in sorted(files):
        relative_path = path.relative_to(root_path).as_posix().encode("utf-8")
        digest.update(relative_path)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_artifact_attestation(
    artifact_path: Path,
    *,
    source_kind: str | None,
) -> ArtifactAttestation:
    """Fingerprint one concrete firmware artifact."""

    resolved_path = Path(artifact_path).expanduser().resolve(strict=True)
    if not resolved_path.is_file():
        raise FirmwareReleasePolicyError(f"artifact not found: {resolved_path}")
    return ArtifactAttestation(
        path=str(resolved_path),
        sha256=sha256_file(resolved_path),
        size_bytes=resolved_path.stat().st_size,
        source_kind=source_kind,
    )


def create_build_attestation(
    *,
    artifact_path: Path,
    firmware_root: Path,
    firmware_git_commit: str,
    firmware_git_tag: str | None,
    expected_firmware_revision: str,
    app_root: Path,
    app_tree_sha256: str,
    toolchain: str,
    firmware_patch_paths: tuple[Path, ...] = (),
) -> BuildAttestation:
    """Create one dev-lane build attestation from pinned source inputs."""

    return BuildAttestation(
        schema_version=SCHEMA_VERSION,
        kind=BUILD_ATTESTATION_KIND,
        app_name=APP_NAME,
        lane=LANE_DEV,
        created_at_utc=_utc_now_text(),
        artifact=build_artifact_attestation(artifact_path, source_kind="bin"),
        source=BuildSourceAttestation(
            firmware_root=str(Path(firmware_root).expanduser().resolve()),
            firmware_git_commit=_require_text(
                firmware_git_commit,
                field_name="firmware_git_commit",
            ),
            firmware_git_tag=str(firmware_git_tag).strip() or None,
            expected_firmware_revision=_require_text(
                expected_firmware_revision,
                field_name="expected_firmware_revision",
            ),
            app_root=str(Path(app_root).expanduser().resolve()),
            app_tree_sha256=_normalize_sha256(app_tree_sha256),
            toolchain=_require_text(toolchain, field_name="toolchain"),
            firmware_patch_paths=tuple(
                str(Path(path).expanduser().resolve()) for path in firmware_patch_paths
            ),
            firmware_patch_sha256s=tuple(
                sha256_file(Path(path).expanduser().resolve()) for path in firmware_patch_paths
            ),
        ),
    )


def create_release_attestation(
    *,
    artifact_path: Path,
    lane: str,
    release_id: str,
    source_attestation: Attestation,
    source_attestation_path: Path,
    approved_by: str,
    reason: str,
    validation_evidence: tuple[str, ...],
) -> ReleaseAttestation:
    """Create one bench/operator promotion attestation."""

    normalized_lane = _normalize_lane(lane)
    if normalized_lane not in {LANE_BENCH, LANE_OPERATOR}:
        raise FirmwareReleasePolicyError(
            f"promotion only supports {LANE_BENCH} or {LANE_OPERATOR}, got {lane}"
        )

    if normalized_lane == LANE_BENCH:
        if not isinstance(source_attestation, BuildAttestation):
            raise FirmwareReleasePolicyError(
                "bench promotion requires a build attestation as input"
            )
        if source_attestation.lane != LANE_DEV:
            raise FirmwareReleasePolicyError(
                f"bench promotion requires a dev build attestation, got {source_attestation.lane}"
            )
    else:
        if not isinstance(source_attestation, ReleaseAttestation):
            raise FirmwareReleasePolicyError(
                "operator promotion requires a bench release attestation as input"
            )
        if source_attestation.lane != LANE_BENCH:
            raise FirmwareReleasePolicyError(
                "operator promotion requires a bench release attestation"
            )
        if source_attestation.source.firmware_git_tag is None:
            raise FirmwareReleasePolicyError(
                "operator promotion requires a source build pinned to an exact Bitcraze firmware tag"
            )

    artifact = build_artifact_attestation(artifact_path, source_kind="bin")
    if artifact.sha256 != source_attestation.artifact.sha256:
        raise FirmwareReleasePolicyError(
            "promotion artifact does not match the source attestation SHA-256"
        )
    if artifact.size_bytes != source_attestation.artifact.size_bytes:
        raise FirmwareReleasePolicyError(
            "promotion artifact does not match the source attestation size"
        )

    normalized_release_id = _require_text(release_id, field_name="release_id")
    normalized_evidence = tuple(
        _require_text(item, field_name="validation_evidence[]")
        for item in validation_evidence
    )
    if not normalized_evidence:
        raise FirmwareReleasePolicyError(
            "promotion requires at least one validation evidence entry"
        )

    return ReleaseAttestation(
        schema_version=SCHEMA_VERSION,
        kind=RELEASE_ATTESTATION_KIND,
        app_name=APP_NAME,
        lane=normalized_lane,
        release_id=normalized_release_id,
        created_at_utc=_utc_now_text(),
        artifact=artifact,
        source=source_attestation.source,
        source_attestation_kind=source_attestation.kind,
        source_attestation_path=str(Path(source_attestation_path).expanduser().resolve()),
        source_attestation_sha256=sha256_file(source_attestation_path),
        approval=PromotionApproval(
            approved_by=_require_text(approved_by, field_name="approved_by"),
            reason=_require_text(reason, field_name="reason"),
            validation_evidence=normalized_evidence,
        ),
    )


def write_attestation(path: Path, attestation: Attestation) -> Path:
    """Persist one build or release attestation as JSON."""

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(attestation), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_attestation(path: Path) -> Attestation:
    """Load one build or release attestation from disk."""

    attestation_path = Path(path).expanduser().resolve(strict=True)
    payload = _read_json_mapping(attestation_path)
    schema_version = int(payload.get("schema_version", 0))
    if schema_version != SCHEMA_VERSION:
        raise FirmwareReleasePolicyError(
            f"unsupported attestation schema version {schema_version} in {attestation_path}"
        )
    kind = str(payload.get("kind", "")).strip()
    if kind == BUILD_ATTESTATION_KIND:
        return BuildAttestation(
            schema_version=schema_version,
            kind=kind,
            app_name=_require_text(
                str(payload.get("app_name", "")),
                field_name="app_name",
            ),
            lane=_normalize_lane(str(payload.get("lane", ""))),
            created_at_utc=_require_text(
                str(payload.get("created_at_utc", "")),
                field_name="created_at_utc",
            ),
            artifact=_artifact_from_mapping(
                payload.get("artifact", {})
                if isinstance(payload.get("artifact"), Mapping)
                else {}
            ),
            source=_source_from_mapping(
                payload.get("source", {})
                if isinstance(payload.get("source"), Mapping)
                else {}
            ),
        )
    if kind == RELEASE_ATTESTATION_KIND:
        return ReleaseAttestation(
            schema_version=schema_version,
            kind=kind,
            app_name=_require_text(
                str(payload.get("app_name", "")),
                field_name="app_name",
            ),
            lane=_normalize_lane(str(payload.get("lane", ""))),
            release_id=_require_text(
                str(payload.get("release_id", "")),
                field_name="release_id",
            ),
            created_at_utc=_require_text(
                str(payload.get("created_at_utc", "")),
                field_name="created_at_utc",
            ),
            artifact=_artifact_from_mapping(
                payload.get("artifact", {})
                if isinstance(payload.get("artifact"), Mapping)
                else {}
            ),
            source=_source_from_mapping(
                payload.get("source", {})
                if isinstance(payload.get("source"), Mapping)
                else {}
            ),
            source_attestation_kind=_require_text(
                str(payload.get("source_attestation_kind", "")),
                field_name="source_attestation_kind",
            ),
            source_attestation_path=_require_text(
                str(payload.get("source_attestation_path", "")),
                field_name="source_attestation_path",
            ),
            source_attestation_sha256=_normalize_sha256(
                str(payload.get("source_attestation_sha256", ""))
            ),
            approval=_approval_from_mapping(
                payload.get("approval", {})
                if isinstance(payload.get("approval"), Mapping)
                else {}
            ),
        )
    raise FirmwareReleasePolicyError(
        f"unsupported attestation kind {kind!r} in {attestation_path}"
    )


def authorize_flash_request(
    *,
    lane: str,
    device_role: str,
    artifact_path: Path,
    artifact_sha256: str,
    artifact_size_bytes: int,
    artifact_source_kind: str | None,
    artifact_release: str | None,
    artifact_repository: str | None,
    artifact_platform: str | None,
    artifact_target: str | None,
    artifact_firmware_type: str | None,
    attestation_path: Path | None,
) -> FlashAuthorization:
    """Fail closed unless the lane, artifact, and attestation all agree."""

    normalized_lane = _normalize_lane(lane)
    normalized_device_role = _normalize_device_role(device_role)
    normalized_artifact_sha = _normalize_sha256(artifact_sha256)
    normalized_artifact_path = Path(artifact_path).expanduser().resolve(strict=True)

    if normalized_lane == LANE_RECOVERY:
        if attestation_path is not None:
            raise FirmwareReleasePolicyError(
                "recovery lane does not accept a custom attestation; use the official Bitcraze release ZIP directly"
            )
        if artifact_source_kind not in {"zip", "manifest"}:
            raise FirmwareReleasePolicyError(
                "recovery lane requires an official Bitcraze release ZIP or manifest, not a raw .bin"
            )
        if (
            str(artifact_repository).strip() != "crazyflie-firmware"
            or str(artifact_platform).strip() != "cf2"
            or str(artifact_target).strip() != "stm32"
            or str(artifact_firmware_type).strip() != "fw"
        ):
            raise FirmwareReleasePolicyError(
                "recovery lane only allows official crazyflie-firmware cf2/stm32/fw artifacts"
            )
        if not str(artifact_release).strip():
            raise FirmwareReleasePolicyError(
                "recovery lane requires release metadata from the Bitcraze manifest"
            )
        return FlashAuthorization(
            lane=normalized_lane,
            device_role=normalized_device_role,
            attestation_kind=None,
            attestation_path=None,
            release_id=str(artifact_release).strip(),
            source_firmware_git_commit=None,
            source_firmware_git_tag=None,
            source_app_tree_sha256=None,
        )

    if artifact_source_kind != "bin":
        raise FirmwareReleasePolicyError(
            f"{normalized_lane} lane requires a raw .bin artifact produced from the promoted Twinr build"
        )

    if normalized_lane == LANE_DEV and normalized_device_role != DEVICE_ROLE_DEV:
        raise FirmwareReleasePolicyError(
            "dev lane may only flash dedicated dev devices"
        )
    if normalized_lane == LANE_BENCH and normalized_device_role != DEVICE_ROLE_BENCH:
        raise FirmwareReleasePolicyError(
            "bench lane may only flash dedicated bench devices"
        )
    if normalized_lane == LANE_OPERATOR and normalized_device_role != DEVICE_ROLE_OPERATOR:
        raise FirmwareReleasePolicyError(
            "operator lane may only flash declared operator devices"
        )

    if attestation_path is None:
        raise FirmwareReleasePolicyError(
            f"{normalized_lane} lane requires an attestation path"
        )

    loaded_attestation = load_attestation(attestation_path)
    if normalized_lane == LANE_DEV:
        if not isinstance(loaded_attestation, BuildAttestation):
            raise FirmwareReleasePolicyError(
                "dev lane requires a build attestation"
            )
        if loaded_attestation.lane != LANE_DEV:
            raise FirmwareReleasePolicyError(
                f"dev lane requires a dev build attestation, got {loaded_attestation.lane}"
            )
    else:
        if not isinstance(loaded_attestation, ReleaseAttestation):
            raise FirmwareReleasePolicyError(
                f"{normalized_lane} lane requires a promoted release attestation"
            )
        if loaded_attestation.lane != normalized_lane:
            raise FirmwareReleasePolicyError(
                f"{normalized_lane} lane requires a {normalized_lane} release attestation, got {loaded_attestation.lane}"
            )
        if loaded_attestation.source.firmware_git_tag is None and normalized_lane == LANE_OPERATOR:
            raise FirmwareReleasePolicyError(
                "operator lane requires a promoted build pinned to an exact Bitcraze firmware tag"
            )

    if loaded_attestation.app_name != APP_NAME:
        raise FirmwareReleasePolicyError(
            f"unexpected app_name in attestation: {loaded_attestation.app_name}"
        )
    if loaded_attestation.artifact.sha256 != normalized_artifact_sha:
        raise FirmwareReleasePolicyError(
            "attestation SHA-256 does not match the requested artifact"
        )
    if loaded_attestation.artifact.size_bytes != int(artifact_size_bytes):
        raise FirmwareReleasePolicyError(
            "attestation artifact size does not match the requested artifact"
        )
    if Path(loaded_attestation.artifact.path).expanduser().resolve() != normalized_artifact_path:
        raise FirmwareReleasePolicyError(
            "attestation artifact path does not match the requested artifact path"
        )

    return FlashAuthorization(
        lane=normalized_lane,
        device_role=normalized_device_role,
        attestation_kind=loaded_attestation.kind,
        attestation_path=str(Path(attestation_path).expanduser().resolve()),
        release_id=loaded_attestation.release_id
        if isinstance(loaded_attestation, ReleaseAttestation)
        else None,
        source_firmware_git_commit=loaded_attestation.source.firmware_git_commit,
        source_firmware_git_tag=loaded_attestation.source.firmware_git_tag,
        source_app_tree_sha256=loaded_attestation.source.app_tree_sha256,
    )
