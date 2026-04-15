#!/usr/bin/env python3
"""Promote one attested Crazyflie failsafe artifact into bench/operator lanes.

Purpose
-------
Split the firmware workflow into explicit lanes. A local build attestation may
be promoted into the `bench` lane after validation, and only a bench release may
be promoted into the `operator` lane. The script never rebuilds or reflashes
anything; it only attests that one existing binary is allowed into a stricter
lane.

Usage
-----
Command-line examples::

    python hardware/bitcraze/promote_on_device_failsafe_release.py \
        --artifact hardware/bitcraze/twinr_on_device_failsafe/build/twinr_on_device_failsafe.bin \
        --source-attestation hardware/bitcraze/twinr_on_device_failsafe/build/twinr_on_device_failsafe.build-attestation.json \
        --lane bench \
        --release-id twinr-failsafe-bench-2026-04-11-r1 \
        --approved-by thh \
        --reason "bounded bench validation passed" \
        --validation-evidence test:test_bitcraze_flash_on_device_failsafe.py

Inputs
------
- Existing firmware artifact path
- Existing build or bench-release attestation
- Target promotion lane (`bench` or `operator`)
- Approval metadata and at least one validation evidence entry

Outputs
-------
- A promoted release attestation JSON on disk
- Exit code `0` on success, `1` on validation or write failure
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from firmware_release_policy import (  # noqa: E402
    ArtifactAttestation,
    BuildSourceAttestation,
    LANE_BENCH,
    LANE_OPERATOR,
    create_release_attestation,
    load_attestation,
    write_attestation,
)


DEFAULT_ARTIFACT = (
    SCRIPT_DIR / "twinr_on_device_failsafe" / "build" / "twinr_on_device_failsafe.bin"
)
DEFAULT_SOURCE_ATTESTATION = (
    SCRIPT_DIR
    / "twinr_on_device_failsafe"
    / "build"
    / "twinr_on_device_failsafe.build-attestation.json"
)


@dataclass(frozen=True)
class PromotionReport:
    """Summarize one completed release promotion."""

    output_path: str
    lane: str
    release_id: str
    artifact: ArtifactAttestation
    source: BuildSourceAttestation
    source_attestation_path: str
    source_attestation_kind: str


def _default_output_path(artifact_path: Path, lane: str, release_id: str) -> Path:
    artifact = Path(artifact_path).expanduser().resolve()
    return artifact.with_name(f"{artifact.stem}.{lane}.{release_id}.release.json")


def promote_on_device_failsafe_release(
    *,
    artifact_path: Path,
    source_attestation_path: Path,
    lane: str,
    release_id: str,
    approved_by: str,
    reason: str,
    validation_evidence: tuple[str, ...],
    output_path: Path | None = None,
) -> PromotionReport:
    """Create one promoted bench/operator release attestation."""

    loaded_attestation = load_attestation(source_attestation_path)
    release_attestation = create_release_attestation(
        artifact_path=artifact_path,
        lane=lane,
        release_id=release_id,
        source_attestation=loaded_attestation,
        source_attestation_path=source_attestation_path,
        approved_by=approved_by,
        reason=reason,
        validation_evidence=validation_evidence,
    )
    destination_path = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else _default_output_path(artifact_path, release_attestation.lane, release_attestation.release_id)
    )
    written_path = write_attestation(destination_path, release_attestation)
    return PromotionReport(
        output_path=str(written_path),
        lane=release_attestation.lane,
        release_id=release_attestation.release_id,
        artifact=release_attestation.artifact,
        source=release_attestation.source,
        source_attestation_path=release_attestation.source_attestation_path,
        source_attestation_kind=release_attestation.source_attestation_kind,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Promote one attested Crazyflie failsafe artifact into the bench or operator lane."
    )
    parser.add_argument(
        "--artifact",
        default=str(DEFAULT_ARTIFACT),
        help=f"Binary artifact to promote (default: {DEFAULT_ARTIFACT}).",
    )
    parser.add_argument(
        "--source-attestation",
        default=str(DEFAULT_SOURCE_ATTESTATION),
        help=(
            "Existing build or bench-release attestation to promote from "
            f"(default: {DEFAULT_SOURCE_ATTESTATION})."
        ),
    )
    parser.add_argument(
        "--lane",
        required=True,
        choices=(LANE_BENCH, LANE_OPERATOR),
        help="Target promotion lane.",
    )
    parser.add_argument(
        "--release-id",
        required=True,
        help="Stable release identifier written into the promoted attestation.",
    )
    parser.add_argument(
        "--approved-by",
        required=True,
        help="Human/operator identifier approving the promotion.",
    )
    parser.add_argument(
        "--reason",
        required=True,
        help="Short approval reason for the promoted release.",
    )
    parser.add_argument(
        "--validation-evidence",
        action="append",
        default=[],
        help="Repeatable proof token (test path, report id, task id, etc.).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Destination JSON path. Defaults to <artifact>.<lane>.<release-id>.release.json.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the promotion report as JSON.")
    return parser


def main() -> int:
    """Parse args, create one promoted release attestation, and print a report."""

    args = _build_parser().parse_args()
    report = promote_on_device_failsafe_release(
        artifact_path=Path(str(args.artifact).strip()),
        source_attestation_path=Path(str(args.source_attestation).strip()),
        lane=str(args.lane).strip(),
        release_id=str(args.release_id).strip(),
        approved_by=str(args.approved_by).strip(),
        reason=str(args.reason).strip(),
        validation_evidence=tuple(
            str(item).strip() for item in args.validation_evidence if str(item).strip()
        ),
        output_path=None if args.output is None else Path(str(args.output).strip()),
    )
    if args.json:
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
    else:
        print(f"output_path={report.output_path}")
        print(f"lane={report.lane}")
        print(f"release_id={report.release_id}")
        print(f"artifact.sha256={report.artifact.sha256}")
        print(f"artifact.size_bytes={report.artifact.size_bytes}")
        print(f"source_attestation_kind={report.source_attestation_kind}")
        print(f"source_attestation_path={report.source_attestation_path}")
        print(f"source.firmware_git_commit={report.source.firmware_git_commit}")
        if report.source.firmware_git_tag is not None:
            print(f"source.firmware_git_tag={report.source.firmware_git_tag}")
        print(f"source.app_tree_sha256={report.source.app_tree_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
