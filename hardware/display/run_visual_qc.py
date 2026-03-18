#!/usr/bin/env python3
"""Run screenshot-backed visual QC for Twinr's visible HDMI surface.

Purpose
-------
Capture a deterministic HDMI scene set from the real visible Twinr surface,
compute image-diff metrics, and store the screenshots plus a durable report
under the repository report artifacts.

Usage
-----
Command-line invocation examples::

    python hardware/display/run_visual_qc.py --env-file .env
    python hardware/display/run_visual_qc.py --env-file /twinr/.env --image-path /tmp/family_photo.png

Inputs
------
- ``--env-file`` path to the Twinr environment file with HDMI settings
- ``--image-path`` optional image used for the fullscreen image scene
- ``--report-title`` optional title override for the generated report
- ``--workdir`` optional scratch directory for raw screenshots before report import
- ``--keep-workdir`` preserve the scratch directory after the report is created

Outputs
-------
- Captures the visible HDMI Wayland surface with ``grim``
- Creates a durable report under ``artifacts/reports/report/<RPT...>/`` when the
  repo report tool is available, otherwise under
  ``artifacts/reports/display_visual_qc/<RUN_ID>/``
- Prints the report id, markdown path, and key QC metrics
- Exit code 0 on success
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.display.visual_qc import (
    DisplayVisualQcRunner,
    build_visual_qc_report_markdown,
    build_visual_qc_report_payload,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the display visual-QC script."""

    parser = argparse.ArgumentParser(description="Run Twinr HDMI visual QC and persist a report-backed artifact bundle")
    parser.add_argument("--env-file", default=Path(__file__).resolve().parents[2] / ".env")
    parser.add_argument("--image-path", default=None, help="Optional image path for the fullscreen image scene.")
    parser.add_argument("--report-title", default=None, help="Optional report title override.")
    parser.add_argument("--workdir", default=None, help="Optional scratch directory for raw screenshots before report import.")
    parser.add_argument("--keep-workdir", action="store_true", help="Preserve the scratch workdir after the report is created.")
    return parser


def _run_json(command: list[str], *, cwd: Path) -> dict[str, object]:
    """Run one JSON-emitting repo tool command and return its decoded payload."""

    completed = subprocess.run(
        command,
        cwd=str(cwd),
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(completed.stdout.strip() or "{}")


def _report_tool_path(repo_root: Path) -> Path:
    """Return the repo-local report CLI path."""

    tool_path = repo_root / "agentic_tools" / "report"
    if not tool_path.exists():
        raise RuntimeError(f"Report CLI was not found at {tool_path}")
    return tool_path


def _report_command(report_tool: Path, *args: str) -> list[str]:
    """Build one report-tool command pinned to the current Python interpreter."""

    return [sys.executable, str(report_tool), *args]


def _utc_report_stamp() -> str:
    """Return one UTC timestamp for fallback artifact bundle ids."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _compact_error(exc: BaseException) -> str:
    """Normalize one exception into a short single-line diagnostic string."""

    text = " ".join(str(exc).split())
    return text or exc.__class__.__name__


def _write_filesystem_report_bundle(
    *,
    repo_root: Path,
    report_payload: dict[str, object],
    report_markdown: str,
    attachments: list[str],
) -> dict[str, object]:
    """Persist one report-like artifact bundle without the report tool."""

    report_id = f"display_visual_qc_{_utc_report_stamp()}"
    bundle_root = repo_root / "artifacts" / "reports" / "display_visual_qc" / report_id
    assets_root = bundle_root / "assets"
    assets_root.mkdir(parents=True, exist_ok=True)
    copied_assets: list[str] = []
    for attachment in attachments:
        src = Path(attachment).resolve()
        dst = assets_root / src.name
        if dst != src:
            shutil.copyfile(src, dst)
        copied_assets.append(str(dst.relative_to(repo_root)))
    md_path = bundle_root / "report.md"
    md_path.write_text(report_markdown, encoding="utf-8")
    data_path = bundle_root / "report_data.json"
    data_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "report_backend": "filesystem",
        "report_id": report_id,
        "md_rel_path": str(md_path.relative_to(repo_root)),
        "assets_rel_paths": copied_assets,
        "report_data_rel_path": str(data_path.relative_to(repo_root)),
    }


def main() -> int:
    """Execute the HDMI visual QC workflow and persist its report artifacts."""

    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config = TwinrConfig.from_env(Path(args.env_file))
    if config.display_driver != "hdmi_wayland":
        raise RuntimeError(
            f"Display visual QC requires TWINR_DISPLAY_DRIVER=hdmi_wayland, got {config.display_driver!r}."
        )

    workdir_context = None
    if args.workdir:
        workdir = Path(args.workdir).expanduser().resolve()
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        (repo_root / "artifacts").mkdir(parents=True, exist_ok=True)
        workdir_context = tempfile.TemporaryDirectory(prefix="display_visual_qc_", dir=repo_root / "artifacts")
        workdir = Path(workdir_context.name).resolve()

    try:
        runner = DisplayVisualQcRunner(config)
        result = runner.run(workdir, emit=print, image_path=args.image_path)
        report_markdown = build_visual_qc_report_markdown(result)
        report_payload = build_visual_qc_report_payload(
            result,
            title=args.report_title,
            task_id="808cb48943b5",
        )
        report_md_path = workdir / "report.md"
        report_md_path.write_text(report_markdown, encoding="utf-8")
        report_data_path = workdir / "report_data.json"
        report_data_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        attachment_paths = [str(path) for path in result.attachment_paths()]
        try:
            report_tool = _report_tool_path(repo_root)
            _run_json(_report_command(report_tool, "init"), cwd=repo_root)
            create_command = _report_command(
                report_tool,
                "create",
                "--data-file",
                str(report_data_path),
                "--body-md-file",
                str(report_md_path),
            )
            for attachment_path in attachment_paths:
                create_command.extend(["--attach", attachment_path])
            create_payload = _run_json(create_command, cwd=repo_root)
            report_id = str(create_payload.get("report_id") or "").strip()
            if not report_id:
                raise RuntimeError(f"Report creation did not return a report_id: {create_payload}")
            _run_json(_report_command(report_tool, "finalize", "--id", report_id), cwd=repo_root)
            report_backend = "agentic_tools"
        except Exception as exc:
            create_payload = _write_filesystem_report_bundle(
                repo_root=repo_root,
                report_payload=report_payload,
                report_markdown=report_markdown,
                attachments=attachment_paths,
            )
            report_id = str(create_payload.get("report_id") or "").strip()
            report_backend = "filesystem"
            print(f"report_warning={_compact_error(exc)}")

        print("display_visual_qc=ok")
        print(f"report_backend={report_backend}")
        print(f"report_id={report_id}")
        print(f"report_md={create_payload.get('md_rel_path', '')}")
        print(f"report_assets={','.join(create_payload.get('assets_rel_paths', []))}")
        print(f"captures_total={len(result.captures)}")
        print(f"transition_pairs_total={len(result.diffs)}")
        for diff in result.diffs:
            print(
                "transition="
                f"{diff.from_key}:{diff.to_key}"
                f" changed_pixels={diff.changed_pixels}"
                f" changed_ratio={diff.changed_ratio:.6f}"
            )
        return 0
    finally:
        if workdir_context is not None and not args.keep_workdir:
            workdir_context.cleanup()
        elif args.keep_workdir:
            print(f"visual_qc_workdir={workdir}")


if __name__ == "__main__":
    raise SystemExit(main())
