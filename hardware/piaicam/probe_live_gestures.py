#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///

"""Capture a bounded live gesture probe from Twinr's Pi AI-camera path.

Purpose
-------
Record the raw MediaPipe pose/fine-hand result that the local AI-camera adapter
cached for the current frame *and* the final composed Twinr observation that
runtime layers consume. This gives one calibration/debug view that answers
whether a missing gesture was lost in the recognizer itself or only later in
surface stabilization / HDMI acknowledgement policy.

Usage
-----
Command-line invocation examples::

    PYTHONPATH=src python3 hardware/piaicam/probe_live_gestures.py --env-file /twinr/.env
    PYTHONPATH=src python3 hardware/piaicam/probe_live_gestures.py --duration-s 8 --interval-s 0.2
    PYTHONPATH=src python3 hardware/piaicam/probe_live_gestures.py --output /tmp/twinr_gesture_probe.jsonl

Inputs
------
- ``--env-file`` Twinr env file used to build the local AI-camera adapter
- ``--duration-s`` bounded probe duration
- ``--interval-s`` bounded capture cadence between observations
- ``--output`` optional JSONL path for the collected probe lines

Outputs
-------
- Prints one JSON line per sampled frame
- Optionally writes the same JSON lines to ``--output``
- Exit code 0 when the bounded probe completes, 1 on setup failure

Notes
-----
This script is intentionally diagnostic-only. It reads the adapter's cached
raw ``_last_pose_result`` after each observation so operator calibration can
compare raw gesture inference with the final observation contract without
teaching that debug path to the production runtime.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.hardware.camera_ai.adapter import LocalAICameraAdapter, PoseResult
from twinr.hardware.camera_ai.models import AICameraObservation


@dataclass(frozen=True, slots=True)
class GestureProbeSample:
    """Describe one raw-plus-final gesture sample from the local camera path."""

    index: int
    elapsed_s: float
    person_count: int
    final_coarse: str
    final_coarse_conf: float | None
    final_fine: str
    final_fine_conf: float | None
    raw_coarse: str | None
    raw_coarse_conf: float | None
    raw_fine: str | None
    raw_fine_conf: float | None
    hand_near: bool
    showing_intent: bool | None
    camera_error: str | None


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the bounded live gesture probe."""

    parser = argparse.ArgumentParser(description="Capture bounded live gesture probe samples")
    parser.add_argument("--env-file", default=".env", help="Twinr env file path")
    parser.add_argument("--duration-s", type=float, default=12.0, help="Total bounded probe duration")
    parser.add_argument("--interval-s", type=float, default=0.2, help="Delay between observations")
    parser.add_argument("--output", type=Path, help="Optional JSONL output path")
    return parser


def _coerce_duration(value: float) -> float:
    return max(0.5, min(60.0, float(value)))


def _coerce_interval(value: float) -> float:
    return max(0.05, min(2.0, float(value)))


def _build_sample(
    *,
    index: int,
    started_at: float,
    observation: AICameraObservation,
    raw_pose: PoseResult | None,
) -> GestureProbeSample:
    """Convert one adapter observation plus cached raw pose into JSON-safe fields."""

    return GestureProbeSample(
        index=index,
        elapsed_s=round(time.time() - started_at, 3),
        person_count=int(observation.person_count),
        final_coarse=observation.gesture_event.value,
        final_coarse_conf=observation.gesture_confidence,
        final_fine=observation.fine_hand_gesture.value,
        final_fine_conf=observation.fine_hand_gesture_confidence,
        raw_coarse=(None if raw_pose is None else raw_pose.gesture_event.value),
        raw_coarse_conf=(None if raw_pose is None else raw_pose.gesture_confidence),
        raw_fine=(None if raw_pose is None else raw_pose.fine_hand_gesture.value),
        raw_fine_conf=(None if raw_pose is None else raw_pose.fine_hand_gesture_confidence),
        hand_near=bool(observation.hand_or_object_near_camera),
        showing_intent=observation.showing_intent_likely,
        camera_error=observation.camera_error,
    )


def main() -> int:
    """Run the bounded gesture probe and emit JSONL samples."""

    args = build_parser().parse_args()
    duration_s = _coerce_duration(args.duration_s)
    interval_s = _coerce_interval(args.interval_s)
    output_path = args.output.resolve() if args.output else None

    config = TwinrConfig.from_env(args.env_file)
    adapter = LocalAICameraAdapter.from_config(config)
    started_at = time.time()
    samples: list[str] = []
    index = 0
    try:
        while (time.time() - started_at) < duration_s:
            observation = adapter.observe()
            raw_pose = adapter._last_pose_result
            line = json.dumps(asdict(_build_sample(
                index=index,
                started_at=started_at,
                observation=observation,
                raw_pose=raw_pose,
            )))
            print(line, flush=True)
            samples.append(line)
            index += 1
            time.sleep(interval_s)
    finally:
        adapter.close()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(samples) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
