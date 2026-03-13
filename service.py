from __future__ import annotations

from dataclasses import dataclass
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.runtime_state import RuntimeSnapshot, RuntimeSnapshotStore
from twinr.display.waveshare_v2 import WaveshareEPD4In2V2


def _default_emit(line: str) -> None:
    print(line, flush=True)


@dataclass(slots=True)
class TwinrStatusDisplayLoop:
    config: TwinrConfig
    display: WaveshareEPD4In2V2
    snapshot_store: RuntimeSnapshotStore
    emit: callable = _default_emit
    sleep: callable = time.sleep

    @classmethod
    def from_config(
        cls,
        config: TwinrConfig,
        *,
        emit=None,
        sleep=time.sleep,
    ) -> "TwinrStatusDisplayLoop":
        return cls(
            config=config,
            display=WaveshareEPD4In2V2.from_config(config),
            snapshot_store=RuntimeSnapshotStore(config.runtime_state_path),
            emit=emit or _default_emit,
            sleep=sleep,
        )

    def run(self, *, duration_s: float | None = None, max_cycles: int | None = None) -> int:
        started_at = time.monotonic()
        last_signature: tuple[str, int] | None = None
        cycles = 0
        while True:
            if duration_s is not None and (time.monotonic() - started_at) >= duration_s:
                return 0
            if max_cycles is not None and cycles >= max_cycles:
                return 0

            snapshot = self.snapshot_store.load()
            frame = self._animation_frame(snapshot.status)
            signature = (snapshot.status, frame)
            if signature != last_signature:
                headline, details = self._build_status_content(snapshot)
                self.display.show_status(
                    snapshot.status,
                    headline=headline,
                    details=details,
                    animation_frame=frame,
                )
                self.emit(f"display_status={snapshot.status}")
                last_signature = signature
            cycles += 1
            self.sleep(self.config.display_poll_interval_s)

    def _build_status_content(self, snapshot: RuntimeSnapshot) -> tuple[str, tuple[str, ...]]:
        status = snapshot.status.lower()
        if status == "waiting":
            return "Waiting", ()
        if status == "listening":
            return "Listening", ()
        if status == "processing":
            return "Processing", ()
        if status == "answering":
            return "Answering", ()
        if status == "printing":
            return "Printing", ()
        if status == "error":
            return "Error", ()
        return snapshot.status.title(), ()

    def _detail_lines(self, *values: str | None) -> tuple[str, ...]:
        lines: list[str] = []
        for value in values:
            if not value:
                continue
            compact = " ".join(value.split())
            if not compact:
                continue
            while compact and len(lines) < 4:
                lines.append(compact[:32])
                compact = compact[32:].lstrip()
        return tuple(lines[:4])

    def _animation_frame(self, status: str) -> int:
        normalized = status.lower()
        now = time.monotonic()
        if normalized == "waiting":
            return int(now / 12.0) % 8
        if normalized in {"answering", "listening"}:
            return int(now / 2.0) % 4
        if normalized in {"processing", "printing"}:
            return int(now / 4.0) % 4
        return 0
