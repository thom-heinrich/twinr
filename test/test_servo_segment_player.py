from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.servo_segment_player import BoundedServoPulseSegmentPlayer


class FakeServoPulseWriter:
    def __init__(self) -> None:
        self.writes: list[tuple[str, int, int]] = []
        self.disables: list[tuple[str, int]] = []

    def write(self, *, gpio_chip: str, gpio: int, pulse_width_us: int) -> None:
        self.writes.append((gpio_chip, gpio, pulse_width_us))

    def disable(self, *, gpio_chip: str, gpio: int) -> None:
        self.disables.append((gpio_chip, gpio))


class ImmediateThread:
    def __init__(self, target) -> None:
        self._target = target

    def start(self) -> None:
        self._target()


class ManualThread:
    def __init__(self, target) -> None:
        self._target = target

    def start(self) -> None:
        return


class BoundedServoPulseSegmentPlayerTests(unittest.TestCase):
    def test_start_segment_writes_then_reports_completion_after_exact_wait(self) -> None:
        writer = FakeServoPulseWriter()
        waits: list[float] = []
        def immediate_wait(_cancel_event, timeout_s: float) -> bool:
            waits.append(timeout_s)
            return False
        player = BoundedServoPulseSegmentPlayer(
            pulse_writer=writer,
            gpio_chip="gpiochip0",
            gpio=18,
            monotonic_fn=lambda: 10.0,
            wait_fn=immediate_wait,
            thread_factory=lambda target: ImmediateThread(target),
        )

        playback = player.start_segment(pulse_width_us=1610, duration_s=0.8)
        completion = player.consume_completion()

        self.assertEqual(playback.pulse_width_us, 1610)
        self.assertAlmostEqual(playback.duration_s, 0.8)
        self.assertEqual(writer.writes, [("gpiochip0", 18, 1610)])
        self.assertEqual(writer.disables, [("gpiochip0", 18)])
        self.assertEqual(len(waits), 1)
        self.assertAlmostEqual(waits[0], 0.8)
        self.assertIsNotNone(completion)
        if completion is None:
            self.fail("expected bounded segment completion")
        self.assertIsNone(completion.error)
        self.assertEqual(completion.playback, playback)
        self.assertIsNone(player.active_segment())

    def test_cancel_clears_active_segment_without_emitting_completion(self) -> None:
        writer = FakeServoPulseWriter()
        player = BoundedServoPulseSegmentPlayer(
            pulse_writer=writer,
            gpio_chip="gpiochip0",
            gpio=18,
            monotonic_fn=lambda: 10.0,
            wait_fn=lambda cancel_event, timeout_s: cancel_event.wait(0.0),
            thread_factory=lambda target: ManualThread(target),
        )

        playback = player.start_segment(pulse_width_us=1550, duration_s=0.8)
        cancelled = player.cancel()

        self.assertEqual(cancelled, playback)
        self.assertEqual(writer.writes, [("gpiochip0", 18, 1550)])
        self.assertEqual(writer.disables, [])
        self.assertIsNone(player.active_segment())
        self.assertIsNone(player.consume_completion())
