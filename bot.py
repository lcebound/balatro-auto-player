"""Small Airtest runner for the local, windowed Balatro game.

The first milestone is a reliable observe -> act -> observe loop.  Every
observation and action is recorded so later strategy changes can be audited.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import psutil
import cv2
import win32con
import win32gui
import win32process
from airtest.core.api import connect_device
from recovery import json_default, ObservationError, DesktopUnavailableError


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
VISION_SIZE = (2064, 1191)


def write_observation_png(path, image):
    """Lossless evidence without Pillow's expensive optimize=True pass."""
    ok, encoded = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise ValueError('Cannot encode observation PNG')
    Path(path).write_bytes(encoded.tobytes())


def find_balatro_window() -> int:
    matches: list[int] = []

    def visit(hwnd: int, _: object) -> None:
        if not win32gui.IsWindowVisible(hwnd):
            return
        if win32gui.GetWindowText(hwnd) != "Balatro":
            return
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        try:
            if psutil.Process(pid).name().lower() == "balatro.exe":
                matches.append(hwnd)
        except psutil.Error:
            pass

    win32gui.EnumWindows(visit, None)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one visible Balatro window, found {len(matches)}")
    return matches[0]


class Session:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "events.jsonl"
        self.hwnd = find_balatro_window()
        # The game may restore a different window size after a restart.
        # Vision coordinates and card templates use this calibrated size.
        # Win32 uses logical pixels here; this desktop is at 150% scaling.
        win32gui.MoveWindow(self.hwnd, 0, 0, 1376, 794, True)
        time.sleep(0.5)
        self.device = connect_device(f"Windows:///{self.hwnd}")
        self.size: tuple[int, int] | None = None
        self.input_revision = 0
        self._last_observation = None
        self.counter = max((int(p.name.split('-')[0]) for p in self.run_dir.glob('*.png')
                            if p.name.split('-')[0].isdigit()), default=0)
        journal = self.run_dir / 'pending-action.json'
        self.pending_action = None
        if journal.exists():
            pending = json.loads(journal.read_text(encoding='utf-8'))
            if pending.get('kind'):
                self.pending_action = pending

    def log(self, event: str, **data: object) -> None:
        record = {"time": datetime.now().astimezone().isoformat(), "event": event, **data}
        serialized = json.dumps(record, ensure_ascii=False, default=json_default)
        with self.log_path.open("a", encoding="utf-8") as output:
            output.write(serialized + "\n")
        try:
            print(serialized, flush=True)
        except (BrokenPipeError, UnicodeEncodeError):
            # The durable UTF-8 log is authoritative when IDE output closes.
            pass

    def observe(self, label: str):
        self.focus()
        self.counter += 1
        path = self.run_dir / f"{self.counter:04d}-{label}.png"
        started = time.perf_counter()
        try:
            image = self.device.snapshot()
            if image is None or image.ndim < 2 or not image.size:
                raise ValueError('Empty window snapshot')
            captured = time.perf_counter()
            write_observation_png(path, image)
            saved = time.perf_counter()
        except Exception as error:
            raise ObservationError(f'Cannot capture Balatro: {error}') from error
        height, width = image.shape[:2]
        if self.size is None:
            self.size = (width, height)
        elif self.size != (width, height):
            raise ObservationError(f"Window size changed from {self.size} to {(width, height)}")
        if (width, height) != VISION_SIZE:
            image = cv2.resize(image, VISION_SIZE, interpolation=cv2.INTER_NEAREST)
        self.log("observe", label=label, screenshot=str(path), width=width, height=height,
                 vision_width=VISION_SIZE[0], vision_height=VISION_SIZE[1],
                 capture_seconds=round(captured-started,4),save_seconds=round(saved-captured,4))
        self._last_observation = (image, self.input_revision, time.monotonic(),
                                  str(path), width, height)
        return image

    def reuse_observation(self, label, image):
        """Reuse one recent immutable frame only until the next UI input.

        This never substitutes yesterday's card/state cache for live pixels.
        The alias retains the exact source screenshot for later diagnosis.
        """
        previous = self._last_observation
        if previous is not None:
            frame, revision, captured, path, width, height = previous
            age = time.monotonic() - captured
            if (image is frame and revision == self.input_revision
                    and 0 <= age <= 3):
                self.focus()
                if revision == self.input_revision:
                    self.log('observe', label=label, screenshot=path, width=width, height=height,
                             vision_width=VISION_SIZE[0], vision_height=VISION_SIZE[1],
                             reused=True, age_seconds=round(age, 3))
                    return frame
        return self.observe(label)

    def focus(self) -> None:
        if not win32gui.IsWindow(self.hwnd):
            raise ObservationError('Balatro window was closed; game state must be reconciled before resuming')
        try:
            if (win32gui.GetForegroundWindow() == self.hwnd
                    and not win32gui.IsIconic(self.hwnd)):
                return
            self.input_revision += 1
            win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)
        except Exception as error:
            raise DesktopUnavailableError('Balatro desktop is unavailable') from error
        for attempt in range(3):
            try:
                win32gui.SetForegroundWindow(self.hwnd)
            except Exception:
                if attempt == 2:
                    raise DesktopUnavailableError('Cannot focus Balatro')
            time.sleep(0.25)
            if win32gui.GetForegroundWindow() == self.hwnd:
                break
        if win32gui.GetForegroundWindow() != self.hwnd:
            raise DesktopUnavailableError("Balatro is not the foreground window; waiting for desktop access")

    def tap(self, x: float, y: float, reason: str) -> None:
        if self.size is None:
            raise RuntimeError("Observe the window before tapping")
        if not (0 < x < 1 and 0 < y < 1):
            raise ValueError("Tap coordinates must be normalized to 0..1")
        self.focus()
        self.input_revision += 1
        self.log("tap", x=x, y=y, reason=reason)
        try:
            self.device.touch((x, y))
        except Exception as error:
            raise ObservationError(f'Click needs outcome confirmation: {error}') from error
        time.sleep(0.6)

    def hover(self, x: float, y: float, reason: str) -> None:
        if self.size is None or not (0 < x < 1 and 0 < y < 1):
            raise ValueError("Observe first and supply normalized hover coordinates")
        self.focus()
        self.input_revision += 1
        self.log("hover", x=x, y=y, reason=reason)
        # mouse_move unlike touch/swipe does not convert fractional coordinates.
        try:
            self.device.mouse_move((round(x * self.size[0]), round(y * self.size[1])))
        except Exception as error:
            raise ObservationError(f'Cannot inspect tooltip: {error}') from error
        time.sleep(.5)

    def drag(self, start, end, reason):
        self.focus()
        self.input_revision += 1
        self.log("drag", start=start, end=end, reason=reason)
        try:
            self.device.swipe(tuple(start), tuple(end), duration=.65, steps=12)
        except Exception as error:
            raise ObservationError(f'Drag needs position recollection: {error}') from error
        time.sleep(.5)

    def restore_geometry(self):
        self.focus()
        self.input_revision += 1
        win32gui.MoveWindow(self.hwnd, 0, 0, 1376, 794, True)
        self.size = None
        time.sleep(.5)
        self.log('window_geometry_restored')


def main() -> None:
    parser = argparse.ArgumentParser(description="Balatro Airtest runner")
    parser.add_argument("command", choices=["probe"])
    args = parser.parse_args()
    run_dir = RUNS / datetime.now().strftime("%Y%m%d-%H%M%S")
    session = Session(run_dir)
    session.observe(args.command)
    print(f"Saved log and screenshot in {run_dir}")


if __name__ == "__main__":
    main()
