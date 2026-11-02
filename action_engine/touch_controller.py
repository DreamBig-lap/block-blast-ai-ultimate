"""ADB Touch Controller for BLOCK BLAST AI ULTIMATE.

Provides reliable low-level touch actuation over ADB (tap, swipe/drag, keyevent)
with timeout protection, dry-run mode, and detailed execution logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
import subprocess
import time
from typing import Optional, Sequence

logger = logging.getLogger(__name__)


class TouchControllerError(Exception):
    """Raised when an ADB touch command fails or times out."""
    pass


@dataclass
class TouchRecord:
    """Audit log entry for an executed touch action."""
    action_type: str
    params: dict[str, int | str]
    timestamp: float
    duration_ms: float
    success: bool
    returncode: int
    error_message: Optional[str] = None


class TouchController:
    """Manages low-level touch input injection on the target Android device via ADB."""

    def __init__(
        self,
        adb_path: str = "adb",
        device_id: Optional[str] = None,
        timeout: float = 5.0,
        dry_run: bool = False,
    ) -> None:
        self.adb_path = adb_path
        self.device_id = device_id
        self.timeout = timeout
        self.dry_run = dry_run
        self.history: list[TouchRecord] = []

    def _build_adb_cmd(self, shell_args: Sequence[str]) -> list[str]:
        cmd = [self.adb_path]
        if self.device_id:
            cmd.extend(["-s", self.device_id])
        cmd.append("shell")
        cmd.extend(shell_args)
        return cmd

    def _execute_shell(self, shell_args: Sequence[str], action_name: str, params: dict) -> bool:
        """Execute an ADB shell input command with error checking and logging."""
        t0 = time.perf_counter()

        if self.dry_run:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            rec = TouchRecord(
                action_type=action_name,
                params=params,
                timestamp=time.time(),
                duration_ms=elapsed_ms,
                success=True,
                returncode=0,
                error_message="DRY_RUN",
            )
            self.history.append(rec)
            logger.debug("[DRY RUN] %s: %s", action_name, params)
            return True

        if not os.path.isfile(self.adb_path):
            raise TouchControllerError(f"ADB binary not found at: {self.adb_path}")

        full_cmd = self._build_adb_cmd(shell_args)

        try:
            res = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            success = (res.returncode == 0)
            err_msg = res.stderr.strip() if res.stderr else None

            rec = TouchRecord(
                action_type=action_name,
                params=params,
                timestamp=time.time(),
                duration_ms=elapsed_ms,
                success=success,
                returncode=res.returncode,
                error_message=err_msg,
            )
            self.history.append(rec)

            if not success:
                logger.error("ADB touch failed: %s (code %d: %s)", action_name, res.returncode, err_msg)
                raise TouchControllerError(f"ADB {action_name} failed (code {res.returncode}): {err_msg}")

            return True

        except subprocess.TimeoutExpired as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            rec = TouchRecord(
                action_type=action_name,
                params=params,
                timestamp=time.time(),
                duration_ms=elapsed_ms,
                success=False,
                returncode=-1,
                error_message=f"Timeout after {self.timeout}s",
            )
            self.history.append(rec)
            raise TouchControllerError(f"ADB {action_name} timed out after {self.timeout}s") from exc

    def tap(self, x: int, y: int) -> bool:
        """Inject a single tap at (x, y) coordinates on the touchscreen."""
        return self._execute_shell(
            shell_args=["input", "tap", str(int(x)), str(int(y))],
            action_name="TAP",
            params={"x": int(x), "y": int(y)},
        )

    def drag_and_drop(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        pickup_delay_sec: float = 0.08,
        settle_delay_sec: float = 0.18,
        duration_ms: int = 320,
        action_name: str = "DRAG_AND_DROP",
    ) -> bool:
        """Inject a high-reliability drag-and-drop gesture using unified motionevent.

        Includes mandatory pre-pickup dwell (to register block grab) and pre-release dwell
        (to allow Unity physics grid snap before touch lift). Uses direct single-stroke MOVE
        without artificial interpolation to eliminate process spawn overhead.
        """
        params = {
            "x1": int(x1), "y1": int(y1),
            "x2": int(x2), "y2": int(y2),
            "duration_ms": int(duration_ms),
            "pickup_sec": pickup_delay_sec,
            "settle_sec": settle_delay_sec,
        }
        if self.dry_run:
            return self._execute_shell([], action_name, params)

        # Single unified command chain in android shell: DOWN -> sleep -> MOVE -> sleep -> UP
        shell_script = (
            f"input motionevent DOWN {int(x1)} {int(y1)}; "
            f"sleep {pickup_delay_sec:.2f}; "
            f"input motionevent MOVE {int(x2)} {int(y2)}; "
            f"sleep {settle_delay_sec:.2f}; "
            f"input motionevent UP {int(x2)} {int(y2)}"
        )
        return self._execute_shell(
            shell_args=[shell_script],
            action_name=action_name,
            params=params,
        )

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 380) -> bool:
        """Inject a drag/swipe gesture from (x1, y1) to (x2, y2).

        Uses high-reliability drag_and_drop motionevent sequence with SWIPE audit type.
        """
        return self.drag_and_drop(x1, y1, x2, y2, duration_ms=duration_ms, action_name="SWIPE")

    def keyevent(self, keycode: str | int) -> bool:
        """Inject an Android hardware or navigation keyevent."""
        return self._execute_shell(
            shell_args=["input", "keyevent", str(keycode)],
            action_name="KEYEVENT",
            params={"keycode": str(keycode)},
        )

    def clear_history(self) -> None:
        """Clear recorded touch history."""
        self.history.clear()
