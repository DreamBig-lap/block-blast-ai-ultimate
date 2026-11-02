"""Console Dashboard & Live Control Interface for BLOCK BLAST AI ULTIMATE.

Provides real-time terminal monitoring of:
- ADB device connection status
- Autonomous state machine (IDLE, CAPTURING, RECOGNIZING, THINKING, EXECUTING, VERIFYING, PAUSED, GAME_OVER, ERROR)
- 8x8 Board ASCII view and occupancy bitboard
- Dynamic tray pieces (1, 2, or 3 pieces)
- AI planned move details
- Session score delta, lines cleared, turn count, success rate
- Latency breakdown: Capture, Vision, AI, Drag, Verify, Total
- Activity log history
- Interactive controls: START [S], PAUSE [P], STEP [Space], STOP/QUIT [Q]
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Optional, Union

from action_engine.autonomous_controller import (
    AutonomousGameController,
    AutonomousState,
    TurnLatencyMetrics,
)
from ai_engine.decision import DecisionResult, PlannedMove
from computer_vision.models import FrameRecognitionResult


class AutonomousDashboard:
    """Live interactive terminal dashboard for AutonomousGameController."""

    def __init__(self, controller: AutonomousGameController) -> None:
        self.controller = controller
        self._is_active = False
        self._stop_event = threading.Event()
        self._kb_thread: Optional[threading.Thread] = None

    def render_view(self) -> str:
        """Render a formatted ASCII string of current operational state."""
        ctrl = self.controller
        stats = ctrl.stats
        state_str = ctrl.state.value

        lines: list[str] = []
        lines.append("=" * 74)
        lines.append("  BLOCK BLAST AI ULTIMATE — FULL AUTONOMOUS GAME LOOP")
        lines.append("=" * 74)

        # 1. Device & State Header
        dev = ctrl.config.device_id or "Default"
        dry_flag = " [DRY-RUN]" if ctrl.config.dry_run else " [LIVE]"
        uptime = int(stats.uptime_seconds)

        lines.append(f"  Device: {dev:<15}{dry_flag:<10} State: [{state_str:<11}] Uptime: {uptime}s")
        reuse_str = f" | Reuse: {stats.frame_reuse_hits} saved" if getattr(stats, "frame_reuse_hits", 0) > 0 else ""
        lines.append(
            f"  Turns: {stats.successful_turns}/{stats.total_turns} ({stats.success_rate:.1f}% OK) | "
            f"Score Delta: +{stats.total_score_delta} | Lines: {stats.total_lines_cleared} | "
            f"Refills: {stats.rounds_refilled}{reuse_str}"
        )

        # Latency breakdown
        if stats.recent_latencies:
            last_lat = stats.recent_latencies[-1]
            reuse_tag = " [REUSED]" if getattr(last_lat, "frame_reused", False) else ""
            if last_lat.adb_transfer_ms > 0:
                lines.append(
                    f"  Turn Latency: {last_lat.total_turn_ms:.0f}ms{reuse_tag} (ADB: {last_lat.adb_transfer_ms:.0f}ms | "
                    f"Dec: {last_lat.png_decode_ms:.0f}ms | Vis: {last_lat.vision_ms:.0f}ms | "
                    f"AI: {last_lat.ai_ms:.0f}ms | Drag: {last_lat.drag_ms:.0f}ms | Ver: {last_lat.verify_ms:.0f}ms)"
                )
            else:
                lines.append(
                    f"  Turn Latency: {last_lat.total_turn_ms:.0f}ms{reuse_tag} (Cap: {last_lat.capture_ms:.0f}ms | "
                    f"Vis: {last_lat.vision_ms:.0f}ms | AI: {last_lat.ai_ms:.0f}ms | "
                    f"Drag: {last_lat.drag_ms:.0f}ms | Ver: {last_lat.verify_ms:.0f}ms)"
                )
        else:
            lines.append("  Turn Latency: (Awaiting first turn execution...)")

        lines.append("-" * 74)

        # 2. Board ASCII State
        if ctrl.last_recognition and ctrl.last_recognition.board:
            b = ctrl.last_recognition.board
            conf_pct = int(ctrl.last_recognition.overall_confidence * 100)
            status_tag = " [VERIFYING POST-MOVE...]" if ctrl.state == AutonomousState.VERIFYING else ""
            lines.append(f"  BOARD STATE{status_tag} (Occupied: {bin(ctrl.last_bitboard).count('1')}/64 | Bitboard: 0x{ctrl.last_bitboard:016X} | Conf: {conf_pct}%):")
            lines.append("     0 1 2 3 4 5 6 7")
            for r in range(8):
                row_str = " ".join("#" if (ctrl.last_bitboard & (1 << (r * 8 + c))) else "." for c in range(8))
                lines.append(f"  {r}  {row_str}")
        else:
            lines.append("  BOARD: Waiting for perception capture...")

        lines.append("-" * 74)

        # 3. Dynamic Tray Pieces
        if ctrl.last_recognition:
            lines.append("  TRAY PIECES:")
            tray_parts = []
            for p in ctrl.last_recognition.tray_pieces:
                if ctrl.state == AutonomousState.VERIFYING and ctrl.last_planned_move and p.slot_idx == ctrl.last_planned_move.slot_idx:
                    tray_parts.append(f"Slot {p.slot_idx}: [PLACED -> VERIFYING]")
                elif p.is_empty:
                    tray_parts.append(f"Slot {p.slot_idx}: [EMPTY]")
                else:
                    conf_p = int(p.confidence * 100)
                    tray_parts.append(f"Slot {p.slot_idx}: {p.matched_name} ({conf_p}%)")
            lines.append("  " + "  |  ".join(tray_parts))
        else:
            lines.append("  TRAY: No data")

        lines.append("-" * 74)

        # 4. AI Planned Move
        if ctrl.state == AutonomousState.GAME_OVER:
            lines.append("  AI PLANNED MOVE: GAME OVER (Mathematical verification: 0 legal placements for current tray)")
        elif ctrl.last_planned_move:
            m = ctrl.last_planned_move
            lines.append(f"  AI PLANNED MOVE (Strategy: {ctrl.config.strategy}):")
            lines.append(
                f"    -> Slot {m.slot_idx} [{m.piece_name}] -> Board (Row {m.row}, Col {m.col}) "
                f"[Score: +{m.predicted_score_delta}, Cleared: {m.lines_cleared} lines]"
            )
        else:
            lines.append("  AI PLANNED MOVE: (None active)")

        lines.append("-" * 74)

        # 5. Recent Activity Logs
        lines.append("  RECENT ACTIONS:")
        if ctrl.action_log:
            for entry in ctrl.action_log[-5:]:
                lines.append(f"    * [{entry['timestamp']}] {entry['message']}")
        else:
            lines.append("    * (Ready)")

        lines.append("=" * 74)
        lines.append("  CONTROLS: [S] Start / Resume   [P] Pause   [Space] Single Step   [Q] Stop")
        lines.append("=" * 74)

        return "\n".join(lines)

    def print_update(self) -> None:
        """Print refreshed view to console."""
        print(self.render_view())

    def listen_keyboard(self) -> None:
        """Non-blocking keyboard listener thread for Windows."""
        try:
            import msvcrt
            while self._is_active:
                try:
                    if msvcrt.kbhit():
                        ch = msvcrt.getch()
                        if ch in (b'\x00', b'\xe0'):
                            msvcrt.getch()
                            continue
                        key = ch.decode("utf-8", errors="ignore").lower()
                        self.handle_key(key)
                except Exception:
                    pass
                time.sleep(0.05)
        except ImportError:
            pass

    def handle_key(self, key: str) -> None:
        """Handle dashboard keyboard shortcut."""
        if key == 's':
            if self.controller.state == AutonomousState.PAUSED:
                print("\n>>> RESUMING AUTONOMOUS PLAY...")
                self.controller.resume("User pressed 's'")
                if not self.controller.is_running():
                    self.controller.start()
            elif self.controller.state in (AutonomousState.GAME_OVER, AutonomousState.ERROR):
                print("\n>>> RESTARTING AUTONOMOUS PLAY FOR NEW GAME...")
                self.controller.stop("Restart requested after game over/error")
                self.controller.start()
            elif self.controller.is_running():
                print("\n>>> AUTONOMOUS PLAY IS ALREADY RUNNING...")
            else:
                print("\n>>> STARTING AUTONOMOUS PLAY...")
                self.controller.start()
        elif key == 'p':
            print("\n>>> PAUSING AUTONOMOUS PLAY...")
            self.controller.pause("User pressed 'p'")
        elif key == ' ':
            print("\n>>> STEPPING SINGLE TURN...")
            self.controller.step_single_turn()
        elif key == 'q':
            print("\n>>> STOPPING AND EXITING...")
            self.stop()

    def stop(self) -> None:
        """Stop dashboard interactive loop and halt controller."""
        self._is_active = False
        self._stop_event.set()
        self.controller.stop("User requested dashboard stop")

    def start_interactive(self, refresh_interval_sec: float = 1.0) -> None:
        """Start interactive dashboard view and listener."""
        self._is_active = True
        self._stop_event.clear()
        self._kb_thread = threading.Thread(target=self.listen_keyboard, daemon=True)
        self._kb_thread.start()

        try:
            while self._is_active and not self._stop_event.is_set() and not self.controller._stop_requested:
                self.print_update()
                if self._stop_event.wait(timeout=refresh_interval_sec):
                    break
        except KeyboardInterrupt:
            print("\n>>> KeyboardInterrupt received. Stopping...")
            self.stop()
        finally:
            self._is_active = False
            self._stop_event.set()
            if self._kb_thread is not None and self._kb_thread.is_alive():
                try:
                    self._kb_thread.join(timeout=1.0)
                except Exception:
                    pass
                self._kb_thread = None


# Backward compatibility alias
ConsoleDashboard = AutonomousDashboard
