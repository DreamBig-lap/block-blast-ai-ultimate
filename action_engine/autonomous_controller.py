"""Autonomous Game Controller & State Machine for BLOCK BLAST AI ULTIMATE.

Orchestrates the complete autonomous game loop:
CAPTURE -> RECOGNIZE -> VALIDATE -> AI DECISION -> SELECT PIECE -> CALCULATE DESTINATION -> EXECUTE DRAG -> VERIFY RESULT -> UPDATE STATE -> REPEAT

Features:
- Thread-safe state machine: IDLE, INITIALIZING, CAPTURING, RECOGNIZING, VALIDATING,
  THINKING, EXECUTING, VERIFYING, PAUSED, GAME_OVER, ERROR.
- Full cycle on every turn: fresh perception on every move (never reuses stale frames).
- Dynamic tray handling: 1, 2, or 3 pieces.
- High-precision closed-loop drag actuation with empirical 2D calibration (Gain X, Gain Y, Lift Y).
- Rigorous post-action verification (bitboard bit-for-bit check, slot consumption, tray refill).
- Zero disk thrashing: zero-copy in-memory frame processing.
- CPU-first efficiency: minimal latency on Intel i5 / 8GB RAM hardware.
- Safe controls: START, PAUSE, RESUME, STOP, STEP. Fail-safe touch release on any interruption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import enum
import logging
import math
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from action_engine.action_verifier import ActionVerifier, VerificationResult, VerificationStatus
from action_engine.closed_loop_controller import (
    ClosedLoopController,
    ClosedLoopStatus,
    ClosedLoopStepResult,
    DragCalibrationConfig,
)
from action_engine.held_piece_tracker import HeldPieceTracker, TrackingStatus
from action_engine.touch_controller import TouchController
from ai_engine.decision import AIDecisionEngine, DecisionResult, DecisionStatus, PlannedMove
from computer_vision.capture import ScreenCapture, ScreenCaptureError
from computer_vision.config import DEFAULT_VISION_CONFIG, VisionConfig
from computer_vision.diagnostic_saver import PerceptionDiagnosticSaver
from computer_vision.models import CellState, FrameRecognitionResult, RecognitionStatus
from computer_vision.recognizer import BlockBlastRecognizer
from game_engine.board import popcount
from game_engine.pieces import Piece
from game_engine.simulator import GameSimulator, StepResult
from game_engine.state import GameState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State Machine & Models
# ---------------------------------------------------------------------------

class AutonomousState(enum.Enum):
    """Operational states of the autonomous game controller."""

    IDLE = "IDLE"                      # System initialized, waiting for START
    INITIALIZING = "INITIALIZING"      # Setting up ADB connection and validating environment
    CAPTURING = "CAPTURING"            # Acquiring live frame from device via memory stream
    RECOGNIZING = "RECOGNIZING"        # Computer vision detecting board and tray pieces
    VALIDATING = "VALIDATING"          # Verifying state sanity, confidence, and piece legality
    THINKING = "THINKING"              # AI Decision Engine computing optimal move
    EXECUTING = "EXECUTING"            # Closed-loop drag-and-drop actuation
    VERIFYING = "VERIFYING"            # Post-move bitboard and slot verification
    PAUSED = "PAUSED"                  # User paused or safety pause triggered (touch released)
    GAME_OVER = "GAME_OVER"            # Game over reached or no legal placements possible
    ERROR = "ERROR"                    # Fatal error / ADB connection lost


@dataclass(frozen=True, slots=True)
class TurnLatencyMetrics:
    """Latency breakdown for a single autonomous game cycle."""

    turn_index: int
    capture_ms: float
    vision_ms: float
    ai_ms: float
    drag_ms: float
    verify_ms: float
    total_turn_ms: float
    adb_transfer_ms: float = 0.0
    png_decode_ms: float = 0.0
    frame_reused: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "capture_ms": round(self.capture_ms, 1),
            "adb_transfer_ms": round(self.adb_transfer_ms, 1),
            "png_decode_ms": round(self.png_decode_ms, 1),
            "vision_ms": round(self.vision_ms, 1),
            "ai_ms": round(self.ai_ms, 1),
            "drag_ms": round(self.drag_ms, 1),
            "verify_ms": round(self.verify_ms, 1),
            "total_turn_ms": round(self.total_turn_ms, 1),
            "frame_reused": self.frame_reused,
        }


@dataclass
class AutonomousSessionStats:
    """Session statistics tracking game progression."""

    total_turns: int = 0
    successful_turns: int = 0
    failed_turns: int = 0
    total_score_delta: int = 0
    total_lines_cleared: int = 0
    rounds_refilled: int = 0
    session_start_time: float = field(default_factory=time.time)
    last_turn_timestamp: float = field(default_factory=time.time)
    recent_latencies: List[TurnLatencyMetrics] = field(default_factory=list)

    # Phase 1.3: Verified Frame Reuse Telemetry
    frame_reuse_hits: int = 0
    frame_reuse_misses: int = 0
    frame_reuse_rejections: int = 0
    frame_reuse_rejection_reasons: Dict[str, int] = field(default_factory=dict)
    captures_saved: int = 0

    def frame_reuse_summary(self) -> Dict[str, Any]:
        """Summary of frame reuse efficiency for telemetry and reporting."""
        return {
            "reuse_hits": self.frame_reuse_hits,
            "fallback_misses": self.frame_reuse_misses,
            "rejections": self.frame_reuse_rejections,
            "rejection_reasons": dict(self.frame_reuse_rejection_reasons),
            "captures_saved": self.captures_saved,
        }

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.session_start_time

    @property
    def success_rate(self) -> float:
        total = self.successful_turns + self.failed_turns
        return (self.successful_turns / total * 100.0) if total > 0 else 100.0

    @property
    def average_turn_time_ms(self) -> float:
        if not self.recent_latencies:
            return 0.0
        return sum(m.total_turn_ms for m in self.recent_latencies) / len(self.recent_latencies)


@dataclass
class AutonomousConfig:
    """Configuration options for AutonomousGameController."""

    device_id: Optional[str] = None
    adb_path: str = "adb"
    strategy: str = "beam_search"
    beam_width: int = 15
    dry_run: bool = False
    max_turns: Optional[int] = None
    min_overall_confidence: float = 0.65
    min_piece_confidence: float = 0.70
    post_move_settle_sec: float = 0.20
    normal_settle_sec: float = 0.20
    line_clear_settle_sec: float = 1.20
    consecutive_frame_delay_sec: float = 0.15
    settle_timeout_sec: float = 3.00
    refill_wait_sec: float = 0.80
    loop_idle_sleep_sec: float = 0.05
    auto_restart_on_refill: bool = True
    fast_gesture_mode: bool = True  # High-reliability direct gesture with smooth interpolation and confirmed UP
    calibration_config: Optional[DragCalibrationConfig] = None
    project_root: Optional[Path] = None


# ---------------------------------------------------------------------------
# Central Autonomous Game Controller
# ---------------------------------------------------------------------------

class AutonomousGameController:
    """Master orchestrator for autonomous Block Blast play."""

    def __init__(
        self,
        config: Optional[AutonomousConfig] = None,
        screen_capture: Optional[ScreenCapture] = None,
        recognizer: Optional[BlockBlastRecognizer] = None,
        ai_engine: Optional[AIDecisionEngine] = None,
        closed_loop_controller: Optional[ClosedLoopController] = None,
        touch_controller: Optional[TouchController] = None,
        action_verifier: Optional[ActionVerifier] = None,
    ) -> None:
        self.config = config or AutonomousConfig()

        # Dependencies
        self.capture = screen_capture or ScreenCapture()
        self.recognizer = recognizer or BlockBlastRecognizer()
        self.ai_engine = ai_engine or AIDecisionEngine(default_beam_width=self.config.beam_width)

        calib = self.config.calibration_config or DragCalibrationConfig.create_samsung_m21_calibrated()
        self.closed_loop = closed_loop_controller or ClosedLoopController(
            config=calib,
            dry_run=self.config.dry_run,
        )
        self.touch = touch_controller or TouchController(
            adb_path=self.config.adb_path,
            device_id=self.config.device_id,
            dry_run=self.config.dry_run,
        )
        self.verifier = action_verifier or ActionVerifier(
            recognizer=self.recognizer,
            capture=self.capture,
            normal_settle_sec=self.config.normal_settle_sec,
            line_clear_settle_sec=self.config.line_clear_settle_sec,
            consecutive_frame_delay_sec=self.config.consecutive_frame_delay_sec,
            settle_timeout_sec=self.config.settle_timeout_sec,
        )

        # Project Root & Debug Directories
        if self.config.project_root is not None:
            self.project_root = Path(self.config.project_root).resolve()
        else:
            self.project_root = Path(__file__).resolve().parent.parent

        self.debug_dir = self.project_root / "debug"
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.diagnostic_saver = PerceptionDiagnosticSaver(
            config=self.recognizer.config,
            base_dir=self.debug_dir / "perception_failures",
        )

        # State & Threading
        self._state = AutonomousState.IDLE
        self._state_lock = threading.Lock()
        self._turn_lock = threading.Lock()
        self._is_running = False
        self._stop_requested = False
        self._pause_requested = False
        self._loop_thread: Optional[threading.Thread] = None
        self._consecutive_empty_trays: int = 0

        # Live State Tracking
        self.stats = AutonomousSessionStats()
        self.last_frame: Optional[np.ndarray] = None
        self.last_recognition: Optional[FrameRecognitionResult] = None
        self.last_bitboard: int = 0
        self.last_decision: Optional[DecisionResult] = None
        self.last_planned_move: Optional[PlannedMove] = None
        self.last_step_result: Optional[ClosedLoopStepResult] = None
        self.last_error_message: Optional[str] = None
        self.last_pause_reason: str = ""
        self.action_log: List[Dict[str, Any]] = []

        # Phase 1.3: Verified Frame Reuse State
        self._pending_verified_frame: Optional[FrameRecognitionResult] = None
        self._pending_verified_img: Optional[np.ndarray] = None

        # Interactive Shell for fast chained touch commands
        self._shell_proc: Optional[subprocess.Popen] = None

        # Pointer & Gesture State Tracking
        self._touch_pointer_down: bool = False
        self._pointer_pos: Tuple[int, int] = (500, 1000)

        # Observer callbacks
        self.on_state_changed: Optional[Callable[[AutonomousState, str], None]] = None
        self.on_turn_completed: Optional[Callable[[TurnLatencyMetrics, PlannedMove, bool], None]] = None
        self.on_log_message: Optional[Callable[[str], None]] = None

    # -------------------------------------------------------------------------
    # State Machine Properties & Transitions
    # -------------------------------------------------------------------------

    @property
    def state(self) -> AutonomousState:
        with self._state_lock:
            return self._state

    def set_state(self, new_state: AutonomousState, reason: str = "") -> None:
        """Atomically transition state with audit logging."""
        with self._state_lock:
            old_state = self._state
            self._state = new_state

        msg = f"[STATE] {old_state.value} -> {new_state.value}"
        if reason:
            msg += f" ({reason})"
        self.log(msg)

        if self.on_state_changed:
            try:
                self.on_state_changed(new_state, reason)
            except Exception as exc:
                logger.error("Error in on_state_changed callback: %s", exc)

    def log(self, message: str) -> None:
        """Record entry in activity log."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        entry = {"timestamp": timestamp, "message": message}
        self.action_log.append(entry)
        if len(self.action_log) > 50:
            self.action_log.pop(0)
        logger.info(message)
        if self.on_log_message:
            try:
                self.on_log_message(f"[{timestamp}] {message}")
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Lifecycle Controls: START, PAUSE, RESUME, STOP
    # -------------------------------------------------------------------------

    def start(self) -> bool:
        """Start autonomous game loop in background thread."""
        if self.is_running():
            logger.warning("Start requested but controller is already running.")
            return False

        self._stop_requested = False
        self._pause_requested = False
        self._is_running = True
        self.last_planned_move = None
        self._consecutive_empty_trays = 0
        self._pending_verified_frame = None
        self._pending_verified_img = None
        if not self.config.dry_run:
            self.last_frame = None
        self.set_state(AutonomousState.INITIALIZING, "User START requested")

        self._loop_thread = threading.Thread(target=self._run_loop, name="AutonomousLoop", daemon=True)
        self._loop_thread.start()
        return True

    def pause(self, reason: str = "User PAUSE requested") -> None:
        """Pause automated execution and release any active touch pointer."""
        self._pause_requested = True
        self.last_planned_move = None
        self.last_pause_reason = reason
        self._pending_verified_frame = None
        self._pending_verified_img = None
        self._release_pointer_failsafe(*self._pointer_pos)
        self._touch_pointer_down = False
        self.set_state(AutonomousState.PAUSED, reason)

    def resume(self, reason: str = "User RESUME requested") -> None:
        """Resume autonomous execution from PAUSED state with fresh perception."""
        self._pause_requested = False
        self.last_planned_move = None
        self._consecutive_empty_trays = 0
        self.last_recognition = None
        self.last_bitboard = 0
        self.last_frame = None  # Guarantee fresh capture
        self._pending_verified_frame = None
        self._pending_verified_img = None
        self.set_state(AutonomousState.CAPTURING, reason)
        self.log(f"Resuming autonomous loop: {reason}")

    def stop(self, reason: str = "User STOP requested") -> None:
        """Gracefully stop and terminate the autonomous game loop."""
        self._stop_requested = True
        self._is_running = False
        self.last_planned_move = None
        self._consecutive_empty_trays = 0
        self._pending_verified_frame = None
        self._pending_verified_img = None
        if self._loop_thread is not None and self._loop_thread.is_alive():
            try:
                self._loop_thread.join(timeout=2.0)
            except Exception:
                pass
            self._loop_thread = None
        self._release_pointer_failsafe(*self._pointer_pos)
        self._touch_pointer_down = False
        self._close_shell()
        self.set_state(AutonomousState.IDLE, reason)

    def is_running(self) -> bool:
        return self._is_running and (self._loop_thread is not None and self._loop_thread.is_alive())

    # -------------------------------------------------------------------------
    # Single-Turn Execution (Perceive -> Plan -> Act -> Verify)
    # -------------------------------------------------------------------------

    def step_single_turn(self) -> bool:
        """Execute exactly one complete perception-decision-action-verification cycle.

        Returns:
            True if move succeeded and game continues; False if paused/game over/failed.
        """
        if not self._turn_lock.acquire(blocking=False):
            logger.warning("step_single_turn ignored: another turn is currently executing.")
            return False

        try:
            return self._step_single_turn_internal()
        except Exception as exc:
            self.log(f"Unexpected error in single-turn: {exc}")
            self.last_error_message = str(exc)
            self.last_planned_move = None
            self._release_pointer_failsafe()
            self.pause(f"Turn error: {exc}")
            return False
        finally:
            self._turn_lock.release()

    def _step_single_turn_internal(self) -> bool:
        t_turn_start = time.perf_counter()

        # ---------------------------------------------------------------------
        # 1. CAPTURE
        # ---------------------------------------------------------------------
        self.set_state(AutonomousState.CAPTURING)
        t0 = time.perf_counter()

        adb_transfer_ms = 0.0
        png_decode_ms = 0.0
        frame_reused = False
        reused_recognition: Optional[FrameRecognitionResult] = None

        if self._pending_verified_frame is not None:
            # Check pending frame and consume immediately (single-use guarantee)
            candidate_rec = self._pending_verified_frame
            candidate_img = self._pending_verified_img
            self._pending_verified_frame = None
            self._pending_verified_img = None

            if candidate_rec is not None and candidate_rec.status == RecognitionStatus.SUCCESS:
                frame_reused = True
                reused_recognition = candidate_rec
                self.stats.frame_reuse_hits += 1
                self.stats.captures_saved += 1
                self.log("[FRAME_REUSE] HIT")
                if candidate_img is not None:
                    frame = candidate_img
                elif self.last_frame is not None:
                    frame = self.last_frame
                else:
                    h, w = candidate_rec.image_shape if (candidate_rec.image_shape and candidate_rec.image_shape[0] > 0) else (2340, 1080)
                    frame = np.zeros((h, w, 3), dtype=np.uint8)
                self.last_frame = frame
                t_capture_ms = 0.0

        if not frame_reused:
            self.stats.frame_reuse_misses += 1
            self.log("[FRAME_REUSE] MISS")
            t_adb_start = time.perf_counter()
            try:
                if self.config.dry_run and self.last_frame is not None:
                    frame = self.last_frame.copy()
                else:
                    frame = self.capture.capture_from_adb(
                        device_id=self.config.device_id,
                        adb_path=self.config.adb_path,
                    )
                    adb_transfer_ms = getattr(self.capture, "last_adb_transfer_ms", 0.0)
                    png_decode_ms = getattr(self.capture, "last_png_decode_ms", 0.0)
            except (ScreenCaptureError, Exception) as exc:
                self.log(f"Capture failed: {exc}")
                self.last_error_message = str(exc)
                self.last_planned_move = None
                self._pending_verified_frame = None
                self._pending_verified_img = None
                self.pause(f"ADB Capture error: {exc}")
                return False

            if frame is None or frame.size == 0:
                self.last_planned_move = None
                self._pending_verified_frame = None
                self._pending_verified_img = None
                self.pause("Capture returned empty frame")
                return False

            self.last_frame = frame
            t_capture_ms = (time.perf_counter() - t0) * 1000.0
            t_cap_duration = (time.perf_counter() - t_adb_start) * 1000.0
            diag = getattr(self.capture, "last_telemetry", {})
            cap_mode = diag.get("mode", getattr(self.capture, "last_capture_mode", "RAW"))
            cap_trans = diag.get("transport", getattr(self.capture, "last_transport_mode", "SOCKET"))
            cap_bytes = diag.get("bytes", getattr(self.capture, "last_bytes_count", 0))
            fallback_info = f" fallback={diag.get('fallback')}" if diag.get("fallback") else ""
            self.log(
                f"[ADB_CAPTURE] mode={cap_mode} transport={cap_trans} bytes={cap_bytes} "
                f"transfer_ms={adb_transfer_ms:.1f} (total_ms={t_cap_duration:.1f}){fallback_info}"
            )

        # ---------------------------------------------------------------------
        # 2. RECOGNIZE
        # ---------------------------------------------------------------------
        self.set_state(AutonomousState.RECOGNIZING)
        t0 = time.perf_counter()

        if frame_reused and reused_recognition is not None:
            rec_res = reused_recognition
            self.last_recognition = rec_res
            if rec_res.board is not None:
                self.last_bitboard = rec_res.board.bitboard
            t_vision_ms = 0.0
        else:
            rec_res = self.recognizer.recognize_frame(frame)
            self.last_recognition = rec_res
            if rec_res.board is not None:
                self.last_bitboard = rec_res.board.bitboard
            t_vision_ms = (time.perf_counter() - t0) * 1000.0

        # ---------------------------------------------------------------------
        # 3. VALIDATE
        # ---------------------------------------------------------------------
        self.set_state(AutonomousState.VALIDATING)

        board_failed = (rec_res.status == RecognitionStatus.FAILED or rec_res.board is None)
        low_conf = (rec_res.status == RecognitionStatus.LOW_CONFIDENCE)
        conf_below_min = (rec_res.overall_confidence < self.config.min_overall_confidence)
        unknown_slots = [
            p.slot_idx for p in rec_res.tray_pieces
            if not p.is_empty and (p.piece is None or p.matched_name == "UNKNOWN" or p.confidence < self.config.min_piece_confidence)
        ]

        if board_failed or low_conf or conf_below_min or unknown_slots:
            if board_failed:
                pause_msg = "Vision failed to detect 8x8 game board"
            elif low_conf:
                pause_msg = f"Perception low confidence: {rec_res.error_message}. Pausing for safety."
            elif conf_below_min:
                pause_msg = f"Low recognition confidence: {rec_res.overall_confidence:.2f}"
            else:
                pause_msg = (
                    f"Perception uncertain: Slot(s) {unknown_slots} contain UNKNOWN piece or confidence below "
                    f"{self.config.min_piece_confidence:.2f}. Pausing to prevent false Game Over."
                )

            # Minimal diagnostic capture: save full failure artifacts package safely
            try:
                saved_dir = self.diagnostic_saver.save_diagnostic_failure(
                    frame=frame,
                    recognition_result=rec_res,
                    pause_reason=pause_msg,
                    piece_detector=self.recognizer.piece_detector,
                )
                if saved_dir:
                    self.log(f"Perception diagnostic artifacts saved to: {saved_dir}")
            except Exception as diag_err:
                self.log(f"Warning: Failed to save perception diagnostic artifacts: {diag_err}")

            self.log(pause_msg)
            self.last_planned_move = None
            self.pause(pause_msg)
            return False

        board_bbox = rec_res.board.board_bbox
        cell_size = rec_res.board.cell_size
        bx, by, bw, bh = board_bbox

        # Board bounding box drift guard: prevent 1-row vertical grid shift
        if frame.shape[0] >= 1000:
            calib_rel = getattr(getattr(self.recognizer, "config", None), "calibrated_board_rel", None)
            if not isinstance(calib_rel, (tuple, list)):
                calib_rel = DEFAULT_VISION_CONFIG.calibrated_board_rel
            expected_board_y = int(round(frame.shape[0] * calib_rel[1]))
            if abs(by - expected_board_y) > 15:
                msg = (
                    f"Board bounding box drift detected: detected y={by} vs calibrated expected y={expected_board_y} "
                    f"(deviation {abs(by - expected_board_y)}px > 15px). Pausing to prevent row shift."
                )
                self.log(msg)
                self.last_planned_move = None
                self.pause(msg)
                return False

        # Board uncertain cell guard:
        if getattr(rec_res.board, "has_uncertain_cells", False) or (
            rec_res.board.cells and any(
                c.state == CellState.UNCERTAIN for row in rec_res.board.cells for c in row
            )
        ):
            msg = "Perception uncertain: Board contains UNCERTAIN cells. Pausing to prevent collision."
            self.log(msg)
            self.last_planned_move = None
            self.pause(msg)
            return False

        # Canonical bitboard from verified CellClassifier
        semantic_bitboard = rec_res.board.bitboard
        self.last_bitboard = semantic_bitboard

        # Dynamic Tray Handling (1, 2, or 3 pieces)
        visible_tray_pieces = []
        tray_pieces_for_state: List[Optional[Piece]] = []
        for p in rec_res.tray_pieces:
            if not p.is_empty and p.piece is not None:
                visible_tray_pieces.append(p)
                tray_pieces_for_state.append(p.piece)
            else:
                tray_pieces_for_state.append(None)

        while len(tray_pieces_for_state) < 3:
            tray_pieces_for_state.append(None)
        tray_tuple = tuple(tray_pieces_for_state[:3])

        visible_count = len(visible_tray_pieces)

        # Handle Tray Empty (Round Refill Animation settling)
        if visible_count == 0:
            self._consecutive_empty_trays += 1
            self.last_planned_move = None
            if self._consecutive_empty_trays > 5:
                self.pause("Refill timeout: Tray remained empty for 5 consecutive checks. Halting for safety.")
                return False

            self.log(f"Tray is empty (check {self._consecutive_empty_trays}/5). Waiting for round refill...")
            if self._consecutive_empty_trays == 1:
                self.stats.rounds_refilled += 1

            if self.config.dry_run:
                # Deal 3 new canonical pieces in dry-run mode
                sim = GameSimulator()
                new_hand = sim.deal_hand()
                cfg_v = self.recognizer.config
                h_img, w_img = frame.shape[:2]
                y0 = int(h_img * cfg_v.tray_expected_y_min_rel)
                y1 = int(h_img * cfg_v.tray_expected_y_max_rel)
                frame[y0:y1, :] = (77, 114, 157)
                bsize = 56
                y_mid = int((y0 + y1) / 2)
                for s_idx, p in enumerate(new_hand):
                    x_min_rel, x_max_rel = cfg_v.tray_slot_x_splits[s_idx]
                    cx = int(w_img * (x_min_rel + x_max_rel) / 2)
                    ox = cx - int(p.width * bsize / 2)
                    oy = y_mid - int(p.height * bsize / 2)
                    for dr, dc in p.cells:
                        x0, y0_ = ox + dc * bsize, oy + dr * bsize
                        cv2.rectangle(frame, (x0 + 2, y0_ + 2), (x0 + bsize - 2, y0_ + bsize - 2), (0, 215, 255), -1)
                self.last_frame = frame
                self._consecutive_empty_trays = 0
                self.log(f"Dry-run round refilled ({self.stats.rounds_refilled}): {[p.name for p in new_hand]}")
            else:
                self.last_frame = None  # Guarantee fresh capture from phone on next turn
                self._pending_verified_frame = None
                self._pending_verified_img = None
            time.sleep(self.config.refill_wait_sec)
            return True
        else:
            self._consecutive_empty_trays = 0

        # ---------------------------------------------------------------------
        # 4. THINKING (AI DECISION)
        # ---------------------------------------------------------------------
        self.set_state(AutonomousState.THINKING)
        t0 = time.perf_counter()

        game_state = GameState(board=semantic_bitboard, tray=tray_tuple)
        decision = self.ai_engine.decide(game_state, strategy=self.config.strategy)
        self.last_decision = decision
        t_ai_ms = (time.perf_counter() - t0) * 1000.0

        if decision.status in (DecisionStatus.GAME_OVER, DecisionStatus.NO_VALID_MOVE):
            self.last_planned_move = None
            self.log(f"GAME OVER detected by AI ({decision.status.value}). No legal placements remain for known pieces.")
            self.set_state(AutonomousState.GAME_OVER, f"Game Over: {decision.status.value}")
            return False

        if decision.status != DecisionStatus.SUCCESS or decision.move_count == 0:
            self.last_planned_move = None
            self.log(f"AI produced no moves: {decision.rationale}")
            self.pause(f"AI produced no moves: {decision.rationale}")
            return False

        # Select Move 0 (first move of current state)
        move = decision.moves[0]
        self.last_planned_move = move

        chosen_slot_piece = next((p for p in visible_tray_pieces if p.slot_idx == move.slot_idx), None)
        if chosen_slot_piece is None:
            self.pause(f"Target slot {move.slot_idx} not found in visible tray pieces")
            return False

        piece_model = chosen_slot_piece.piece
        placement_mask = piece_model.placement_masks[move.row][move.col]
        if placement_mask is None:
            self.pause(f"Pre-actuation guard: Move ({move.row}, {move.col}) out of 8x8 boundaries")
            return False

        # Multi-layer Pre-Actuation Collision Guard:
        # 1. Bitboard collision check against both semantic_bitboard and vision raw bitboard
        if (semantic_bitboard & placement_mask) != 0 or (rec_res.board.bitboard & placement_mask) != 0:
            overlap = (semantic_bitboard | rec_res.board.bitboard) & placement_mask
            self.pause(f"Pre-actuation guard: Target area ({move.row}, {move.col}) collides with bitboard (mask 0x{overlap:X})")
            return False

        # 2. Occupancy matrix and cell state cross-check for every target unit cell
        for dr, dc in piece_model.cells:
            target_r = move.row + dr
            target_c = move.col + dc
            if not (0 <= target_r < 8 and 0 <= target_c < 8):
                self.pause(f"Pre-actuation guard: Cell ({target_r}, {target_c}) exceeds 8x8 grid bounds")
                return False

            if rec_res.board.occupancy_matrix and len(rec_res.board.occupancy_matrix) > target_r:
                if rec_res.board.occupancy_matrix[target_r][target_c] != 0:
                    self.pause(
                        f"Pre-actuation guard: Cell ({target_r}, {target_c}) marked occupied in occupancy_matrix. "
                        f"Halting touch."
                    )
                    return False

            if (
                rec_res.board.cells
                and len(rec_res.board.cells) > target_r
                and len(rec_res.board.cells[target_r]) > target_c
            ):
                cell_info = rec_res.board.cells[target_r][target_c]
                if cell_info.state != CellState.EMPTY:
                    self.pause(
                        f"Pre-actuation guard: Cell ({target_r}, {target_c}) has non-empty state ({cell_info.state.name}, "
                        f"gray={cell_info.mean_gray:.1f}). Halting touch."
                    )
                    return False

        self.log(f"Planned Move: Slot {move.slot_idx} ({move.piece_name}) -> ({move.row}, {move.col})")

        # ---------------------------------------------------------------------
        # 5. EXECUTING (CLOSED-LOOP DRAG)
        # ---------------------------------------------------------------------
        self.set_state(AutonomousState.EXECUTING)
        t0 = time.perf_counter()

        s_bx, s_by, s_bw, s_bh = chosen_slot_piece.bbox_in_screen
        slot_center_x = s_bx + s_bw // 2
        slot_center_y = s_by + s_bh // 2

        start_x, start_y, finger_x_0, finger_y_0 = self.closed_loop.plan_initial_trajectory(
            move=move,
            slot_center_x=slot_center_x,
            slot_center_y=slot_center_y,
            piece_width_cells=piece_model.width,
            piece_height_cells=piece_model.height,
        )

        step_res = self._execute_physical_gesture(
            move=move,
            piece_model=piece_model,
            base_frame=frame,
            start_x=start_x,
            start_y=start_y,
            finger_x_0=finger_x_0,
            finger_y_0=finger_y_0,
        )
        self.last_step_result = step_res
        t_drag_ms = (time.perf_counter() - t0) * 1000.0

        if step_res.requires_failsafe_release or step_res.status in (
            ClosedLoopStatus.UNKNOWN,
            ClosedLoopStatus.CORRECTION_LIMIT_EXCEEDED,
            ClosedLoopStatus.INVALID_CALIBRATION,
        ):
            self.last_planned_move = None
            self.pause(f"Tracker uncertain/unknown during flight ({step_res.status.value}): {step_res.message}")
            return False

        # ---------------------------------------------------------------------
        # 6. VERIFYING
        # ---------------------------------------------------------------------
        self.set_state(AutonomousState.VERIFYING)
        t0 = time.perf_counter()

        # Simulation of expected state
        sim = GameSimulator()
        try:
            sim_res = sim.apply_move(game_state, move.slot_idx, move.row, move.col)
            expected_bitboard = sim_res.state_after.board
        except Exception as exc:
            self.pause(f"Game simulation failed: {exc}")
            return False

        verification_status_str = "VERIFIED"
        fail_reason = ""
        post_semantic_bitboard = 0

        # Fresh post-action frame & verification
        try:
            if self.config.dry_run:
                post_frame = frame.copy()
                # Blank out placed slot partition completely using true tray background color (77, 114, 157)
                cfg_vision = getattr(self.recognizer, "config", None)
                if (
                    cfg_vision is None
                    or not hasattr(cfg_vision, "tray_slot_x_splits")
                    or not isinstance(getattr(cfg_vision, "tray_slot_x_splits", None), (list, tuple))
                ):
                    cfg_vision = DEFAULT_VISION_CONFIG
                h_img, w_img = post_frame.shape[:2]
                y_tray_start = int(h_img * getattr(cfg_vision, "tray_expected_y_min_rel", 0.70))
                y_tray_end = int(h_img * getattr(cfg_vision, "tray_expected_y_max_rel", 0.90))
                splits = getattr(cfg_vision, "tray_slot_x_splits", [(0.05, 0.35), (0.35, 0.65), (0.65, 0.95)])
                if 0 <= move.slot_idx < len(splits):
                    x_rel_min, x_rel_max = splits[move.slot_idx]
                else:
                    x_rel_min, x_rel_max = (0.05, 0.35)
                sx0 = int(w_img * x_rel_min)
                sx1 = int(w_img * x_rel_max)
                post_frame[y_tray_start:y_tray_end, sx0:sx1] = (77, 114, 157)

                # Paint placed piece cells onto board
                for dr, dc in piece_model.cells:
                    tr = move.row + dr
                    tc = move.col + dc
                    cx0 = int(round(bx + tc * cell_size + 2))
                    cx1 = int(round(bx + (tc + 1) * cell_size - 2))
                    cy0 = int(round(by + tr * cell_size + 2))
                    cy1 = int(round(by + (tr + 1) * cell_size - 2))
                    cv2.rectangle(post_frame, (cx0, cy0), (cx1, cy1), (0, 215, 255), -1)

                # Clear any full lines if applicable
                for r in sim_res.cleared_rows:
                    cy0 = int(round(by + r * cell_size + 2))
                    cy1 = int(round(by + (r + 1) * cell_size - 2))
                    cv2.rectangle(post_frame, (bx, cy0), (bx + bw, cy1), (35, 35, 35), -1)
                for c in sim_res.cleared_cols:
                    cx0 = int(round(bx + c * cell_size + 2))
                    cx1 = int(round(bx + (c + 1) * cell_size - 2))
                    cv2.rectangle(post_frame, (cx0, by), (cx1, by + bh), (35, 35, 35), -1)

                post_semantic_bitboard = expected_bitboard
                move_verified = True
                self.last_frame = post_frame
            else:
                verification_res = self.verifier.verify_move(
                    step_result=sim_res,
                    device_id=self.config.device_id,
                    step_index=self.stats.total_turns,
                )
                move_verified = verification_res.is_success
                verification_status_str = verification_res.status.value
                post_semantic_bitboard = verification_res.actual_bitboard
                fail_reason = verification_res.error_message or verification_res.status.value
                path_desc = "Fast-Path (1 frame)" if verification_res.is_fast_path else f"Two-Frame ({verification_res.captures_count} captures)"
                self.log(
                    f"[VERIFICATION] status={verification_status_str} path={path_desc} "
                    f"settle_ms={verification_res.settle_ms:.1f} cap_ms={verification_res.capture_ms:.1f} "
                    f"rec_ms={verification_res.recognition_ms:.1f} (total_ms={verification_res.elapsed_ms:.1f})"
                )
                if verification_res.new_frame_result is not None:
                    self.last_recognition = verification_res.new_frame_result
                    if verification_res.new_frame_result.board is not None:
                        self.last_bitboard = verification_res.new_frame_result.board.bitboard
                    if getattr(verification_res, "new_frame", None) is not None:
                        self.last_frame = verification_res.new_frame
                    else:
                        self.last_frame = None

                # Phase 1.3: Evaluate Verified Frame Reuse candidate
                if move_verified:
                    lines_cleared_val = getattr(move, "lines_cleared", 0) or getattr(sim_res, "lines_cleared", 0) or 0
                    can_reuse, reject_reason = self._can_reuse_verified_frame(
                        verification_res=verification_res,
                        lines_cleared=lines_cleared_val,
                    )
                    if can_reuse:
                        self._pending_verified_frame = verification_res.new_frame_result
                        self._pending_verified_img = getattr(verification_res, "new_frame", None)
                        self.log("[FRAME_REUSE] READY: Pending verified frame cached for next turn")
                    else:
                        self._pending_verified_frame = None
                        self._pending_verified_img = None
                        self.stats.frame_reuse_rejections += 1
                        reason_key = reject_reason.split(":")[0].split("(")[0].strip()
                        self.stats.frame_reuse_rejection_reasons[reason_key] = (
                            self.stats.frame_reuse_rejection_reasons.get(reason_key, 0) + 1
                        )
                        self.log(f"[FRAME_REUSE] REJECT: {reject_reason}")
                else:
                    self._pending_verified_frame = None
                    self._pending_verified_img = None

        except Exception as exc:
            self._pending_verified_frame = None
            self._pending_verified_img = None
            self.pause(f"Post-action verification exception: {exc}")
            return False

        t_verify_ms = (time.perf_counter() - t0) * 1000.0
        t_total_ms = (time.perf_counter() - t_turn_start) * 1000.0

        metrics = TurnLatencyMetrics(
            turn_index=self.stats.total_turns + 1,
            capture_ms=t_capture_ms,
            adb_transfer_ms=adb_transfer_ms,
            png_decode_ms=png_decode_ms,
            vision_ms=t_vision_ms,
            ai_ms=t_ai_ms,
            drag_ms=t_drag_ms,
            verify_ms=t_verify_ms,
            total_turn_ms=t_total_ms,
            frame_reused=frame_reused,
        )

        self.stats.total_turns += 1
        self.stats.recent_latencies.append(metrics)
        if len(self.stats.recent_latencies) > 20:
            self.stats.recent_latencies.pop(0)

        if move_verified:
            self.stats.successful_turns += 1
            self.stats.total_score_delta += move.predicted_score_delta
            self.stats.total_lines_cleared += move.lines_cleared
            self.stats.last_turn_timestamp = time.time()
            self.log(
                f"Turn {self.stats.total_turns} VERIFIED: +{move.predicted_score_delta} pts, "
                f"{move.lines_cleared} lines | Total: {t_total_ms:.1f}ms"
            )
            if self.on_turn_completed:
                self.on_turn_completed(metrics, move, True)

            # Auto loop progression or manual step state transition
            if self._pause_requested or self.state == AutonomousState.PAUSED:
                self.set_state(AutonomousState.PAUSED, f"Turn {self.stats.total_turns} verified; paused as requested")
            elif self.is_running():
                self.set_state(AutonomousState.CAPTURING, f"Turn {self.stats.total_turns} verified; preparing next capture")
            else:
                self.set_state(AutonomousState.VERIFYING, f"Single step {self.stats.total_turns} complete")
            return True
        else:
            self.stats.failed_turns += 1
            if verification_status_str == "TRAY_REJECTED":
                self.log(
                    f"Turn {self.stats.total_turns} REJECTED BY GAME: Piece returned to tray slot {move.slot_idx}. "
                    f"Halting for safety."
                )
            else:
                self.log(
                    f"Turn {self.stats.total_turns} VERIFICATION FAILED ({verification_status_str}): "
                    f"{fail_reason}. Expected {hex(expected_bitboard)}, Got {hex(post_semantic_bitboard)}. Pausing for safety."
                )
            if self.on_turn_completed:
                self.on_turn_completed(metrics, move, False)
            self.pause(f"Post-move verification failed: {verification_status_str}")
            return False

    # -------------------------------------------------------------------------
    # Verified Frame Reuse Safety Gate (Phase 1.3)
    # -------------------------------------------------------------------------

    def _can_reuse_verified_frame(
        self,
        verification_res: VerificationResult,
        lines_cleared: int = 0,
    ) -> tuple[bool, str]:
        """Validate whether the verified post-action frame can be safely reused for the next turn.

        Mandatory safety criteria for Frame Reuse (Phase 1.3):
        1. Verification status must be VERIFIED.
        2. No line clears occurred (lines_cleared == 0).
        3. new_frame_result exists and status is SUCCESS.
        4. No error_message or abnormal status reported.
        5. Board exists with full 64 cells.
        6. Every cell has confidence >= 0.70 and is not UNKNOWN or UNCERTAIN.
        7. Overall board and cell confidence >= 0.70.
        8. Board bounding box is valid.
        9. Every tray slot has confidence >= 0.70 and is not UNKNOWN.
        10. Tray still contains at least one valid remaining piece (refill not needed).
        """
        # Condition 1: Verification status must be VERIFIED
        if verification_res.status != VerificationStatus.VERIFIED:
            return False, f"Verification status is {verification_res.status.value} (expected VERIFIED)"

        # Condition 2: No lines cleared
        if lines_cleared > 0:
            return False, f"Line-clear detected ({lines_cleared} lines cleared); board animations require fresh capture"

        # Condition 3 & 4: new_frame_result exists and is SUCCESS without errors
        new_res = verification_res.new_frame_result
        if new_res is None:
            return False, "new_frame_result is None"

        if new_res.status != RecognitionStatus.SUCCESS:
            return False, f"Recognition status is {new_res.status.value} (expected SUCCESS)"

        if new_res.error_message:
            return False, f"Frame reported error: {new_res.error_message}"

        # Condition 5: Board exists with 64 cells
        board = new_res.board
        if board is None:
            return False, "Board detection is None"

        if not hasattr(board, "cells") or not isinstance(board.cells, (tuple, list)):
            return False, "Board cells missing or invalid"

        # Condition 7: Board overall and average cell confidence
        min_conf = 0.70
        cfg_v = getattr(self.recognizer, "config", None)
        if cfg_v is not None:
            cfg_val = getattr(cfg_v, "min_cell_confidence", None)
            if isinstance(cfg_val, (int, float)) and math.isfinite(cfg_val):
                min_conf = float(cfg_val)

        if (
            not isinstance(board.board_confidence, (int, float))
            or not math.isfinite(board.board_confidence)
            or board.board_confidence < min_conf
        ):
            return False, f"Board confidence ({getattr(board, 'board_confidence', None)}) < {min_conf}"

        if (
            not isinstance(board.cell_confidence, (int, float))
            or not math.isfinite(board.cell_confidence)
            or board.cell_confidence < min_conf
        ):
            return False, f"Average cell confidence ({getattr(board, 'cell_confidence', None)}) < {min_conf}"

        # Condition 6: Validate every individual cell of the 8x8 grid
        total_cells_count = 0
        for r_idx, row in enumerate(board.cells):
            if not isinstance(row, (tuple, list)) or len(row) != 8:
                return False, f"Board row {r_idx} is not a valid sequence of length 8"
            for c_idx, cell in enumerate(row):
                total_cells_count += 1
                c_conf = getattr(cell, "confidence", None)
                if not isinstance(c_conf, (int, float)) or not math.isfinite(c_conf):
                    return False, f"Cell ({r_idx}, {c_idx}) has missing or invalid confidence: {c_conf}"
                if c_conf < min_conf:
                    return False, f"Cell ({r_idx}, {c_idx}) confidence ({c_conf:.2f}) < {min_conf:.2f}"
                c_state = getattr(cell, "state", None)
                if c_state == CellState.UNCERTAIN or getattr(c_state, "name", "") in ("UNCERTAIN", "UNKNOWN"):
                    return False, f"Cell ({r_idx}, {c_idx}) is uncertain or unknown (state={c_state})"

        if total_cells_count != 64:
            return False, f"Board cell count invalid: found {total_cells_count} cells (expected 64)"

        # Condition 8: Board bbox geometry check
        if hasattr(board, "board_bbox") and board.board_bbox:
            if len(board.board_bbox) == 4:
                bx, by, bw, bh = board.board_bbox
                if bw > 0 and bh > 0:
                    ar = bw / bh
                    if not (0.90 <= ar <= 1.10):
                        return False, f"Abnormal board aspect ratio: {ar:.2f} (expected 0.90-1.10)"

        # Condition 9: Tray slots confidence >= 0.70 and not UNKNOWN
        if not hasattr(new_res, "tray_pieces") or not isinstance(new_res.tray_pieces, (tuple, list)):
            return False, "Tray pieces missing or invalid"

        valid_remaining_pieces = 0
        for i, slot in enumerate(new_res.tray_pieces):
            slot_conf = getattr(slot, "confidence", None)
            if not isinstance(slot_conf, (int, float)) or not math.isfinite(slot_conf):
                return False, f"Tray slot {i} has missing or invalid confidence: {slot_conf}"
            if slot_conf < min_conf:
                return False, f"Tray slot {i} confidence ({slot_conf:.2f}) < {min_conf:.2f}"
            slot_name = getattr(slot, "matched_name", None)
            if slot_name == "UNKNOWN":
                return False, f"Tray slot {i} is UNKNOWN"
            if not getattr(slot, "is_empty", False):
                if slot.piece is None or slot_name == "UNKNOWN":
                    return False, f"Tray slot {i} contains invalid/unknown piece"
                valid_remaining_pieces += 1

        # Condition 10: Tray after move must still have valid pieces (refill required if 0)
        if valid_remaining_pieces == 0:
            return False, "Tray has no valid remaining pieces (refill required)"

        return True, ""

    # -------------------------------------------------------------------------
    # Physical Gesture Execution Helper
    # -------------------------------------------------------------------------

    def _execute_physical_gesture(
        self,
        move: PlannedMove,
        piece_model: Piece,
        base_frame: np.ndarray,
        start_x: int,
        start_y: int,
        finger_x_0: int,
        finger_y_0: int,
    ) -> ClosedLoopStepResult:
        """Executes touch gesture via interactive shell or simulated dry-run."""
        if self.config.dry_run:
            # Synthetic dry-run evaluation: return simulated successful convergence
            x_target_grid = self.closed_loop.config.board_bbox[0] + (move.col + piece_model.width / 2.0) * self.closed_loop.config.cell_size_px
            y_target_grid = self.closed_loop.config.board_bbox[1] + (move.row + piece_model.height / 2.0) * self.closed_loop.config.cell_size_px
            return ClosedLoopStepResult(
                status=ClosedLoopStatus.CONVERGED,
                step_index=0,
                piece_name=move.piece_name,
                slot_idx=move.slot_idx,
                target_grid_coord=(move.row, move.col),
                target_screen_center=(int(round(x_target_grid)), int(round(y_target_grid))),
                initial_finger_down=(start_x, start_y),
                current_finger_pos=(finger_x_0, finger_y_0),
                detected_piece_center=(x_target_grid, y_target_grid),
                detected_bbox=(
                    int(x_target_grid - piece_model.width * 60),
                    int(y_target_grid - piece_model.height * 60),
                    int(piece_model.width * 120),
                    int(piece_model.height * 120),
                ),
                error_x_px=0.0,
                error_y_px=0.0,
                proposed_correction_move=None,
                correction_count=0,
                tracker_detection=None,
                is_dry_run=True,
                message="[DRY-RUN] Simulated closed-loop convergence.",
            )

        # Pointer safety check before starting new gesture
        if self._touch_pointer_down:
            self.log("WARNING: Active touch pointer detected before new gesture! Releasing failsafe pointer.")
            self._release_pointer_failsafe(*self._pointer_pos)
            self._touch_pointer_down = False
            time.sleep(0.08)

        t_act_start = time.perf_counter()
        cur_x, cur_y = finger_x_0, finger_y_0
        self._pointer_pos = (start_x, start_y)
        touch_down = False
        shell = self._get_interactive_shell()

        # Compute smooth trajectory waypoints (3 intermediate interpolation steps)
        steps_count = 3
        waypoints: list[tuple[int, int]] = []
        for i in range(1, steps_count + 1):
            alpha = i / float(steps_count)
            wx = int(round(start_x + alpha * (finger_x_0 - start_x)))
            wy = int(round(start_y + alpha * (finger_y_0 - start_y)))
            waypoints.append((wx, wy))

        # ---------------------------------------------------------------------
        # MODE 1: Fast Gesture Mode (Default for continuous AUTO play)
        # ---------------------------------------------------------------------
        if self.config.fast_gesture_mode:
            try:
                # Step A: DOWN at tray slot
                t_down = time.perf_counter()
                self.log(f"[GESTURE] DOWN ({start_x}, {start_y}) [pointer_active=True]")
                shell.stdin.write(f"input motionevent DOWN {start_x} {start_y}\n")
                shell.stdin.flush()
                touch_down = True
                self._touch_pointer_down = True
                time.sleep(0.08)  # Pickup dwell for piece grab registration

                # Step B: Intermediate smooth trajectory MOVEs
                for idx, (wx, wy) in enumerate(waypoints[:-1]):
                    t_move = time.perf_counter()
                    dt_ms = (t_move - t_down) * 1000.0
                    self.log(f"[GESTURE] MOVE step {idx + 1} ({wx}, {wy}) [t={dt_ms:.1f}ms]")
                    shell.stdin.write(f"input motionevent MOVE {wx} {wy}\n")
                    shell.stdin.flush()
                    self._pointer_pos = (wx, wy)
                    time.sleep(0.04)

                # Step C: Final target snap MOVE
                end_x, end_y = waypoints[-1]
                t_final_move = time.perf_counter()
                dt_final_ms = (t_final_move - t_down) * 1000.0
                self.log(f"[GESTURE] MOVE target ({end_x}, {end_y}) [t={dt_final_ms:.1f}ms]")
                shell.stdin.write(f"input motionevent MOVE {end_x} {end_y}\n")
                shell.stdin.flush()
                self._pointer_pos = (end_x, end_y)
                cur_x, cur_y = end_x, end_y
                time.sleep(0.12)  # Settle dwell at target snap position

                # Step D: Confirmed UP with synchronization token
                t_up = time.perf_counter()
                dt_up_ms = (t_up - t_down) * 1000.0
                self.log(f"[GESTURE] UP ({end_x}, {end_y}) [t={dt_up_ms:.1f}ms, awaiting confirmation]")
                shell.stdin.write(f"input motionevent UP {end_x} {end_y} && echo GESTURE_UP_OK\n")
                shell.stdin.flush()

                token_ok = self._read_shell_until_token(shell, token="GESTURE_UP_OK", timeout_sec=2.0)
                if not token_ok:
                    self.log("ERROR: Interactive shell timed out waiting for GESTURE_UP_OK. Sending emergency UP.")
                    self._close_shell_hard()
                    self._release_pointer_failsafe(end_x, end_y)
                    touch_down = False
                    self._touch_pointer_down = False
                    return ClosedLoopStepResult(
                        status=ClosedLoopStatus.UNKNOWN,
                        step_index=0,
                        piece_name=move.piece_name,
                        slot_idx=move.slot_idx,
                        target_grid_coord=(move.row, move.col),
                        target_screen_center=(end_x, end_y),
                        initial_finger_down=(start_x, start_y),
                        current_finger_pos=(end_x, end_y),
                        detected_piece_center=None,
                        detected_bbox=None,
                        error_x_px=0.0,
                        error_y_px=0.0,
                        proposed_correction_move=None,
                        correction_count=0,
                        tracker_detection=None,
                        is_dry_run=False,
                        message="Gesture UP confirmation timed out; emergency release sent.",
                    )

                touch_down = False
                self._touch_pointer_down = False
                t_total_act_ms = (time.perf_counter() - t_act_start) * 1000.0
                self.log(f"[GESTURE] UP CONFIRMED at ({end_x}, {end_y}) [total={t_total_act_ms:.1f}ms, pointer_active=False]")

                return ClosedLoopStepResult(
                    status=ClosedLoopStatus.CONVERGED,
                    step_index=0,
                    piece_name=move.piece_name,
                    slot_idx=move.slot_idx,
                    target_grid_coord=(move.row, move.col),
                    target_screen_center=(end_x, end_y),
                    initial_finger_down=(start_x, start_y),
                    current_finger_pos=(end_x, end_y),
                    detected_piece_center=(end_x, end_y),
                    detected_bbox=(
                        int(end_x - piece_model.width * 60),
                        int(end_y - piece_model.height * 60),
                        int(piece_model.width * 120),
                        int(piece_model.height * 120),
                    ),
                    error_x_px=0.0,
                    error_y_px=0.0,
                    proposed_correction_move=None,
                    correction_count=0,
                    tracker_detection=None,
                    is_dry_run=False,
                    message=f"Fast gesture completed in {t_total_act_ms:.1f}ms",
                )
            except Exception as exc:
                self.log(f"Exception during physical actuation: {exc}")
                if touch_down or self._touch_pointer_down:
                    self._release_pointer_failsafe(cur_x, cur_y)
                    touch_down = False
                    self._touch_pointer_down = False
                self._close_shell_hard()
                return ClosedLoopStepResult(
                    status=ClosedLoopStatus.UNKNOWN,
                    step_index=0,
                    piece_name=move.piece_name,
                    slot_idx=move.slot_idx,
                    target_grid_coord=(move.row, move.col),
                    target_screen_center=(0.0, 0.0),
                    initial_finger_down=(start_x, start_y),
                    current_finger_pos=(cur_x, cur_y),
                    detected_piece_center=None,
                    detected_bbox=None,
                    error_x_px=0.0,
                    error_y_px=0.0,
                    proposed_correction_move=None,
                    correction_count=0,
                    tracker_detection=None,
                    is_dry_run=False,
                    message=f"Actuation exception: {exc}",
                )
            finally:
                if touch_down or self._touch_pointer_down:
                    self._release_pointer_failsafe(cur_x, cur_y)
                    touch_down = False
                    self._touch_pointer_down = False

        # ---------------------------------------------------------------------
        # MODE 2: In-Flight Hold Evaluation Mode (with Screencap and Tracker)
        # ---------------------------------------------------------------------
        try:
            # Step A: DOWN at tray slot
            t_down = time.perf_counter()
            self.log(f"[GESTURE] DOWN ({start_x}, {start_y}) [pointer_active=True]")
            shell.stdin.write(f"input motionevent DOWN {start_x} {start_y}\n")
            shell.stdin.flush()
            touch_down = True
            self._touch_pointer_down = True
            time.sleep(0.15)

            # Step B: Intermediate smooth trajectory MOVEs
            for idx, (wx, wy) in enumerate(waypoints[:-1]):
                shell.stdin.write(f"input motionevent MOVE {wx} {wy}\n")
                shell.stdin.flush()
                self._pointer_pos = (wx, wy)
                time.sleep(0.04)

            # Target MOVE
            end_x, end_y = waypoints[-1]
            shell.stdin.write(f"input motionevent MOVE {end_x} {end_y}\n")
            shell.stdin.flush()
            self._pointer_pos = (end_x, end_y)
            cur_x, cur_y = end_x, end_y
            time.sleep(0.18)

            # Step C: Capture mid_hold while holding touch
            shell.stdin.write("screencap -p /sdcard/bb_mid.png && echo MID1_OK\n")
            shell.stdin.flush()

            token_ok = self._read_shell_until_token(shell, token="MID1_OK", timeout_sec=4.0)
            if not token_ok:
                self.log("ERROR: Interactive shell timed out waiting for in-flight screencap.")
                self._close_shell_hard()
                self._release_pointer_failsafe(cur_x, cur_y)
                touch_down = False
                self._touch_pointer_down = False
                return ClosedLoopStepResult(
                    status=ClosedLoopStatus.UNKNOWN,
                    step_index=0,
                    piece_name=move.piece_name,
                    slot_idx=move.slot_idx,
                    target_grid_coord=(move.row, move.col),
                    target_screen_center=(0.0, 0.0),
                    initial_finger_down=(start_x, start_y),
                    current_finger_pos=(cur_x, cur_y),
                    detected_piece_center=None,
                    detected_bbox=None,
                    error_x_px=0.0,
                    error_y_px=0.0,
                    proposed_correction_move=None,
                    correction_count=0,
                    tracker_detection=None,
                    is_dry_run=False,
                    message="Interactive shell timed out waiting for screencap; emergency UP sent.",
                )

            # Pull mid frame
            mid_local = str(self.debug_dir / "live_mid.png")
            subprocess.run([self.config.adb_path, "-s", self.config.device_id, "pull", "/sdcard/bb_mid.png", mid_local],
                           capture_output=True, timeout=5.0)
            subprocess.run([self.config.adb_path, "-s", self.config.device_id, "shell", "rm -f /sdcard/bb_mid.png"],
                           capture_output=True, timeout=4.0)

            mid_img = cv2.imread(mid_local)
            if mid_img is None:
                raise RuntimeError("Failed to decode in-flight hold image")

            step1_res = self.closed_loop.evaluate_held_frame(
                move=move,
                slot_center=(start_x, start_y),
                current_finger_pos=(finger_x_0, finger_y_0),
                base_frame=base_frame,
                mid_hold_frame=mid_img,
                correction_count=0,
                piece_width_cells=piece_model.width,
                piece_height_cells=piece_model.height,
            )

            # Decision
            if step1_res.status == ClosedLoopStatus.CONVERGED:
                time.sleep(0.12)
                shell.stdin.write(f"input motionevent UP {finger_x_0} {finger_y_0} && echo GESTURE_UP_OK\n")
                shell.stdin.flush()
                self._read_shell_until_token(shell, token="GESTURE_UP_OK", timeout_sec=2.0)
                touch_down = False
                self._touch_pointer_down = False
                return step1_res

            elif step1_res.status == ClosedLoopStatus.NEEDS_CORRECTION and step1_res.proposed_correction_move:
                corr_x, corr_y = step1_res.proposed_correction_move
                cur_x, cur_y = corr_x, corr_y
                shell.stdin.write(f"input motionevent MOVE {corr_x} {corr_y}\n")
                shell.stdin.flush()
                self._pointer_pos = (corr_x, corr_y)
                time.sleep(0.18)

                # Safe release UP after micro-correction
                time.sleep(0.12)
                shell.stdin.write(f"input motionevent UP {corr_x} {corr_y} && echo GESTURE_UP_OK\n")
                shell.stdin.flush()
                self._read_shell_until_token(shell, token="GESTURE_UP_OK", timeout_sec=2.0)
                touch_down = False
                self._touch_pointer_down = False
                return step1_res

            else:
                # Emergency safe release
                shell.stdin.write(f"input motionevent UP {finger_x_0} {finger_y_0} && echo GESTURE_UP_OK\n")
                shell.stdin.flush()
                self._read_shell_until_token(shell, token="GESTURE_UP_OK", timeout_sec=2.0)
                touch_down = False
                self._touch_pointer_down = False
                return step1_res

        except Exception as exc:
            self.log(f"Exception during physical actuation: {exc}")
            if touch_down or self._touch_pointer_down:
                self._release_pointer_failsafe(cur_x, cur_y)
                touch_down = False
                self._touch_pointer_down = False
            self._close_shell_hard()
            return ClosedLoopStepResult(
                status=ClosedLoopStatus.UNKNOWN,
                step_index=0,
                piece_name=move.piece_name,
                slot_idx=move.slot_idx,
                target_grid_coord=(move.row, move.col),
                target_screen_center=(0.0, 0.0),
                initial_finger_down=(start_x, start_y),
                current_finger_pos=(cur_x, cur_y),
                detected_piece_center=None,
                detected_bbox=None,
                error_x_px=0.0,
                error_y_px=0.0,
                proposed_correction_move=None,
                correction_count=0,
                tracker_detection=None,
                is_dry_run=False,
                message=f"Actuation exception: {exc}",
            )
        finally:
            if touch_down or self._touch_pointer_down:
                self._release_pointer_failsafe(cur_x, cur_y)
                touch_down = False
                self._touch_pointer_down = False

    # -------------------------------------------------------------------------
    # Internal Main Loop Thread
    # -------------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Background worker thread executing continuous turns."""
        self.log("Autonomous Game Loop thread started.")

        try:
            while self._is_running and not self._stop_requested:
                if self._pause_requested or self.state == AutonomousState.PAUSED:
                    time.sleep(self.config.loop_idle_sleep_sec)
                    continue

                if self.state in (AutonomousState.GAME_OVER, AutonomousState.ERROR):
                    time.sleep(self.config.loop_idle_sleep_sec)
                    break

                # Execute one turn
                success = self.step_single_turn()

                if self.config.max_turns and self.stats.total_turns >= self.config.max_turns:
                    self.log(f"Reached maximum requested turns ({self.config.max_turns}).")
                    self.pause("Max turns reached")
                    break

                if not success and self.state not in (AutonomousState.PAUSED, AutonomousState.GAME_OVER):
                    time.sleep(self.config.loop_idle_sleep_sec)

        except Exception as exc:
            self.log(f"Fatal error in autonomous loop: {exc}")
            self.last_error_message = str(exc)
            self.set_state(AutonomousState.ERROR, str(exc))
        finally:
            self._is_running = False
            self._release_pointer_failsafe()
            self._close_shell()
            self.log("Autonomous Game Loop thread terminated.")

    # -------------------------------------------------------------------------
    # ADB Shell Management & Fail-Safe Pointer Release
    # -------------------------------------------------------------------------

    def _get_interactive_shell(self) -> subprocess.Popen:
        """Retrieve or spawn persistent ADB interactive shell."""
        if self._shell_proc is None or self._shell_proc.poll() is not None:
            self._shell_proc = subprocess.Popen(
                [self.config.adb_path, "-s", self.config.device_id, "shell"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        return self._shell_proc

    def _read_shell_until_token(
        self,
        shell: subprocess.Popen,
        token: str = "MID1_OK",
        timeout_sec: float = 4.0,
    ) -> bool:
        """Reads lines from shell.stdout until `token` appears or timeout occurs."""
        line_queue: queue.Queue[Optional[str]] = queue.Queue()

        def _reader() -> None:
            try:
                while True:
                    line = shell.stdout.readline()
                    if not line:
                        line_queue.put(None)
                        break
                    line_queue.put(line.strip())
                    if token in line:
                        break
            except Exception:
                line_queue.put(None)

        t = threading.Thread(target=_reader, daemon=True)
        t.start()

        deadline = time.perf_counter() + timeout_sec
        while time.perf_counter() < deadline:
            remaining = max(0.05, deadline - time.perf_counter())
            try:
                val = line_queue.get(timeout=remaining)
                if val is None:
                    return False
                if token in val:
                    return True
            except queue.Empty:
                break
        return False

    def _close_shell_hard(self) -> None:
        """Forcibly kill the interactive shell process and close file descriptors."""
        if self._shell_proc is not None:
            try:
                self._shell_proc.kill()
            except Exception:
                pass
            for stream_name in ("stdin", "stdout", "stderr"):
                try:
                    stream = getattr(self._shell_proc, stream_name, None)
                    if stream and not stream.closed:
                        stream.close()
                except Exception:
                    pass
            self._shell_proc = None

    def _close_shell(self) -> None:
        if self._shell_proc is not None:
            try:
                self._shell_proc.stdin.write("exit\n")
                self._shell_proc.stdin.flush()
                self._shell_proc.wait(timeout=1.5)
            except Exception:
                try:
                    self._shell_proc.kill()
                except Exception:
                    pass
            for stream_name in ("stdin", "stdout", "stderr"):
                try:
                    stream = getattr(self._shell_proc, stream_name, None)
                    if stream and not stream.closed:
                        stream.close()
                except Exception:
                    pass
            self._shell_proc = None

    def _release_pointer_failsafe(self, x: int = 500, y: int = 1000) -> None:
        """Guarantees touch UP is injected via dedicated one-shot ADB command."""
        self._touch_pointer_down = False
        if self.config.dry_run:
            return
        try:
            subprocess.run(
                [self.config.adb_path, "-s", self.config.device_id, "shell", f"input motionevent UP {x} {y}"],
                capture_output=True,
                timeout=4.0,
            )
        except Exception:
            pass
