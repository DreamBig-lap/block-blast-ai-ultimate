"""Auto Play Controller & State Machine for BLOCK BLAST AI ULTIMATE.

Orchestrates the continuous game loop across ASSIST, AUTO, and PAUSE modes.
Integrates live ADB capture, Computer Vision recognition, AI Decision planning,
touch drag actuation, and post-move bitboard verification with emergency stop controls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import enum
import logging
import threading
import time
from typing import Callable, Optional

from action_engine.action_verifier import ActionVerifier, VerificationResult, VerificationStatus
from action_engine.coordinate_mapper import CoordinateMapper, DragTrajectory
from action_engine.touch_controller import TouchController
from ai_engine.decision import AIDecisionEngine, DecisionResult, DecisionStatus, PlannedMove
from computer_vision.capture import ScreenCapture, ScreenCaptureError
from computer_vision.config import DEFAULT_VISION_CONFIG, VisionConfig
from computer_vision.models import FrameRecognitionResult, RecognitionStatus
from computer_vision.recognizer import BlockBlastRecognizer
from game_engine.simulator import StepResult

logger = logging.getLogger(__name__)


class PlayMode(enum.Enum):
    """Operational mode of the auto-play system."""
    ASSIST = "ASSIST"    # AI recommends optimal moves; no touchscreen actuation
    AUTO = "AUTO"        # AI autonomously executes moves with verified drag-and-drop
    PAUSE = "PAUSE"      # System frozen; waits for user instruction or mode switch


@dataclass
class AutoPlayConfig:
    """Runtime configuration options for AutoPlayer."""
    mode: PlayMode = PlayMode.PAUSE
    strategy: str = "beam_search"
    beam_width: int = 15
    device_id: Optional[str] = None
    adb_path: str = "adb"
    vertical_offset_ratio: float = 1.65
    drag_duration_ms: int = 380
    post_move_settle_sec: float = 0.35
    max_rounds: Optional[int] = None
    dry_run: bool = False
    refill_wait_sec: float = 0.50
    loop_idle_sleep_sec: float = 0.20


@dataclass
class AutoPlayStats:
    """Cumulative operational metrics for the active session."""
    total_rounds: int = 0
    total_moves: int = 0
    successful_moves: int = 0
    failed_moves: int = 0
    total_score_delta: int = 0
    total_lines_cleared: int = 0
    session_start_time: float = field(default_factory=time.time)
    last_action_timestamp: float = field(default_factory=time.time)

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.session_start_time

    @property
    def verification_success_rate(self) -> float:
        total = self.successful_moves + self.failed_moves
        return (self.successful_moves / total * 100.0) if total > 0 else 100.0


class AutoPlayer:
    """Master controller managing the auto-play lifecycle, modes, and execution safety."""

    def __init__(
        self,
        config: Optional[AutoPlayConfig] = None,
        vision_config: Optional[VisionConfig] = None,
        touch_controller: Optional[TouchController] = None,
        coordinate_mapper: Optional[CoordinateMapper] = None,
        action_verifier: Optional[ActionVerifier] = None,
        recognizer: Optional[BlockBlastRecognizer] = None,
        ai_engine: Optional[AIDecisionEngine] = None,
    ) -> None:
        self.config = config or AutoPlayConfig()
        v_cfg = vision_config or DEFAULT_VISION_CONFIG

        self.recognizer = recognizer or BlockBlastRecognizer(config=v_cfg)
        self.touch = touch_controller or TouchController(
            adb_path=self.config.adb_path,
            device_id=self.config.device_id,
            dry_run=self.config.dry_run,
        )
        self.mapper = coordinate_mapper or CoordinateMapper(
            vertical_offset_ratio=self.config.vertical_offset_ratio,
            default_drag_duration_ms=self.config.drag_duration_ms,
        )
        self.verifier = action_verifier or ActionVerifier(
            recognizer=self.recognizer,
            post_move_settle_sec=self.config.post_move_settle_sec,
        )
        self.ai_engine = ai_engine or AIDecisionEngine()

        self.stats = AutoPlayStats()
        self._mode = self.config.mode
        self._is_running = False
        self._emergency_stop = False
        self._lock = threading.Lock()

        # Observable callbacks for UI / CLI dashboards
        self.on_frame_update: Optional[Callable[[FrameRecognitionResult], None]] = None
        self.on_plan_update: Optional[Callable[[DecisionResult], None]] = None
        self.on_move_executed: Optional[Callable[[PlannedMove, DragTrajectory, VerificationResult], None]] = None
        self.on_status_change: Optional[Callable[[PlayMode, str], None]] = None

    @property
    def mode(self) -> PlayMode:
        with self._lock:
            return self._mode

    def set_mode(self, new_mode: PlayMode, reason: str = "") -> None:
        """Switch operational mode (e.g., ASSIST -> AUTO -> PAUSE)."""
        with self._lock:
            old_mode = self._mode
            self._mode = new_mode

        logger.info("PlayMode switched: %s -> %s (%s)", old_mode.value, new_mode.value, reason)
        if self.on_status_change:
            self.on_status_change(new_mode, reason)

    def trigger_emergency_stop(self, reason: str = "User Emergency Stop") -> None:
        """Immediately halts all automated operations and enters safe PAUSE."""
        with self._lock:
            self._emergency_stop = True
            self._mode = PlayMode.PAUSE

        logger.warning("EMERGENCY STOP TRIGGERED: %s", reason)
        if self.on_status_change:
            self.on_status_change(PlayMode.PAUSE, f"EMERGENCY STOP: {reason}")

    def reset_emergency_stop(self) -> None:
        """Clear emergency stop flag to allow resuming."""
        with self._lock:
            self._emergency_stop = False

    def is_running(self) -> bool:
        return self._is_running

    def stop(self) -> None:
        """Request the main loop to terminate cleanly."""
        self._is_running = False

    def step_single_cycle(self) -> Optional[DecisionResult]:
        """Execute one complete perceive-plan-act cycle.

        Used in ASSIST mode, step-by-step testing, or inside run_loop().

        Returns:
            DecisionResult if planning succeeded, None otherwise.
        """
        if self._emergency_stop:
            logger.warning("Cycle skipped: Emergency stop active.")
            return None

        # -----------------------------------------------------------------
        # 1. Acquire Frame via ADB
        # -----------------------------------------------------------------
        try:
            img = self.recognizer.capture.capture_from_adb(device_id=self.config.device_id)
        except ScreenCaptureError as exc:
            logger.error("Screen capture failed: %s", exc)
            self.set_mode(PlayMode.PAUSE, f"Screen capture error: {exc}")
            return None

        # -----------------------------------------------------------------
        # 2. Vision Recognition
        # -----------------------------------------------------------------
        frame_res = self.recognizer.recognize_frame(img)
        if self.on_frame_update:
            self.on_frame_update(frame_res)

        if not frame_res.is_usable_for_ai():
            logger.warning("Frame not usable for AI (%s): %s", frame_res.status.value, frame_res.error_message)
            try:
                self.recognizer.save_diagnostic_failure(
                    img,
                    frame_res,
                    pause_reason=f"Unusable vision state: {frame_res.error_message}",
                )
            except Exception as e:
                logger.warning("Failed to save perception diagnostic in auto player: %s", e)

            # If in AUTO mode, safely pause on uncertain or unknown piece detection
            if self.mode == PlayMode.AUTO:
                self.set_mode(PlayMode.PAUSE, f"Unreliable vision state: {frame_res.error_message}")
            return None

        # Check if all tray pieces are empty (waiting for refill animation)
        active_pieces = [p for p in frame_res.tray_pieces if not p.is_empty]
        if not active_pieces:
            logger.debug("Tray is empty. Waiting for refill...")
            time.sleep(self.config.refill_wait_sec)
            return None

        # -----------------------------------------------------------------
        # 3. AI Decision Engine Planning
        # -----------------------------------------------------------------
        decision = self.recognizer.plan_best_moves(
            frame_res,
            strategy=self.config.strategy,
            beam_width=self.config.beam_width,
        )

        if self.on_plan_update:
            self.on_plan_update(decision)

        if decision.status in (DecisionStatus.GAME_OVER, DecisionStatus.NO_VALID_MOVE):
            logger.warning("Game Over / No Valid Moves detected (%s).", decision.status.value)
            self.set_mode(PlayMode.PAUSE, f"Game Over ({decision.status.value})")
            return decision

        if decision.status != DecisionStatus.SUCCESS or decision.move_count == 0:
            logger.warning("AI produced no moves: %s", decision.rationale)
            return decision

        logger.info(
            "AI Plan [%s]: %d moves, predicted +%d pts, rationale: %s",
            self.config.strategy, decision.move_count, decision.predicted_score_delta, decision.rationale
        )

        # -----------------------------------------------------------------
        # 4. Actuation & Verification (Only in AUTO mode)
        # -----------------------------------------------------------------
        if self.mode != PlayMode.AUTO:
            logger.debug("Mode is %s: Plan displayed; skipping physical touch actuation.", self.mode.value)
            return decision

        self.stats.total_rounds += 1
        current_frame = frame_res

        for i, move in enumerate(decision.moves):
            if self._emergency_stop or self.mode != PlayMode.AUTO:
                logger.info("Execution interrupted: mode changed or emergency stop.")
                break

            step_res = decision.step_results[i] if i < len(decision.step_results) else None

            # Calculate physical touch trajectory
            try:
                trajectory = self.mapper.map_move(move, current_frame)
            except Exception as exc:
                logger.error("Coordinate mapping failed for step %d: %s", i, exc)
                self.set_mode(PlayMode.PAUSE, f"Mapping error on step {i}: {exc}")
                break

            # Execute Drag via TouchController
            try:
                self.touch.swipe(
                    x1=trajectory.start_x,
                    y1=trajectory.start_y,
                    x2=trajectory.end_x,
                    y2=trajectory.end_y,
                    duration_ms=trajectory.duration_ms,
                )
            except Exception as exc:
                logger.error("Touch injection failed: %s", exc)
                self.set_mode(PlayMode.PAUSE, f"Touch error on step {i}: {exc}")
                break

            # Post-Action Move Verification
            if step_res is not None and not self.config.dry_run:
                verification = self.verifier.verify_move(step_res, device_id=self.config.device_id, step_index=i)
            else:
                # Dry run synthetic verification
                verification = VerificationResult(
                    status=VerificationStatus.VERIFIED,
                    step_index=move.step_index,
                    slot_idx=move.slot_idx,
                    piece_name=move.piece_name,
                    expected_bitboard=step_res.state_after.board if step_res else 0,
                    actual_bitboard=step_res.state_after.board if step_res else 0,
                    mismatch_count=0,
                    mismatched_cells=(),
                    tray_slot_cleared=True,
                    attempts=1,
                    elapsed_ms=0.0,
                )

            # Record stats
            if verification.is_success:
                self.stats.successful_moves += 1
                self.stats.total_score_delta += move.predicted_score_delta
                self.stats.total_lines_cleared += move.lines_cleared
            else:
                self.stats.failed_moves += 1

            self.stats.total_moves += 1
            self.stats.last_action_timestamp = time.time()

            if self.on_move_executed:
                self.on_move_executed(move, trajectory, verification)

            # Verification Failure Handling
            if not verification.is_success:
                logger.error(
                    "Step %d verification FAILED (%s): %s. Halting for safety.",
                    i, verification.status.value, verification.error_message
                )
                self.set_mode(PlayMode.PAUSE, f"Step {i} verification failed: {verification.status.value}")
                break

            # Update current_frame for subsequent moves in this round
            if verification.new_frame_result is not None:
                current_frame = verification.new_frame_result

        # Wait briefly for tray refill if round finished cleanly
        if self.mode == PlayMode.AUTO and not self._emergency_stop:
            time.sleep(self.config.refill_wait_sec)

        return decision

    def run_loop(self) -> None:
        """Continuous execution loop until stopped or max rounds reached."""
        self._is_running = True
        logger.info("AutoPlayer main loop started in mode: %s", self.mode.value)

        rounds_completed = 0
        try:
            while self._is_running:
                if self._emergency_stop:
                    time.sleep(self.config.loop_idle_sleep_sec)
                    continue

                if self.mode == PlayMode.PAUSE:
                    time.sleep(self.config.loop_idle_sleep_sec)
                    continue

                # Run one perceive-plan-act cycle
                self.step_single_cycle()
                rounds_completed += 1

                if self.config.max_rounds and rounds_completed >= self.config.max_rounds:
                    logger.info("Reached maximum requested rounds (%d). Stopping loop.", self.config.max_rounds)
                    self.set_mode(PlayMode.PAUSE, "Max rounds reached")
                    break

        except KeyboardInterrupt:
            logger.info("Received KeyboardInterrupt. Halting...")
            self.trigger_emergency_stop("KeyboardInterrupt")
        finally:
            self._is_running = False
            logger.info("AutoPlayer main loop exited.")
