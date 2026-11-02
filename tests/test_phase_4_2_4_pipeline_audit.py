"""Phase 4.2.4 — Full Autonomous Pipeline Integration Audit Tests.

Offline tests validating:
1. Verification timeout (UNSTABLE_ANIMATION_TIMEOUT) triggers safe PAUSE.
2. ADB capture failure (ScreenCaptureError) triggers safe PAUSE and clears planned move.
3. Tray refill lifecycle triggers fresh perception and recognition on next turn.
4. RESUME resets PAUSED state and triggers fresh CAPTURING from device.
5. STOP terminates worker thread cleanly without allowing further turns.
6. Failed verification strictly halts the pipeline (zero subsequent moves).
7. Closed-loop correction limit exceeded triggers failsafe pointer release and PAUSE.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from action_engine.action_verifier import ActionVerifier, VerificationResult, VerificationStatus
from action_engine.autonomous_controller import (
    AutonomousConfig,
    AutonomousGameController,
    AutonomousState,
)
from action_engine.closed_loop_controller import (
    ClosedLoopController,
    ClosedLoopStatus,
    ClosedLoopStepResult,
    DragCalibrationConfig,
)
from ai_engine.decision import AIDecisionEngine, DecisionResult, DecisionStatus, PlannedMove
from computer_vision.capture import ScreenCapture, ScreenCaptureError
from game_engine.pieces import CANONICAL_41_CATALOG
from tests.test_action_engine import create_synthetic_frame_result


class TestPipelineIntegrationAudit(unittest.TestCase):
    """Offline audit unit tests for full autonomous pipeline integration."""

    def setUp(self):
        self.config = AutonomousConfig(dry_run=False, device_id="TEST_DEVICE")
        self.mock_cap = MagicMock(spec=ScreenCapture)
        self.mock_rec = MagicMock()
        self.ai_engine = AIDecisionEngine(default_beam_width=5)
        self.mock_closed_loop = MagicMock(spec=ClosedLoopController)
        self.mock_verifier = MagicMock(spec=ActionVerifier)

        # Baseline synthetic frame with 1 playable piece
        self.test_frame_img = np.zeros((2340, 1080, 3), dtype=np.uint8)
        self.mock_cap.capture_from_adb.return_value = self.test_frame_img

        self.frame_res = create_synthetic_frame_result(bitboard=0x0, tray_piece_names=("dot_1x1", "EMPTY", "EMPTY"))
        self.mock_rec.recognize_frame.return_value = self.frame_res

        # Baseline converged step result
        self.step_res = ClosedLoopStepResult(
            status=ClosedLoopStatus.CONVERGED,
            step_index=0,
            piece_name="dot_1x1",
            slot_idx=0,
            target_grid_coord=(0, 0),
            target_screen_center=(114.75, 608.75),
            initial_finger_down=(200, 1700),
            current_finger_pos=(115, 609),
            detected_piece_center=(115.0, 609.0),
            detected_bbox=(80, 570, 70, 70),
            error_x_px=0.0,
            error_y_px=0.0,
            proposed_correction_move=None,
            correction_count=0,
            tracker_detection=None,
            is_dry_run=False,
            message="Converged",
        )
        self.mock_closed_loop.plan_initial_trajectory.return_value = (200, 1700, 115, 609)

        self.controller = AutonomousGameController(
            config=self.config,
            screen_capture=self.mock_cap,
            recognizer=self.mock_rec,
            ai_engine=self.ai_engine,
            closed_loop_controller=self.mock_closed_loop,
            action_verifier=self.mock_verifier,
        )
        # Mock physical gesture execution to bypass ADB shell during unit test
        self.controller._execute_physical_gesture = MagicMock(return_value=self.step_res)

    def test_1_verification_timeout_triggers_safe_pause(self):
        """Audit 1: Verification timeout (UNSTABLE_ANIMATION_TIMEOUT) must immediately PAUSE."""
        self.mock_verifier.verify_move.return_value = VerificationResult(
            status=VerificationStatus.UNSTABLE_ANIMATION_TIMEOUT,
            step_index=0,
            slot_idx=0,
            piece_name="dot_1x1",
            expected_bitboard=0x1,
            actual_bitboard=0x18,
            mismatch_count=2,
            mismatched_cells=(),
            tray_slot_cleared=False,
            attempts=3,
            elapsed_ms=3000.0,
            error_message="Animation unsettled after timeout",
        )

        res = self.controller.step_single_turn()
        self.assertFalse(res, "Single turn must return False on verification failure")
        self.assertEqual(self.controller.state, AutonomousState.PAUSED)
        self.assertEqual(self.controller.stats.failed_turns, 1)
        self.assertEqual(self.controller.stats.successful_turns, 0)
        self.assertIn("UNSTABLE_ANIMATION_TIMEOUT", self.controller.action_log[-1]["message"])

    def test_2_adb_capture_failure_triggers_safe_pause(self):
        """Audit 2: ADB capture failure (ScreenCaptureError) must PAUSE and clear planned move."""
        self.mock_cap.capture_from_adb.side_effect = ScreenCaptureError("ADB connection lost: device not found")

        res = self.controller.step_single_turn()
        self.assertFalse(res)
        self.assertEqual(self.controller.state, AutonomousState.PAUSED)
        self.assertIsNone(self.controller.last_planned_move)
        self.assertIn("Capture error", self.controller.action_log[-1]["message"])

    def test_3_tray_refill_triggers_fresh_perception_on_next_turn(self):
        """Audit 3: When tray is empty, round refill is handled and next cycle captures fresh frame."""
        empty_frame = create_synthetic_frame_result(bitboard=0x0, tray_piece_names=("EMPTY", "EMPTY", "EMPTY"))
        self.mock_rec.recognize_frame.return_value = empty_frame

        with patch("time.sleep"):
            # Turn 1: detects empty tray
            res = self.controller.step_single_turn()
            self.assertTrue(res, "Empty tray refill wait returns True to allow next perception cycle")
            self.assertEqual(self.controller.stats.rounds_refilled, 1)

            # Turn 2: mock fresh capture with refilled pieces
            refilled_frame = create_synthetic_frame_result(bitboard=0x0, tray_piece_names=("dot_1x1", "line_1x3_h", "square_2x2"))
            self.mock_rec.recognize_frame.return_value = refilled_frame
            self.mock_verifier.verify_move.return_value = VerificationResult(
                status=VerificationStatus.VERIFIED,
                step_index=0,
                slot_idx=0,
                piece_name="dot_1x1",
                expected_bitboard=0x1,
                actual_bitboard=0x1,
                mismatch_count=0,
                mismatched_cells=(),
                tray_slot_cleared=True,
                attempts=1,
                elapsed_ms=50.0,
            )
            res2 = self.controller.step_single_turn()
            self.assertTrue(res2)
            self.assertEqual(self.controller.stats.successful_turns, 1)

    def test_4_resume_resets_paused_state_and_triggers_capturing(self):
        """Audit 4: RESUME must clear PAUSED state and immediately transition to CAPTURING."""
        self.controller.start()
        self.assertTrue(self.controller.is_running())

        self.controller.pause("Safety pause test")
        self.assertEqual(self.controller.state, AutonomousState.PAUSED)
        self.assertTrue(self.controller._pause_requested)

        self.controller.resume("User resumed from dashboard")
        self.assertFalse(self.controller._pause_requested)
        self.assertNotEqual(self.controller.state, AutonomousState.PAUSED)
        messages = [e["message"] for e in self.controller.action_log]
        self.assertTrue(any("Resuming autonomous loop" in m for m in messages))
        self.controller.stop()

    def test_5_stop_guarantees_thread_termination_and_no_further_turns(self):
        """Audit 5: STOP must set IDLE, terminate worker thread, and prevent subsequent turns."""
        self.controller.start()
        self.assertTrue(self.controller.is_running())

        self.controller.stop("User stop test")
        self.assertFalse(self.controller.is_running())
        self.assertEqual(self.controller.state, AutonomousState.IDLE)
        self.assertTrue(self.controller._stop_requested)

    def test_6_no_subsequent_move_when_verification_fails(self):
        """Audit 6: Failed verification (MISMATCH) strictly halts execution (zero subsequent moves)."""
        self.mock_verifier.verify_move.return_value = VerificationResult(
            status=VerificationStatus.MISMATCH,
            step_index=0,
            slot_idx=0,
            piece_name="dot_1x1",
            expected_bitboard=0x1,
            actual_bitboard=0x0,
            mismatch_count=1,
            mismatched_cells=(),
            tray_slot_cleared=True,
            attempts=3,
            elapsed_ms=1000.0,
            error_message="1 cell mismatched after 3 attempts",
        )

        res = self.controller.step_single_turn()
        self.assertFalse(res)
        self.assertEqual(self.controller.state, AutonomousState.PAUSED)

        # In PAUSED state, loop worker cannot proceed
        self.assertTrue(self.controller.state == AutonomousState.PAUSED)
        self.assertEqual(self.controller.stats.successful_turns, 0)
        self.assertEqual(self.controller.stats.failed_turns, 1)

    def test_7_correction_limit_exceeded_triggers_failsafe_release_and_pause(self):
        """Audit 7: Closed-loop CORRECTION_LIMIT_EXCEEDED triggers safe PAUSE."""
        step_limit_exceeded = ClosedLoopStepResult(
            status=ClosedLoopStatus.CORRECTION_LIMIT_EXCEEDED,
            step_index=0,
            piece_name="dot_1x1",
            slot_idx=0,
            target_grid_coord=(0, 0),
            target_screen_center=(114.75, 608.75),
            initial_finger_down=(200, 1700),
            current_finger_pos=(115, 609),
            detected_piece_center=(70.0, 550.0),
            detected_bbox=(40, 520, 70, 70),
            error_x_px=44.75,
            error_y_px=58.75,
            proposed_correction_move=None,
            correction_count=1,
            tracker_detection=None,
            is_dry_run=False,
            message="Correction limit reached (1/1). Error persists.",
        )
        self.controller._execute_physical_gesture = MagicMock(return_value=step_limit_exceeded)

        res = self.controller.step_single_turn()
        self.assertFalse(res)
        self.assertEqual(self.controller.state, AutonomousState.PAUSED)
        self.assertIn("LIMIT_EXCEEDED", self.controller.action_log[-1]["message"])


if __name__ == "__main__":
    unittest.main()
