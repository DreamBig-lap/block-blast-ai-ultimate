"""Unit tests for Phase 4.2 AUTO LOOP and Gesture Reliability.

Covers:
1. AUTO loop continuous execution for >= 10 turns without manual stepping.
2. PAUSE command safely halting the loop and releasing touch pointers.
3. RESUME command clearing stale recognition/frame and re-perceiving from scratch.
4. Single-step mode executing exactly one turn without auto-advancing.
5. Physical gesture execution always releasing pointer on successful dispatch.
6. Physical gesture execution guaranteed failsafe pointer release on unexpected exception.
7. Verification failure (MISMATCH or TRAY_REJECTED) gating and halting the loop.
8. Stuck pointer failsafe release before starting any new gesture.
"""

from __future__ import annotations

import io
import threading
import time
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
from ai_engine.decision import AIDecisionEngine, PlannedMove
from computer_vision.capture import ScreenCapture
from computer_vision.models import FrameRecognitionResult, RecognitionStatus
from game_engine.board import coord_to_bit
from game_engine.pieces import CANONICAL_41_CATALOG
from tests.test_action_engine import create_synthetic_frame_result


class TestAutoLoopAndGestureReliability(unittest.TestCase):
    """Test suite for continuous AUTO loop and gesture reliability guards."""

    def setUp(self) -> None:
        self.catalog = CANONICAL_41_CATALOG
        self.dot = self.catalog.get_by_name("dot_1x1")

        self.mock_cap = MagicMock(spec=ScreenCapture)
        self.mock_rec = MagicMock()
        self.ai_engine = AIDecisionEngine(default_beam_width=5)
        self.mock_closed_loop = MagicMock(spec=ClosedLoopController)
        self.mock_verifier = MagicMock(spec=ActionVerifier)

        self.test_frame = np.zeros((2340, 1080, 3), dtype=np.uint8)
        self.mock_cap.capture_from_adb.return_value = self.test_frame

        self.mock_closed_loop.config = DragCalibrationConfig.create_samsung_m21_calibrated()
        self.mock_closed_loop.plan_initial_trajectory.return_value = (500, 1800, 200, 700)

        # Base synthetic frame result with single dot in slot 0
        self.base_frame_res = create_synthetic_frame_result(
            bitboard=0x0,
            tray_piece_names=("dot_1x1", "EMPTY", "EMPTY"),
        )
        self.mock_rec.recognize_frame.return_value = self.base_frame_res

        self.step_res = ClosedLoopStepResult(
            status=ClosedLoopStatus.CONVERGED,
            step_index=0,
            piece_name="dot_1x1",
            slot_idx=0,
            target_grid_coord=(0, 0),
            target_screen_center=(200, 700),
            initial_finger_down=(500, 1800),
            current_finger_pos=(200, 700),
            detected_piece_center=(200, 700),
            detected_bbox=(150, 650, 100, 100),
            error_x_px=0.0,
            error_y_px=0.0,
            proposed_correction_move=None,
            correction_count=0,
            tracker_detection=None,
            is_dry_run=False,
            message="Converged",
        )
        self.mock_closed_loop.evaluate_held_frame.return_value = self.step_res

        self.success_ver_res = VerificationResult(
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
            elapsed_ms=10.0,
        )
        self.mock_verifier.verify_move.return_value = self.success_ver_res

    def _create_controller(self, max_turns: int = 10, dry_run: bool = True) -> AutonomousGameController:
        config = AutonomousConfig(
            device_id="TEST_DEVICE",
            adb_path="adb",
            max_turns=max_turns,
            dry_run=dry_run,
            loop_idle_sleep_sec=0.01,
            fast_gesture_mode=True,
        )
        return AutonomousGameController(
            config=config,
            screen_capture=self.mock_cap,
            recognizer=self.mock_rec,
            ai_engine=self.ai_engine,
            closed_loop_controller=self.mock_closed_loop,
            action_verifier=self.mock_verifier,
        )

    # =========================================================================
    # PART 1: AUTO LOOP CONTINUOUS EXECUTION (>= 10 TURNS)
    # =========================================================================

    def test_01_auto_loop_continuous_execution_10_turns(self) -> None:
        """Verify AUTO loop executes >= 10 turns continuously without pressing Space."""
        ctrl = self._create_controller(max_turns=10, dry_run=True)

        started = ctrl.start()
        self.assertTrue(started)
        self.assertTrue(ctrl.is_running())

        # Wait for max_turns execution (up to 5 seconds)
        t_start = time.time()
        while ctrl.stats.total_turns < 10 and (time.time() - t_start) < 5.0:
            time.sleep(0.05)

        # Stop controller
        ctrl.stop("Test finished")

        self.assertEqual(ctrl.stats.total_turns, 10)
        self.assertEqual(ctrl.stats.successful_turns, 10)
        self.assertEqual(ctrl.stats.failed_turns, 0)
        self.assertEqual(ctrl.state, AutonomousState.IDLE)

    def test_02_pause_safely_halts_auto_loop_and_releases_pointer(self) -> None:
        """Verify PAUSE stops the loop immediately and releases pointer."""
        ctrl = self._create_controller(max_turns=20, dry_run=True)
        ctrl.start()

        # Let it run 2 turns
        t_start = time.time()
        while ctrl.stats.total_turns < 2 and (time.time() - t_start) < 3.0:
            time.sleep(0.02)

        # Trigger PAUSE
        ctrl.pause("User pressed 'p'")
        time.sleep(0.05)  # Allow in-flight turn to settle
        turns_at_pause = ctrl.stats.total_turns
        self.assertGreaterEqual(turns_at_pause, 2)
        self.assertEqual(ctrl.state, AutonomousState.PAUSED)
        self.assertFalse(ctrl._touch_pointer_down)

        # Wait a moment and confirm no new turns are executed
        time.sleep(0.15)
        self.assertEqual(ctrl.stats.total_turns, turns_at_pause)

        ctrl.stop()

    def test_03_resume_clears_stale_data_and_resumes_loop(self) -> None:
        """Verify RESUME re-captures fresh perception and continues AUTO loop."""
        ctrl = self._create_controller(max_turns=6, dry_run=True)
        ctrl.start()

        # Let it run 1 turn and pause
        t_start = time.time()
        while ctrl.stats.total_turns < 1 and (time.time() - t_start) < 3.0:
            time.sleep(0.02)

        ctrl.pause("Paused for testing")
        self.assertEqual(ctrl.state, AutonomousState.PAUSED)

        # Set fake stale frame and recognition
        ctrl.last_frame = np.ones((10, 10, 3), dtype=np.uint8)
        ctrl.last_recognition = self.base_frame_res
        ctrl.last_planned_move = PlannedMove(
            step_index=0, slot_idx=0, piece_name="dot_1x1",
            row=0, col=0, cell_count=1, lines_cleared=0,
            unique_cells_cleared=1, predicted_score_delta=10,
            combo_streak_after=0, is_game_over=False,
        )

        # Resume
        ctrl.resume("User resumed")

        # Confirm stale plan and frame were cleared
        self.assertIsNone(ctrl.last_planned_move)
        self.assertIsNone(ctrl.last_recognition)
        self.assertIsNone(ctrl.last_frame)

        # Wait until remaining turns complete
        t_start = time.time()
        while ctrl.stats.total_turns < 6 and (time.time() - t_start) < 4.0:
            time.sleep(0.05)

        ctrl.stop()
        self.assertGreaterEqual(ctrl.stats.total_turns, 6)

    def test_04_single_step_executes_exactly_one_turn(self) -> None:
        """Verify step_single_turn() executes exactly 1 turn and does not auto-advance."""
        ctrl = self._create_controller(max_turns=10, dry_run=True)
        self.assertEqual(ctrl.state, AutonomousState.IDLE)
        self.assertFalse(ctrl.is_running())

        # Execute single step
        success = ctrl.step_single_turn()

        self.assertTrue(success)
        self.assertEqual(ctrl.stats.total_turns, 1)
        self.assertEqual(ctrl.stats.successful_turns, 1)
        self.assertFalse(ctrl.is_running())

        # Sleep to confirm no background loop was started
        time.sleep(0.1)
        self.assertEqual(ctrl.stats.total_turns, 1)

    # =========================================================================
    # PART 2: GESTURE RELIABILITY & POINTER RELEASE GUARANTEES
    # =========================================================================

    def test_05_gesture_guarantees_pointer_release_on_success(self) -> None:
        """Verify fast physical gesture completes and confirms pointer release."""
        ctrl = self._create_controller(dry_run=False)

        mock_shell = MagicMock()
        mock_shell.poll.return_value = None
        mock_shell.stdin = MagicMock()
        ctrl._shell_proc = mock_shell

        planned_move = PlannedMove(
            step_index=0, slot_idx=0, piece_name="dot_1x1",
            row=0, col=0, cell_count=1, lines_cleared=0,
            unique_cells_cleared=1, predicted_score_delta=10,
            combo_streak_after=0, is_game_over=False,
        )

        # Simulate shell token reader returning True for GESTURE_UP_OK
        with patch.object(ctrl, "_read_shell_until_token", return_value=True):
            res = ctrl._execute_physical_gesture(
                move=planned_move,
                piece_model=self.dot,
                base_frame=self.test_frame,
                start_x=500,
                start_y=1800,
                finger_x_0=200,
                finger_y_0=700,
            )

            self.assertEqual(res.status, ClosedLoopStatus.CONVERGED)
            self.assertFalse(ctrl._touch_pointer_down)

            # Inspect commands written to shell stdin
            written_commands = "".join(call[0][0] for call in mock_shell.stdin.write.call_args_list)
            self.assertIn("input motionevent DOWN 500 1800", written_commands)
            self.assertIn("input motionevent MOVE", written_commands)
            self.assertIn("input motionevent UP 200 700 && echo GESTURE_UP_OK", written_commands)

    def test_06_gesture_guarantees_pointer_release_on_exception(self) -> None:
        """Verify gesture failure / exception triggers failsafe release and hard shell close."""
        ctrl = self._create_controller(dry_run=False)

        mock_shell = MagicMock()
        mock_shell.poll.return_value = None
        mock_shell.stdin = MagicMock()
        # Simulate pipe breakage during MOVE
        mock_shell.stdin.write.side_effect = [None, BrokenPipeError("ADB pipe broken")]
        ctrl._shell_proc = mock_shell

        planned_move = PlannedMove(
            step_index=0, slot_idx=0, piece_name="dot_1x1",
            row=0, col=0, cell_count=1, lines_cleared=0,
            unique_cells_cleared=1, predicted_score_delta=10,
            combo_streak_after=0, is_game_over=False,
        )

        with patch.object(ctrl, "_release_pointer_failsafe") as mock_failsafe_release:
            with patch.object(ctrl, "_close_shell_hard") as mock_close_shell:
                res = ctrl._execute_physical_gesture(
                    move=planned_move,
                    piece_model=self.dot,
                    base_frame=self.test_frame,
                    start_x=500,
                    start_y=1800,
                    finger_x_0=200,
                    finger_y_0=700,
                )

                self.assertEqual(res.status, ClosedLoopStatus.UNKNOWN)
                self.assertFalse(ctrl._touch_pointer_down)
                mock_failsafe_release.assert_called()
                mock_close_shell.assert_called_once()

    def test_07_pre_gesture_stuck_pointer_failsafe_release(self) -> None:
        """Verify active pointer from previous interrupted operation is released before starting new gesture."""
        ctrl = self._create_controller(dry_run=False)

        # Intentionally mark pointer active
        ctrl._touch_pointer_down = True
        ctrl._pointer_pos = (550, 1200)

        mock_shell = MagicMock()
        mock_shell.poll.return_value = None
        mock_shell.stdin = MagicMock()
        ctrl._shell_proc = mock_shell

        planned_move = PlannedMove(
            step_index=0, slot_idx=0, piece_name="dot_1x1",
            row=0, col=0, cell_count=1, lines_cleared=0,
            unique_cells_cleared=1, predicted_score_delta=10,
            combo_streak_after=0, is_game_over=False,
        )

        with patch.object(ctrl, "_release_pointer_failsafe") as mock_failsafe_release:
            with patch.object(ctrl, "_read_shell_until_token", return_value=True):
                ctrl._execute_physical_gesture(
                    move=planned_move,
                    piece_model=self.dot,
                    base_frame=self.test_frame,
                    start_x=500,
                    start_y=1800,
                    finger_x_0=200,
                    finger_y_0=700,
                )

                # Pre-gesture failsafe release must have been invoked
                mock_failsafe_release.assert_called_with(550, 1200)

    # =========================================================================
    # PART 3: VERIFICATION GATING (NO BLIND ADVANCE)
    # =========================================================================

    def test_08_verification_mismatch_halts_auto_loop(self) -> None:
        """Verify that when verification fails (bitboard mismatch), next turn is never executed."""
        ctrl = self._create_controller(max_turns=5, dry_run=False)

        # Simulate gesture success
        with patch.object(ctrl, "_execute_physical_gesture", return_value=self.step_res):
            # Simulate verification failure (bitboard mismatch)
            mismatch_ver = VerificationResult(
                status=VerificationStatus.MISMATCH,
                step_index=0,
                slot_idx=0,
                piece_name="dot_1x1",
                expected_bitboard=0x1,
                actual_bitboard=0x0,
                mismatch_count=1,
                mismatched_cells=(),
                tray_slot_cleared=False,
                attempts=2,
                elapsed_ms=50.0,
                error_message="Board bitboard did not reflect placed piece",
            )
            self.mock_verifier.verify_move.return_value = mismatch_ver

            success = ctrl.step_single_turn()

            self.assertFalse(success)
            self.assertEqual(ctrl.stats.total_turns, 1)
            self.assertEqual(ctrl.stats.failed_turns, 1)
            self.assertEqual(ctrl.stats.successful_turns, 0)
            self.assertEqual(ctrl.state, AutonomousState.PAUSED)
            self.assertIn("Post-move verification failed", ctrl.last_pause_reason)

    def test_09_tray_rejected_detection_pauses_game_cleanly(self) -> None:
        """Verify TRAY_REJECTED triggers specialized halt and pauses without retrying."""
        ctrl = self._create_controller(max_turns=5, dry_run=False)

        with patch.object(ctrl, "_execute_physical_gesture", return_value=self.step_res):
            rejected_ver = VerificationResult(
                status=VerificationStatus.TRAY_REJECTED,
                step_index=0,
                slot_idx=0,
                piece_name="dot_1x1",
                expected_bitboard=0x1,
                actual_bitboard=0x0,
                mismatch_count=1,
                mismatched_cells=(),
                tray_slot_cleared=False,
                attempts=1,
                elapsed_ms=30.0,
                error_message="Piece returned to tray",
            )
            self.mock_verifier.verify_move.return_value = rejected_ver

            success = ctrl.step_single_turn()

            self.assertFalse(success)
            self.assertEqual(ctrl.state, AutonomousState.PAUSED)
            self.assertIn("TRAY_REJECTED", ctrl.last_pause_reason)
            self.assertFalse(ctrl._touch_pointer_down)


if __name__ == "__main__":
    unittest.main()
