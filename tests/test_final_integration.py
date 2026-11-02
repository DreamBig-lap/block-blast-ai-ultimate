"""Final Integration Unit Tests for BLOCK BLAST AI ULTIMATE.

Validates:
1. Interactive shell read timeout and emergency TOUCH_UP on hang.
2. Non-blocking turn lock preventing concurrent step_single_turn calls.
3. Consecutive empty tray safety timeout (> 5 empty frames triggers PAUSE).
4. RESUME lifecycle clearing stale plan and forcing fresh ADB capture.
5. Dashboard key handler across all states (IDLE, RUNNING, PAUSED, GAME_OVER, ERROR).
6. Generic capture exception handling (USB disconnect/OSError) triggering safe PAUSE.
"""

from __future__ import annotations

import subprocess
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
from action_engine.dashboard import AutonomousDashboard
from ai_engine.decision import AIDecisionEngine, PlannedMove
from computer_vision.capture import ScreenCapture, ScreenCaptureError
from tests.test_action_engine import create_synthetic_frame_result


class TestFinalIntegration(unittest.TestCase):
    """Unit tests for final integration enhancements."""

    def setUp(self):
        self.config = AutonomousConfig(dry_run=False, device_id="TEST_M21")
        self.mock_cap = MagicMock(spec=ScreenCapture)
        self.mock_rec = MagicMock()
        self.ai_engine = AIDecisionEngine(default_beam_width=5)
        self.mock_closed_loop = MagicMock(spec=ClosedLoopController)
        self.mock_verifier = MagicMock(spec=ActionVerifier)

        self.test_frame = np.zeros((2340, 1080, 3), dtype=np.uint8)
        self.mock_cap.capture_from_adb.return_value = self.test_frame

        self.frame_res = create_synthetic_frame_result(bitboard=0x0, tray_piece_names=("dot_1x1", "EMPTY", "EMPTY"))
        self.mock_rec.recognize_frame.return_value = self.frame_res

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
        self.mock_closed_loop.plan_initial_trajectory.return_value = (500, 1800, 200, 700)
        self.mock_closed_loop.evaluate_held_frame.return_value = self.step_res
        self.mock_closed_loop.config = DragCalibrationConfig.create_samsung_m21_calibrated()

        self.ver_res = VerificationResult(
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
        self.mock_verifier.verify_move.return_value = self.ver_res

    def _create_controller(self) -> AutonomousGameController:
        ctrl = AutonomousGameController(
            config=self.config,
            screen_capture=self.mock_cap,
            recognizer=self.mock_rec,
            ai_engine=self.ai_engine,
            closed_loop_controller=self.mock_closed_loop,
            action_verifier=self.mock_verifier,
        )
        return ctrl

    @patch("subprocess.run")
    def test_shell_read_timeout_and_emergency_touch_up(self, mock_subproc_run):
        """Test that interactive shell timeout triggers hard kill, emergency UP, and PAUSE."""
        ctrl = self._create_controller()

        # Mock shell process
        mock_shell = MagicMock()
        mock_shell.poll.return_value = None
        mock_shell.stdin = MagicMock()
        ctrl._shell_proc = mock_shell

        # Simulate token reader timing out
        with patch.object(ctrl, "_read_shell_until_token", return_value=False):
            with patch.object(ctrl, "_release_pointer_failsafe") as mock_failsafe_up:
                with patch.object(ctrl, "_close_shell_hard") as mock_kill_shell:
                    success = ctrl.step_single_turn()

                    self.assertFalse(success)
                    self.assertEqual(ctrl.state, AutonomousState.PAUSED)
                    mock_kill_shell.assert_called_once()
                    mock_failsafe_up.assert_called()
                    self.assertIn("Tracker uncertain/unknown", ctrl.action_log[-1]["message"])

    def test_turn_lock_prevents_concurrent_execution(self):
        """Test non-blocking turn lock blocks re-entrant or concurrent turn calls."""
        ctrl = self._create_controller()

        turn_started = threading.Event()
        turn_proceed = threading.Event()

        def slow_turn():
            turn_started.set()
            turn_proceed.wait(timeout=2.0)
            return True

        with patch.object(ctrl, "_step_single_turn_internal", side_effect=slow_turn):
            # Start background turn thread
            t = threading.Thread(target=ctrl.step_single_turn)
            t.start()

            self.assertTrue(turn_started.wait(timeout=1.0))

            # Attempt concurrent step_single_turn call
            concurrent_result = ctrl.step_single_turn()
            self.assertFalse(concurrent_result)

            # Let slow turn finish
            turn_proceed.set()
            t.join(timeout=1.0)

    def test_consecutive_empty_tray_timeout(self):
        """Test > 5 consecutive empty tray captures pauses with refill timeout."""
        ctrl = self._create_controller()

        empty_frame_res = create_synthetic_frame_result(bitboard=0x0, tray_piece_names=("EMPTY", "EMPTY", "EMPTY"))
        self.mock_rec.recognize_frame.return_value = empty_frame_res

        # Step 5 times: should wait and return True (handling refill)
        for i in range(5):
            res = ctrl.step_single_turn()
            self.assertTrue(res, f"Step {i+1} should return True while waiting for refill")
            self.assertEqual(ctrl.state, AutonomousState.VALIDATING)

        # 6th consecutive empty tray: should trigger PAUSE
        res6 = ctrl.step_single_turn()
        self.assertFalse(res6)
        self.assertEqual(ctrl.state, AutonomousState.PAUSED)
        self.assertIn("Refill timeout", ctrl.action_log[-1]["message"])

    def test_resume_clears_stale_data_and_forces_fresh_capture(self):
        """Test that resume() clears planned move, resets empty count, and invalidates frame."""
        ctrl = self._create_controller()
        ctrl._is_running = True
        ctrl.last_planned_move = PlannedMove(
            step_index=0,
            slot_idx=0,
            piece_name="dot_1x1",
            row=0,
            col=0,
            cell_count=1,
            lines_cleared=0,
            unique_cells_cleared=1,
            predicted_score_delta=10,
            combo_streak_after=0,
            is_game_over=False,
        )
        ctrl.last_frame = np.ones((100, 100, 3), dtype=np.uint8)
        ctrl._consecutive_empty_trays = 4
        ctrl.set_state(AutonomousState.PAUSED, "Test paused")

        ctrl.resume("User resumed")

        self.assertIsNone(ctrl.last_planned_move)
        self.assertIsNone(ctrl.last_frame)
        self.assertEqual(ctrl._consecutive_empty_trays, 0)
        self.assertEqual(ctrl.state, AutonomousState.CAPTURING)

    def test_generic_capture_exception_handled_safely(self):
        """Test unexpected OSError/ConnectionError during capture pauses game cleanly."""
        ctrl = self._create_controller()
        self.mock_cap.capture_from_adb.side_effect = ConnectionResetError("USB device connection reset")

        success = ctrl.step_single_turn()

        self.assertFalse(success)
        self.assertEqual(ctrl.state, AutonomousState.PAUSED)
        self.assertIsNone(ctrl.last_planned_move)
        self.assertIn("USB device connection reset", ctrl.action_log[-1]["message"])

    def test_dashboard_key_controls_and_safety_guards(self):
        """Test dashboard key actions and guards for IDLE, PAUSED, RUNNING, and GAME_OVER."""
        ctrl = self._create_controller()
        dashboard = AutonomousDashboard(controller=ctrl)

        # 1. 's' in IDLE starts
        dashboard.handle_key('s')
        self.assertTrue(ctrl.is_running())

        # 2. 'p' pauses
        dashboard.handle_key('p')
        self.assertEqual(ctrl.state, AutonomousState.PAUSED)

        # 3. ' ' (Space) in PAUSED steps single turn
        with patch.object(ctrl, "step_single_turn") as mock_step:
            dashboard.handle_key(' ')
            mock_step.assert_called_once()

        # 4. 's' in PAUSED resumes
        dashboard.handle_key('s')
        self.assertEqual(ctrl.state, AutonomousState.CAPTURING)

        # 5. ' ' (Space) calls step_single_turn (non-blocking turn_lock prevents concurrency)
        with patch.object(ctrl, "step_single_turn") as mock_step:
            dashboard.handle_key(' ')
            mock_step.assert_called_once()

        # 6. 's' in GAME_OVER restarts controller
        ctrl.set_state(AutonomousState.GAME_OVER, "Game over")
        with patch.object(ctrl, "start") as mock_start:
            dashboard.handle_key('s')
            mock_start.assert_called_once()

        # 7. 'q' stops and resets to IDLE
        dashboard.handle_key('q')
        self.assertEqual(ctrl.state, AutonomousState.IDLE)


if __name__ == "__main__":
    unittest.main()
