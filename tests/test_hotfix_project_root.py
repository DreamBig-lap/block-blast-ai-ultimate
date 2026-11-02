"""Regression Tests for HOTFIX — PROJECT_ROOT Undefined in Live Physical Actuation.

Validates:
1. Physical actuation code path uses dynamically resolved self.debug_dir and never throws NameError.
2. Actuation exceptions gracefully release pointer, clear last_planned_move, and transition to PAUSED.
3. Custom project_root via AutonomousConfig is respected and creates debug directory.
4. All remaining occurrences of PROJECT_ROOT across the repository are properly defined.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
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
from ai_engine.decision import AIDecisionEngine
from computer_vision.capture import ScreenCapture
from tests.test_action_engine import create_synthetic_frame_result


class TestHotfixProjectRoot(unittest.TestCase):
    """Regression test suite for PROJECT_ROOT definition and actuation safety."""

    def setUp(self):
        self.config = AutonomousConfig(dry_run=False, device_id="TEST_M21", fast_gesture_mode=False)
        self.mock_cap = MagicMock(spec=ScreenCapture)
        self.mock_rec = MagicMock()
        self.ai_engine = AIDecisionEngine(default_beam_width=5)
        self.mock_closed_loop = MagicMock(spec=ClosedLoopController)
        self.mock_verifier = MagicMock(spec=ActionVerifier)

        self.test_frame = np.zeros((2340, 1080, 3), dtype=np.uint8)
        self.mock_cap.capture_from_adb.return_value = self.test_frame

        self.frame_res = create_synthetic_frame_result(bitboard=0x0, tray_piece_names=("dot_1x1", "EMPTY", "EMPTY"))
        self.mock_rec.recognize_frame.return_value = self.frame_res

        self.mock_closed_loop.plan_initial_trajectory.return_value = (500, 1800, 200, 700)
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

    def test_01_physical_actuation_resolves_debug_dir_without_name_error(self):
        """Verifies physical actuation path (dry_run=False) constructs mid_local without PROJECT_ROOT NameError."""
        ctrl = AutonomousGameController(
            config=self.config,
            screen_capture=self.mock_cap,
            recognizer=self.mock_rec,
            ai_engine=self.ai_engine,
            closed_loop_controller=self.mock_closed_loop,
            action_verifier=self.mock_verifier,
        )

        # Check project_root and debug_dir are valid Paths
        self.assertIsInstance(ctrl.project_root, Path)
        self.assertTrue(ctrl.project_root.exists())
        self.assertIsInstance(ctrl.debug_dir, Path)
        self.assertTrue(ctrl.debug_dir.exists())

        # Mock shell process
        mock_shell = MagicMock()
        mock_shell.poll.return_value = None
        mock_shell.stdin = MagicMock()
        ctrl._shell_proc = mock_shell

        step_res_converged = ClosedLoopStepResult(
            status=ClosedLoopStatus.CONVERGED,
            step_index=0,
            piece_name="dot_1x1",
            slot_idx=0,
            target_grid_coord=(0, 0),
            target_screen_center=(200.0, 700.0),
            initial_finger_down=(500, 1800),
            current_finger_pos=(200, 700),
            detected_piece_center=(200.0, 700.0),
            detected_bbox=(150, 650, 100, 100),
            error_x_px=0.0,
            error_y_px=0.0,
            proposed_correction_move=None,
            correction_count=0,
            tracker_detection=None,
            is_dry_run=False,
            message="Converged successfully",
        )
        self.mock_closed_loop.evaluate_held_frame.return_value = step_res_converged

        # Mock token reading, adb pull/rm subprocess calls, and cv2.imread
        synthetic_mid = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch.object(ctrl, "_read_shell_until_token", return_value=True):
            with patch("subprocess.run") as mock_subproc:
                with patch("cv2.imread", return_value=synthetic_mid) as mock_imread:
                    mock_subproc.return_value = MagicMock(returncode=0)

                    # Execute single turn in full live physical actuation mode
                    success = ctrl.step_single_turn()

                    self.assertTrue(success, "Turn must succeed in live mode when components mock successfully")
                    self.assertEqual(ctrl.stats.successful_turns, 1)

                    # Verify cv2.imread received the correct string path to live_mid.png inside debug_dir
                    expected_mid_path = str(ctrl.debug_dir / "live_mid.png")
                    mock_imread.assert_called_with(expected_mid_path)

    def test_02_actuation_exception_safely_releases_pointer_and_pauses(self):
        """Verifies that if an unexpected exception occurs during actuation, touch is released and controller PAUSES."""
        ctrl = AutonomousGameController(
            config=self.config,
            screen_capture=self.mock_cap,
            recognizer=self.mock_rec,
            ai_engine=self.ai_engine,
            closed_loop_controller=self.mock_closed_loop,
            action_verifier=self.mock_verifier,
        )

        mock_shell = MagicMock()
        mock_shell.poll.return_value = None
        mock_shell.stdin = MagicMock()
        ctrl._shell_proc = mock_shell

        # Simulate unexpected error in subprocess.run during pull
        with patch.object(ctrl, "_read_shell_until_token", return_value=True):
            with patch("subprocess.run", side_effect=OSError("Disk write error during ADB pull")):
                with patch.object(ctrl, "_release_pointer_failsafe") as mock_failsafe_up:
                    success = ctrl.step_single_turn()

                    self.assertFalse(success)
                    self.assertEqual(ctrl.state, AutonomousState.PAUSED)
                    self.assertIsNone(ctrl.last_planned_move, "Stale planned move must be cleared")
                    mock_failsafe_up.assert_called()
                    self.assertIn("Disk write error", ctrl.action_log[-1]["message"])

    def test_03_custom_project_root_configured(self):
        """Verifies that passing custom project_root via AutonomousConfig works and creates debug dir."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_root = Path(tmp_dir)
            cfg = AutonomousConfig(dry_run=True, project_root=custom_root)
            ctrl = AutonomousGameController(config=cfg)

            self.assertEqual(ctrl.project_root, custom_root)
            self.assertEqual(ctrl.debug_dir, custom_root / "debug")
            self.assertTrue(ctrl.debug_dir.exists())

    def test_04_no_other_files_have_undefined_project_root(self):
        """Verifies that no other source code file in the repository has an undefined PROJECT_ROOT."""
        repo_root = Path(__file__).resolve().parent.parent

        for py_file in repo_root.glob("**/*.py"):
            # Skip virtual environments if any
            if ".venv" in py_file.parts or "venv" in py_file.parts:
                continue

            content = py_file.read_text(encoding="utf-8", errors="ignore")
            if "PROJECT_ROOT" in content:
                # Must define PROJECT_ROOT or import it
                has_definition = (
                    "PROJECT_ROOT =" in content
                    or "PROJECT_ROOT=" in content
                    or "import PROJECT_ROOT" in content
                )
                self.assertTrue(
                    has_definition,
                    f"File {py_file} uses PROJECT_ROOT but does not define or import it!"
                )


if __name__ == "__main__":
    unittest.main()
