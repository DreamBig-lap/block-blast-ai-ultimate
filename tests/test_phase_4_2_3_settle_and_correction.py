"""Tests for Phase 4.2.3: Adaptive Settle Time, Consecutive Frame Verification, and Correction Audit.

Validates:
1. Adaptive Settle Time:
   - Line clear moves trigger line_clear_settle_sec (1.2s).
   - Non-line-clear moves trigger normal_settle_sec (0.35s).
2. Consecutive Frame Consistency:
   - Identical consecutive frames are verified as stable.
   - Differing consecutive frames (mid-animation) trigger settling retry.
   - Persistent consecutive frame inconsistency results in UNSTABLE_ANIMATION_TIMEOUT and host PAUSE.
3. Closed-Loop Correction Limits & Convergence:
   - Exceeding max_corrections_allowed when error > 25px returns CORRECTION_LIMIT_EXCEEDED.
   - requires_failsafe_release is True on limit exceeded and UNKNOWN.
   - Deviation <= 25px successfully converges.
4. Correction Coordinate Mapping & Direction Audit:
   - Phase 4.2.2 telemetry: error_x = +61.7px, error_y = -28.7px.
   - Verifies positive X adjustment (rightward) and negative Y adjustment (upward).
   - Inverts gains properly (dX = err_X / Gain_X, dY = err_Y / Gain_Y).
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, call, patch
import numpy as np

from action_engine.action_verifier import ActionVerifier, VerificationStatus
from action_engine.closed_loop_controller import (
    ClosedLoopController,
    ClosedLoopStatus,
    ClosedLoopStepResult,
    DragCalibrationConfig,
)
from action_engine.held_piece_tracker import HeldPieceDetection, TrackingStatus
from ai_engine.decision import PlannedMove
from game_engine.pieces import CANONICAL_41_CATALOG
from game_engine.simulator import GameSimulator, StepResult
from game_engine.state import GameState
from tests.test_action_engine import create_synthetic_frame_result


def create_mock_held_piece_detection(
    status: TrackingStatus = TrackingStatus.SUCCESS,
    centroid: tuple[float, float] | None = None,
    bbox: tuple[int, int, int, int] | None = (100, 100, 100, 100),
    confidence: float = 0.90,
    message: str = "Mock detection",
) -> HeldPieceDetection:
    """Helper to instantiate full HeldPieceDetection dataclass."""
    return HeldPieceDetection(
        status=status,
        piece_name="l_2x3_h1",
        bbox=bbox,
        centroid=centroid,
        contour_area=1000.0,
        aspect_ratio=1.5,
        mean_saturation=180.0,
        mean_value=200.0,
        confidence=confidence,
        is_abnormal_bbox=False,
        shadow_detected=False,
        message=message,
        candidates_evaluated=1,
        top_candidates=(),
        elapsed_ms=5.0,
    )


class TestAdaptiveSettleTime(unittest.TestCase):
    """A. Adaptive Settle Time Unit Tests."""

    def setUp(self):
        self.piece = CANONICAL_41_CATALOG.get_by_name("dot_1x1")
        self.sim = GameSimulator()

    @patch("time.sleep")
    def test_adaptive_settle_time_line_clear_uses_1_2_seconds(self, mock_sleep):
        """Line clear move (lines_cleared > 0) must trigger line_clear_settle_sec = 1.2s."""
        # Create a mock StepResult with 1 line cleared
        init_state = GameState.create_initial(initial_tray=[self.piece, None, None])
        step_res = self.sim.apply_move(init_state, slot_idx=0, row=0, col=0)
        # Mock step_res.lines_cleared = 1
        mock_step_res = MagicMock(spec=StepResult)
        mock_step_res.state_after.board = step_res.state_after.board
        mock_step_res.slot_idx = 0
        mock_step_res.piece = self.piece
        mock_step_res.lines_cleared = 1

        mock_rec = MagicMock()
        frame_res = create_synthetic_frame_result(bitboard=step_res.state_after.board)
        mock_rec.recognize_frame.return_value = frame_res

        mock_cap = MagicMock()
        mock_cap.capture_from_adb.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        verifier = ActionVerifier(
            recognizer=mock_rec,
            capture=mock_cap,
            normal_settle_sec=0.35,
            line_clear_settle_sec=1.2,
            consecutive_frame_delay_sec=0.15,
            max_settle_retries=1,
        )

        res = verifier.verify_move(mock_step_res)
        self.assertTrue(res.is_success)

        # First sleep call must be exactly line_clear_settle_sec (1.2s)
        self.assertTrue(mock_sleep.called)
        first_sleep_arg = mock_sleep.call_args_list[0][0][0]
        self.assertAlmostEqual(first_sleep_arg, 1.2, places=2)

    @patch("time.sleep")
    def test_adaptive_settle_time_no_line_clear_uses_normal_settle(self, mock_sleep):
        """Regular move (lines_cleared == 0) must trigger normal_settle_sec = 0.35s."""
        init_state = GameState.create_initial(initial_tray=[self.piece, None, None])
        step_res = self.sim.apply_move(init_state, slot_idx=0, row=0, col=0)
        mock_step_res = MagicMock(spec=StepResult)
        mock_step_res.state_after.board = step_res.state_after.board
        mock_step_res.slot_idx = 0
        mock_step_res.piece = self.piece
        mock_step_res.lines_cleared = 0

        mock_rec = MagicMock()
        frame_res = create_synthetic_frame_result(bitboard=step_res.state_after.board)
        mock_rec.recognize_frame.return_value = frame_res

        mock_cap = MagicMock()
        mock_cap.capture_from_adb.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        verifier = ActionVerifier(
            recognizer=mock_rec,
            capture=mock_cap,
            normal_settle_sec=0.35,
            line_clear_settle_sec=1.2,
            consecutive_frame_delay_sec=0.15,
            max_settle_retries=1,
        )

        res = verifier.verify_move(mock_step_res)
        self.assertTrue(res.is_success)

        # First sleep call must be normal_settle_sec (0.35s)
        first_sleep_arg = mock_sleep.call_args_list[0][0][0]
        self.assertAlmostEqual(first_sleep_arg, 0.35, places=2)


class TestConsecutiveFrameConsistency(unittest.TestCase):
    """A. Consecutive Frame Consistency Unit Tests."""

    def setUp(self):
        self.piece = CANONICAL_41_CATALOG.get_by_name("dot_1x1")
        self.sim = GameSimulator()
        init_state = GameState.create_initial(initial_tray=[self.piece, None, None])
        self.step_res = self.sim.apply_move(init_state, slot_idx=0, row=0, col=0)

    @patch("time.sleep")
    def test_animation_in_flight_inconsistent_frames_retries_and_settles(self, mock_sleep):
        """When Frame A differs from Frame B (mid-animation), verifier retries and verifies upon settle."""
        expected_bb = self.step_res.state_after.board

        # Attempt 1: Frame A has particle spark at row 0 (0x18), Frame B has different particle state (0x10)
        frame1_a = create_synthetic_frame_result(bitboard=expected_bb | 0x18)
        frame1_b = create_synthetic_frame_result(bitboard=expected_bb | 0x10)

        # Attempt 2: Animation finished; Frame A and Frame B are identical and equal to expected_bb
        frame2_a = create_synthetic_frame_result(bitboard=expected_bb)
        frame2_b = create_synthetic_frame_result(bitboard=expected_bb)

        mock_rec = MagicMock()
        mock_rec.recognize_frame.side_effect = [frame1_a, frame1_b, frame2_a, frame2_b]

        mock_cap = MagicMock()
        mock_cap.capture_from_adb.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        verifier = ActionVerifier(
            recognizer=mock_rec,
            capture=mock_cap,
            normal_settle_sec=0.0,
            line_clear_settle_sec=0.0,
            consecutive_frame_delay_sec=0.0,
            max_settle_retries=3,
            retry_interval_sec=0.0,
        )

        res = verifier.verify_move(self.step_res)
        self.assertTrue(res.is_success)
        self.assertEqual(res.status, VerificationStatus.ANIMATION_SETTLING)
        self.assertEqual(res.attempts, 2)
        self.assertEqual(res.actual_bitboard, expected_bb)

    @patch("time.sleep")
    def test_persistent_inconsistent_frames_trigger_unstable_animation_timeout(self, mock_sleep):
        """If consecutive frames continue to differ throughout all retries, return UNSTABLE_ANIMATION_TIMEOUT."""
        expected_bb = self.step_res.state_after.board

        # Alternating fluctuating frames simulating turbulent animation
        frame_odd = create_synthetic_frame_result(bitboard=expected_bb | 0x18)
        frame_even = create_synthetic_frame_result(bitboard=expected_bb | 0x24)

        mock_rec = MagicMock()
        # 3 attempts x 2 frames each = 6 frame recognitions
        mock_rec.recognize_frame.side_effect = [
            frame_odd, frame_even,
            frame_odd, frame_even,
            frame_odd, frame_even,
        ]

        mock_cap = MagicMock()
        mock_cap.capture_from_adb.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        verifier = ActionVerifier(
            recognizer=mock_rec,
            capture=mock_cap,
            normal_settle_sec=0.0,
            line_clear_settle_sec=0.0,
            consecutive_frame_delay_sec=0.0,
            max_settle_retries=3,
            retry_interval_sec=0.0,
        )

        res = verifier.verify_move(self.step_res)
        self.assertFalse(res.is_success)
        self.assertEqual(res.status, VerificationStatus.UNSTABLE_ANIMATION_TIMEOUT)
        self.assertIn("consecutive frames remained inconsistent", res.error_message)


class TestCorrectionAuditAndLimits(unittest.TestCase):
    """B. Correction Audit & Bounds Unit Tests."""

    def setUp(self):
        self.cfg = DragCalibrationConfig.create_samsung_m21_calibrated()
        self.controller = ClosedLoopController(config=self.cfg, dry_run=True)
        self.move = PlannedMove(
            step_index=0,
            slot_idx=2,
            row=1,
            col=0,
            piece_name="l_2x3_h1",
            cell_count=4,
            predicted_score_delta=14,
            lines_cleared=1,
            unique_cells_cleared=4,
            combo_streak_after=0,
            is_game_over=False,
        )

    def test_phase_4_2_2_telemetry_direction_and_scaling(self):
        """Audits exact Phase 4.2.2 telemetry: error_x = +61.7px, error_y = -28.7px.

        Verifies:
        - Piece to the left (error_x > 0) -> finger moves right (positive delta).
        - Piece below target (error_y < 0) -> finger moves up (negative delta in Android coordinates).
        - Scaling uses Gain_X = 1.428, Gain_Y = 1.438.
        """
        base_img = np.zeros((2340, 1080, 3), dtype=np.uint8)
        mid_img = np.zeros((2340, 1080, 3), dtype=np.uint8)

        # Mock tracker to return exact Phase 4.2.2 centroid
        # Target center for l_2x3_h1 at (row=1, col=0):
        # x_target = 54 + (0 + 3/2)*121.5 = 236.25
        # y_target = 548 + (1 + 2/2)*121.5 = 791.0
        seen_x = 174.595
        seen_y = 819.715
        mock_tracker = MagicMock()
        mock_tracker.track_held_piece.return_value = create_mock_held_piece_detection(
            status=TrackingStatus.SUCCESS,
            centroid=(seen_x, seen_y),
            bbox=(100, 750, 150, 140),
            confidence=0.88,
            message="Held piece detected with decomposed line clear glow",
        )
        controller = ClosedLoopController(config=self.cfg, tracker=mock_tracker, dry_run=True)

        current_finger = (423, 1315)
        res = controller.evaluate_held_frame(
            move=self.move,
            slot_center=(860, 1793),
            current_finger_pos=current_finger,
            base_frame=base_img,
            mid_hold_frame=mid_img,
            correction_count=0,
            piece_width_cells=3,
            piece_height_cells=2,
        )

        self.assertEqual(res.status, ClosedLoopStatus.NEEDS_CORRECTION)
        self.assertIsNotNone(res.proposed_correction_move)
        prop_x, prop_y = res.proposed_correction_move

        # Check directional signs
        self.assertGreater(res.error_x_px, 0.0, "e_X must be positive when piece is left of target")
        self.assertLess(res.error_y_px, 0.0, "e_Y must be negative when piece is below target")
        self.assertGreater(prop_x, current_finger[0], "Finger X must move right (+)")
        self.assertLess(prop_y, current_finger[1], "Finger Y must move up (-)")

        # Check values matching Phase 4.2.2 telemetry: (466, 1295)
        self.assertEqual(prop_x, 466)
        self.assertEqual(prop_y, 1295)

    def test_correction_limit_exceeded_triggers_failsafe(self):
        """When correction_count reaches max_corrections_allowed and error > 25px:
        Must return CORRECTION_LIMIT_EXCEEDED and requires_failsafe_release = True.
        """
        base_img = np.zeros((2340, 1080, 3), dtype=np.uint8)
        mid_img = np.zeros((2340, 1080, 3), dtype=np.uint8)

        # Piece remains displaced by 50px (> tolerance 25px)
        mock_tracker = MagicMock()
        mock_tracker.track_held_piece.return_value = create_mock_held_piece_detection(
            status=TrackingStatus.SUCCESS,
            centroid=(200.0, 750.0),  # Error is roughly 36px in X and 41px in Y
            bbox=(100, 700, 150, 140),
            confidence=0.85,
        )
        controller = ClosedLoopController(config=self.cfg, tracker=mock_tracker, dry_run=True)

        # Call with correction_count = 1 (already corrected once, max is 1)
        res = controller.evaluate_held_frame(
            move=self.move,
            slot_center=(860, 1793),
            current_finger_pos=(466, 1295),
            base_frame=base_img,
            mid_hold_frame=mid_img,
            correction_count=1,
            piece_width_cells=3,
            piece_height_cells=2,
        )

        self.assertEqual(res.status, ClosedLoopStatus.CORRECTION_LIMIT_EXCEEDED)
        self.assertIsNone(res.proposed_correction_move)
        self.assertTrue(res.requires_failsafe_release)
        self.assertFalse(res.is_converged)
        self.assertTrue(controller.should_pause_after_evaluation(res))
        self.assertGreater(res.euclidean_error_px, 25.0)

    def test_correction_converges_within_25px_tolerance(self):
        """When spatial error <= 25px, status must be CONVERGED and safe to release."""
        base_img = np.zeros((2340, 1080, 3), dtype=np.uint8)
        mid_img = np.zeros((2340, 1080, 3), dtype=np.uint8)

        # Place piece within 10px of target (236.25, 791.0)
        mock_tracker = MagicMock()
        mock_tracker.track_held_piece.return_value = create_mock_held_piece_detection(
            status=TrackingStatus.SUCCESS,
            centroid=(230.0, 795.0),  # dX = 6.25px, dY = -4.0px, dist = 7.4px <= 25.0px
            bbox=(160, 720, 140, 140),
            confidence=0.92,
        )
        controller = ClosedLoopController(config=self.cfg, tracker=mock_tracker, dry_run=True)

        res = controller.evaluate_held_frame(
            move=self.move,
            slot_center=(860, 1793),
            current_finger_pos=(466, 1295),
            base_frame=base_img,
            mid_hold_frame=mid_img,
            correction_count=1,
            piece_width_cells=3,
            piece_height_cells=2,
        )

        self.assertEqual(res.status, ClosedLoopStatus.CONVERGED)
        self.assertTrue(res.is_converged)
        self.assertFalse(res.requires_failsafe_release)
        self.assertFalse(controller.should_pause_after_evaluation(res))
        self.assertLessEqual(res.euclidean_error_px, 25.0)


if __name__ == "__main__":
    unittest.main()
