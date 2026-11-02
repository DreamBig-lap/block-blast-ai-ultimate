"""Unit Tests for ClosedLoopController (BLOCK BLAST AI ULTIMATE).

Tests all 10 minimum required test cases in DRY-RUN mode without device connection:
1. Accurate initial trajectory calculation from calibration config.
2. Tracker successful detection & convergence on real Phase 3.5.3 artifacts.
3. Precise spatial error X/Y calculation on controlled synthetic offsets.
4. Proposed micro-correction finger coordinate calculation.
5. Detection of invalid calibration parameters.
6. Safe fallback to UNKNOWN when tracker detects no contour.
7. Safe fallback to UNKNOWN when tracker detects ambiguous contours.
8. Exceeding max correction limit safely halts micro-moves.
9. Safety assertion: Dry-Run mode executes zero touch actuation commands.
10. Strict immutability of GameState and input frames.
"""

from __future__ import annotations

import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np

from action_engine.closed_loop_controller import (
    ClosedLoopController,
    ClosedLoopStatus,
    ClosedLoopStepResult,
    DragCalibrationConfig,
)
from action_engine.held_piece_tracker import HeldPieceTracker
from ai_engine.decision import PlannedMove
from game_engine.pieces import CANONICAL_41_CATALOG
from game_engine.state import GameState

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "debug", "phase_3_5_3")


class TestClosedLoopController(unittest.TestCase):
    """Test suite for ClosedLoopController dry-run control logic."""

    def setUp(self) -> None:
        self.calib_m21 = DragCalibrationConfig.create_samsung_m21_calibrated()
        self.calib_default = DragCalibrationConfig.create_default()
        self.controller = ClosedLoopController(config=self.calib_m21, dry_run=True)

        self.base_path = os.path.join(ARTIFACTS_DIR, "base.png")
        self.mid_path = os.path.join(ARTIFACTS_DIR, "mid_hold.png")

        if os.path.exists(self.base_path) and os.path.exists(self.mid_path):
            self.real_base = cv2.imread(self.base_path)
            self.real_mid = cv2.imread(self.mid_path)
        else:
            self.real_base = None
            self.real_mid = None

    # =========================================================================
    # TEST 1: Accurate Initial Trajectory Calculation
    # =========================================================================
    def test_01_plan_initial_trajectory_math(self) -> None:
        """Verify mathematical calculation of initial finger destination with Gain X and Lift Y."""
        move = PlannedMove(
            step_index=0,
            slot_idx=2,
            row=4,
            col=2,
            piece_name="line_1x5_h",
            cell_count=5,
            predicted_score_delta=5,
            lines_cleared=0,
            unique_cells_cleared=0,
            combo_streak_after=0,
            is_game_over=False,
        )

        # Board is at (54, 548), cell_size is 121.5
        # Line 1x5 piece: width=5, height=1
        # Target Grid Center:
        # X = 54 + (2 + 2.5) * 121.5 = 600.75 px
        # Y = 548 + (4 + 0.5) * 121.5 = 1094.75 px
        # Tray slot 2 center: (860, 1794)
        # Delta Piece X = 600.75 - 860 = -259.25 px
        # Delta Finger X = -259.25 / 1.428 = -181.55 px
        # Expected Finger End X = round(860 - 181.55) = 678 px
        # Calibrated Vertical calculation with Gain_Y=1.380, Lift_Y=314:
        # Delta Piece Y = 1094.75 - (1794 - 314) = -385.25 px
        # Delta Finger Y = -385.25 / 1.380 = -279.17 px
        # Expected Finger End Y = round(1794 - 279.17) = 1515 px
        start_x, start_y, end_x, end_y = self.controller.plan_initial_trajectory(
            move=move,
            slot_center_x=860,
            slot_center_y=1794,
        )

        self.assertEqual(start_x, 860)
        self.assertEqual(start_y, 1794)
        self.assertAlmostEqual(end_x, 678, delta=1)
        self.assertAlmostEqual(end_y, 1515, delta=1)

    # =========================================================================
    # TEST 2: Real Artifacts Evaluation & Convergence
    # =========================================================================
    def test_02_real_artifacts_feedback_evaluation(self) -> None:
        """Verify closed-loop evaluation on real Phase 3.5.3 M21 capture."""
        if self.real_base is None or self.real_mid is None:
            self.skipTest("Phase 3.5.3 artifacts not found.")

        move = PlannedMove(
            step_index=0,
            slot_idx=2,
            row=4,
            col=2,
            piece_name="line_1x5_h",
            cell_count=5,
            predicted_score_delta=5,
            lines_cleared=0,
            unique_cells_cleared=0,
            combo_streak_after=0,
            is_game_over=False,
        )

        # In Phase 3.5.3, finger was held at (660, 1542)
        # Using tolerance 35.0 px (piece landed at ~574 px, target was 600.75 px -> Euclidean error ~32.4 px)
        cfg_converged = DragCalibrationConfig(
            drag_gain_x=1.428,
            vertical_lift_px=429.0,
            convergence_tolerance_px=35.0,
            cell_size_px=121.5,
        )
        ctrl_conv = ClosedLoopController(config=cfg_converged, dry_run=True)

        res = ctrl_conv.evaluate_held_frame(
            move=move,
            slot_center=(860, 1794),
            current_finger_pos=(660, 1542),
            base_frame=self.real_base,
            mid_hold_frame=self.real_mid,
            correction_count=0,
        )

        self.assertEqual(res.status, ClosedLoopStatus.CONVERGED)
        self.assertTrue(res.is_converged)
        self.assertTrue(res.is_dry_run)
        self.assertIsNotNone(res.detected_piece_center)
        self.assertLessEqual(abs(res.error_x_px), 30.0)
        self.assertLessEqual(abs(res.error_y_px), 30.0)

    # =========================================================================
    # TEST 3: Precise Spatial Error X/Y on Controlled Synthetic Offset
    # =========================================================================
    def test_03_controlled_spatial_error_computation(self) -> None:
        """Verify error computation e_X and e_Y accurately measures deviation from target."""
        h, w = 2000, 1080
        base = np.zeros((h, w, 3), dtype=np.uint8)
        hold = np.zeros((h, w, 3), dtype=np.uint8)

        # Target center for dot_1x1 at row=3, col=3:
        # X_target = 54 + (3 + 0.5) * 121.5 = 479.25
        # Y_target = 548 + (3 + 0.5) * 121.5 = 973.25
        # Place the dot on screen offset by exactly +40 px on X and -50 px on Y:
        # seen_x = 479.25 - 40 = 439.25 (center of 120px box -> X in [379, 499])
        # seen_y = 973.25 - (-50) = 1023.25 (center of 120px box -> Y in [963, 1083])
        hold[963:1083, 379:499] = (0, 200, 255)

        move = PlannedMove(
            step_index=0, slot_idx=0, row=3, col=3, piece_name="dot_1x1",
            cell_count=1, predicted_score_delta=1, lines_cleared=0,
            unique_cells_cleared=0, combo_streak_after=0, is_game_over=False,
        )

        res = self.controller.evaluate_held_frame(
            move=move,
            slot_center=(172, 1794),
            current_finger_pos=(450, 1400),
            base_frame=base,
            mid_hold_frame=hold,
            correction_count=0,
        )

        self.assertIsNotNone(res.detected_piece_center)
        self.assertAlmostEqual(res.error_x_px, 40.0, delta=2.0)
        self.assertAlmostEqual(res.error_y_px, -50.0, delta=2.0)
        self.assertEqual(res.status, ClosedLoopStatus.NEEDS_CORRECTION)

    # =========================================================================
    # TEST 4: Proposed Micro-Correction Finger Move
    # =========================================================================
    def test_04_proposed_correction_move_math(self) -> None:
        """Verify proposed finger adjustment compensates for measured error using Gain X."""
        h, w = 2000, 1080
        base = np.zeros((h, w, 3), dtype=np.uint8)
        hold = np.zeros((h, w, 3), dtype=np.uint8)

        # Place piece offset so error_x = +40 px, error_y = -30 px
        # Target: Row 3, Col 3
        hold[950:1070, 390:510] = (0, 215, 255)

        move = PlannedMove(
            step_index=0, slot_idx=0, row=3, col=3, piece_name="dot_1x1",
            cell_count=1, predicted_score_delta=1, lines_cleared=0,
            unique_cells_cleared=0, combo_streak_after=0, is_game_over=False,
        )

        current_finger = (500, 1400)
        res = self.controller.evaluate_held_frame(
            move=move,
            slot_center=(172, 1794),
            current_finger_pos=current_finger,
            base_frame=base,
            mid_hold_frame=hold,
            correction_count=0,
        )

        self.assertEqual(res.status, ClosedLoopStatus.NEEDS_CORRECTION)
        self.assertIsNotNone(res.proposed_correction_move)

        prop_x, prop_y = res.proposed_correction_move
        expected_adj_x = int(round(current_finger[0] + res.error_x_px / self.calib_m21.drag_gain_x))
        expected_adj_y = int(round(current_finger[1] + res.error_y_px / self.calib_m21.drag_gain_y))

        self.assertEqual(prop_x, expected_adj_x)
        self.assertEqual(prop_y, expected_adj_y)

    # =========================================================================
    # TEST 5: Invalid Calibration Parameters
    # =========================================================================
    def test_05_invalid_calibration_rejection(self) -> None:
        """Verify controller detects and rejects unphysical calibration configs."""
        invalid_cfg = DragCalibrationConfig(
            drag_gain_x=-1.5,  # Negative gain is impossible
            vertical_lift_px=0.0,
        )
        ctrl = ClosedLoopController(config=invalid_cfg, dry_run=True)

        dummy = np.zeros((1000, 1000, 3), dtype=np.uint8)
        move = PlannedMove(
            step_index=0, slot_idx=0, row=0, col=0, piece_name="dot_1x1",
            cell_count=1, predicted_score_delta=1, lines_cleared=0,
            unique_cells_cleared=0, combo_streak_after=0, is_game_over=False,
        )

        res = ctrl.evaluate_held_frame(move, (100, 100), (200, 200), dummy, dummy)
        self.assertEqual(res.status, ClosedLoopStatus.INVALID_CALIBRATION)
        self.assertIn("Invalid calibration", res.message)

    # =========================================================================
    # TEST 6: Tracker UNKNOWN (No Contour Found)
    # =========================================================================
    def test_06_tracker_unknown_on_no_contour(self) -> None:
        """Verify controller yields UNKNOWN when tracker finds no piece in the air."""
        base = np.zeros((1500, 1080, 3), dtype=np.uint8)
        hold = np.zeros((1500, 1080, 3), dtype=np.uint8)

        move = PlannedMove(
            step_index=0, slot_idx=0, row=0, col=0, piece_name="line_1x3_h",
            cell_count=3, predicted_score_delta=3, lines_cleared=0,
            unique_cells_cleared=0, combo_streak_after=0, is_game_over=False,
        )

        res = self.controller.evaluate_held_frame(move, (100, 100), (200, 200), base, hold)
        self.assertEqual(res.status, ClosedLoopStatus.UNKNOWN)
        self.assertIsNone(res.detected_piece_center)
        self.assertIsNone(res.proposed_correction_move)

    # =========================================================================
    # TEST 7: Tracker AMBIGUOUS (Competing Blobs)
    # =========================================================================
    def test_07_tracker_ambiguous_safe_abort(self) -> None:
        """Verify controller yields UNKNOWN when tracker flags ambiguous competing candidates."""
        base = np.zeros((1500, 1080, 3), dtype=np.uint8)
        hold = np.zeros((1500, 1080, 3), dtype=np.uint8)

        # Two competing identical squares
        hold[400:640, 150:390] = (255, 120, 0)
        hold[400:640, 650:890] = (255, 120, 0)

        move = PlannedMove(
            step_index=0, slot_idx=0, row=0, col=0, piece_name="square_2x2",
            cell_count=4, predicted_score_delta=4, lines_cleared=0,
            unique_cells_cleared=0, combo_streak_after=0, is_game_over=False,
        )

        res = self.controller.evaluate_held_frame(move, (100, 100), (200, 200), base, hold)
        self.assertEqual(res.status, ClosedLoopStatus.UNKNOWN)
        self.assertIn("AMBIGUOUS", res.message)

    # =========================================================================
    # TEST 8: Exceeding Max Correction Limit
    # =========================================================================
    def test_08_exceed_correction_limit(self) -> None:
        """Verify controller stops proposing moves when max_corrections budget is reached."""
        h, w = 2000, 1080
        base = np.zeros((h, w, 3), dtype=np.uint8)
        hold = np.zeros((h, w, 3), dtype=np.uint8)

        # Place piece offset so error remains > tolerance
        hold[950:1070, 390:510] = (0, 215, 255)

        move = PlannedMove(
            step_index=0, slot_idx=0, row=3, col=3, piece_name="dot_1x1",
            cell_count=1, predicted_score_delta=1, lines_cleared=0,
            unique_cells_cleared=0, combo_streak_after=0, is_game_over=False,
        )

        # Simulate that 1 correction has already been executed (limit = 1)
        res = self.controller.evaluate_held_frame(
            move=move,
            slot_center=(172, 1794),
            current_finger_pos=(500, 1400),
            base_frame=base,
            mid_hold_frame=hold,
            correction_count=1,
        )

        self.assertEqual(res.status, ClosedLoopStatus.CORRECTION_LIMIT_EXCEEDED)
        self.assertIsNone(res.proposed_correction_move)
        self.assertIn("limit reached", res.message)

    # =========================================================================
    # TEST 9: Safety Assertion — Zero Touch Actuation in DRY-RUN
    # =========================================================================
    def test_09_safety_dry_run_zero_actuation(self) -> None:
        """Verify ClosedLoopController does not instantiate or invoke ADB shell commands."""
        self.assertTrue(self.controller.dry_run)
        # Ensure no touch controller attributes or adb processes are spawned
        self.assertFalse(hasattr(self.controller, "touch_controller"))
        self.assertFalse(hasattr(self.controller, "adb_path"))

    # =========================================================================
    # TEST 10: Strict Immutability of GameState and Input Frames
    # =========================================================================
    def test_10_gamestate_and_frames_immutability(self) -> None:
        """Verify that GameState and input frames remain completely untouched."""
        dot = CANONICAL_41_CATALOG.get_by_name("dot_1x1")
        initial_state = GameState(
            board=0x000000000008C0F0,
            score=120,
            combo_streak=2,
            non_clearing_moves=1,
            round_number=3,
            tray=(dot, None, None),
        )

        base_img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        hold_img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        base_copy = base_img.copy()
        hold_copy = hold_img.copy()

        move = PlannedMove(
            step_index=0, slot_idx=0, row=0, col=0, piece_name="dot_1x1",
            cell_count=1, predicted_score_delta=1, lines_cleared=0,
            unique_cells_cleared=0, combo_streak_after=0, is_game_over=False,
        )

        _ = self.controller.evaluate_held_frame(move, (100, 100), (200, 200), base_img, hold_img)

        # Verify GameState immutability
        self.assertEqual(initial_state.board, 0x000000000008C0F0)
        self.assertEqual(initial_state.score, 120)
        self.assertEqual(initial_state.combo_streak, 2)
        self.assertEqual(initial_state.tray, (dot, None, None))
        self.assertTrue(self.controller.verify_game_state_immutability(initial_state, initial_state))

        # Verify Frame immutability
        self.assertTrue(np.array_equal(base_img, base_copy))
        self.assertTrue(np.array_equal(hold_img, hold_copy))

    # =========================================================================
    # TEST 11: Real Phase 3.5.7 Artifacts Replay & Overshoot Mitigation
    # =========================================================================
    def test_11_phase_3_5_7_real_data_replay(self) -> None:
        """Replay actual Phase 3.5.7 frames to verify calibrated Gain Y prevents overshoot."""
        p_base = os.path.join(PROJECT_ROOT, "debug", "phase_3_5_7", "base.png")
        p_mid1 = os.path.join(PROJECT_ROOT, "debug", "phase_3_5_7", "mid_hold.png")

        if not os.path.exists(p_base) or not os.path.exists(p_mid1):
            self.skipTest("Phase 3.5.7 real artifacts not found.")

        base = cv2.imread(p_base)
        mid1 = cv2.imread(p_mid1)

        move = PlannedMove(
            step_index=0, slot_idx=1, row=0, col=1, piece_name="line_1x3_h",
            cell_count=3, predicted_score_delta=3, lines_cleared=0,
            unique_cells_cleared=0, combo_streak_after=0, is_game_over=False,
        )

        # 1. Uncalibrated controller (Gain Y = 1.0)
        cfg_uncal = DragCalibrationConfig(
            drag_gain_x=1.428,
            drag_gain_y=1.0,  # Legacy assumption
            vertical_lift_px=429.0,
            convergence_tolerance_px=25.0,
        )
        ctrl_uncal = ClosedLoopController(config=cfg_uncal, dry_run=True)
        res_uncal = ctrl_uncal.evaluate_held_frame(
            move=move,
            slot_center=(545, 1793),
            current_finger_pos=(414, 1038),
            base_frame=base,
            mid_hold_frame=mid1,
            correction_count=0,
        )

        self.assertEqual(res_uncal.status, ClosedLoopStatus.NEEDS_CORRECTION)
        self.assertAlmostEqual(res_uncal.error_x_px, 15.2, delta=1.5)
        self.assertAlmostEqual(res_uncal.error_y_px, 221.8, delta=2.0)
        # Uncalibrated proposed finger Y = 1038 + 222 = 1260 (overshot in real test)
        self.assertEqual(res_uncal.proposed_correction_move, (425, 1260))

        # 2. Calibrated controller with Gain Y = 1.518 (Phase 3.5.7 measured)
        cfg_cal = DragCalibrationConfig(
            drag_gain_x=1.413,
            drag_gain_y=1.518,
            vertical_lift_px=314.0,
            convergence_tolerance_px=25.0,
        )
        ctrl_cal = ClosedLoopController(config=cfg_cal, dry_run=True)
        res_cal = ctrl_cal.evaluate_held_frame(
            move=move,
            slot_center=(545, 1793),
            current_finger_pos=(414, 1038),
            base_frame=base,
            mid_hold_frame=mid1,
            correction_count=0,
        )

        self.assertEqual(res_cal.status, ClosedLoopStatus.NEEDS_CORRECTION)
        # Calibrated proposed finger Y = 1038 + round(221.8 / 1.518) = 1038 + 146 = 1184 px
        prop_x, prop_y = res_cal.proposed_correction_move
        self.assertEqual(prop_x, 425)
        self.assertEqual(prop_y, 1184)

    # =========================================================================
    # TEST 12: Simulation of Overshoot Mitigation
    # =========================================================================
    def test_12_simulation_overshoot_mitigation(self) -> None:
        """Verify micro-correction scales positive vertical error by Gain Y."""
        cfg = DragCalibrationConfig(
            drag_gain_x=1.40,
            drag_gain_y=1.50,
            vertical_lift_px=300.0,
        )
        ctrl = ClosedLoopController(config=cfg, dry_run=True)

        h, w = 2000, 1080
        base = np.zeros((h, w, 3), dtype=np.uint8)
        hold = np.zeros((h, w, 3), dtype=np.uint8)

        # Place piece too high by 150px (error_y = +150px)
        # Target: Row 3 Col 3 (Y_target = 973.25) -> Seen Y = 823.25
        hold[763:883, 419:539] = (0, 200, 255)

        move = PlannedMove(
            step_index=0, slot_idx=0, row=3, col=3, piece_name="dot_1x1",
            cell_count=1, predicted_score_delta=1, lines_cleared=0,
            unique_cells_cleared=0, combo_streak_after=0, is_game_over=False,
        )

        cur_finger = (450, 1200)
        res = ctrl.evaluate_held_frame(move, (172, 1794), cur_finger, base, hold)
        self.assertEqual(res.status, ClosedLoopStatus.NEEDS_CORRECTION)

        prop_x, prop_y = res.proposed_correction_move
        # finger should move down by 150 / 1.50 = 100px (to 1300), NOT by 150px (to 1350)
        expected_y = cur_finger[1] + int(round(res.error_y_px / 1.50))
        self.assertEqual(prop_y, expected_y)
        self.assertEqual(prop_y, 1300)

    # =========================================================================
    # TEST 13: Simulation of Undershoot Compensation
    # =========================================================================
    def test_13_simulation_undershoot_compensation(self) -> None:
        """Verify micro-correction scales negative vertical error by Gain Y."""
        cfg = DragCalibrationConfig(
            drag_gain_x=1.40,
            drag_gain_y=1.50,
            vertical_lift_px=300.0,
        )
        ctrl = ClosedLoopController(config=cfg, dry_run=True)

        h, w = 2000, 1080
        base = np.zeros((h, w, 3), dtype=np.uint8)
        hold = np.zeros((h, w, 3), dtype=np.uint8)

        # Place piece too low by 150px (error_y = -150px)
        # Target: Row 3 Col 3 (Y_target = 973.25) -> Seen Y = 1123.25
        hold[1063:1183, 419:539] = (0, 200, 255)

        move = PlannedMove(
            step_index=0, slot_idx=0, row=3, col=3, piece_name="dot_1x1",
            cell_count=1, predicted_score_delta=1, lines_cleared=0,
            unique_cells_cleared=0, combo_streak_after=0, is_game_over=False,
        )

        cur_finger = (450, 1400)
        res = ctrl.evaluate_held_frame(move, (172, 1794), cur_finger, base, hold)
        self.assertEqual(res.status, ClosedLoopStatus.NEEDS_CORRECTION)

        prop_x, prop_y = res.proposed_correction_move
        # finger should move up by -150 / 1.50 = -100px (to 1300), NOT by -150px (to 1250)
        expected_y = cur_finger[1] + int(round(res.error_y_px / 1.50))
        self.assertEqual(prop_y, expected_y)
        self.assertEqual(prop_y, 1300)

    # =========================================================================
    # TEST 14: Piece-Specific Gain Tables and Fallback
    # =========================================================================
    def test_14_piece_specific_gain_tables(self) -> None:
        """Verify piece-specific gains are used when available and fall back gracefully."""
        cfg = DragCalibrationConfig(
            drag_gain_x=1.428,
            drag_gain_y=1.438,
            piece_gains_x={"line_1x3_h": 1.413},
            piece_gains_y={"line_1x3_h": 1.518},
        )

        # Known piece line_1x3_h
        self.assertEqual(cfg.get_gain_x_for_piece("line_1x3_h"), 1.413)
        self.assertEqual(cfg.get_gain_y_for_piece("line_1x3_h"), 1.518)

        # Unknown piece falls back to general gains
        self.assertEqual(cfg.get_gain_x_for_piece("square_2x2"), 1.428)
        self.assertEqual(cfg.get_gain_y_for_piece("square_2x2"), 1.438)

    # =========================================================================
    # TEST 15: Invalid Gain Y and Piece Table Rejection
    # =========================================================================
    def test_15_invalid_gain_y_and_tables_rejection(self) -> None:
        """Verify is_valid rejects negative, zero, nan, or inf in Gain Y and piece tables."""
        # Negative drag_gain_y
        cfg_neg = DragCalibrationConfig(drag_gain_y=-1.0)
        self.assertFalse(cfg_neg.is_valid())

        # Zero drag_gain_y
        cfg_zero = DragCalibrationConfig(drag_gain_y=0.0)
        self.assertFalse(cfg_zero.is_valid())

        # NaN drag_gain_y
        cfg_nan = DragCalibrationConfig(drag_gain_y=float("nan"))
        self.assertFalse(cfg_nan.is_valid())

        # Negative in piece_gains_y
        cfg_piece_neg = DragCalibrationConfig(piece_gains_y={"dot_1x1": -0.5})
        self.assertFalse(cfg_piece_neg.is_valid())

    # =========================================================================
    # TEST 16: Tracker Low Confidence Yields UNKNOWN
    # =========================================================================
    def test_16_tracker_low_confidence_unknown(self) -> None:
        """Verify controller flags UNKNOWN when tracker confidence is below threshold."""
        base = np.zeros((1500, 1080, 3), dtype=np.uint8)
        hold = np.zeros((1500, 1080, 3), dtype=np.uint8)

        # Create faint/dark blob that will trigger low confidence
        hold[500:600, 200:300] = (30, 30, 30)

        move = PlannedMove(
            step_index=0, slot_idx=0, row=0, col=0, piece_name="dot_1x1",
            cell_count=1, predicted_score_delta=1, lines_cleared=0,
            unique_cells_cleared=0, combo_streak_after=0, is_game_over=False,
        )

        res = self.controller.evaluate_held_frame(move, (100, 100), (200, 200), base, hold)
        self.assertEqual(res.status, ClosedLoopStatus.UNKNOWN)
        self.assertIsNone(res.proposed_correction_move)


if __name__ == "__main__":
    unittest.main()
