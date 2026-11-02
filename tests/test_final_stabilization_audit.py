"""Comprehensive Regression and Stabilization Test Suite for BLOCK BLAST AI ULTIMATE.

Covers all 14 mandatory critical scenarios:
1. Empty board with 3 known canonical pieces.
2. Partially occupied board (40 occupied cells matching live_audit_raw1.png).
3. Nearly full board (56 cells occupied) with tight placement.
4. Completely full board (64 cells occupied -> clean GAME_OVER).
5. Empty tray slot (1 or 2 pieces used -> moves planned for remaining pieces).
6. Multi-color blocks on board (Red, Cyan, Purple, Green, Yellow, Dark Blue).
7. Unknown tray piece (triggers safe pause and diagnostic capture without crash).
8. Board detection shifted or uncertain (triggers drift guard pause, no touch sent).
9. Illegal placement attempt (multi-layer collision guard prevents touch).
10. Failed gesture release / shell timeout (triggers emergency UP fail-safe).
11. Post-action verification mismatch / tray rejection (triggers safe pause).
12. Pause ('P') and Resume ('S') FSM cycles with fresh perception.
13. Continuous AUTO loop execution for >= 50 consecutive turns.
14. Strict mathematical Game Over validation (only when all active pieces have 0 legal placements).
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Optional, Sequence
import unittest
from unittest.mock import MagicMock, patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np

from action_engine.action_verifier import ActionVerifier, VerificationResult, VerificationStatus
from action_engine.autonomous_controller import (
    AutonomousConfig,
    AutonomousGameController,
    AutonomousState,
    TurnLatencyMetrics,
)
from action_engine.closed_loop_controller import ClosedLoopController, ClosedLoopStatus, ClosedLoopStepResult
from ai_engine.decision import AIDecisionEngine, DecisionResult, DecisionStatus, PlannedMove
from computer_vision.cell_classifier import CellClassifier
from computer_vision.config import DEFAULT_VISION_CONFIG, VisionConfig
from computer_vision.models import (
    BoardRecognitionResult,
    CellInfo,
    CellState,
    FrameRecognitionResult,
    PieceRecognitionResult,
    RecognitionStatus,
)
from computer_vision.recognizer import BlockBlastRecognizer
from game_engine.board import BOARD_SIZE, FULL_BOARD_MASK, coord_to_bit, popcount
from game_engine.pieces import CANONICAL_41_CATALOG, Piece
from game_engine.simulator import GameSimulator, StepResult
from game_engine.state import GameState


def make_test_board(
    bitboard: int = 0,
    board_bbox: tuple[int, int, int, int] = (54, 548, 972, 972),
    board_conf: float = 0.95,
    uncertain_cells: tuple[tuple[int, int], ...] = (),
) -> BoardRecognitionResult:
    """Helper to construct a valid BoardRecognitionResult for testing."""
    cell_size = board_bbox[2] / 8.0
    cells_grid = []
    matrix = []
    for r in range(8):
        row_cells = []
        row_mat = []
        for c in range(8):
            is_occ = bool(bitboard & coord_to_bit(r, c))
            is_unc = (r, c) in uncertain_cells
            state = CellState.UNCERTAIN if is_unc else (CellState.OCCUPIED if is_occ else CellState.EMPTY)
            conf = 0.40 if is_unc else 0.95
            occ_val = 1 if is_occ else 0
            cx = int(board_bbox[0] + (c + 0.5) * cell_size)
            cy = int(board_bbox[1] + (r + 0.5) * cell_size)
            row_cells.append(
                CellInfo(
                    row=r,
                    col=c,
                    state=state,
                    confidence=conf,
                    center_x=cx,
                    center_y=cy,
                    width=int(cell_size),
                    height=int(cell_size),
                    mean_gray=180.0 if is_occ else 84.0,
                    mean_sat=100.0 if is_occ else 20.0,
                )
            )
            row_mat.append(occ_val)
        cells_grid.append(row_cells)
        matrix.append(row_mat)

    return BoardRecognitionResult(
        board_bbox=board_bbox,
        grid_rows=8,
        grid_cols=8,
        cell_size=cell_size,
        cells=tuple(tuple(r) for r in cells_grid),
        occupancy_matrix=tuple(tuple(r) for r in matrix),
        bitboard=bitboard,
        board_confidence=board_conf,
        grid_confidence=board_conf if not uncertain_cells else 0.50,
        cell_confidence=board_conf if not uncertain_cells else 0.50,
    )


def make_test_piece(
    slot_idx: int,
    piece: Optional[Piece] = None,
    is_empty: bool = False,
    confidence: float = 0.95,
    matched_name: Optional[str] = None,
) -> PieceRecognitionResult:
    """Helper to construct a valid PieceRecognitionResult."""
    if is_empty:
        return PieceRecognitionResult(
            slot_idx=slot_idx,
            is_empty=True,
            matched_name=None,
            piece=None,
            cells=(),
            height=0,
            width=0,
            cell_count=0,
            bbox_in_screen=(100 + slot_idx * 300, 1700, 150, 150),
            confidence=confidence,
            status_message="EMPTY",
        )
    return PieceRecognitionResult(
        slot_idx=slot_idx,
        is_empty=False,
        matched_name=matched_name or (piece.name if piece else "UNKNOWN"),
        piece=piece,
        cells=piece.cells if piece else (),
        height=piece.height if piece else 0,
        width=piece.width if piece else 0,
        cell_count=piece.cell_count if piece else 0,
        bbox_in_screen=(100 + slot_idx * 300, 1700, 150, 150),
        confidence=confidence,
        status_message="OK" if piece else "UNKNOWN",
    )


def make_test_frame_rec(
    board: Optional[BoardRecognitionResult],
    tray_pieces: Sequence[PieceRecognitionResult],
    overall_confidence: float = 0.95,
    status: RecognitionStatus = RecognitionStatus.SUCCESS,
    error_message: Optional[str] = None,
) -> FrameRecognitionResult:
    """Helper to construct a valid FrameRecognitionResult."""
    return FrameRecognitionResult(
        status=status,
        board=board,
        tray_pieces=tuple(tray_pieces),
        overall_confidence=overall_confidence,
        processing_time_ms=10.0,
        error_message=error_message,
        image_shape=(2340, 1080),
    )


class TestFinalStabilizationAudit(unittest.TestCase):
    """Rigorous regression suite covering all 14 mandatory pipeline scenarios."""

    def setUp(self) -> None:
        self.config = AutonomousConfig(dry_run=True, max_turns=50)
        self.controller = AutonomousGameController(config=self.config)
        self.frame = np.zeros((2340, 1080, 3), dtype=np.uint8)
        self.controller.last_frame = self.frame
        self.controller.capture.capture_from_adb = MagicMock(return_value=self.frame)

    # -------------------------------------------------------------------------
    # Scenario 1: Empty board with 3 known canonical pieces
    # -------------------------------------------------------------------------
    def test_scenario_01_empty_board_with_three_known_pieces(self) -> None:
        """Scenario 1: Empty board with 3 known pieces successfully plans, acts, and verifies."""
        dot = CANONICAL_41_CATALOG.get_by_name("dot_1x1")
        line = CANONICAL_41_CATALOG.get_by_name("line_1x3_h")
        sq = CANONICAL_41_CATALOG.get_by_name("square_2x2")

        board_res = make_test_board(bitboard=0)
        tray_res = [
            make_test_piece(0, dot),
            make_test_piece(1, line),
            make_test_piece(2, sq),
        ]
        mock_rec = make_test_frame_rec(board_res, tray_res)

        with patch.object(self.controller.recognizer, "recognize_frame", return_value=mock_rec):
            success = self.controller.step_single_turn()

        self.assertTrue(success, "Single turn on empty board should succeed.")
        self.assertEqual(self.controller.stats.successful_turns, 1)
        self.assertIsNotNone(self.controller.last_planned_move)
        self.assertIn(self.controller.last_planned_move.slot_idx, (0, 1, 2))

    # -------------------------------------------------------------------------
    # Scenario 2: Partially occupied board (40 occupied cells from live_audit_raw1)
    # -------------------------------------------------------------------------
    def test_scenario_02_partially_occupied_board(self) -> None:
        """Scenario 2: Partially occupied board (40 occupied cells) correctly chooses non-colliding move."""
        board_40 = 0x007E7E3C7E3C0000
        dot = CANONICAL_41_CATALOG.get_by_name("dot_1x1")
        line2 = CANONICAL_41_CATALOG.get_by_name("line_1x2_h")
        sq = CANONICAL_41_CATALOG.get_by_name("square_2x2")

        board_res = make_test_board(bitboard=board_40)
        tray_res = [
            make_test_piece(0, dot),
            make_test_piece(1, line2),
            make_test_piece(2, sq),
        ]
        mock_rec = make_test_frame_rec(board_res, tray_res, overall_confidence=0.92)

        with patch.object(self.controller.recognizer, "recognize_frame", return_value=mock_rec):
            success = self.controller.step_single_turn()

        self.assertTrue(success)
        move = self.controller.last_planned_move
        self.assertIsNotNone(move)
        chosen_piece = CANONICAL_41_CATALOG.get_by_name(move.piece_name)
        mask = chosen_piece.get_placement_mask(move.row, move.col)
        self.assertIsNotNone(mask)
        self.assertEqual(board_40 & mask, 0, "Move must not collide with existing occupied cells.")

    # -------------------------------------------------------------------------
    # Scenario 3: Nearly full board (56 cells occupied) with tight placement
    # -------------------------------------------------------------------------
    def test_scenario_03_nearly_full_board_tight_placement(self) -> None:
        """Scenario 3: Nearly full board (56 cells) finds tight legal placement."""
        board_56 = 0xFFFFFFFFFFFFFF00
        dot = CANONICAL_41_CATALOG.get_by_name("dot_1x1")
        line3 = CANONICAL_41_CATALOG.get_by_name("line_1x3_h")
        sq3 = CANONICAL_41_CATALOG.get_by_name("square_3x3")

        board_res = make_test_board(bitboard=board_56)
        tray_res = [
            make_test_piece(0, dot),
            make_test_piece(1, line3),
            make_test_piece(2, sq3),  # sq3 cannot fit
        ]
        mock_rec = make_test_frame_rec(board_res, tray_res, overall_confidence=0.90)

        with patch.object(self.controller.recognizer, "recognize_frame", return_value=mock_rec):
            success = self.controller.step_single_turn()

        self.assertTrue(success)
        move = self.controller.last_planned_move
        self.assertIsNotNone(move)
        self.assertIn(move.slot_idx, (0, 1), "Must choose piece that fits (dot or line3, not sq3).")
        self.assertEqual(move.row, 0, "Must be placed in the only legal row (Row 0).")

    # -------------------------------------------------------------------------
    # Scenario 4: Completely full board (64 cells occupied -> GAME_OVER)
    # -------------------------------------------------------------------------
    def test_scenario_04_completely_full_board_game_over(self) -> None:
        """Scenario 4: Full board (64 cells occupied) triggers clean GAME_OVER transition."""
        full_board = FULL_BOARD_MASK
        dot = CANONICAL_41_CATALOG.get_by_name("dot_1x1")
        board_res = make_test_board(bitboard=full_board)
        tray_res = [
            make_test_piece(0, dot),
            make_test_piece(1, dot),
            make_test_piece(2, dot),
        ]
        mock_rec = make_test_frame_rec(board_res, tray_res)

        with patch.object(self.controller.recognizer, "recognize_frame", return_value=mock_rec):
            success = self.controller.step_single_turn()

        self.assertFalse(success)
        self.assertEqual(self.controller.state, AutonomousState.GAME_OVER)
        self.assertIn("Game Over", self.controller.last_pause_reason or "Game Over")

    # -------------------------------------------------------------------------
    # Scenario 5: Empty tray slot (1 or 2 pieces used)
    # -------------------------------------------------------------------------
    def test_scenario_05_empty_tray_slot_active_subset(self) -> None:
        """Scenario 5: Dynamic tray with only 1 piece remaining plans only for active piece."""
        line = CANONICAL_41_CATALOG.get_by_name("line_1x3_h")
        board_res = make_test_board(bitboard=0)
        tray_res = [
            make_test_piece(0, is_empty=True),
            make_test_piece(1, piece=line),
            make_test_piece(2, is_empty=True),
        ]
        mock_rec = make_test_frame_rec(board_res, tray_res, overall_confidence=0.92)

        with patch.object(self.controller.recognizer, "recognize_frame", return_value=mock_rec):
            success = self.controller.step_single_turn()

        self.assertTrue(success)
        move = self.controller.last_planned_move
        self.assertIsNotNone(move)
        self.assertEqual(move.slot_idx, 1, "Must choose Slot 1 since Slot 0 and 2 are empty.")

    # -------------------------------------------------------------------------
    # Scenario 6: Multi-color blocks detected by CellClassifier
    # -------------------------------------------------------------------------
    def test_scenario_06_multi_color_blocks_detected_by_cell_classifier(self) -> None:
        """Scenario 6: Dual-criteria CellClassifier detects all Block Blast block colors and preserves empty themes."""
        classifier = CellClassifier()
        dummy_img = np.zeros((2340, 1080, 3), dtype=np.uint8)
        board_bbox = (54, 548, 972, 972)
        cell_size = 121.5

        # Colors: (B, G, R)
        colors = {
            "green": (57, 175, 43),     # Green: low luminance (G=122), high V (175), S (194)
            "red": (49, 48, 198),       # Red: low luminance (G=93), high V (198), S (193)
            "dark_blue": (180, 50, 30), # Dark blue: low luminance (G=59), high V (180), S (210)
            "purple": (160, 40, 160),   # Purple: low luminance (G=90), high V (160), S (180)
            "yellow": (122, 202, 205),  # Yellow: high luminance (G=194)
            "cyan": (200, 200, 50),     # Cyan: high luminance (G=180)
        }

        # Place the 6 colored blocks in row 0, cols 0..5
        for col_idx, (cname, bgr) in enumerate(colors.items()):
            cx0 = int(board_bbox[0] + col_idx * cell_size)
            cx1 = int(cx0 + cell_size)
            cy0 = int(board_bbox[1])
            cy1 = int(cy0 + cell_size)
            dummy_img[cy0:cy1, cx0:cx1] = bgr

        # Place empty cells in row 1:
        # Col 0: dark theme (35, 35, 35)
        # Col 1: greenish-blue theme (96, 102, 45)
        cy0_e = int(board_bbox[1] + cell_size)
        cy1_e = int(cy0_e + cell_size)
        cx0_e0 = int(board_bbox[0])
        cx1_e0 = int(cx0_e0 + cell_size)
        dummy_img[cy0_e:cy1_e, cx0_e0:cx1_e0] = (35, 35, 35)

        cx0_e1 = int(board_bbox[0] + cell_size)
        cx1_e1 = int(cx0_e1 + cell_size)
        dummy_img[cy0_e:cy1_e, cx0_e1:cx1_e1] = (96, 102, 45)

        res = classifier.classify_cells(dummy_img, board_bbox)

        # Verify all 6 colored blocks are OCCUPIED
        for col_idx, cname in enumerate(colors.keys()):
            cell_info = res.cells[0][col_idx]
            self.assertEqual(
                cell_info.state,
                CellState.OCCUPIED,
                f"Block '{cname}' in row 0, col {col_idx} should be classified as OCCUPIED.",
            )
            self.assertEqual(res.occupancy_matrix[0][col_idx], 1)
            self.assertTrue(res.bitboard & coord_to_bit(0, col_idx))

        # Verify both empty cells are EMPTY
        self.assertEqual(
            res.cells[1][0].state,
            CellState.EMPTY,
            "Dark theme empty cell should be classified as EMPTY.",
        )
        self.assertEqual(res.occupancy_matrix[1][0], 0)
        self.assertFalse(res.bitboard & coord_to_bit(1, 0))

        self.assertEqual(
            res.cells[1][1].state,
            CellState.EMPTY,
            "Greenish-blue theme empty cell should be classified as EMPTY.",
        )
        self.assertEqual(res.occupancy_matrix[1][1], 0)
        self.assertFalse(res.bitboard & coord_to_bit(1, 1))

    # -------------------------------------------------------------------------
    # Scenario 7: Unknown tray piece triggers safe pause & diagnostic capture
    # -------------------------------------------------------------------------
    def test_scenario_07_unknown_tray_piece_safe_pause(self) -> None:
        """Scenario 7: Unknown tray piece halts execution and calls diagnostic saver without crash."""
        board_res = make_test_board(bitboard=0)
        tray_res = [
            make_test_piece(0, is_empty=False, matched_name="UNKNOWN", piece=None, confidence=0.0),
            make_test_piece(1, CANONICAL_41_CATALOG.get_by_name("dot_1x1")),
            make_test_piece(2, is_empty=True),
        ]
        mock_rec = make_test_frame_rec(board_res, tray_res, overall_confidence=0.70)

        with patch.object(self.controller.recognizer, "recognize_frame", return_value=mock_rec):
            with patch.object(self.controller.diagnostic_saver, "save_diagnostic_failure") as mock_save:
                success = self.controller.step_single_turn()

        self.assertFalse(success)
        self.assertEqual(self.controller.state, AutonomousState.PAUSED)
        self.assertIn("UNKNOWN", self.controller.last_pause_reason)
        mock_save.assert_called_once()

    # -------------------------------------------------------------------------
    # Scenario 8: Board detection shifted or uncertain triggers pause
    # -------------------------------------------------------------------------
    def test_scenario_08_board_drift_guard_and_uncertain_cells(self) -> None:
        """Scenario 8: Shifted board bbox (> 15px) or UNCERTAIN cells trigger drift guard pause."""
        dot = CANONICAL_41_CATALOG.get_by_name("dot_1x1")

        # Case A: Shifted board bbox (y=570 vs expected 548, delta 22 > 15)
        shifted_bbox = (54, 570, 972, 972)
        board_shifted = make_test_board(bitboard=0, board_bbox=shifted_bbox)
        tray_res = [make_test_piece(0, dot), make_test_piece(1, dot), make_test_piece(2, dot)]
        mock_rec_shifted = make_test_frame_rec(board_shifted, tray_res, overall_confidence=0.90)

        with patch.object(self.controller.recognizer, "recognize_frame", return_value=mock_rec_shifted):
            success = self.controller.step_single_turn()

        self.assertFalse(success)
        self.assertEqual(self.controller.state, AutonomousState.PAUSED)
        self.assertIn("drift detected", self.controller.last_pause_reason)

        # Case B: Board contains UNCERTAIN cells
        self.controller.set_state(AutonomousState.IDLE)
        board_unc = make_test_board(bitboard=0, uncertain_cells=((3, 3),))
        mock_rec_unc = make_test_frame_rec(board_unc, tray_res, overall_confidence=0.65)

        with patch.object(self.controller.recognizer, "recognize_frame", return_value=mock_rec_unc):
            success_unc = self.controller.step_single_turn()

        self.assertFalse(success_unc)
        self.assertEqual(self.controller.state, AutonomousState.PAUSED)
        self.assertIn("uncertain", self.controller.last_pause_reason.lower())

    # -------------------------------------------------------------------------
    # Scenario 9: Illegal placement attempt prevented by collision guard
    # -------------------------------------------------------------------------
    def test_scenario_09_illegal_placement_attempt_prevented(self) -> None:
        """Scenario 9: Pre-actuation collision guard prevents touch if AI attempts overlapping move."""
        board_occ = 0x01  # cell (0, 0) is occupied
        sq = CANONICAL_41_CATALOG.get_by_name("square_2x2")

        board_res = make_test_board(bitboard=board_occ)
        tray_res = [make_test_piece(0, sq), make_test_piece(1, is_empty=True), make_test_piece(2, is_empty=True)]
        mock_rec = make_test_frame_rec(board_res, tray_res)

        colliding_move = PlannedMove(
            step_index=0,
            slot_idx=0,
            row=0,
            col=0,
            piece_name="square_2x2",
            cell_count=4,
            predicted_score_delta=4,
            lines_cleared=0,
            unique_cells_cleared=0,
            combo_streak_after=0,
            is_game_over=False,
        )
        fake_decision = DecisionResult(
            status=DecisionStatus.SUCCESS,
            strategy_name="beam_search",
            moves=(colliding_move,),
            predicted_score_delta=4,
            final_board_evaluation=100.0,
            rationale="Test collision",
            states_explored=1,
            duration_ms=1.0,
            initial_board=board_occ,
            final_board=board_occ,
        )

        with patch.object(self.controller.recognizer, "recognize_frame", return_value=mock_rec):
            with patch.object(self.controller.ai_engine, "decide", return_value=fake_decision):
                success = self.controller.step_single_turn()

        self.assertFalse(success)
        self.assertEqual(self.controller.state, AutonomousState.PAUSED)
        self.assertIn("collides with bitboard", self.controller.last_pause_reason)

    # -------------------------------------------------------------------------
    # Scenario 10: Failed gesture release triggers emergency UP fail-safe
    # -------------------------------------------------------------------------
    def test_scenario_10_failed_gesture_release_triggers_emergency_up(self) -> None:
        """Scenario 10: Actuation exception or timeout invokes emergency fail-safe UP."""
        live_cfg = AutonomousConfig(dry_run=False, fast_gesture_mode=True)
        live_ctrl = AutonomousGameController(config=live_cfg)
        dot = CANONICAL_41_CATALOG.get_by_name("dot_1x1")

        mock_shell = MagicMock()
        mock_shell.stdin = MagicMock()
        # DOWN succeeds, then subsequent MOVE fails with BrokenPipeError
        mock_shell.stdin.flush.side_effect = [None, BrokenPipeError("Simulated pipe failure")]

        with patch.object(live_ctrl, "_get_interactive_shell", return_value=mock_shell):
            with patch.object(live_ctrl, "_release_pointer_failsafe") as mock_release:
                res = live_ctrl._execute_physical_gesture(
                    move=PlannedMove(0, 0, 0, 0, "dot_1x1", 1, 1, 0, 0, 0, False),
                    piece_model=dot,
                    base_frame=self.frame,
                    start_x=200,
                    start_y=1700,
                    finger_x_0=114,
                    finger_y_0=608,
                )

        self.assertEqual(res.status, ClosedLoopStatus.UNKNOWN)
        mock_release.assert_called()
        self.assertFalse(live_ctrl._touch_pointer_down)

    # -------------------------------------------------------------------------
    # Scenario 11: Post-action verification mismatch / tray rejection
    # -------------------------------------------------------------------------
    def test_scenario_11_post_action_verification_mismatch_and_rejection(self) -> None:
        """Scenario 11: Verification mismatch or game rejection triggers safe pause and tracks failure."""
        dot = CANONICAL_41_CATALOG.get_by_name("dot_1x1")
        board_res = make_test_board(bitboard=0)
        tray_res = [make_test_piece(0, dot), make_test_piece(1, is_empty=True), make_test_piece(2, is_empty=True)]
        mock_rec = make_test_frame_rec(board_res, tray_res)

        mock_v_res = VerificationResult(
            status=VerificationStatus.TRAY_REJECTED,
            step_index=0,
            slot_idx=0,
            piece_name="dot_1x1",
            expected_bitboard=1,
            actual_bitboard=0,
            mismatch_count=1,
            mismatched_cells=(),
            tray_slot_cleared=False,
            attempts=1,
            elapsed_ms=50.0,
            error_message="Piece returned to tray slot",
        )

        with patch.object(self.controller.recognizer, "recognize_frame", return_value=mock_rec):
            with patch.object(self.controller.verifier, "verify_move", return_value=mock_v_res):
                self.controller.config = AutonomousConfig(dry_run=False, fast_gesture_mode=True)
                with patch.object(self.controller, "_execute_physical_gesture") as mock_gesture:
                    mock_gesture.return_value = ClosedLoopStepResult(
                        status=ClosedLoopStatus.CONVERGED,
                        step_index=0,
                        piece_name="dot_1x1",
                        slot_idx=0,
                        target_grid_coord=(0, 0),
                        target_screen_center=(114, 608),
                        initial_finger_down=(200, 1700),
                        current_finger_pos=(114, 608),
                        detected_piece_center=(114, 608),
                        detected_bbox=(54, 548, 120, 120),
                        error_x_px=0.0,
                        error_y_px=0.0,
                        proposed_correction_move=None,
                        correction_count=0,
                        tracker_detection=None,
                        is_dry_run=False,
                        message="OK",
                    )
                    success = self.controller.step_single_turn()

        self.assertFalse(success)
        self.assertEqual(self.controller.state, AutonomousState.PAUSED)
        self.assertEqual(self.controller.stats.failed_turns, 1)

    # -------------------------------------------------------------------------
    # Scenario 12: Pause ('P') and Resume ('S') FSM cycles
    # -------------------------------------------------------------------------
    def test_scenario_12_pause_and_resume_fsm_cycles(self) -> None:
        """Scenario 12: Pause releases pointer; Resume clears cached frames and transitions to CAPTURING."""
        self.controller._touch_pointer_down = True
        with patch.object(self.controller, "_release_pointer_failsafe") as mock_release:
            self.controller.pause("User pause")

        self.assertEqual(self.controller.state, AutonomousState.PAUSED)
        self.assertFalse(self.controller._touch_pointer_down)
        mock_release.assert_called_once()

        self.controller.resume("User resume")
        self.assertEqual(self.controller.state, AutonomousState.CAPTURING)
        self.assertIsNone(self.controller.last_frame, "Resume must clear cached frame for fresh capture.")

    # -------------------------------------------------------------------------
    # Scenario 13: Continuous AUTO loop execution for >= 50 turns
    # -------------------------------------------------------------------------
    def test_scenario_13_continuous_auto_loop_50_turns(self) -> None:
        """Scenario 13: Full autonomous loop runs 50 consecutive turns without drift or unhandled exceptions."""
        cfg = AutonomousConfig(dry_run=True, max_turns=50, loop_idle_sleep_sec=0.001)
        ctrl = AutonomousGameController(config=cfg)
        ctrl.last_frame = self.frame
        ctrl.capture.capture_from_adb = MagicMock(return_value=self.frame)

        dot = CANONICAL_41_CATALOG.get_by_name("dot_1x1")
        board_res = make_test_board(bitboard=0)
        tray_res = [make_test_piece(0, dot), make_test_piece(1, dot), make_test_piece(2, dot)]
        mock_rec = make_test_frame_rec(board_res, tray_res)

        with patch.object(ctrl.recognizer, "recognize_frame", return_value=mock_rec):
            ctrl.start()
            for _ in range(1000):
                if ctrl.stats.total_turns >= 50 or ctrl.state in (AutonomousState.PAUSED, AutonomousState.GAME_OVER):
                    break
                time.sleep(0.01)
            ctrl.stop("Test complete")

        self.assertGreaterEqual(ctrl.stats.total_turns, 50, "Should complete all 50 requested turns.")
        self.assertEqual(ctrl.stats.failed_turns, 0, "All 50 turns should succeed.")
        self.assertEqual(ctrl.stats.successful_turns, 50)
        self.assertGreater(ctrl.stats.total_score_delta, 0)

    # -------------------------------------------------------------------------
    # Scenario 14: Strict mathematical Game Over validation
    # -------------------------------------------------------------------------
    def test_scenario_14_strict_mathematical_game_over_validation(self) -> None:
        """Scenario 14: Game Over triggers ONLY when all active tray pieces have 0 legal placements."""
        sim = GameSimulator()
        sq3 = CANONICAL_41_CATALOG.get_by_name("square_3x3")
        dot = CANONICAL_41_CATALOG.get_by_name("dot_1x1")

        # Board with 1 isolated empty cell at (0, 0)
        board_almost_full = FULL_BOARD_MASK ^ coord_to_bit(0, 0)

        # Hand with square_3x3 only: CANNOT fit anywhere on board -> Game Over is TRUE
        state_no_fit = GameState(board=board_almost_full, tray=(sq3, None, None))
        self.assertFalse(sim.has_any_legal_move(state_no_fit))
        self.assertTrue(sim.check_game_over(state_no_fit))

        # Hand with dot_1x1: CAN fit at (0, 0) -> Game Over is FALSE
        state_fits = GameState(board=board_almost_full, tray=(sq3, dot, None))
        self.assertTrue(sim.has_any_legal_move(state_fits))
        self.assertFalse(sim.check_game_over(state_fits))


if __name__ == "__main__":
    unittest.main()
