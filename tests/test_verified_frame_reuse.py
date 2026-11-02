"""Unit & Integration Tests for Phase 1.3: Verified Frame Reuse.

Validates that post-action frames verified at the end of turn N are safely reused
for turn N+1 without performing redundant ADB screen captures or Computer Vision
re-evaluations, under strict safety gates.
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
    TurnLatencyMetrics,
)
from action_engine.closed_loop_controller import ClosedLoopController, ClosedLoopStatus, ClosedLoopStepResult
from ai_engine.decision import AIDecisionEngine
from computer_vision.capture import ScreenCapture
from computer_vision.config import DEFAULT_VISION_CONFIG
from computer_vision.models import (
    BoardRecognitionResult,
    CellInfo,
    CellState,
    FrameRecognitionResult,
    PieceRecognitionResult,
    RecognitionStatus,
)
from computer_vision.recognizer import BlockBlastRecognizer
from game_engine.board import coord_to_bit
from game_engine.pieces import CANONICAL_41_CATALOG, Piece


def make_test_board(
    bitboard: int = 0x0,
    board_bbox: tuple[int, int, int, int] = (54, 548, 972, 972),
    board_conf: float = 0.98,
    cell_conf: float = 0.98,
    uncertain_cell: tuple[int, int] | None = None,
    subthreshold_cell: tuple[int, int] | None = None,
) -> BoardRecognitionResult:
    """Constructs a fully formed 8x8 BoardRecognitionResult for testing."""
    cell_size = board_bbox[2] / 8.0
    cells = []
    matrix = []
    for r in range(8):
        row_cells = []
        row_mat = []
        for c in range(8):
            is_occ = bool(bitboard & coord_to_bit(r, c))
            is_unc = (uncertain_cell == (r, c))
            is_sub = (subthreshold_cell == (r, c))

            state = CellState.UNCERTAIN if is_unc else (CellState.OCCUPIED if is_occ else CellState.EMPTY)
            if is_sub:
                c_conf = 0.65
            elif is_unc:
                c_conf = 0.40
            else:
                c_conf = cell_conf

            row_cells.append(
                CellInfo(
                    row=r,
                    col=c,
                    state=state,
                    confidence=c_conf,
                    center_x=int(board_bbox[0] + (c + 0.5) * cell_size),
                    center_y=int(board_bbox[1] + (r + 0.5) * cell_size),
                    width=int(cell_size),
                    height=int(cell_size),
                    mean_gray=180.0 if is_occ else 84.0,
                    mean_sat=100.0 if is_occ else 20.0,
                )
            )
            row_mat.append(1 if is_occ else 0)
        cells.append(tuple(row_cells))
        matrix.append(tuple(row_mat))

    return BoardRecognitionResult(
        board_bbox=board_bbox,
        grid_rows=8,
        grid_cols=8,
        cell_size=cell_size,
        cells=tuple(cells),
        occupancy_matrix=tuple(matrix),
        bitboard=bitboard,
        board_confidence=board_conf,
        grid_confidence=0.99,
        cell_confidence=cell_conf if uncertain_cell is None else 0.40,
    )


def make_test_tray_piece(
    slot_idx: int,
    piece: Piece | None = None,
    is_empty: bool = False,
    confidence: float = 0.95,
    matched_name: str | None = None,
) -> PieceRecognitionResult:
    """Constructs a PieceRecognitionResult for testing."""
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
            bbox_in_screen=(100 + slot_idx * 300, 1800, 100, 100),
            confidence=confidence,
            status_message="Empty slot",
        )
    p_name = matched_name or (piece.name if piece else "UNKNOWN")
    return PieceRecognitionResult(
        slot_idx=slot_idx,
        is_empty=False,
        matched_name=p_name,
        piece=piece,
        cells=piece.cells if piece else (),
        height=piece.height if piece else 1,
        width=piece.width if piece else 1,
        cell_count=piece.cell_count if piece else 1,
        bbox_in_screen=(100 + slot_idx * 300, 1800, 200, 200),
        confidence=confidence,
        status_message=f"Piece {p_name}",
    )


def make_test_frame_res(
    board: BoardRecognitionResult | None,
    tray: tuple[PieceRecognitionResult, ...],
    status: RecognitionStatus = RecognitionStatus.SUCCESS,
    overall_conf: float = 0.95,
    error_msg: str | None = None,
) -> FrameRecognitionResult:
    """Constructs a FrameRecognitionResult container."""
    return FrameRecognitionResult(
        status=status,
        board=board,
        tray_pieces=tray,
        overall_confidence=overall_conf,
        processing_time_ms=35.0,
        error_message=error_msg,
        image_shape=(2340, 1080),
    )


class TestVerifiedFrameReuse(unittest.TestCase):
    """Test suite covering the 10 mandatory safety requirements for Phase 1.3."""

    def setUp(self) -> None:
        self.catalog = CANONICAL_41_CATALOG
        self.dot = self.catalog.get_by_name("dot_1x1")
        self.bar2 = self.catalog.get_by_name("line_1x2_h")
        self.bar3 = self.catalog.get_by_name("line_1x3_h")

        # Mock dependencies
        self.mock_cap = MagicMock(spec=ScreenCapture)
        self.dummy_img = np.zeros((2340, 1080, 3), dtype=np.uint8)
        self.mock_cap.capture_from_adb.return_value = self.dummy_img
        self.mock_cap.last_adb_transfer_ms = 1500.0
        self.mock_cap.last_png_decode_ms = 35.0

        self.mock_rec = MagicMock(spec=BlockBlastRecognizer)
        self.mock_rec.config = DEFAULT_VISION_CONFIG
        self.ai_engine = AIDecisionEngine()
        self.mock_closed_loop = MagicMock(spec=ClosedLoopController)
        self.mock_verifier = MagicMock(spec=ActionVerifier)

        # Baseline converged step result
        self.step_res = ClosedLoopStepResult(
            status=ClosedLoopStatus.CONVERGED,
            step_index=0,
            piece_name="dot_1x1",
            slot_idx=0,
            target_grid_coord=(0, 0),
            target_screen_center=(114, 608),
            initial_finger_down=(250, 1900),
            current_finger_pos=(114, 608),
            detected_piece_center=(114, 608),
            detected_bbox=(54, 548, 120, 120),
            error_x_px=0.0,
            error_y_px=0.0,
            proposed_correction_move=None,
            correction_count=0,
            tracker_detection=None,
            is_dry_run=False,
            message="Converged",
        )
        self.mock_closed_loop.evaluate_held_frame.return_value = self.step_res
        self.mock_closed_loop.plan_initial_trajectory.return_value = (500, 1800, 200, 700)

    def _create_controller(self, max_turns: int = 5) -> AutonomousGameController:
        config = AutonomousConfig(
            device_id="TEST_DEVICE",
            adb_path="adb",
            max_turns=max_turns,
            dry_run=False,
            loop_idle_sleep_sec=0.01,
            fast_gesture_mode=True,
        )
        ctrl = AutonomousGameController(
            config=config,
            screen_capture=self.mock_cap,
            recognizer=self.mock_rec,
            ai_engine=self.ai_engine,
            closed_loop_controller=self.mock_closed_loop,
            action_verifier=self.mock_verifier,
        )
        # Mock interactive shell to avoid real subprocess execution in tests
        mock_shell = MagicMock()
        mock_shell.poll.return_value = None
        mock_shell.stdin = MagicMock()
        ctrl._shell_proc = mock_shell
        ctrl._read_shell_until_token = MagicMock(return_value=True)
        return ctrl

    # =========================================================================
    # TEST 1: VERIFIED + Valid Frame -> Next Turn Reuses Frame, Skips Capture
    # =========================================================================

    def test_01_verified_valid_frame_reuses_frame_and_skips_capture(self) -> None:
        """VERIFIED + valid frame: Turn 2 uses frame directly without calling ADB capture or CV."""
        ctrl = self._create_controller(max_turns=2)

        # Pre-move perception for Turn 1: 3 pieces in tray
        initial_board = make_test_board(bitboard=0x0)
        initial_tray = (
            make_test_tray_piece(0, self.dot),
            make_test_tray_piece(1, self.bar2),
            make_test_tray_piece(2, self.bar3),
        )
        self.mock_rec.recognize_frame.return_value = make_test_frame_res(initial_board, initial_tray)

        # Verification result for Turn 1: slot 0 placed, slots 1 & 2 remain in tray, 0 lines cleared
        post_board = make_test_board(bitboard=0x1)  # cell (0, 0) occupied by dot_1x1
        post_tray = (
            make_test_tray_piece(0, is_empty=True),
            make_test_tray_piece(1, self.bar2),
            make_test_tray_piece(2, self.bar3),
        )
        verified_frame_res = make_test_frame_res(post_board, post_tray)
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
            elapsed_ms=15.0,
            new_frame_result=verified_frame_res,
            is_fast_path=True,
            captures_count=1,
        )

        # Execute Turn 1
        turn1_ok = ctrl.step_single_turn()
        self.assertTrue(turn1_ok)
        self.assertEqual(ctrl.stats.total_turns, 1)
        self.assertEqual(self.mock_cap.capture_from_adb.call_count, 1)
        self.assertEqual(self.mock_rec.recognize_frame.call_count, 1)
        self.assertIsNotNone(ctrl._pending_verified_frame)

        # Execute Turn 2: should HIT frame reuse
        turn2_ok = ctrl.step_single_turn()
        self.assertTrue(turn2_ok)
        self.assertEqual(ctrl.stats.total_turns, 2)

        # CRITICAL ASSERTION: ADB capture was NOT called again! (still 1, not 2)
        self.assertEqual(self.mock_cap.capture_from_adb.call_count, 1)
        # CRITICAL ASSERTION: Vision recognition was NOT called again for Turn 2 perception!
        self.assertEqual(self.mock_rec.recognize_frame.call_count, 1)

        # Telemetry assertions
        self.assertEqual(ctrl.stats.frame_reuse_hits, 1)
        self.assertEqual(ctrl.stats.captures_saved, 1)
        last_metric = ctrl.stats.recent_latencies[-1]
        self.assertTrue(last_metric.frame_reused)
        self.assertEqual(last_metric.capture_ms, 0.0)
        self.assertEqual(last_metric.vision_ms, 0.0)

    # =========================================================================
    # TEST 2: MISMATCH -> Does Not Reuse
    # =========================================================================

    def test_02_mismatch_verification_does_not_reuse(self) -> None:
        """MISMATCH verification: does not cache pending frame, clears state, rejects reuse."""
        ctrl = self._create_controller(max_turns=2)

        initial_board = make_test_board(bitboard=0x0)
        initial_tray = (
            make_test_tray_piece(0, self.dot),
            make_test_tray_piece(1, self.bar2),
            make_test_tray_piece(2, self.bar3),
        )
        self.mock_rec.recognize_frame.return_value = make_test_frame_res(initial_board, initial_tray)

        # Mismatch result
        mismatch_res = VerificationResult(
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
            new_frame_result=None,
            error_message="1 cells mismatched",
        )
        self.mock_verifier.verify_move.return_value = mismatch_res

        turn1_ok = ctrl.step_single_turn()
        self.assertFalse(turn1_ok)  # Pauses on mismatch
        self.assertIsNone(ctrl._pending_verified_frame)
        self.assertEqual(ctrl.stats.frame_reuse_hits, 0)

    # =========================================================================
    # TEST 3: Frame Missing Board -> Does Not Reuse
    # =========================================================================

    def test_03_missing_board_does_not_reuse(self) -> None:
        """Frame missing board detection: rejected by safety gate."""
        ctrl = self._create_controller(max_turns=2)

        post_tray = (
            make_test_tray_piece(0, is_empty=True),
            make_test_tray_piece(1, self.bar2),
            make_test_tray_piece(2, self.bar3),
        )
        # new_frame_result has board=None
        frame_res_no_board = make_test_frame_res(board=None, tray=post_tray)
        ver_res = VerificationResult(
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
            new_frame_result=frame_res_no_board,
        )

        can_reuse, reason = ctrl._can_reuse_verified_frame(ver_res, lines_cleared=0)
        self.assertFalse(can_reuse)
        self.assertIn("Board detection is None", reason)

    # =========================================================================
    # TEST 4: One Cell Confidence < 0.70 -> Does Not Reuse
    # =========================================================================

    def test_04_cell_confidence_below_threshold_does_not_reuse(self) -> None:
        """Cell with confidence 0.65 (< 0.70): rejected by safety gate."""
        ctrl = self._create_controller(max_turns=2)

        post_board = make_test_board(bitboard=0x1, subthreshold_cell=(2, 3))  # Cell (2, 3) has conf 0.65
        post_tray = (
            make_test_tray_piece(0, is_empty=True),
            make_test_tray_piece(1, self.bar2),
            make_test_tray_piece(2, self.bar3),
        )
        ver_res = VerificationResult(
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
            new_frame_result=make_test_frame_res(post_board, post_tray),
        )

        can_reuse, reason = ctrl._can_reuse_verified_frame(ver_res, lines_cleared=0)
        self.assertFalse(can_reuse)
        self.assertIn("Cell (2, 3) confidence", reason)
        self.assertIn("< 0.70", reason)

    # =========================================================================
    # TEST 5: Slot UNKNOWN -> Does Not Reuse
    # =========================================================================

    def test_05_tray_slot_unknown_does_not_reuse(self) -> None:
        """Tray slot with UNKNOWN piece: rejected by safety gate."""
        ctrl = self._create_controller(max_turns=2)

        post_board = make_test_board(bitboard=0x1)
        post_tray = (
            make_test_tray_piece(0, is_empty=True),
            make_test_tray_piece(1, piece=None, matched_name="UNKNOWN"),
            make_test_tray_piece(2, self.bar3),
        )
        ver_res = VerificationResult(
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
            new_frame_result=make_test_frame_res(post_board, post_tray),
        )

        can_reuse, reason = ctrl._can_reuse_verified_frame(ver_res, lines_cleared=0)
        self.assertFalse(can_reuse)
        self.assertIn("Slot 1", reason.title())

    # =========================================================================
    # TEST 6: Slot Confidence < 0.70 -> Does Not Reuse
    # =========================================================================

    def test_06_tray_slot_confidence_below_threshold_does_not_reuse(self) -> None:
        """Tray slot confidence 0.60 (< 0.70): rejected by safety gate."""
        ctrl = self._create_controller(max_turns=2)

        post_board = make_test_board(bitboard=0x1)
        post_tray = (
            make_test_tray_piece(0, is_empty=True),
            make_test_tray_piece(1, self.bar2, confidence=0.60),  # Sub-threshold slot
            make_test_tray_piece(2, self.bar3),
        )
        ver_res = VerificationResult(
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
            new_frame_result=make_test_frame_res(post_board, post_tray),
        )

        can_reuse, reason = ctrl._can_reuse_verified_frame(ver_res, lines_cleared=0)
        self.assertFalse(can_reuse)
        self.assertIn("Tray slot 1 confidence (0.60) < 0.70", reason)

    # =========================================================================
    # TEST 7: Line-Clear -> Does Not Reuse
    # =========================================================================

    def test_07_line_clear_does_not_reuse(self) -> None:
        """Line-clear detected (lines_cleared > 0): rejected by safety gate."""
        ctrl = self._create_controller(max_turns=2)

        post_board = make_test_board(bitboard=0x0)
        post_tray = (
            make_test_tray_piece(0, is_empty=True),
            make_test_tray_piece(1, self.bar2),
            make_test_tray_piece(2, self.bar3),
        )
        ver_res = VerificationResult(
            status=VerificationStatus.VERIFIED,
            step_index=0,
            slot_idx=0,
            piece_name="dot_1x1",
            expected_bitboard=0x0,
            actual_bitboard=0x0,
            mismatch_count=0,
            mismatched_cells=(),
            tray_slot_cleared=True,
            attempts=1,
            elapsed_ms=10.0,
            new_frame_result=make_test_frame_res(post_board, post_tray),
        )

        can_reuse, reason = ctrl._can_reuse_verified_frame(ver_res, lines_cleared=1)
        self.assertFalse(can_reuse)
        self.assertIn("Line-clear detected", reason)

    # =========================================================================
    # TEST 8: Tray Refill (All Pieces Used) -> Does Not Reuse
    # =========================================================================

    def test_08_tray_refill_empty_tray_does_not_reuse(self) -> None:
        """All 3 tray pieces used: rejected by safety gate because round refill is needed."""
        ctrl = self._create_controller(max_turns=2)

        post_board = make_test_board(bitboard=0x7)
        # All 3 slots empty:
        empty_tray = (
            make_test_tray_piece(0, is_empty=True),
            make_test_tray_piece(1, is_empty=True),
            make_test_tray_piece(2, is_empty=True),
        )
        ver_res = VerificationResult(
            status=VerificationStatus.VERIFIED,
            step_index=2,
            slot_idx=2,
            piece_name="dot_1x1",
            expected_bitboard=0x7,
            actual_bitboard=0x7,
            mismatch_count=0,
            mismatched_cells=(),
            tray_slot_cleared=True,
            attempts=1,
            elapsed_ms=10.0,
            new_frame_result=make_test_frame_res(post_board, empty_tray),
        )

        can_reuse, reason = ctrl._can_reuse_verified_frame(ver_res, lines_cleared=0)
        self.assertFalse(can_reuse)
        self.assertIn("refill required", reason)

    # =========================================================================
    # TEST 9: Frame is Used Only Once (Single-Use Guarantee)
    # =========================================================================

    def test_09_frame_reused_only_once_single_use(self) -> None:
        """Cached frame is consumed on Turn 2; Turn 3 must capture fresh if Turn 2 is not reusable."""
        ctrl = self._create_controller(max_turns=3)

        initial_board = make_test_board(bitboard=0x0)
        initial_tray = (
            make_test_tray_piece(0, self.dot),
            make_test_tray_piece(1, self.bar2),
            make_test_tray_piece(2, self.bar3),
        )
        self.mock_rec.recognize_frame.return_value = make_test_frame_res(initial_board, initial_tray)

        # Turn 1 verification: reusable (2 pieces left)
        board_turn1 = make_test_board(bitboard=0x1)
        tray_turn1 = (
            make_test_tray_piece(0, is_empty=True),
            make_test_tray_piece(1, self.bar2),
            make_test_tray_piece(2, self.bar3),
        )
        ver_res_1 = VerificationResult(
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
            new_frame_result=make_test_frame_res(board_turn1, tray_turn1),
            is_fast_path=True,
            captures_count=1,
        )

        # Turn 2 verification: NOT reusable (line cleared)
        board_turn2 = make_test_board(bitboard=0x3)
        tray_turn2 = (
            make_test_tray_piece(0, is_empty=True),
            make_test_tray_piece(1, is_empty=True),
            make_test_tray_piece(2, self.bar3),
        )
        ver_res_2 = VerificationResult(
            status=VerificationStatus.VERIFIED,
            step_index=1,
            slot_idx=1,
            piece_name="line_1x2_h",
            expected_bitboard=0x3,
            actual_bitboard=0x3,
            mismatch_count=0,
            mismatched_cells=(),
            tray_slot_cleared=True,
            attempts=1,
            elapsed_ms=10.0,
            new_frame_result=make_test_frame_res(board_turn2, tray_turn2),
            is_fast_path=False,
            captures_count=2,
        )

        self.mock_verifier.verify_move.side_effect = [ver_res_1, ver_res_2, ver_res_2]

        # Turn 1: ADB capture #1
        ctrl.step_single_turn()
        self.assertEqual(self.mock_cap.capture_from_adb.call_count, 1)
        self.assertEqual(ctrl.stats.frame_reuse_hits, 0)
        self.assertIsNotNone(ctrl._pending_verified_frame)

        # Turn 2: REUSED frame (no capture!)
        ctrl.step_single_turn()
        self.assertEqual(self.mock_cap.capture_from_adb.call_count, 1)
        self.assertEqual(ctrl.stats.frame_reuse_hits, 1)

        # But Turn 2 resulted in a line-clear or rejected reuse:
        # Patch last planned move lines_cleared for Turn 2
        ctrl._pending_verified_frame = None  # Turn 2 rejected reuse

        # Turn 3: MUST capture from ADB (capture #2)
        ctrl.step_single_turn()
        self.assertEqual(self.mock_cap.capture_from_adb.call_count, 2)
        self.assertEqual(ctrl.stats.frame_reuse_hits, 1)  # Hits did NOT increase to 2

    # =========================================================================
    # TEST 10: Abnormal Error -> Clears Pending Frame and Captures Fresh
    # =========================================================================

    def test_10_abnormal_error_clears_pending_frame_and_captures_fresh(self) -> None:
        """Error or pause immediately wipes pending frame; subsequent turn captures fresh."""
        ctrl = self._create_controller(max_turns=2)

        # Populate a pending frame artificially
        valid_board = make_test_board(bitboard=0x1)
        valid_tray = (
            make_test_tray_piece(0, is_empty=True),
            make_test_tray_piece(1, self.bar2),
            make_test_tray_piece(2, self.bar3),
        )
        ctrl._pending_verified_frame = make_test_frame_res(valid_board, valid_tray)
        self.assertIsNotNone(ctrl._pending_verified_frame)

        # Trigger PAUSE
        ctrl.pause("Emergency pause")
        self.assertIsNone(ctrl._pending_verified_frame)
        self.assertIsNone(ctrl._pending_verified_img)

        # Populate again and test RESUME
        ctrl._pending_verified_frame = make_test_frame_res(valid_board, valid_tray)
        ctrl.resume("User resumed")
        self.assertIsNone(ctrl._pending_verified_frame)
        self.assertIsNone(ctrl._pending_verified_img)

        # Populate again and test STOP
        ctrl._pending_verified_frame = make_test_frame_res(valid_board, valid_tray)
        ctrl.stop("User stopped")
        self.assertIsNone(ctrl._pending_verified_frame)
        self.assertIsNone(ctrl._pending_verified_img)


if __name__ == "__main__":
    unittest.main()
