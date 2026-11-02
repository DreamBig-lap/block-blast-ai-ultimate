"""Unit Tests for Phase 1: Safe Single-Frame Verification (Fast-Path).

Tests all 10 mandatory edge cases:
1. Normal move, Frame A matches completely -> Fast-path confirmed (captures=1, is_fast_path=True, VERIFIED).
2. Frame A bitboard mismatch -> Fast-path skipped, falls back to two-frame check.
3. Frame A low confidence / uncertain cells -> Fast-path skipped, falls back to two-frame / retry.
4. Tray does not match expected tray -> Fast-path skipped, falls back to two-frame check.
5. Dragged slot not empty in Frame A -> Fast-path skipped, falls back to two-frame check.
6. Move with line clear (lines_cleared > 0) -> Fast-path skipped, executes two-frame pipeline.
7. Frame A capture/recognition failure -> Fallback to VISION_FAILED, halts safely.
8. Frame B unstable (consecutive frame inconsistency) -> Retry and UNSTABLE_ANIMATION_TIMEOUT.
9. Timeout ceiling reached -> Never returns VERIFIED, halts with MISMATCH or timeout.
10. PAUSE / STOP behavior preserved -> Appropriate error status and message returned.
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Any
import unittest
from unittest.mock import MagicMock, patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from action_engine.action_verifier import ActionVerifier, VerificationResult, VerificationStatus
from computer_vision.config import DEFAULT_VISION_CONFIG
from computer_vision.models import (
    BoardRecognitionResult,
    CellInfo,
    CellState,
    FrameRecognitionResult,
    PieceRecognitionResult,
    RecognitionStatus,
)
from game_engine.board import coord_to_bit
from game_engine.pieces import CANONICAL_41_CATALOG, Piece
from game_engine.simulator import GameSimulator, StepResult
from game_engine.state import GameState


def make_test_board(
    bitboard: int = 0,
    board_bbox: tuple[int, int, int, int] = (54, 548, 972, 972),
    board_conf: float = 0.98,
    cell_conf: float = 0.98,
    has_uncertain: bool = False,
) -> BoardRecognitionResult:
    """Helper to build BoardRecognitionResult for testing."""
    cell_size = board_bbox[2] / 8.0
    cells = []
    matrix = []
    for r in range(8):
        row_cells = []
        row_mat = []
        for c in range(8):
            is_occ = bool(bitboard & coord_to_bit(r, c))
            is_unc = has_uncertain and (r == 0 and c == 0)
            state = CellState.UNCERTAIN if is_unc else (CellState.OCCUPIED if is_occ else CellState.EMPTY)
            conf = 0.40 if is_unc else cell_conf
            row_cells.append(
                CellInfo(
                    row=r,
                    col=c,
                    state=state,
                    confidence=conf,
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
        cell_confidence=cell_conf if not has_uncertain else 0.40,
    )


def make_test_tray_piece(
    slot_idx: int,
    piece: Piece | None = None,
    is_empty: bool = False,
    confidence: float = 0.95,
    matched_name: str | None = None,
) -> PieceRecognitionResult:
    """Helper to build PieceRecognitionResult for testing."""
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
            status_message="Slot is empty",
        )
    p_name = matched_name or (piece.name if piece else "UNKNOWN")
    return PieceRecognitionResult(
        slot_idx=slot_idx,
        is_empty=False,
        matched_name=p_name,
        piece=piece,
        cells=piece.cells if piece else (),
        height=piece.height if piece else 0,
        width=piece.width if piece else 0,
        cell_count=piece.cell_count if piece else 0,
        bbox_in_screen=(100 + slot_idx * 300, 1700, 150, 150),
        confidence=confidence,
        status_message=f"Matched: {p_name}",
    )


def make_test_frame(
    bitboard: int,
    tray_pieces: tuple[PieceRecognitionResult, ...],
    board_bbox: tuple[int, int, int, int] = (54, 548, 972, 972),
    board_conf: float = 0.98,
    cell_conf: float = 0.98,
    has_uncertain: bool = False,
    status: RecognitionStatus | None = None,
    error_message: str | None = None,
) -> FrameRecognitionResult:
    """Helper to build full FrameRecognitionResult for testing."""
    board = make_test_board(
        bitboard=bitboard,
        board_bbox=board_bbox,
        board_conf=board_conf,
        cell_conf=cell_conf,
        has_uncertain=has_uncertain,
    )
    if status is None:
        if has_uncertain or board_conf < 0.70 or cell_conf < 0.70:
            eff_status = RecognitionStatus.LOW_CONFIDENCE
            if error_message is None:
                error_message = "Perception uncertain: Board contains UNCERTAIN cells."
        else:
            eff_status = RecognitionStatus.SUCCESS
    else:
        eff_status = status

    return FrameRecognitionResult(
        status=eff_status,
        board=board,
        tray_pieces=tray_pieces,
        overall_confidence=min(board_conf, cell_conf),
        processing_time_ms=15.0,
        error_message=error_message,
        image_shape=(2340, 1080),
    )


class TestSingleFrameVerification(unittest.TestCase):
    """Unit tests for Safe Single-Frame Verification."""

    def setUp(self):
        self.p0 = CANONICAL_41_CATALOG.get_by_name("dot_1x1")
        self.p1 = CANONICAL_41_CATALOG.get_by_name("line_1x2_h")
        self.p2 = CANONICAL_41_CATALOG.get_by_name("square_2x2")
        self.sim = GameSimulator()

        # Standard initial state: 3 pieces in tray, empty board
        self.init_state = GameState.create_initial(initial_tray=[self.p0, self.p1, self.p2])
        # Place dot_1x1 at (0, 0), slot 0 -> no line clears
        self.step_res = self.sim.apply_move(self.init_state, slot_idx=0, row=0, col=0)
        self.expected_bb = self.step_res.state_after.board

    def _create_matching_frame_a(self) -> FrameRecognitionResult:
        """Create a Frame A that matches step_res.state_after perfectly."""
        tray = (
            make_test_tray_piece(slot_idx=0, is_empty=True),
            make_test_tray_piece(slot_idx=1, piece=self.p1),
            make_test_tray_piece(slot_idx=2, piece=self.p2),
        )
        return make_test_frame(bitboard=self.expected_bb, tray_pieces=tray)

    # -------------------------------------------------------------------------
    # Scenario 1: Normal move, Frame A matches completely -> Fast-Path Verified
    # -------------------------------------------------------------------------
    def test_scenario_1_fast_path_success(self):
        """Move with no line clear and perfect Frame A matches fast-path criteria."""
        frame_a = self._create_matching_frame_a()

        mock_rec = MagicMock()
        mock_rec.recognize_frame.return_value = frame_a
        mock_rec.config = DEFAULT_VISION_CONFIG

        mock_cap = MagicMock()
        mock_cap.capture_from_adb.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        verifier = ActionVerifier(
            recognizer=mock_rec,
            capture=mock_cap,
            normal_settle_sec=0.0,
            consecutive_frame_delay_sec=0.15,
            max_settle_retries=2,
        )

        res = verifier.verify_move(self.step_res)

        # Assert fast-path was taken
        self.assertTrue(res.is_success)
        self.assertEqual(res.status, VerificationStatus.VERIFIED)
        self.assertTrue(res.is_fast_path)
        self.assertEqual(res.captures_count, 1)
        self.assertEqual(res.attempts, 1)
        self.assertEqual(res.mismatch_count, 0)
        self.assertTrue(res.tray_slot_cleared)
        self.assertIsNone(res.fast_path_rejection_reason)
        # Exactly 1 capture and 1 recognize call
        self.assertEqual(mock_cap.capture_from_adb.call_count, 1)
        self.assertEqual(mock_rec.recognize_frame.call_count, 1)

    # -------------------------------------------------------------------------
    # Scenario 2: Frame A bitboard mismatch -> Fallback to Two-Frame
    # -------------------------------------------------------------------------
    def test_scenario_2_bitboard_mismatch_skips_fast_path(self):
        """Frame A bitboard mismatch rejects fast-path and falls back to two-frame check."""
        # Frame A has an extra bit (unsettled particle)
        tray = (
            make_test_tray_piece(slot_idx=0, is_empty=True),
            make_test_tray_piece(slot_idx=1, piece=self.p1),
            make_test_tray_piece(slot_idx=2, piece=self.p2),
        )
        frame_a = make_test_frame(bitboard=self.expected_bb | 0x80, tray_pieces=tray)
        # Frame B in attempt 1 is also mismatched
        frame_b = make_test_frame(bitboard=self.expected_bb | 0x80, tray_pieces=tray)
        # In attempt 2, particle settles
        frame2_a = make_test_frame(bitboard=self.expected_bb, tray_pieces=tray)
        frame2_b = make_test_frame(bitboard=self.expected_bb, tray_pieces=tray)

        mock_rec = MagicMock()
        mock_rec.recognize_frame.side_effect = [frame_a, frame_b, frame2_a, frame2_b]
        mock_rec.config = DEFAULT_VISION_CONFIG

        mock_cap = MagicMock()
        mock_cap.capture_from_adb.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        verifier = ActionVerifier(
            recognizer=mock_rec,
            capture=mock_cap,
            normal_settle_sec=0.0,
            consecutive_frame_delay_sec=0.0,
            retry_interval_sec=0.0,
            max_settle_retries=2,
        )

        res = verifier.verify_move(self.step_res)

        self.assertTrue(res.is_success)
        self.assertFalse(res.is_fast_path)
        self.assertEqual(res.attempts, 2)
        self.assertEqual(res.captures_count, 4)
        self.assertIsNotNone(res.fast_path_rejection_reason)
        self.assertIn("Bitboard mismatch", res.fast_path_rejection_reason)

    # -------------------------------------------------------------------------
    # Scenario 3: Frame A low confidence / uncertain cell -> Fallback
    # -------------------------------------------------------------------------
    def test_scenario_3_low_confidence_skips_fast_path(self):
        """Frame A having uncertain cell or low confidence rejects fast-path."""
        tray = (
            make_test_tray_piece(slot_idx=0, is_empty=True),
            make_test_tray_piece(slot_idx=1, piece=self.p1),
            make_test_tray_piece(slot_idx=2, piece=self.p2),
        )
        # Frame A has uncertain cells
        frame_a = make_test_frame(bitboard=self.expected_bb, tray_pieces=tray, has_uncertain=True)
        # Frame B also has uncertain cells
        frame_b = make_test_frame(bitboard=self.expected_bb, tray_pieces=tray, has_uncertain=True)

        mock_rec = MagicMock()
        mock_rec.recognize_frame.return_value = frame_a
        mock_rec.config = DEFAULT_VISION_CONFIG

        mock_cap = MagicMock()
        mock_cap.capture_from_adb.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        verifier = ActionVerifier(
            recognizer=mock_rec,
            capture=mock_cap,
            normal_settle_sec=0.0,
            consecutive_frame_delay_sec=0.0,
            retry_interval_sec=0.0,
            max_settle_retries=1,
        )

        res = verifier.verify_move(self.step_res)

        self.assertFalse(res.is_success)
        self.assertFalse(res.is_fast_path)
        self.assertEqual(res.status, VerificationStatus.UNKNOWN)
        self.assertIsNotNone(res.fast_path_rejection_reason)
        self.assertTrue(
            "LOW_CONFIDENCE" in res.fast_path_rejection_reason
            or "UNCERTAIN" in res.fast_path_rejection_reason
        )

    # -------------------------------------------------------------------------
    # Scenario 4: Tray does not match expected tray -> Fallback
    # -------------------------------------------------------------------------
    def test_scenario_4_tray_mismatch_skips_fast_path(self):
        """Tray in Frame A having wrong piece in slot 1 rejects fast-path."""
        # Slot 1 has dot_1x1 instead of line_1x2_h
        tray = (
            make_test_tray_piece(slot_idx=0, is_empty=True),
            make_test_tray_piece(slot_idx=1, piece=self.p0),  # Mismatched piece!
            make_test_tray_piece(slot_idx=2, piece=self.p2),
        )
        frame_a = make_test_frame(bitboard=self.expected_bb, tray_pieces=tray)
        frame_b = make_test_frame(bitboard=self.expected_bb, tray_pieces=tray)

        mock_rec = MagicMock()
        mock_rec.recognize_frame.return_value = frame_a
        mock_rec.config = DEFAULT_VISION_CONFIG

        mock_cap = MagicMock()
        mock_cap.capture_from_adb.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        verifier = ActionVerifier(
            recognizer=mock_rec,
            capture=mock_cap,
            normal_settle_sec=0.0,
            consecutive_frame_delay_sec=0.0,
            max_settle_retries=1,
        )

        res = verifier.verify_move(self.step_res)

        # Bitboard matched, but fast-path was skipped due to tray piece mismatch
        self.assertFalse(res.is_fast_path)
        self.assertEqual(res.captures_count, 2)
        self.assertIsNotNone(res.fast_path_rejection_reason)
        self.assertIn("Tray slot 1 piece mismatch", res.fast_path_rejection_reason)

    # -------------------------------------------------------------------------
    # Scenario 5: Dragged slot not empty in Frame A -> Fallback
    # -------------------------------------------------------------------------
    def test_scenario_5_dragged_slot_not_empty_skips_fast_path(self):
        """If dragged slot 0 is not empty in Frame A, fast-path is rejected."""
        # Slot 0 still contains dot_1x1 (drag dropped back)
        tray = (
            make_test_tray_piece(slot_idx=0, piece=self.p0),  # Not empty!
            make_test_tray_piece(slot_idx=1, piece=self.p1),
            make_test_tray_piece(slot_idx=2, piece=self.p2),
        )
        frame_a = make_test_frame(bitboard=0x0, tray_pieces=tray)
        frame_b = make_test_frame(bitboard=0x0, tray_pieces=tray)

        mock_rec = MagicMock()
        mock_rec.recognize_frame.return_value = frame_a
        mock_rec.config = DEFAULT_VISION_CONFIG

        mock_cap = MagicMock()
        mock_cap.capture_from_adb.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        verifier = ActionVerifier(
            recognizer=mock_rec,
            capture=mock_cap,
            normal_settle_sec=0.0,
            consecutive_frame_delay_sec=0.0,
            max_settle_retries=1,
        )

        res = verifier.verify_move(self.step_res)

        self.assertFalse(res.is_success)
        self.assertFalse(res.is_fast_path)
        self.assertEqual(res.status, VerificationStatus.TRAY_REJECTED)
        self.assertIsNotNone(res.fast_path_rejection_reason)

    # -------------------------------------------------------------------------
    # Scenario 6: Move with line clear (lines_cleared > 0) -> Must Use Two-Frame
    # -------------------------------------------------------------------------
    def test_scenario_6_line_clear_skips_fast_path(self):
        """Move that clears lines must NOT use fast-path; requires two-frame verification."""
        # Create a step_res with lines_cleared = 1
        mock_step_res = MagicMock(spec=StepResult)
        mock_step_res.state_after.board = self.expected_bb
        mock_step_res.state_after.tray = (None, self.p1, self.p2)
        mock_step_res.slot_idx = 0
        mock_step_res.piece = self.p0
        mock_step_res.lines_cleared = 1

        frame_a = self._create_matching_frame_a()
        frame_b = self._create_matching_frame_a()

        mock_rec = MagicMock()
        mock_rec.recognize_frame.side_effect = [frame_a, frame_b]
        mock_rec.config = DEFAULT_VISION_CONFIG

        mock_cap = MagicMock()
        mock_cap.capture_from_adb.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        verifier = ActionVerifier(
            recognizer=mock_rec,
            capture=mock_cap,
            normal_settle_sec=0.0,
            line_clear_settle_sec=0.0,
            consecutive_frame_delay_sec=0.0,
            max_settle_retries=1,
        )

        res = verifier.verify_move(mock_step_res)

        self.assertTrue(res.is_success)
        self.assertEqual(res.status, VerificationStatus.VERIFIED)
        self.assertFalse(res.is_fast_path)
        self.assertEqual(res.captures_count, 2)
        self.assertIsNotNone(res.fast_path_rejection_reason)
        self.assertIn("particle settling requires two-frame verification", res.fast_path_rejection_reason)

    # -------------------------------------------------------------------------
    # Scenario 7: Frame A capture / vision failure -> Fallback to VISION_FAILED
    # -------------------------------------------------------------------------
    def test_scenario_7_frame_a_capture_failure_halts_safely(self):
        """ADB capture failure on Frame A returns VISION_FAILED immediately."""
        mock_rec = MagicMock()
        mock_cap = MagicMock()
        mock_cap.capture_from_adb.side_effect = RuntimeError("ADB transport error")

        verifier = ActionVerifier(
            recognizer=mock_rec,
            capture=mock_cap,
            normal_settle_sec=0.0,
            max_settle_retries=1,
        )

        res = verifier.verify_move(self.step_res)

        self.assertFalse(res.is_success)
        self.assertEqual(res.status, VerificationStatus.VISION_FAILED)
        self.assertFalse(res.is_fast_path)
        self.assertIn("ADB transport error", res.error_message)

    # -------------------------------------------------------------------------
    # Scenario 8: Frame B unstable -> Retry & UNSTABLE_ANIMATION_TIMEOUT
    # -------------------------------------------------------------------------
    def test_scenario_8_consecutive_inconsistency_triggers_unstable_timeout(self):
        """Persistent consecutive frame inconsistency returns UNSTABLE_ANIMATION_TIMEOUT."""
        # Bitboard differs between A and B, so fast-path skipped and consecutive check fails
        tray = (
            make_test_tray_piece(slot_idx=0, is_empty=True),
            make_test_tray_piece(slot_idx=1, piece=self.p1),
            make_test_tray_piece(slot_idx=2, piece=self.p2),
        )
        frame_a = make_test_frame(bitboard=self.expected_bb | 0x10, tray_pieces=tray)
        frame_b = make_test_frame(bitboard=self.expected_bb | 0x20, tray_pieces=tray)

        mock_rec = MagicMock()
        mock_rec.recognize_frame.side_effect = [frame_a, frame_b, frame_a, frame_b]
        mock_rec.config = DEFAULT_VISION_CONFIG

        mock_cap = MagicMock()
        mock_cap.capture_from_adb.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        verifier = ActionVerifier(
            recognizer=mock_rec,
            capture=mock_cap,
            normal_settle_sec=0.0,
            consecutive_frame_delay_sec=0.0,
            retry_interval_sec=0.0,
            max_settle_retries=2,
        )

        res = verifier.verify_move(self.step_res)

        self.assertFalse(res.is_success)
        self.assertEqual(res.status, VerificationStatus.UNSTABLE_ANIMATION_TIMEOUT)
        self.assertFalse(res.is_fast_path)
        self.assertEqual(res.attempts, 2)

    # -------------------------------------------------------------------------
    # Scenario 9: Timeout ceiling reached -> Never returns VERIFIED
    # -------------------------------------------------------------------------
    def test_scenario_9_timeout_ceiling_aborts_retries(self):
        """When settle_timeout_sec is exceeded on retry, verification aborts without VERIFIED."""
        tray = (
            make_test_tray_piece(slot_idx=0, is_empty=True),
            make_test_tray_piece(slot_idx=1, piece=self.p1),
            make_test_tray_piece(slot_idx=2, piece=self.p2),
        )
        frame_mismatch = make_test_frame(bitboard=self.expected_bb | 0xFF, tray_pieces=tray)

        mock_rec = MagicMock()
        mock_rec.recognize_frame.return_value = frame_mismatch
        mock_rec.config = DEFAULT_VISION_CONFIG

        mock_cap = MagicMock()
        mock_cap.capture_from_adb.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        verifier = ActionVerifier(
            recognizer=mock_rec,
            capture=mock_cap,
            normal_settle_sec=0.0,
            consecutive_frame_delay_sec=0.0,
            retry_interval_sec=0.0,
            settle_timeout_sec=0.001,  # Ultra-short timeout
            max_settle_retries=5,
        )

        # Mock time.perf_counter to simulate timeout after attempt 1
        real_perf_counter = time.perf_counter
        call_times = [0.0, 0.0, 0.0, 0.0, 10.0]  # Jump to 10s on second attempt check
        def mock_time():
            if call_times:
                return call_times.pop(0)
            return 10.0

        with patch("time.perf_counter", side_effect=mock_time):
            res = verifier.verify_move(self.step_res)

        self.assertFalse(res.is_success)
        self.assertNotEqual(res.status, VerificationStatus.VERIFIED)
        self.assertEqual(res.status, VerificationStatus.MISMATCH)

    # -------------------------------------------------------------------------
    # Scenario 10: PAUSE / STOP behavior preserved
    # -------------------------------------------------------------------------
    def test_scenario_10_pause_stop_safety_preserved(self):
        """Verification failure statuses (MISMATCH, TRAY_REJECTED, UNKNOWN) are strictly preserved."""
        # Case A: Tray rejected
        tray_rejected = (
            make_test_tray_piece(slot_idx=0, piece=self.p0),  # Returned to slot 0
            make_test_tray_piece(slot_idx=1, piece=self.p1),
            make_test_tray_piece(slot_idx=2, piece=self.p2),
        )
        frame_tr = make_test_frame(bitboard=0x0, tray_pieces=tray_rejected)

        mock_rec = MagicMock()
        mock_rec.recognize_frame.return_value = frame_tr
        mock_rec.config = DEFAULT_VISION_CONFIG
        mock_cap = MagicMock()
        mock_cap.capture_from_adb.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        verifier = ActionVerifier(
            recognizer=mock_rec,
            capture=mock_cap,
            normal_settle_sec=0.0,
            consecutive_frame_delay_sec=0.0,
            max_settle_retries=1,
        )

        res = verifier.verify_move(self.step_res)
        self.assertEqual(res.status, VerificationStatus.TRAY_REJECTED)
        self.assertFalse(res.is_success)
        self.assertFalse(res.tray_slot_cleared)

        # Case B: Status UNKNOWN on vision LOW_CONFIDENCE
        frame_low = FrameRecognitionResult(
            status=RecognitionStatus.LOW_CONFIDENCE,
            board=None,
            tray_pieces=(),
            overall_confidence=0.40,
            processing_time_ms=10.0,
            error_message="Sub-threshold confidence",
        )
        mock_rec.recognize_frame.return_value = frame_low
        res_low = verifier.verify_move(self.step_res)
        self.assertEqual(res_low.status, VerificationStatus.UNKNOWN)
        self.assertFalse(res_low.is_success)


class TestFastPathConfidenceValidation(unittest.TestCase):
    """Specific tests for Phase 1.1: Per-Cell and Tray Slot Confidence Validation."""

    def setUp(self):
        self.p0 = CANONICAL_41_CATALOG.get_by_name("dot_1x1")
        self.p1 = CANONICAL_41_CATALOG.get_by_name("line_1x2_h")
        self.p2 = CANONICAL_41_CATALOG.get_by_name("square_2x2")
        self.sim = GameSimulator()

        self.init_state = GameState.create_initial(initial_tray=[self.p0, self.p1, self.p2])
        self.step_res = self.sim.apply_move(self.init_state, slot_idx=0, row=0, col=0)
        self.expected_bb = self.step_res.state_after.board

        self.verifier = ActionVerifier(
            normal_settle_sec=0.0,
            consecutive_frame_delay_sec=0.0,
            max_settle_retries=1,
        )

    def _build_board_with_cell_confs(
        self,
        default_conf: float = 0.98,
        confs_map: dict[tuple[int, int], Any] | None = None,
    ) -> BoardRecognitionResult:
        """Construct an 8x8 BoardRecognitionResult with customizable per-cell confidences."""
        confs_map = confs_map or {}
        cells = []
        matrix = []
        total_conf = 0.0
        for r in range(8):
            row_cells = []
            row_mat = []
            for c in range(8):
                is_occ = bool(self.expected_bb & coord_to_bit(r, c))
                conf = confs_map.get((r, c), default_conf)
                total_conf += conf if (isinstance(conf, (int, float)) and math.isfinite(conf)) else 0.0
                row_cells.append(
                    CellInfo(
                        row=r,
                        col=c,
                        state=CellState.OCCUPIED if is_occ else CellState.EMPTY,
                        confidence=conf,
                        center_x=int(54 + (c + 0.5) * 121.5),
                        center_y=int(548 + (r + 0.5) * 121.5),
                        width=121,
                        height=121,
                        mean_gray=180.0 if is_occ else 84.0,
                        mean_sat=100.0 if is_occ else 20.0,
                    )
                )
                row_mat.append(1 if is_occ else 0)
            cells.append(tuple(row_cells))
            matrix.append(tuple(row_mat))

        return BoardRecognitionResult(
            board_bbox=(54, 548, 972, 972),
            grid_rows=8,
            grid_cols=8,
            cell_size=121.5,
            cells=tuple(cells),
            occupancy_matrix=tuple(matrix),
            bitboard=self.expected_bb,
            board_confidence=0.98,
            grid_confidence=0.98,
            cell_confidence=total_conf / 64.0,
        )

    def _build_matching_frame(
        self,
        board: BoardRecognitionResult | None = None,
        tray: tuple[PieceRecognitionResult, ...] | None = None,
    ) -> FrameRecognitionResult:
        """Construct full FrameRecognitionResult matching expected step_res."""
        if board is None:
            board = self._build_board_with_cell_confs(default_conf=0.98)
        if tray is None:
            tray = (
                make_test_tray_piece(slot_idx=0, is_empty=True, confidence=0.95),
                make_test_tray_piece(slot_idx=1, piece=self.p1, confidence=0.95),
                make_test_tray_piece(slot_idx=2, piece=self.p2, confidence=0.95),
            )
        return FrameRecognitionResult(
            status=RecognitionStatus.SUCCESS,
            board=board,
            tray_pieces=tray,
            overall_confidence=0.98,
            processing_time_ms=15.0,
            image_shape=(2340, 1080),
        )

    # 1. 64 cell đều confidence 0.70: được phép qua điều kiện confidence
    def test_1_all_64_cells_confidence_0_70_passes(self):
        board = self._build_board_with_cell_confs(default_conf=0.70)
        frame = self._build_matching_frame(board=board)
        can_fp, reason = self.verifier._can_fast_path_verify(frame, self.step_res, lines_cleared=0)
        self.assertTrue(can_fp, f"Expected fast-path with 64 cells @ 0.70, but got: {reason}")

    # 2. Một cell confidence 0.69: từ chối
    def test_2_one_cell_confidence_0_69_rejected(self):
        board = self._build_board_with_cell_confs(default_conf=0.80, confs_map={(3, 4): 0.69})
        frame = self._build_matching_frame(board=board)
        can_fp, reason = self.verifier._can_fast_path_verify(frame, self.step_res, lines_cleared=0)
        self.assertFalse(can_fp)
        self.assertIn("0.69", reason)
        self.assertIn("(3, 4)", reason)

    # 3. Một cell confidence 0.65, 63 cell còn lại 0.98: từ chối
    def test_3_one_cell_0_65_and_63_cells_0_98_rejected(self):
        board = self._build_board_with_cell_confs(default_conf=0.98, confs_map={(1, 2): 0.65})
        # Note: average cell confidence is still high (~0.975 >= 0.70)
        self.assertGreater(board.cell_confidence, 0.90)
        frame = self._build_matching_frame(board=board)
        can_fp, reason = self.verifier._can_fast_path_verify(frame, self.step_res, lines_cleared=0)
        self.assertFalse(can_fp)
        self.assertIn("0.65", reason)
        self.assertIn("(1, 2)", reason)

    # 4. Một cell thiếu confidence: từ chối
    def test_4_one_cell_missing_confidence_rejected(self):
        board = self._build_board_with_cell_confs(default_conf=0.98, confs_map={(0, 0): None})
        frame = self._build_matching_frame(board=board)
        can_fp, reason = self.verifier._can_fast_path_verify(frame, self.step_res, lines_cleared=0)
        self.assertFalse(can_fp)
        self.assertIn("missing or invalid confidence", reason)

    # 5. Một cell confidence NaN: từ chối
    def test_5_one_cell_nan_confidence_rejected(self):
        board = self._build_board_with_cell_confs(default_conf=0.98, confs_map={(2, 2): float("nan")})
        frame = self._build_matching_frame(board=board)
        can_fp, reason = self.verifier._can_fast_path_verify(frame, self.step_res, lines_cleared=0)
        self.assertFalse(can_fp)
        self.assertIn("missing or invalid confidence", reason)

    # 6. Slot rỗng confidence 0.95: được phép qua điều kiện confidence
    def test_6_empty_slot_confidence_0_95_passes(self):
        tray = (
            make_test_tray_piece(slot_idx=0, is_empty=True, confidence=0.95),
            make_test_tray_piece(slot_idx=1, piece=self.p1, confidence=0.95),
            make_test_tray_piece(slot_idx=2, piece=self.p2, confidence=0.95),
        )
        frame = self._build_matching_frame(tray=tray)
        can_fp, reason = self.verifier._can_fast_path_verify(frame, self.step_res, lines_cleared=0)
        self.assertTrue(can_fp, f"Expected fast-path with empty slot @ 0.95, but got: {reason}")

    # 7. Slot rỗng confidence 0.60: từ chối
    def test_7_empty_slot_confidence_0_60_rejected(self):
        tray = (
            make_test_tray_piece(slot_idx=0, is_empty=True, confidence=0.60),  # Debris / contour < 20px
            make_test_tray_piece(slot_idx=1, piece=self.p1, confidence=0.95),
            make_test_tray_piece(slot_idx=2, piece=self.p2, confidence=0.95),
        )
        frame = self._build_matching_frame(tray=tray)
        can_fp, reason = self.verifier._can_fast_path_verify(frame, self.step_res, lines_cleared=0)
        self.assertFalse(can_fp)
        self.assertIn("0.60", reason)
        self.assertIn("Tray slot 0 confidence", reason)

    # 8. Slot rỗng thiếu confidence: từ chối
    def test_8_empty_slot_missing_confidence_rejected(self):
        tray = (
            make_test_tray_piece(slot_idx=0, is_empty=True, confidence=None),
            make_test_tray_piece(slot_idx=1, piece=self.p1, confidence=0.95),
            make_test_tray_piece(slot_idx=2, piece=self.p2, confidence=0.95),
        )
        frame = self._build_matching_frame(tray=tray)
        can_fp, reason = self.verifier._can_fast_path_verify(frame, self.step_res, lines_cleared=0)
        self.assertFalse(can_fp)
        self.assertIn("missing or invalid confidence", reason)

    # 9. Slot có confidence NaN: từ chối
    def test_9_slot_nan_confidence_rejected(self):
        tray = (
            make_test_tray_piece(slot_idx=0, is_empty=True, confidence=0.95),
            make_test_tray_piece(slot_idx=1, piece=self.p1, confidence=float("nan")),
            make_test_tray_piece(slot_idx=2, piece=self.p2, confidence=0.95),
        )
        frame = self._build_matching_frame(tray=tray)
        can_fp, reason = self.verifier._can_fast_path_verify(frame, self.step_res, lines_cleared=0)
        self.assertFalse(can_fp)
        self.assertIn("missing or invalid confidence", reason)

    # 10. Tray sai thứ tự: từ chối
    def test_10_tray_wrong_order_rejected(self):
        # Expected after move: Slot 0 empty, Slot 1 = p1 (line_1x2_h), Slot 2 = p2 (square_2x2)
        # Actual frame has pieces swapped: Slot 1 has p2, Slot 2 has p1
        tray = (
            make_test_tray_piece(slot_idx=0, is_empty=True, confidence=0.95),
            make_test_tray_piece(slot_idx=1, piece=self.p2, confidence=0.95),  # p2 instead of p1!
            make_test_tray_piece(slot_idx=2, piece=self.p1, confidence=0.95),  # p1 instead of p2!
        )
        frame = self._build_matching_frame(tray=tray)
        can_fp, reason = self.verifier._can_fast_path_verify(frame, self.step_res, lines_cleared=0)
        self.assertFalse(can_fp)
        self.assertIn("piece mismatch", reason)

    # 11. Production flow: 64 cells @ 0.70 verified via Fast-Path
    def test_11_production_flow_fast_path_verified(self):
        board = self._build_board_with_cell_confs(default_conf=0.70)
        frame = self._build_matching_frame(board=board)
        mock_rec = MagicMock()
        mock_rec.recognize_frame.return_value = frame
        mock_rec.config = DEFAULT_VISION_CONFIG
        mock_cap = MagicMock()
        mock_cap.capture_from_adb.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        v = ActionVerifier(recognizer=mock_rec, capture=mock_cap, normal_settle_sec=0.0)
        res = v.verify_move(self.step_res)
        self.assertTrue(res.is_success)
        self.assertTrue(res.is_fast_path)
        self.assertEqual(res.captures_count, 1)

    # 12. Production flow: Empty slot 0.60 rejects Fast-Path, falls back to two-frame
    def test_12_production_flow_empty_slot_0_60_falls_back_to_two_frame(self):
        tray_0_60 = (
            make_test_tray_piece(slot_idx=0, is_empty=True, confidence=0.60),
            make_test_tray_piece(slot_idx=1, piece=self.p1, confidence=0.95),
            make_test_tray_piece(slot_idx=2, piece=self.p2, confidence=0.95),
        )
        frame = self._build_matching_frame(tray=tray_0_60)
        mock_rec = MagicMock()
        mock_rec.recognize_frame.return_value = frame
        mock_rec.config = DEFAULT_VISION_CONFIG
        mock_cap = MagicMock()
        mock_cap.capture_from_adb.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        v = ActionVerifier(recognizer=mock_rec, capture=mock_cap, normal_settle_sec=0.0, consecutive_frame_delay_sec=0.0)
        res = v.verify_move(self.step_res)
        self.assertFalse(res.is_fast_path)
        self.assertEqual(res.captures_count, 2)
        self.assertIn("0.60", res.fast_path_rejection_reason)


if __name__ == "__main__":
    unittest.main()
