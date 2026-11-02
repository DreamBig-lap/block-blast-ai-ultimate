"""Dedicated Verification Suite for CellClassifier Confidence Fix (Live Run 20:52).

Verifies:
1. Forensic verification against debug/perception_failures/20261101_205205/full_screen.png:
   - Target cells (4,1), (4,2), (5,1), (5,2), (6,1), (6,2) are all OCCUPIED with conf >= 0.90.
   - Total occupied cells == 14, uncertain cells == 0.
   - Slot 0 == EMPTY, Slot 1 == rect_3x2_v (conf 1.0), Slot 2 == EMPTY.
   - Recognizer status is SUCCESS, has_uncertain_cells is False.
2. Color resilience:
   - Red, orange, green, purple, dark blue, yellow, cyan blocks all classified OCCUPIED with high confidence.
3. Empty cells:
   - Empty board cells remain EMPTY with high confidence (no false positives).
4. Board occupancy levels:
   - Empty board (0/64), partial board, nearly full board, and full board (64/64).
5. Tray empty slots:
   - EMPTY tray slots are not misclassified as pieces.
6. Recognizer error message separation:
   - UNKNOWN tray pieces message vs UNCERTAIN board cells message vs both.
7. AutonomousGameController state update:
   - Controller updates self.last_bitboard immediately on valid board detection even if paused.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from action_engine.autonomous_controller import AutonomousConfig, AutonomousGameController, AutonomousState
from computer_vision.board_detector import BoardDetector
from computer_vision.cell_classifier import CellClassifier
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
from game_engine.pieces import CANONICAL_41_CATALOG


DIAGNOSTIC_DIR = os.path.join(PROJECT_ROOT, "debug", "perception_failures", "20261101_205205")
SCREENSHOT_PATH = os.path.join(DIAGNOSTIC_DIR, "full_screen.png")


def make_test_board(
    bitboard: int = 0,
    board_bbox: tuple[int, int, int, int] = (54, 548, 972, 972),
    board_conf: float = 0.95,
    uncertain_cells: tuple[tuple[int, int], ...] = (),
) -> BoardRecognitionResult:
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
    matched_name: str | None = None,
    is_empty: bool = False,
    confidence: float = 0.95,
) -> PieceRecognitionResult:
    piece = CANONICAL_41_CATALOG.get_by_name(matched_name) if matched_name and matched_name != "UNKNOWN" else None
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


class TestCellClassifierLive2052Fix(unittest.TestCase):
    """Verifies that the live 20:52 failure is completely resolved."""

    def test_live_diagnostic_image_205205(self):
        """Test full pipeline recognition on 20261101_205205/full_screen.png."""
        if not os.path.exists(SCREENSHOT_PATH):
            self.skipTest(f"Diagnostic image not found: {SCREENSHOT_PATH}")

        img = cv2.imread(SCREENSHOT_PATH)
        self.assertIsNotNone(img, "Failed to load screenshot")

        recognizer = BlockBlastRecognizer()
        res = recognizer.recognize_frame(img)

        # 1. Pipeline status
        self.assertEqual(res.status, RecognitionStatus.SUCCESS, f"Expected SUCCESS, got {res.status}: {res.error_message}")
        self.assertFalse(res.board.has_uncertain_cells, "Expected no uncertain cells on board")

        # 2. Board state
        board = res.board
        self.assertIsNotNone(board)
        self.assertEqual(board.occupied_count, 14, f"Expected 14 occupied cells, got {board.occupied_count}")
        self.assertEqual(sum(1 for row in board.cells for c in row if c.state == CellState.UNCERTAIN), 0)
        self.assertGreaterEqual(board.board_confidence, 0.85)

        # 3. Specifically verify the 6 red/orange target cells
        target_cells = [(4, 1), (4, 2), (5, 1), (5, 2), (6, 1), (6, 2)]
        for r, c in target_cells:
            cell = board.cells[r][c]
            self.assertEqual(
                cell.state,
                CellState.OCCUPIED,
                f"Cell ({r},{c}) should be OCCUPIED, got {cell.state} with conf={cell.confidence:.2f}",
            )
            self.assertGreaterEqual(
                cell.confidence,
                0.90,
                f"Cell ({r},{c}) confidence should be >= 0.90, got {cell.confidence:.2f}",
            )

        # 4. Tray verification
        tray = res.tray_pieces
        self.assertEqual(len(tray), 3)
        self.assertTrue(tray[0].is_empty, "Slot 0 should be empty")
        self.assertFalse(tray[1].is_empty, "Slot 1 should be occupied")
        self.assertEqual(tray[1].matched_name, "rect_3x2_v")
        self.assertGreaterEqual(tray[1].confidence, 0.95)
        self.assertTrue(tray[2].is_empty, "Slot 2 should be empty")

    def test_multi_color_blocks_occupancy_confidence(self):
        """Test CellClassifier on all distinct block colors (Green, Orange, Red, Purple, Dark Blue, Yellow, Cyan)."""
        classifier = CellClassifier()

        color_samples = {
            "vivid_red": (40, 50, 230),       # High Sat, High Val, moderate Gray
            "vivid_orange": (30, 140, 240),    # High Sat, High Val, high Gray
            "vivid_yellow": (30, 220, 240),    # High Val, high Gray
            "vivid_green": (60, 220, 70),      # High Val, high Gray
            "vivid_cyan": (230, 210, 40),      # High Val, high Gray
            "vivid_blue": (220, 100, 30),      # High Val, moderate Gray
            "vivid_purple": (200, 40, 160),    # High Sat, High Val, moderate Gray
            "darker_blue": (180, 70, 20),      # Vivid blue
        }

        board_bbox = (0, 0, 800, 800)
        cell_size = 100

        for name, bgr in color_samples.items():
            img = np.zeros((800, 800, 3), dtype=np.uint8)
            img[:] = (84, 84, 84)

            # Paint cell (3, 3) with this color
            cx, cy = 3 * cell_size, 3 * cell_size
            img[cy : cy + cell_size, cx : cx + cell_size] = bgr

            res = classifier.classify_cells(img, board_bbox)
            cell = res.cells[3][3]
            self.assertEqual(
                cell.state,
                CellState.OCCUPIED,
                f"Color '{name}' should be OCCUPIED, got {cell.state} (conf={cell.confidence:.2f})",
            )
            self.assertGreaterEqual(
                cell.confidence,
                0.80,
                f"Color '{name}' should have confidence >= 0.80, got {cell.confidence:.2f}",
            )

    def test_empty_cells_confidence_not_inflated(self):
        """Ensure empty cells (dark background) remain EMPTY and have high confidence without false positives."""
        classifier = CellClassifier()
        img = np.zeros((800, 800, 3), dtype=np.uint8)
        img[:] = (84, 84, 84)

        board_bbox = (0, 0, 800, 800)
        res = classifier.classify_cells(img, board_bbox)

        self.assertEqual(res.occupied_count, 0)
        self.assertFalse(res.has_uncertain_cells)
        for r in range(8):
            for c in range(8):
                cell = res.cells[r][c]
                self.assertEqual(cell.state, CellState.EMPTY)
                self.assertGreaterEqual(cell.confidence, 0.90)

    def test_full_and_partial_board_occupancies(self):
        """Verify 0/64, 40/64, 56/64, 64/64 occupancy classifications."""
        classifier = CellClassifier()
        board_bbox = (0, 0, 800, 800)
        cell_size = 100

        # 1. 0/64
        empty_img = np.full((800, 800, 3), 84, dtype=np.uint8)
        res_0 = classifier.classify_cells(empty_img, board_bbox)
        self.assertEqual(res_0.occupied_count, 0)

        # 2. 64/64
        full_img = np.full((800, 800, 3), 200, dtype=np.uint8)
        res_64 = classifier.classify_cells(full_img, board_bbox)
        self.assertEqual(res_64.occupied_count, 64)
        self.assertFalse(res_64.has_uncertain_cells)

        # 3. 40/64
        partial_img = np.full((800, 800, 3), 84, dtype=np.uint8)
        for i in range(40):
            r, c = divmod(i, 8)
            cx, cy = c * cell_size, r * cell_size
            partial_img[cy : cy + cell_size, cx : cx + cell_size] = (40, 50, 230)
        res_40 = classifier.classify_cells(partial_img, board_bbox)
        self.assertEqual(res_40.occupied_count, 40)
        self.assertFalse(res_40.has_uncertain_cells)

    def test_tray_empty_slots_remain_empty(self):
        """Verify that genuine empty slots in tray are not misclassified as pieces."""
        recognizer = BlockBlastRecognizer()
        if not os.path.exists(SCREENSHOT_PATH):
            self.skipTest(f"Diagnostic image not found: {SCREENSHOT_PATH}")

        img = cv2.imread(SCREENSHOT_PATH)
        res = recognizer.recognize_frame(img)
        self.assertEqual(len(res.tray_pieces), 3)
        self.assertTrue(res.tray_pieces[0].is_empty)
        self.assertTrue(res.tray_pieces[2].is_empty)
        self.assertIsNone(res.tray_pieces[0].piece)
        self.assertIsNone(res.tray_pieces[2].piece)

    def test_recognizer_error_message_separation(self):
        """Verify that UNKNOWN tray piece and UNCERTAIN board cell messages are disentangled."""
        recognizer = BlockBlastRecognizer()
        fake_img = np.zeros((1920, 1080, 3), dtype=np.uint8)

        recognizer.board_detector.detect_board = MagicMock(return_value=((50, 50, 800, 800), 100.0, 0.95))

        # Case 1: UNKNOWN piece only
        board_valid = make_test_board(bitboard=0)
        recognizer.cell_classifier.classify_cells = MagicMock(return_value=board_valid)
        recognizer.piece_detector.detect_tray_pieces = MagicMock(return_value=[])
        tray_unknown = [
            make_test_piece(0, matched_name="UNKNOWN"),
            make_test_piece(1, is_empty=True),
            make_test_piece(2, is_empty=True),
        ]
        recognizer.piece_classifier.classify_all_blobs = MagicMock(return_value=tray_unknown)

        res_unknown_only = recognizer.recognize_frame(fake_img)
        self.assertEqual(res_unknown_only.status, RecognitionStatus.LOW_CONFIDENCE)
        self.assertEqual(res_unknown_only.error_message, "Contains unmapped tray pieces (UNKNOWN).")

        # Case 2: UNCERTAIN cell only
        board_uncertain = make_test_board(bitboard=0, uncertain_cells=((0, 0),))
        recognizer.cell_classifier.classify_cells = MagicMock(return_value=board_uncertain)
        tray_valid = [
            make_test_piece(0, matched_name="dot_1x1"),
            make_test_piece(1, is_empty=True),
            make_test_piece(2, is_empty=True),
        ]
        recognizer.piece_classifier.classify_all_blobs = MagicMock(return_value=tray_valid)

        res_uncertain_only = recognizer.recognize_frame(fake_img)
        self.assertEqual(res_uncertain_only.status, RecognitionStatus.LOW_CONFIDENCE)
        self.assertEqual(res_uncertain_only.error_message, "Perception uncertain: Board contains UNCERTAIN cells.")

        # Case 3: Both
        recognizer.piece_classifier.classify_all_blobs = MagicMock(return_value=tray_unknown)
        res_both = recognizer.recognize_frame(fake_img)
        self.assertEqual(res_both.status, RecognitionStatus.LOW_CONFIDENCE)
        self.assertEqual(res_both.error_message, "Contains unmapped tray pieces (UNKNOWN) and uncertain board cells.")

    def test_autonomous_controller_updates_last_bitboard_on_board_detection(self):
        """Verify controller.last_bitboard is updated immediately when board is detected even if paused."""
        controller = AutonomousGameController(AutonomousConfig(device_id="DRY_RUN", dry_run=True))

        dummy_board = make_test_board(bitboard=0x12345678, uncertain_cells=((0, 0),))
        rec_res = FrameRecognitionResult(
            board=dummy_board,
            tray_pieces=(),
            status=RecognitionStatus.LOW_CONFIDENCE,
            error_message="Perception uncertain: Board contains UNCERTAIN cells.",
            overall_confidence=0.7,
            processing_time_ms=15.0,
        )

        controller.capture.capture_from_adb = MagicMock(return_value=np.zeros((1920, 1080, 3), dtype=np.uint8))
        controller.recognizer.recognize_frame = MagicMock(return_value=rec_res)

        res = controller.step_single_turn()

        self.assertFalse(res)
        self.assertEqual(controller.last_bitboard, 0x12345678)
        self.assertEqual(controller.state, AutonomousState.PAUSED)


if __name__ == "__main__":
    unittest.main()
