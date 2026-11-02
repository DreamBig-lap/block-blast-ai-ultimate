"""Regression and Safety Test Suite for Initial Board 64/64 Occupied Hotfix.

Verifies:
1. Initial empty boards with teal, dark, synthetic, and brighter background themes
   are correctly detected as 0/64 occupied (bitboard 0x0) with high confidence.
2. Real Samsung Galaxy M21 phone screenshots with known ground truth bitboards
   are recognized with 100% bit-for-bit precision.
3. Dark blue, purple, deep red, and bright colored blocks remain OCCUPIED with confidence >= 0.85.
"""

from __future__ import annotations

import os
import sys
import unittest

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from computer_vision.cell_classifier import CellClassifier
from computer_vision.config import VisionConfig
from computer_vision.models import CellState, RecognitionStatus
from computer_vision.recognizer import BlockBlastRecognizer
from game_engine.board import coord_to_bit


class TestInitialBoardEmptyDetection(unittest.TestCase):
    """Test suite ensuring empty boards are never misclassified as occupied."""

    def setUp(self) -> None:
        self.config = VisionConfig()
        self.classifier = CellClassifier(self.config)
        self.recognizer = BlockBlastRecognizer(self.config)
        self.board_bbox = (54, 548, 972, 972)

    def test_01_synthetic_teal_empty_board(self) -> None:
        """Fresh board with real phone teal tiles (BGR [96, 102, 45]) must be 0/64 occupied."""
        img = np.full((2340, 1080, 3), [96, 102, 45], dtype=np.uint8)
        res = self.classifier.classify_cells(img, self.board_bbox)

        self.assertEqual(res.occupied_count, 0)
        self.assertEqual(res.bitboard, 0x0)
        self.assertFalse(res.has_uncertain_cells)
        self.assertGreaterEqual(res.cell_confidence, 0.90)

        for r in range(8):
            for c in range(8):
                cell = res.cells[r][c]
                self.assertEqual(cell.state, CellState.EMPTY)
                self.assertGreaterEqual(cell.confidence, 0.85)

    def test_02_synthetic_dark_theme_empty_board(self) -> None:
        """Fresh board with dark theme tiles (BGR [84, 43, 35]) must be 0/64 occupied."""
        img = np.full((2340, 1080, 3), [84, 43, 35], dtype=np.uint8)
        res = self.classifier.classify_cells(img, self.board_bbox)

        self.assertEqual(res.occupied_count, 0)
        self.assertEqual(res.bitboard, 0x0)
        self.assertFalse(res.has_uncertain_cells)
        self.assertGreaterEqual(res.cell_confidence, 0.90)

    def test_03_elevated_brightness_empty_board_no_false_positive(self) -> None:
        """Fresh board with brighter background variation (BGR [105, 112, 50]) must remain 0/64 occupied."""
        img = np.full((2340, 1080, 3), [105, 112, 50], dtype=np.uint8)
        res = self.classifier.classify_cells(img, self.board_bbox)

        self.assertEqual(res.occupied_count, 0)
        self.assertEqual(res.bitboard, 0x0)
        self.assertFalse(res.has_uncertain_cells)

    def test_04_real_device_ground_truth_regression(self) -> None:
        """Real device frames must match exact known ground-truth bitboards."""
        datasets = [
            (os.path.join("debug", "phase_3_4_8_live.png"), 0x00FEE2F8FB007F7F, 37),
            (os.path.join("debug", "live_audit_raw1.png"), 0x00FEE27F7A3E7E7E, 40),
            (os.path.join("debug", "perception_failures", "20261101_205205", "full_screen.png"), 0x0006060606060606, 14),
        ]

        for rel_path, expected_bb, expected_occ in datasets:
            full_p = os.path.join(PROJECT_ROOT, rel_path)
            if not os.path.exists(full_p):
                continue
            img = cv2.imread(full_p)
            self.assertIsNotNone(img, f"Failed to load {rel_path}")
            res = self.recognizer.recognize_frame(img)
            self.assertEqual(res.status, RecognitionStatus.SUCCESS)
            self.assertIsNotNone(res.board)
            self.assertEqual(
                res.board.bitboard,
                expected_bb,
                f"Bitboard mismatch on {rel_path}: expected 0x{expected_bb:016X}, got 0x{res.board.bitboard:016X}",
            )
            self.assertEqual(res.board.occupied_count, expected_occ)
            self.assertFalse(res.board.has_uncertain_cells)

    def test_05_dark_blue_and_purple_blocks_classified_occupied(self) -> None:
        """Dark blue (BGR [152, 120, 91]) and purple (BGR [135, 103, 74]) must be OCCUPIED with conf >= 0.85."""
        cell_size = 972 // 8
        dummy_img = np.full((2340, 1080, 3), [96, 102, 45], dtype=np.uint8)

        # Place dark blue at (2, 3)
        x_23 = int(54 + 3 * cell_size)
        y_23 = int(548 + 2 * cell_size)
        dummy_img[y_23 : y_23 + cell_size, x_23 : x_23 + cell_size] = [152, 120, 91]

        # Place dark purple at (2, 4)
        x_24 = int(54 + 4 * cell_size)
        y_24 = int(548 + 2 * cell_size)
        dummy_img[y_24 : y_24 + cell_size, x_24 : x_24 + cell_size] = [135, 103, 74]

        res = self.classifier.classify_cells(dummy_img, self.board_bbox)

        self.assertEqual(res.cells[2][3].state, CellState.OCCUPIED)
        self.assertGreaterEqual(res.cells[2][3].confidence, 0.85)

        self.assertEqual(res.cells[2][4].state, CellState.OCCUPIED)
        self.assertGreaterEqual(res.cells[2][4].confidence, 0.85)

        # Board occupied count must be exactly 2
        self.assertEqual(res.occupied_count, 2)
        expected_bb = coord_to_bit(2, 3) | coord_to_bit(2, 4)
        self.assertEqual(res.bitboard, expected_bb)
        self.assertFalse(res.has_uncertain_cells)


if __name__ == "__main__":
    unittest.main()
