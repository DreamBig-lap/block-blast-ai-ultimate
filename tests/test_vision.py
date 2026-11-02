"""Unit and Integration Tests for computer_vision Package.

Verifies:
- Board geometry tiling and coordinate calculations.
- Cell classification with synthetic and real data.
- Piece shape normalization and canonical catalog matching.
- Unknown shape fallback (no guessing).
- Confidence thresholding and rejection of unreliable data.
- Invalid / corrupted screenshot handling.
- Conversion to Game Engine GameState and AI Decision planning.
- Real-world verification on C:\\platform-tools\\blockblast_test.png.
"""

from __future__ import annotations

import os
from pathlib import Path
import unittest

import cv2
import numpy as np

from ai_engine.decision import AIDecisionEngine, DecisionStatus
from computer_vision.board_detector import BoardDetector
from computer_vision.capture import ScreenCapture, ScreenCaptureError
from computer_vision.cell_classifier import CellClassifier
from computer_vision.config import DEFAULT_VISION_CONFIG, VisionConfig
from computer_vision.models import (
    CellState,
    FrameRecognitionResult,
    PieceRecognitionResult,
    RecognitionStatus,
)
from computer_vision.piece_classifier import PieceClassifier
from computer_vision.piece_detector import PieceDetector, RawPieceBlob
from computer_vision.recognizer import BlockBlastRecognizer
from game_engine.board import popcount
from game_engine.pieces import CANONICAL_41_CATALOG


REAL_SCREENSHOT_PATH = r"C:\platform-tools\blockblast_test.png"


class TestBoardGeometry(unittest.TestCase):
    """Test mathematical geometry of board and cell coordinate mapping."""

    def setUp(self) -> None:
        self.detector = BoardDetector()
        self.board_bbox = (50, 500, 960, 960)

    def test_cell_geometry_tiling(self):
        """Cell bounding boxes must tile the entire board without gaps."""
        bx, by, bw, bh = self.board_bbox
        for r in range(8):
            for c in range(8):
                cx, cy, cw, ch = self.detector.get_cell_bounding_box(self.board_bbox, r, c)
                center_x, center_y = self.detector.get_cell_center(self.board_bbox, r, c)

                # Cell box within board
                self.assertGreaterEqual(cx, bx)
                self.assertLess(cx + cw, bx + bw + 2)
                self.assertGreaterEqual(cy, by)
                self.assertLess(cy + ch, by + bh + 2)

                # Center within box
                self.assertGreater(center_x, cx)
                self.assertLess(center_x, cx + cw)
                self.assertGreater(center_y, cy)
                self.assertLess(center_y, cy + ch)


class TestCellClassification(unittest.TestCase):
    """Test classification of empty vs occupied cells."""

    def setUp(self) -> None:
        self.classifier = CellClassifier()

    def test_synthetic_cell_classification(self):
        """Classify a synthetic 8x8 image with a known checkerboard pattern."""
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        # Fill background with ~150
        img[:] = 150

        board_bbox = (100, 100, 800, 800)
        cell_size = 100

        # Draw empty (gray 84) and occupied (gray 190) cells
        expected_occupied: set[tuple[int, int]] = set()
        for r in range(8):
            for c in range(8):
                cx = 100 + c * cell_size
                cy = 100 + r * cell_size
                is_occ = (r + c) % 2 == 1
                color = 190 if is_occ else 84
                img[cy : cy + cell_size, cx : cx + cell_size] = color
                if is_occ:
                    expected_occupied.add((r, c))

        res = self.classifier.classify_cells(img, board_bbox)
        self.assertEqual(res.occupied_count, len(expected_occupied))
        self.assertEqual(res.empty_count, 64 - len(expected_occupied))

        for r in range(8):
            for c in range(8):
                expected_state = 1 if (r, c) in expected_occupied else 0
                self.assertEqual(res.occupancy_matrix[r][c], expected_state)


class TestPieceNormalizationAndMatching(unittest.TestCase):
    """Test discretization of binary masks and mapping to canonical catalog."""

    def setUp(self) -> None:
        self.classifier = PieceClassifier()

    def test_match_canonical_pieces(self):
        """Synthetic masks of 2x2 square and 1x3 line must match 100%."""
        # 1. Square 2x2: 120x120 mask
        mask_2x2 = np.ones((120, 120), dtype=np.uint8) * 255
        blob_2x2 = RawPieceBlob(
            slot_idx=0,
            is_empty=False,
            bbox_in_screen=(100, 1700, 120, 120),
            binary_mask=mask_2x2,
            confidence=0.95,
            message="Test 2x2",
        )
        res_2x2 = self.classifier.classify_blob(blob_2x2)
        self.assertEqual(res_2x2.matched_name, "square_2x2")
        self.assertEqual(res_2x2.height, 2)
        self.assertEqual(res_2x2.width, 2)
        self.assertAlmostEqual(res_2x2.confidence, 1.0, places=2)

        # 2. Horizontal line 1x3: 60x180 mask
        mask_1x3 = np.ones((60, 175), dtype=np.uint8) * 255
        blob_1x3 = RawPieceBlob(
            slot_idx=1,
            is_empty=False,
            bbox_in_screen=(400, 1700, 175, 60),
            binary_mask=mask_1x3,
            confidence=0.95,
            message="Test 1x3",
        )
        res_1x3 = self.classifier.classify_blob(blob_1x3)
        self.assertEqual(res_1x3.matched_name, "line_1x3_h")
        self.assertEqual(res_1x3.height, 1)
        self.assertEqual(res_1x3.width, 3)

    def test_unknown_piece_fallback(self):
        """An irregular non-polyomino or unmapped shape must return UNKNOWN instead of guessing."""
        # A 7-cell H-shape polyomino (height=3, width=3, cell_count=7)
        # There are zero 7-cell pieces in the canonical 41 catalog.
        u = 56
        mask_h_shape = np.zeros((3 * u, 3 * u), dtype=np.uint8)
        # Left col: 3 cells
        mask_h_shape[0 : 3 * u, 0 : u] = 255
        # Middle crossbar: 1 cell at (1, 1)
        mask_h_shape[u : 2 * u, u : 2 * u] = 255
        # Right col: 3 cells
        mask_h_shape[0 : 3 * u, 2 * u : 3 * u] = 255

        blob_unknown = RawPieceBlob(
            slot_idx=2,
            is_empty=False,
            bbox_in_screen=(700, 1700, 3 * u, 3 * u),
            binary_mask=mask_h_shape,
            confidence=0.90,
            message="Test Unknown H-Shape",
        )
        res_unknown = self.classifier.classify_blob(blob_unknown)
        self.assertEqual(res_unknown.matched_name, "UNKNOWN")
        self.assertIsNone(res_unknown.piece)


class TestConfidenceAndErrorHandling(unittest.TestCase):
    """Test confidence thresholds, invalid inputs, and error safety."""

    def setUp(self) -> None:
        self.recognizer = BlockBlastRecognizer()

    def test_invalid_screenshot_handling(self):
        """Corrupted or tiny image raises error cleanly."""
        cap = ScreenCapture()
        with self.assertRaises(ScreenCaptureError):
            cap.load_from_file("non_existent_file_path.png")

        # Blank random noise image
        noise = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        res = self.recognizer.recognize_frame(noise)
        self.assertIn(res.status, (RecognitionStatus.FAILED, RecognitionStatus.LOW_CONFIDENCE))
        self.assertFalse(res.is_usable_for_ai())

    def test_cannot_build_game_state_when_unusable(self):
        """Calling to_game_state() on an unusable result raises ValueError."""
        unusable_result = FrameRecognitionResult(
            status=RecognitionStatus.FAILED,
            board=None,
            tray_pieces=(),
            overall_confidence=0.2,
            processing_time_ms=10.0,
            error_message="Board not found",
        )
        with self.assertRaises(ValueError):
            unusable_result.to_game_state()


class TestRealScreenshotRecognition(unittest.TestCase):
    """End-to-end integration test on real phone screenshot C:\\platform-tools\\blockblast_test.png."""

    @unittest.skipUnless(os.path.isfile(REAL_SCREENSHOT_PATH), "Real screenshot file not present")
    def test_real_screenshot_pipeline_end_to_end(self):
        """Full end-to-end recognition on the authentic phone screenshot."""
        recognizer = BlockBlastRecognizer()
        result = recognizer.recognize_file(REAL_SCREENSHOT_PATH)

        # 1. Pipeline status
        self.assertEqual(result.status, RecognitionStatus.SUCCESS)
        self.assertGreaterEqual(result.overall_confidence, 0.90)
        self.assertLess(result.processing_time_ms, 300.0)  # Must be fast on CPU

        # 2. Board geometry
        self.assertIsNotNone(result.board)
        b = result.board
        self.assertEqual(b.grid_rows, 8)
        self.assertEqual(b.grid_cols, 8)
        self.assertEqual(b.board_bbox, (54, 548, 972, 972))
        self.assertEqual(b.occupied_count, 40)
        self.assertEqual(b.empty_count, 24)

        # 3. Tray pieces
        self.assertEqual(len(result.tray_pieces), 3)

        p0 = result.tray_pieces[0]
        self.assertFalse(p0.is_empty)
        self.assertEqual(p0.matched_name, "line_1x5_h")
        self.assertEqual(p0.cell_count, 5)

        p1 = result.tray_pieces[1]
        self.assertFalse(p1.is_empty)
        self.assertEqual(p1.matched_name, "l_3x2_v4")
        self.assertEqual(p1.cell_count, 4)

        p2 = result.tray_pieces[2]
        self.assertFalse(p2.is_empty)
        self.assertEqual(p2.matched_name, "line_1x4_v")
        self.assertEqual(p2.cell_count, 4)

        # 4. Conversion to Game Engine GameState
        self.assertTrue(result.is_usable_for_ai())
        state = result.to_game_state()
        self.assertEqual(popcount(state.board), 40)
        self.assertEqual(state.active_piece_count(), 3)

        # 5. Plan moves with AI Decision Engine
        ai_engine = AIDecisionEngine()
        decision = ai_engine.decide(state, strategy="beam_search", beam_width=10)
        self.assertEqual(decision.status, DecisionStatus.SUCCESS)
        self.assertEqual(decision.move_count, 3)
        self.assertGreater(decision.predicted_score_delta, 0)
        self.assertGreaterEqual(decision.total_lines_cleared, 1)


class TestColorAwareTraySegmentation(unittest.TestCase):
    """Test suite for color-aware BGR Euclidean distance tray segmentation."""

    def setUp(self) -> None:
        self.detector = PieceDetector()
        self.classifier = PieceClassifier()
        self.recognizer = BlockBlastRecognizer()

    def test_red_piece_solid_segmentation(self) -> None:
        """Verify red pieces (low grayscale contrast) produce solid masks without hollow holes."""
        h, w = 2340, 1080
        # Slate blue tray background
        frame = np.full((h, w, 3), (148, 83, 59), dtype=np.uint8)
        # Draw red 2x2 square in slot 0 (x: 100..210, y: 1720..1830)
        # Red center BGR: [49, 48, 198]
        frame[1720:1830, 100:210] = (49, 48, 198)

        blobs = self.detector.detect_tray_pieces(frame)
        blob_s0 = blobs[0]
        self.assertFalse(blob_s0.is_empty)
        self.assertIsNotNone(blob_s0.binary_mask)

        # Check mask fill ratio: must be a solid body (> 85%), NOT hollow
        mask = blob_s0.binary_mask
        fill = np.count_nonzero(mask) / float(mask.size)
        self.assertGreater(fill, 0.85)

        # Classify: must match square_2x2
        res = self.classifier.classify_blob(blob_s0)
        self.assertEqual(res.matched_name, "square_2x2")
        self.assertGreaterEqual(res.confidence, 0.85)

    def test_empty_tray_background_rejection(self) -> None:
        """Verify clean background noise does not generate false piece contours."""
        h, w = 2340, 1080
        # Slate blue background with slight Gaussian noise (+/- 5 BGR)
        frame = np.full((h, w, 3), (148, 83, 59), dtype=np.uint8)
        noise = np.random.randint(-4, 5, size=(h, w, 3), dtype=np.int16)
        noisy_frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        blobs = self.detector.detect_tray_pieces(noisy_frame)
        for i, b in enumerate(blobs):
            self.assertTrue(b.is_empty, f"Slot {i} should be empty on background noise")
            self.assertGreaterEqual(b.confidence, 0.90)

    def test_all_game_colors_segmented_accurately(self) -> None:
        """Verify red, cyan, orange, yellow, green, purple, and blue are all correctly segmented."""
        colors = {
            "red": (49, 48, 198),
            "cyan": (227, 181, 45),
            "orange": (30, 130, 240),
            "yellow": (30, 210, 240),
            "green": (40, 200, 80),
            "purple": (160, 40, 120),
            "dark_blue": (200, 100, 30),
        }
        h, w = 2340, 1080
        for name, bgr in colors.items():
            frame = np.full((h, w, 3), (148, 83, 59), dtype=np.uint8)
            # Draw 1x3 horizontal line in Slot 1 (w: ~165, h: ~55)
            frame[1750:1805, 450:615] = bgr

            blobs = self.detector.detect_tray_pieces(frame)
            b1 = blobs[1]
            self.assertFalse(b1.is_empty, f"Color {name} failed to segment")
            res = self.classifier.classify_blob(b1)
            self.assertEqual(res.matched_name, "line_1x3_h", f"Color {name} failed classification")
            self.assertGreaterEqual(res.confidence, 0.85)

    def test_re_run_failed_diagnostic_20261101_180516(self) -> None:
        """Verify the exact live failure screenshot from 20261101_180516 is now completely recovered."""
        fail_path = Path(__file__).resolve().parent.parent / "debug" / "perception_failures" / "20261101_180516" / "full_screen.png"
        if not fail_path.exists():
            self.skipTest("Diagnostic capture file not found on disk")

        img = cv2.imread(str(fail_path))
        result = self.recognizer.recognize_frame(img)

        # Full pipeline must succeed
        self.assertEqual(result.status, RecognitionStatus.SUCCESS)
        self.assertTrue(result.is_usable_for_ai())
        self.assertGreater(result.overall_confidence, 0.90)

        # Slot 0: Cyan diagonal_3_tr
        s0 = result.tray_pieces[0]
        self.assertFalse(s0.is_empty)
        self.assertEqual(s0.matched_name, "diagonal_3_tr")
        self.assertEqual(s0.confidence, 1.0)

        # Slot 1: Empty
        s1 = result.tray_pieces[1]
        self.assertTrue(s1.is_empty)

        # Slot 2: Red diagonal_3_tl (previously UNKNOWN)
        s2 = result.tray_pieces[2]
        self.assertFalse(s2.is_empty)
        self.assertEqual(s2.matched_name, "diagonal_3_tl")
        self.assertEqual(s2.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
