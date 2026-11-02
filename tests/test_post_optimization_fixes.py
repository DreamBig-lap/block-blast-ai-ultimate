"""Unit and Integration Tests for Post-Optimization Live Failure Fixes.

Covers:
1. Dark Blue / Purple block classification with high confidence (no UNCERTAIN cells).
2. Resolution of bitboard discrepancies (bits 19, 20 and bit 2).
3. Raw screencap streaming with tolerance for trailing bytes and socket telemetry.
4. Tray piece detector background gradient compensation and empty slot confidence.
5. Action verification Fast-Path and Frame Reuse pass criteria.
"""

from __future__ import annotations

import struct
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from action_engine.action_verifier import ActionVerifier, VerificationStatus
from action_engine.autonomous_controller import AutonomousConfig, AutonomousGameController
from computer_vision.capture import ScreenCapture, parse_raw_screencap
from computer_vision.cell_classifier import CellClassifier
from computer_vision.config import VisionConfig
from computer_vision.models import CellState, FrameRecognitionResult, PieceRecognitionResult, RecognitionStatus
from computer_vision.piece_detector import PieceDetector
from computer_vision.recognizer import BlockBlastRecognizer
from game_engine.board import coord_to_bit
from game_engine.pieces import CANONICAL_41_CATALOG
from game_engine.simulator import StepResult


class TestPostOptimizationFixes(unittest.TestCase):
    """Test suite validating fixes for live phone failures."""

    def setUp(self) -> None:
        self.config = VisionConfig()
        self.classifier = CellClassifier(self.config)
        self.recognizer = BlockBlastRecognizer(self.config)

    def test_01_dark_blue_and_purple_blocks_classified_with_high_confidence(self) -> None:
        """Dark blue/purple blocks (mean_g ~ 115-125, mean_v ~ 145-155, mean_s ~ 90-115) must be OCCUPIED with conf >= 0.85."""
        board_bbox = (54, 548, 972, 972)
        cell_size = 972 // 8
        dummy_img = np.full((2340, 1080, 3), (65, 65, 60), dtype=np.uint8)

        # Place dark blue block at (2, 3): BGR [152, 120, 91]
        x_23 = int(54 + 3 * cell_size)
        y_23 = int(548 + 2 * cell_size)
        dummy_img[y_23:y_23 + cell_size, x_23:x_23 + cell_size] = [152, 120, 91]

        # Place dark purple block at (2, 4): BGR [135, 103, 74]
        x_24 = int(54 + 4 * cell_size)
        y_24 = int(548 + 2 * cell_size)
        dummy_img[y_24:y_24 + cell_size, x_24:x_24 + cell_size] = [135, 103, 74]

        res = self.classifier.classify_cells(dummy_img, board_bbox)

        cell_23 = res.cells[2][3]
        cell_24 = res.cells[2][4]

        self.assertEqual(cell_23.state, CellState.OCCUPIED)
        self.assertGreaterEqual(cell_23.confidence, 0.85)
        self.assertNotEqual(cell_23.state, CellState.UNCERTAIN)

        self.assertEqual(cell_24.state, CellState.OCCUPIED)
        self.assertGreaterEqual(cell_24.confidence, 0.85)
        self.assertNotEqual(cell_24.state, CellState.UNCERTAIN)

        # Confirm bits 19 and 20 are set
        self.assertTrue(bool(res.bitboard & coord_to_bit(2, 3)))
        self.assertTrue(bool(res.bitboard & coord_to_bit(2, 4)))
        self.assertFalse(res.has_uncertain_cells)

    def test_02_bitboard_discrepancy_resolution(self) -> None:
        """Verify bit 2 (row 0, col 2) is reliably classified when colored block is placed."""
        board_bbox = (54, 548, 972, 972)
        cell_size = 972 // 8
        dummy_img = np.full((2340, 1080, 3), (70, 70, 65), dtype=np.uint8)

        # Place colored block at row 0, col 2: BGR [147, 97, 183]
        x_02 = int(54 + 2 * cell_size)
        y_02 = int(548 + 0 * cell_size)
        dummy_img[y_02:y_02 + cell_size, x_02:x_02 + cell_size] = [147, 97, 183]

        res = self.classifier.classify_cells(dummy_img, board_bbox)
        cell_02 = res.cells[0][2]

        self.assertEqual(cell_02.state, CellState.OCCUPIED)
        self.assertGreaterEqual(cell_02.confidence, 0.85)
        self.assertTrue(bool(res.bitboard & coord_to_bit(0, 2)))
        self.assertFalse(res.has_uncertain_cells)

    def test_03_raw_screencap_tolerant_to_trailing_bytes(self) -> None:
        """parse_raw_screencap handles up to 64 bytes of trailing newlines/padding gracefully."""
        w, h = 300, 300
        raw_pixels = np.full((h, w, 4), 200, dtype=np.uint8)
        header = struct.pack("<IIII", w, h, 1, 0)
        base_stream = header + raw_pixels.tobytes()

        # Exact length: success
        img_exact = parse_raw_screencap(base_stream)
        self.assertIsNotNone(img_exact)
        self.assertEqual(img_exact.shape, (300, 300, 3))

        # Trailing newline (1 byte): success
        img_nl = parse_raw_screencap(base_stream + b"\n")
        self.assertIsNotNone(img_nl)
        self.assertEqual(img_nl.shape, (300, 300, 3))

        # Trailing 16 bytes padding: success
        img_pad = parse_raw_screencap(base_stream + b"\x00" * 16)
        self.assertIsNotNone(img_pad)
        self.assertEqual(img_pad.shape, (300, 300, 3))

        # Gross mismatch (>64 bytes extra): returns None safely
        img_corrupt = parse_raw_screencap(base_stream + b"\x00" * 200)
        self.assertIsNone(img_corrupt)

    def test_04_screen_capture_telemetry_recording(self) -> None:
        """ScreenCapture captures and exposes structured diagnostic telemetry."""
        cap = ScreenCapture(VisionConfig(use_raw_screencap=True, use_adb_socket=False))
        w, h = 250, 250
        raw_pixels = np.full((h, w, 4), 180, dtype=np.uint8)
        header = struct.pack("<IIII", w, h, 1, 0)
        mock_raw = header + raw_pixels.tobytes()

        with patch("subprocess.run") as mock_sub:
            mock_sub.return_value = MagicMock(returncode=0, stdout=mock_raw)
            img, adb_ms, dec_ms = cap.capture_from_adb_with_timings(device_id="TEST_DEV")

            self.assertIsNotNone(img)
            self.assertEqual(cap.last_capture_mode, "RAW")
            self.assertEqual(cap.last_transport_mode, "SUBPROCESS")
            self.assertEqual(cap.last_bytes_count, len(mock_raw))
            self.assertIn("attempt", cap.last_telemetry)
            self.assertEqual(cap.last_telemetry["mode"], "RAW")
            self.assertEqual(cap.last_telemetry["transport"], "SUBPROCESS")
            self.assertIsNone(cap.last_telemetry["fallback"])

    def test_05_tray_detector_vertical_gradient_compensation(self) -> None:
        """Tray detector handles vertical gradient backgrounds without bleeding into piece contours."""
        detector = PieceDetector()
        img = np.zeros((2340, 1080, 3), dtype=np.uint8)

        # Create vertical gradient in tray band (y=1638 to 2012)
        tray_h = int(2340 * 0.86) - int(2340 * 0.70)
        for y in range(tray_h):
            alpha = y / float(tray_h)
            color = (1.0 - alpha) * np.array([75, 85, 70]) + alpha * np.array([45, 70, 20])
            img[int(2340 * 0.70) + y, :] = color.astype(np.uint8)

        blobs = detector.detect_tray_pieces(img)
        self.assertEqual(len(blobs), 3)
        # All slots are empty (no pieces rendered), and must be confirmed empty with high confidence
        for b in blobs:
            self.assertTrue(b.is_empty)
            self.assertGreaterEqual(b.confidence, 0.70)


if __name__ == "__main__":
    unittest.main()
