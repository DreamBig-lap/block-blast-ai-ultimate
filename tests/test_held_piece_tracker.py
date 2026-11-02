"""Unit Tests for HeldPieceTracker (BLOCK BLAST AI ULTIMATE).

Tests all 10 minimum required test cases offline without device connection:
1. Detection of correct 5-cell horizontal line ('line_1x5_h') on real Phase 3.5.3 artifacts.
2. Centroid calculation (moments centroid vs bbox center).
3. Abnormally large bounding box detection.
4. No contour found in flight region.
5. Multiple competing ambiguous contours.
6. Invalid input images (None, empty, shape mismatch, channel mismatch).
7. Base and hold images identical (zero diff).
8. Color noise and drop shadow separation (saturation gating).
9. Low confidence fallback to UNKNOWN / LOW_CONFIDENCE.
10. Immutability of input image arrays.
11. Multi-polyomino shape support (testing against canonical catalog).
"""

from __future__ import annotations

import os
import sys
import unittest

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np

from action_engine.held_piece_tracker import (
    HeldPieceDetection,
    HeldPieceTracker,
    HeldPieceTrackerConfig,
    TrackingStatus,
)
from game_engine.pieces import CANONICAL_41_CATALOG

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "debug", "phase_3_5_3")


class TestHeldPieceTracker(unittest.TestCase):
    """Test suite for HeldPieceTracker offline vision algorithm."""

    def setUp(self) -> None:
        self.tracker = HeldPieceTracker()
        self.base_path = os.path.join(ARTIFACTS_DIR, "base.png")
        self.mid_path = os.path.join(ARTIFACTS_DIR, "mid_hold.png")

        if os.path.exists(self.base_path) and os.path.exists(self.mid_path):
            self.real_base = cv2.imread(self.base_path)
            self.real_mid = cv2.imread(self.mid_path)
        else:
            self.real_base = None
            self.real_mid = None

    # =========================================================================
    # TEST 1: Real-World 1x5 Horizontal Line Detection (Phase 3.5.3 Artifacts)
    # =========================================================================
    def test_01_detect_line_1x5_h_real_artifacts(self) -> None:
        """Verify successful localization of line_1x5_h on actual Samsung M21 capture."""
        if self.real_base is None or self.real_mid is None:
            self.skipTest("Phase 3.5.3 artifacts not found on disk.")

        detection = self.tracker.track_held_piece(
            base_image=self.real_base,
            hold_image=self.real_mid,
            expected_piece_name="line_1x5_h",
            cell_size_px=121.5,
        )

        self.assertEqual(detection.status, TrackingStatus.SUCCESS)
        self.assertTrue(detection.is_success)
        self.assertEqual(detection.piece_name, "line_1x5_h")
        self.assertIsNotNone(detection.bbox)
        self.assertIsNotNone(detection.centroid)

        bx, by, bw, bh = detection.bbox
        self.assertGreaterEqual(bw, 550, "Width should be close to 607.5 px")
        self.assertLessEqual(bw, 700)
        self.assertGreaterEqual(detection.aspect_ratio, 3.0, "Aspect ratio must be strongly horizontal")
        self.assertGreaterEqual(detection.confidence, 0.70, "Confidence should exceed 70%")
        self.assertGreaterEqual(detection.mean_saturation, 100.0, "Real piece must have high saturation")
        self.assertLess(detection.elapsed_ms, 80.0, "Inference latency should be under 80ms")

    # =========================================================================
    # TEST 2: Centroid Calculation (Moments Centroid vs BBox Center)
    # =========================================================================
    def test_02_centroid_calculation_moments(self) -> None:
        """Verify spatial moments centroid calculation accurately tracks mass center."""
        # Create synthetic base and hold images
        h, w = 1000, 1000
        base = np.zeros((h, w, 3), dtype=np.uint8)
        hold = np.zeros((h, w, 3), dtype=np.uint8)

        # Draw an L-shaped piece where centroid is strictly different from bbox center
        # L-shape: 3 vertical blocks (200, 100) to (500, 200), and 1 bottom block (400, 200) to (500, 300)
        # BBox spans X: [100, 300] (W=200), Y: [200, 500] (H=300). Center = (200, 350)
        hold[200:500, 100:200] = (0, 165, 255)  # Orange vertical stem
        hold[400:500, 200:300] = (0, 165, 255)  # Orange bottom foot

        cfg = HeldPieceTrackerConfig(tray_mask_y_start=900, top_mask_y_end=50, min_absolute_area=1000.0)
        tracker = HeldPieceTracker(config=cfg)

        detection = tracker.track_held_piece(
            base_image=base,
            hold_image=hold,
            expected_piece_name="corner_2x2_bl",
            cell_size_px=100.0,
        )

        self.assertIsNotNone(detection.centroid)
        cx, cy = detection.centroid
        bx, by, bw, bh = detection.bbox
        bbox_cx = bx + bw / 2.0
        bbox_cy = by + bh / 2.0

        # In an L-shape with stem at X in [100, 200] and foot at [200, 300],
        # mass is heavier on the left, so cx must be < bbox_cx (200.0)
        self.assertLess(cx, bbox_cx, "Moments centroid X must reflect asymmetric mass distribution")
        self.assertGreater(cy, 300.0)

    # =========================================================================
    # TEST 3: Abnormally Large Bounding Box Detection
    # =========================================================================
    def test_03_detect_abnormal_bounding_box(self) -> None:
        """Verify tracker flags abnormally oversized bounding box when multiple items merge."""
        h, w = 2000, 1080
        base = np.zeros((h, w, 3), dtype=np.uint8)
        hold = np.zeros((h, w, 3), dtype=np.uint8)

        # Draw a massive blob (W=800, H=500) representing a merged piece + giant shadow
        hold[600:1100, 100:900] = (0, 200, 255)

        detection = self.tracker.track_held_piece(
            base_image=base,
            hold_image=hold,
            expected_piece_name="line_1x3_h",  # Expected W = 364.5, H = 121.5
            cell_size_px=121.5,
        )

        self.assertEqual(detection.status, TrackingStatus.ABNORMAL_BOUNDS)
        self.assertTrue(detection.is_abnormal_bbox)
        self.assertFalse(detection.is_success)

    # =========================================================================
    # TEST 4: No Contour Found in Flight Region
    # =========================================================================
    def test_04_no_contour_found(self) -> None:
        """Verify tracker returns NO_CONTOUR_FOUND when differential change is negligible."""
        h, w = 2000, 1080
        base = np.full((h, w, 3), 40, dtype=np.uint8)
        hold = base.copy()

        # Add tiny 5px noise speckles below min_absolute_area threshold
        hold[500:505, 500:505] = 255

        detection = self.tracker.track_held_piece(
            base_image=base,
            hold_image=hold,
            expected_piece_name="dot_1x1",
            cell_size_px=121.5,
        )

        self.assertEqual(detection.status, TrackingStatus.NO_CONTOUR_FOUND)
        self.assertEqual(detection.confidence, 0.0)
        self.assertIsNone(detection.bbox)
        self.assertIsNone(detection.centroid)

    # =========================================================================
    # TEST 5: Multiple Competing Ambiguous Contours
    # =========================================================================
    def test_05_ambiguous_competing_contours(self) -> None:
        """Verify tracker detects ambiguity when two identical candidates compete."""
        h, w = 2000, 1080
        base = np.zeros((h, w, 3), dtype=np.uint8)
        hold = np.zeros((h, w, 3), dtype=np.uint8)

        # Draw two identical 2x2 squares with high saturation in flight zone
        hold[500:740, 200:440] = (255, 100, 0)
        hold[500:740, 600:840] = (255, 100, 0)

        detection = self.tracker.track_held_piece(
            base_image=base,
            hold_image=hold,
            expected_piece_name="square_2x2",
            cell_size_px=120.0,
        )

        self.assertEqual(detection.status, TrackingStatus.AMBIGUOUS_CONTOURS)
        self.assertFalse(detection.is_success)
        self.assertGreaterEqual(detection.candidates_evaluated, 2)
        self.assertIn("Ambiguous candidates", detection.message)

    # =========================================================================
    # TEST 6: Invalid Input Images
    # =========================================================================
    def test_06_invalid_input_images(self) -> None:
        """Verify tracker gracefully handles corrupted or invalid input structures."""
        valid_img = np.zeros((1000, 1000, 3), dtype=np.uint8)

        # None inputs
        res1 = self.tracker.track_held_piece(None, valid_img, "dot_1x1")
        self.assertEqual(res1.status, TrackingStatus.INVALID_INPUT)

        # Dimension mismatch
        small_img = np.zeros((800, 800, 3), dtype=np.uint8)
        res2 = self.tracker.track_held_piece(valid_img, small_img, "dot_1x1")
        self.assertEqual(res2.status, TrackingStatus.INVALID_INPUT)

        # 2D array (grayscale instead of 3-channel BGR)
        gray_img = np.zeros((1000, 1000), dtype=np.uint8)
        res3 = self.tracker.track_held_piece(gray_img, gray_img, "dot_1x1")
        self.assertEqual(res3.status, TrackingStatus.INVALID_INPUT)

        # Tiny image below minimum threshold
        tiny_img = np.zeros((100, 100, 3), dtype=np.uint8)
        res4 = self.tracker.track_held_piece(tiny_img, tiny_img, "dot_1x1")
        self.assertEqual(res4.status, TrackingStatus.INVALID_INPUT)

    # =========================================================================
    # TEST 7: Base and Hold Images Identical (Zero Differential Change)
    # =========================================================================
    def test_07_identical_base_and_hold(self) -> None:
        """Verify tracker immediately returns zero confidence on identical frames."""
        dummy = np.full((1200, 1080, 3), 75, dtype=np.uint8)

        detection = self.tracker.track_held_piece(
            base_image=dummy,
            hold_image=dummy,
            expected_piece_name="line_1x5_h",
        )

        self.assertEqual(detection.status, TrackingStatus.NO_CONTOUR_FOUND)
        self.assertEqual(detection.confidence, 0.0)
        self.assertIn("identical", detection.message)

    # =========================================================================
    # TEST 8: Drop Shadow & Color Noise Separation (Saturation Gating)
    # =========================================================================
    def test_08_shadow_and_noise_separation(self) -> None:
        """Verify drop shadow (low saturation) is penalized and rejected."""
        h, w = 2000, 1080
        base = np.zeros((h, w, 3), dtype=np.uint8)
        hold = np.zeros((h, w, 3), dtype=np.uint8)

        # Shadow-like blob: dark gray, saturation = 0, value = 50
        hold[500:620, 200:800] = (50, 50, 50)

        detection = self.tracker.track_held_piece(
            base_image=base,
            hold_image=hold,
            expected_piece_name="line_1x5_h",
            cell_size_px=120.0,
        )

        # Pure shadow must not produce high confidence SUCCESS
        self.assertNotEqual(detection.status, TrackingStatus.SUCCESS)
        self.assertLess(detection.confidence, 0.60)
        self.assertLess(detection.mean_saturation, 10.0)

    # =========================================================================
    # TEST 9: Low Confidence Returns LOW_CONFIDENCE / UNKNOWN
    # =========================================================================
    def test_09_mismatched_shape_yields_low_confidence(self) -> None:
        """Verify that looking for line_1x5_h when a square_2x2 is present yields LOW_CONFIDENCE."""
        h, w = 2000, 1080
        base = np.zeros((h, w, 3), dtype=np.uint8)
        hold = np.zeros((h, w, 3), dtype=np.uint8)

        # Draw a bright 2x2 square (W=240, H=240, Aspect=1.0)
        hold[600:840, 400:640] = (0, 220, 255)

        # Query tracker for a line_1x5_h (W=607.5, H=121.5, Aspect=5.0)
        detection = self.tracker.track_held_piece(
            base_image=base,
            hold_image=hold,
            expected_piece_name="line_1x5_h",
            cell_size_px=121.5,
        )

        self.assertEqual(detection.status, TrackingStatus.LOW_CONFIDENCE)
        self.assertFalse(detection.is_success)
        self.assertLess(detection.confidence, 0.60)

    # =========================================================================
    # TEST 10: Immutability of Input Image Arrays
    # =========================================================================
    def test_10_input_immutability(self) -> None:
        """Verify tracker does not modify input numpy arrays under any circumstance."""
        if self.real_base is not None and self.real_mid is not None:
            base_copy = self.real_base.copy()
            mid_copy = self.real_mid.copy()
            _ = self.tracker.track_held_piece(self.real_base, self.real_mid, "line_1x5_h")
            self.assertTrue(np.array_equal(self.real_base, base_copy), "Base image was mutated!")
            self.assertTrue(np.array_equal(self.real_mid, mid_copy), "Hold image was mutated!")

        synth_base = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        synth_hold = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        sb_copy = synth_base.copy()
        sh_copy = synth_hold.copy()

        _ = self.tracker.track_held_piece(synth_base, synth_hold, "square_2x2")
        self.assertTrue(np.array_equal(synth_base, sb_copy), "Synthetic base image was mutated!")
        self.assertTrue(np.array_equal(synth_hold, sh_copy), "Synthetic hold image was mutated!")

    # =========================================================================
    # TEST 11: Multi-Polyomino Shape Catalog Support
    # =========================================================================
    def test_11_multi_polyomino_catalog_support(self) -> None:
        """Verify tracker supports various shapes from CANONICAL_41_CATALOG."""
        shapes_to_test = [
            ("dot_1x1", 120, 120),
            ("line_1x3_h", 360, 120),
            ("line_1x3_v", 120, 360),
            ("square_2x2", 240, 240),
            ("rect_2x3_h", 360, 240),
        ]

        h, w = 2000, 1080
        for piece_name, p_w, p_h in shapes_to_test:
            base = np.zeros((h, w, 3), dtype=np.uint8)
            hold = np.zeros((h, w, 3), dtype=np.uint8)

            hold[600:600+p_h, 300:300+p_w] = (0, 215, 255)  # Vivid yellow block

            detection = self.tracker.track_held_piece(
                base_image=base,
                hold_image=hold,
                expected_piece_name=piece_name,
                cell_size_px=120.0,
            )

            self.assertEqual(
                detection.status,
                TrackingStatus.SUCCESS,
                f"Failed for shape {piece_name}: {detection.message}",
            )
            self.assertGreaterEqual(detection.confidence, 0.70)

    # =========================================================================
    # TEST 12: Real Phase 4.2 Line-Clear Preview Decomposition
    # =========================================================================
    def test_12_line_clear_preview_decomposition(self) -> None:
        """Verify tracker decomposes thin line-clear preview glow when a move completes a full row."""
        p42_dir = os.path.join(PROJECT_ROOT, "debug", "phase_4_2_first_real_move")
        base_path = os.path.join(p42_dir, "before.png")
        hold_path = os.path.join(p42_dir, "mid_hold.png")

        if not os.path.isfile(base_path) or not os.path.isfile(hold_path):
            self.skipTest("Phase 4.2 real artifacts not found on disk.")

        base_img = cv2.imread(base_path)
        hold_img = cv2.imread(hold_path)
        self.assertIsNotNone(base_img)
        self.assertIsNotNone(hold_img)

        # Expected piece: line_1x4_h (W ~ 486px, H ~ 121.5px)
        detection = self.tracker.track_held_piece(
            base_image=base_img,
            hold_image=hold_img,
            expected_piece_name="line_1x4_h",
            cell_size_px=121.5,
        )

        self.assertEqual(
            detection.status,
            TrackingStatus.SUCCESS,
            f"Line-clear preview decomposition failed: {detection.message}",
        )
        self.assertGreaterEqual(detection.confidence, 0.70)
        self.assertFalse(detection.is_abnormal_bbox)
        self.assertIsNotNone(detection.bbox)
        self.assertIsNotNone(detection.centroid)

        # Verify bounding box matches expected line_1x4_h dimensions (~486 x 121.5)
        bx, by, bw, bh = detection.bbox
        self.assertAlmostEqual(bw, 486.0, delta=60.0, msg="Width should be close to 4 cells")
        self.assertAlmostEqual(bh, 121.5, delta=30.0, msg="Height should be close to 1 cell")

        # Verify centroid is close to actual physical placement (~281, 610)
        cx, cy = detection.centroid
        self.assertAlmostEqual(cx, 281.2, delta=20.0)
        self.assertAlmostEqual(cy, 609.9, delta=20.0)


if __name__ == "__main__":
    unittest.main()
