"""Unit and Integration Tests for PerceptionDiagnosticSaver.

Verifies:
- Trigger conditions for should_save() (LOW_CONFIDENCE, UNKNOWN slot, low confidence).
- Non-overwrite unique folder generation.
- Complete artifact creation:
  1. full_screen.png
  2. tray_band.png
  3. slot_0.png, slot_1.png, slot_2.png
  4. slot_0_mask.png, slot_1_mask.png, slot_2_mask.png
  5. tray_contours.png
  6. recognition_report.json
- Consistency between saved artifacts and in-memory detector state.
- AutonomousGameController integration: pauses and writes diagnostics safely on UNKNOWN.
- Safe error handling: disk write failure does not crash game controller.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from action_engine.autonomous_controller import (
    AutonomousConfig,
    AutonomousGameController,
    AutonomousState,
)
from computer_vision.config import DEFAULT_VISION_CONFIG, VisionConfig
from computer_vision.diagnostic_saver import PerceptionDiagnosticSaver
from computer_vision.models import (
    BoardRecognitionResult,
    FrameRecognitionResult,
    PieceRecognitionResult,
    RecognitionStatus,
)
from computer_vision.piece_detector import PieceDetector, RawPieceBlob
from computer_vision.recognizer import BlockBlastRecognizer


class TestPerceptionDiagnosticSaver(unittest.TestCase):
    """Test suite for PerceptionDiagnosticSaver."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_diag_saver_"))
        self.config = DEFAULT_VISION_CONFIG
        self.saver = PerceptionDiagnosticSaver(config=self.config, base_dir=self.temp_dir)

    def tearDown(self) -> None:
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_should_save_triggers(self) -> None:
        """Verify all low-confidence/UNKNOWN conditions return should_save() == True."""
        # 1. Status LOW_CONFIDENCE
        res_low_conf = FrameRecognitionResult(
            status=RecognitionStatus.LOW_CONFIDENCE,
            board=None,
            tray_pieces=(),
            overall_confidence=0.50,
            processing_time_ms=5.0,
            error_message="Low confidence",
        )
        self.assertTrue(self.saver.should_save(res_low_conf))

        # 2. Status FAILED
        res_failed = FrameRecognitionResult(
            status=RecognitionStatus.FAILED,
            board=None,
            tray_pieces=(),
            overall_confidence=0.20,
            processing_time_ms=5.0,
            error_message="Board not found",
        )
        self.assertTrue(self.saver.should_save(res_failed))

        # 3. Overall confidence below threshold
        res_below_min = FrameRecognitionResult(
            status=RecognitionStatus.SUCCESS,
            board=MagicMock(board_bbox=(54, 548, 972, 972)),
            tray_pieces=(),
            overall_confidence=0.65,  # Below 0.75
            processing_time_ms=5.0,
        )
        self.assertTrue(self.saver.should_save(res_below_min))

        # 4. UNKNOWN tray piece
        unknown_piece = PieceRecognitionResult(
            slot_idx=0,
            is_empty=False,
            matched_name="UNKNOWN",
            piece=None,
            cells=(),
            height=0,
            width=0,
            cell_count=0,
            bbox_in_screen=(100, 1700, 100, 100),
            confidence=0.0,
            status_message="No match",
        )
        res_unknown_slot = FrameRecognitionResult(
            status=RecognitionStatus.SUCCESS,
            board=MagicMock(board_bbox=(54, 548, 972, 972)),
            tray_pieces=(unknown_piece,),
            overall_confidence=0.88,
            processing_time_ms=5.0,
        )
        self.assertTrue(self.saver.should_save(res_unknown_slot))

        # 5. Clean success should NOT trigger save
        clean_piece = PieceRecognitionResult(
            slot_idx=0,
            is_empty=False,
            matched_name="square_2x2",
            piece=MagicMock(),
            cells=((0, 0), (0, 1), (1, 0), (1, 1)),
            height=2,
            width=2,
            cell_count=4,
            bbox_in_screen=(100, 1700, 118, 118),
            confidence=1.0,
            status_message="Matched square_2x2",
        )
        res_clean = FrameRecognitionResult(
            status=RecognitionStatus.SUCCESS,
            board=MagicMock(board_bbox=(54, 548, 972, 972)),
            tray_pieces=(clean_piece,),
            overall_confidence=0.95,
            processing_time_ms=5.0,
        )
        self.assertFalse(self.saver.should_save(res_clean))

    def test_unique_folder_creation_no_overwrite(self) -> None:
        """Verify folders never overwrite existing directories with the same timestamp."""
        fixed_time = datetime(2026, 11, 1, 17, 42, 0)
        dir1 = self.saver.create_unique_failure_dir(timestamp=fixed_time)
        dir2 = self.saver.create_unique_failure_dir(timestamp=fixed_time)
        dir3 = self.saver.create_unique_failure_dir(timestamp=fixed_time)

        self.assertEqual(dir1.name, "20261101_174200")
        self.assertEqual(dir2.name, "20261101_174200_1")
        self.assertEqual(dir3.name, "20261101_174200_2")
        self.assertTrue(dir1.is_dir())
        self.assertTrue(dir2.is_dir())
        self.assertTrue(dir3.is_dir())

    def test_save_diagnostic_failure_artifacts(self) -> None:
        """Verify all 6 required artifacts are saved and contain accurate data."""
        # Create synthetic 1080x2340 image with tray background and a synthetic square block in Slot 1
        h, w = 2340, 1080
        frame = np.full((h, w, 3), (158, 114, 76), dtype=np.uint8)  # Tray background color
        # Draw bright block in Slot 1 (x: 480..580, y: 1750..1850)
        frame[1750:1850, 480:580] = (30, 200, 240)  # Bright yellow/orange block

        # Run detector so intermediate details are generated
        detector = PieceDetector(self.config)
        blobs = detector.detect_tray_pieces(frame)
        self.assertIsNotNone(detector.last_detection_details)

        # Mock recognition result with Slot 0 as UNKNOWN
        unknown_piece = PieceRecognitionResult(
            slot_idx=0,
            is_empty=False,
            matched_name="UNKNOWN",
            piece=None,
            cells=(),
            height=0,
            width=0,
            cell_count=0,
            bbox_in_screen=(100, 1700, 120, 120),
            confidence=0.0,
            status_message="Piece shape does not match any canonical catalog piece with sufficient confidence.",
        )
        empty_slot2 = PieceRecognitionResult(
            slot_idx=2,
            is_empty=True,
            matched_name=None,
            piece=None,
            cells=(),
            height=0,
            width=0,
            cell_count=0,
            bbox_in_screen=(691, 1661, 345, 351),
            confidence=0.95,
            status_message="Tray slot is empty.",
        )
        piece_slot1 = PieceRecognitionResult(
            slot_idx=1,
            is_empty=False,
            matched_name="square_2x2",
            piece=MagicMock(),
            cells=((0, 0), (0, 1), (1, 0), (1, 1)),
            height=2,
            width=2,
            cell_count=4,
            bbox_in_screen=(480, 1750, 100, 100),
            confidence=1.0,
            status_message="Matched square_2x2",
        )
        rec_res = FrameRecognitionResult(
            status=RecognitionStatus.LOW_CONFIDENCE,
            board=BoardRecognitionResult(
                board_bbox=(54, 548, 972, 972),
                grid_rows=8,
                grid_cols=8,
                cell_size=121.5,
                cells=(),
                occupancy_matrix=tuple((0,) * 8 for _ in range(8)),
                bitboard=0x0,
                board_confidence=0.95,
                grid_confidence=0.95,
                cell_confidence=0.95,
            ),
            tray_pieces=(unknown_piece, piece_slot1, empty_slot2),
            overall_confidence=0.82,
            processing_time_ms=15.0,
            error_message="Contains unmapped tray pieces (UNKNOWN)",
        )

        pause_reason = "Perception uncertain: Slot(s) [0] contain UNKNOWN piece or confidence below 0.70. Pausing to prevent false Game Over."
        saved_dir = self.saver.save_diagnostic_failure(
            frame=frame,
            recognition_result=rec_res,
            pause_reason=pause_reason,
            piece_detector=detector,
            timestamp=datetime(2026, 11, 1, 17, 45, 0),
        )

        self.assertIsNotNone(saved_dir)
        self.assertTrue(saved_dir.exists())

        # 1. full_screen.png
        p_full = saved_dir / "full_screen.png"
        self.assertTrue(p_full.exists())
        img_full = cv2.imread(str(p_full))
        self.assertEqual(img_full.shape, (2340, 1080, 3))

        # 2. tray_band.png
        p_tray = saved_dir / "tray_band.png"
        self.assertTrue(p_tray.exists())
        img_tray = cv2.imread(str(p_tray))
        expected_tray_h = int(2340 * self.config.tray_expected_y_max_rel) - int(2340 * self.config.tray_expected_y_min_rel)
        self.assertEqual(img_tray.shape[0], expected_tray_h)
        self.assertEqual(img_tray.shape[1], 1080)

        # 3. slot_0.png, slot_1.png, slot_2.png
        for i in range(3):
            p_slot = saved_dir / f"slot_{i}.png"
            self.assertTrue(p_slot.exists())
            img_slot = cv2.imread(str(p_slot))
            self.assertGreater(img_slot.shape[1], 50)
            self.assertGreater(img_slot.shape[0], 50)

        # 4. slot_0_mask.png, slot_1_mask.png, slot_2_mask.png
        for i in range(3):
            p_mask = saved_dir / f"slot_{i}_mask.png"
            self.assertTrue(p_mask.exists())
            img_mask = cv2.imread(str(p_mask), cv2.IMREAD_GRAYSCALE)
            self.assertIsNotNone(img_mask)
            unique_vals = set(np.unique(img_mask))
            self.assertTrue(unique_vals.issubset({0, 255}))

        # 5. tray_contours.png
        p_contours = saved_dir / "tray_contours.png"
        self.assertTrue(p_contours.exists())
        img_contours = cv2.imread(str(p_contours))
        self.assertEqual(img_contours.shape, img_tray.shape)

        # 6. recognition_report.json
        p_report = saved_dir / "recognition_report.json"
        self.assertTrue(p_report.exists())
        with open(p_report, "r", encoding="utf-8") as f:
            report = json.load(f)

        self.assertEqual(report["folder_name"], "20261101_174500")
        self.assertEqual(report["pause_reason"], pause_reason)
        self.assertEqual(report["recognition_status"], "LOW_CONFIDENCE")
        self.assertEqual(report["overall_confidence"], 0.82)
        self.assertEqual(report["screenshot"]["height"], 2340)
        self.assertEqual(report["screenshot"]["width"], 1080)
        self.assertTrue(report["board"]["detected"])
        self.assertEqual(report["board"]["occupied_count"], 0)
        self.assertEqual(report["board"]["bitboard_hex"], "0x0000000000000000")
        self.assertEqual(len(report["slots"]), 3)
        self.assertEqual(report["slots"][0]["classification_result"]["matched_name"], "UNKNOWN")
        self.assertEqual(
            report["slots"][0]["classifier_rejection_reason"],
            "Piece shape does not match any canonical catalog piece with sufficient confidence.",
        )
        self.assertIn("tray_bg_diff_threshold", report["segmentation_thresholds_and_config"])
        self.assertIn("tray_min_contour_area", report["segmentation_thresholds_and_config"])

    def test_controller_integration_on_unknown_piece(self) -> None:
        """Verify AutonomousGameController automatically saves diagnostics on UNKNOWN slot pause."""
        config = AutonomousConfig(dry_run=True, project_root=str(self.temp_dir))
        controller = AutonomousGameController(config=config)

        # Prepare synthetic frame and mock recognizer
        h, w = 2340, 1080
        synthetic_frame = np.full((h, w, 3), 120, dtype=np.uint8)

        unknown_piece = PieceRecognitionResult(
            slot_idx=0,
            is_empty=False,
            matched_name="UNKNOWN",
            piece=None,
            cells=(),
            height=0,
            width=0,
            cell_count=0,
            bbox_in_screen=(100, 1700, 100, 100),
            confidence=0.0,
            status_message="No match in catalog",
        )
        mock_rec = FrameRecognitionResult(
            status=RecognitionStatus.LOW_CONFIDENCE,
            board=BoardRecognitionResult(
                board_bbox=(54, 548, 972, 972),
                grid_rows=8,
                grid_cols=8,
                cell_size=121.5,
                cells=(),
                occupancy_matrix=tuple((0,) * 8 for _ in range(8)),
                bitboard=0x0,
                board_confidence=0.95,
                grid_confidence=0.95,
                cell_confidence=0.95,
            ),
            tray_pieces=(unknown_piece,),
            overall_confidence=0.82,
            processing_time_ms=10.0,
            error_message="Contains unmapped tray pieces (UNKNOWN)",
        )

        controller.capture.capture_from_adb = MagicMock(return_value=synthetic_frame)
        controller.recognizer.recognize_frame = MagicMock(return_value=mock_rec)

        # Step single turn
        result = controller.step_single_turn()
        self.assertFalse(result)
        self.assertEqual(controller.state, AutonomousState.PAUSED)
        self.assertTrue(
            "Perception low confidence" in controller.last_pause_reason
            or "Perception uncertain" in controller.last_pause_reason
        )

        # Confirm diagnostic folder was created
        failures_dir = self.temp_dir / "debug" / "perception_failures"
        self.assertTrue(failures_dir.exists())
        subdirs = list(failures_dir.iterdir())
        self.assertEqual(len(subdirs), 1)

        saved_folder = subdirs[0]
        self.assertTrue((saved_folder / "full_screen.png").exists())
        self.assertTrue((saved_folder / "tray_band.png").exists())
        self.assertTrue((saved_folder / "slot_0.png").exists())
        self.assertTrue((saved_folder / "slot_0_mask.png").exists())
        self.assertTrue((saved_folder / "tray_contours.png").exists())
        self.assertTrue((saved_folder / "recognition_report.json").exists())

    def test_error_handling_does_not_crash_controller(self) -> None:
        """Verify errors during diagnostic saving do not crash AutonomousGameController."""
        config = AutonomousConfig(dry_run=True, project_root=str(self.temp_dir))
        controller = AutonomousGameController(config=config)

        # Make diagnostic_saver raise an unhandled Exception
        controller.diagnostic_saver.save_diagnostic_failure = MagicMock(
            side_effect=IOError("Disk write simulated failure")
        )

        h, w = 2340, 1080
        synthetic_frame = np.full((h, w, 3), 120, dtype=np.uint8)
        mock_rec = FrameRecognitionResult(
            status=RecognitionStatus.FAILED,
            board=None,
            tray_pieces=(),
            overall_confidence=0.1,
            processing_time_ms=5.0,
            error_message="Board missing",
        )

        controller.capture.capture_from_adb = MagicMock(return_value=synthetic_frame)
        controller.recognizer.recognize_frame = MagicMock(return_value=mock_rec)

        # Must not raise exception, must pause cleanly
        result = controller.step_single_turn()
        self.assertFalse(result)
        self.assertEqual(controller.state, AutonomousState.PAUSED)
        self.assertIn("Vision failed to detect 8x8 game board", controller.last_pause_reason)


if __name__ == "__main__":
    unittest.main()
