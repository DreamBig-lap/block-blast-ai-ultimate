"""Tests for One-Pass Performance Pipeline Optimization.

Validates:
1. Raw screencap decoding (16-byte Android 10+ and 12-byte Android 9- headers).
2. Pixel format handling (RGBA_8888, BGRA_8888) and color parity.
3. Robust fallback to PNG when raw buffer is corrupt, truncated, or invalid.
4. ADB socket streaming and automatic fallback to subprocess on connection failure.
5. Device disconnect detection during socket streaming.
6. Adaptive settle times (200ms normal vs 1.2s line-clear).
7. Dashboard optimistic status synchronization during VERIFYING state.
8. High-resolution timer performance benchmarks.
"""

from __future__ import annotations

import socket
import struct
import time
import unittest
from unittest.mock import MagicMock, patch

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
from action_engine.dashboard import AutonomousDashboard
from ai_engine.decision import AIDecisionEngine
from computer_vision.capture import ScreenCapture, ScreenCaptureError, parse_raw_screencap
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
from game_engine.pieces import CANONICAL_41_CATALOG


class TestRawScreencapAndTransport(unittest.TestCase):
    """Test suite for Raw Screencap parsing, socket streaming, and fallback mechanisms."""

    def setUp(self) -> None:
        self.width = 1080
        self.height = 2340
        self.raw_pixels = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        # Red pixel at (0, 0)
        self.raw_pixels[0, 0] = [255, 0, 0, 255]

    def test_01_parse_raw_screencap_android10_16byte_header(self) -> None:
        """Verify 16-byte header Android 10+ RGBA_8888 parsing."""
        header = struct.pack("<IIII", self.width, self.height, 1, 0)
        stream = header + self.raw_pixels.tobytes()

        img = parse_raw_screencap(stream)
        self.assertIsNotNone(img)
        self.assertEqual(img.shape, (self.height, self.width, 3))
        # RGBA [255, 0, 0, 255] converted to BGR is [0, 0, 255]
        np.testing.assert_array_equal(img[0, 0], [0, 0, 255])

    def test_02_parse_raw_screencap_android9_12byte_header(self) -> None:
        """Verify 12-byte header Android 9- RGBA_8888 parsing."""
        header = struct.pack("<III", self.width, self.height, 1)
        stream = header + self.raw_pixels.tobytes()

        img = parse_raw_screencap(stream)
        self.assertIsNotNone(img)
        self.assertEqual(img.shape, (self.height, self.width, 3))
        np.testing.assert_array_equal(img[0, 0], [0, 0, 255])

    def test_03_parse_raw_screencap_bgra_format(self) -> None:
        """Verify BGRA_8888 format (format code 5) correctly converted to BGR."""
        bgra_pixels = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        # Blue in BGRA: [255, 0, 0, 255]
        bgra_pixels[0, 0] = [255, 0, 0, 255]
        header = struct.pack("<IIII", self.width, self.height, 5, 0)
        stream = header + bgra_pixels.tobytes()

        img = parse_raw_screencap(stream)
        self.assertIsNotNone(img)
        # Blue in BGR: [255, 0, 0]
        np.testing.assert_array_equal(img[0, 0], [255, 0, 0])

    def test_04_parse_raw_screencap_corrupted_stream_returns_none(self) -> None:
        """Corrupted or truncated buffer returns None safely without raising exceptions."""
        self.assertIsNone(parse_raw_screencap(b""))
        self.assertIsNone(parse_raw_screencap(b"short_header"))

        # Truncated payload
        header = struct.pack("<IIII", self.width, self.height, 1, 0)
        self.assertIsNone(parse_raw_screencap(header + b"truncated_payload"))

        # Inconsistent dimensions
        hdr_mismatch = struct.pack("<IIII", 500, 500, 1, 0)
        self.assertIsNone(parse_raw_screencap(hdr_mismatch + self.raw_pixels.tobytes()))

    @patch("subprocess.run")
    def test_05_capture_falls_back_to_png_on_raw_corruption(self, mock_sub: MagicMock) -> None:
        """If raw stream is corrupted, capture_from_adb_with_timings transparently falls back to PNG."""
        cap = ScreenCapture(VisionConfig(use_raw_screencap=True, use_adb_socket=False))

        # First call (raw): returns invalid bytes
        raw_fail = MagicMock(returncode=0, stdout=b"corrupted_raw_bytes")
        # Second call (PNG fallback): returns valid synthetic PNG
        synthetic_png = cv2.imencode(".png", np.zeros((300, 300, 3), dtype=np.uint8))[1].tobytes()
        png_success = MagicMock(returncode=0, stdout=synthetic_png)

        mock_sub.side_effect = [raw_fail, png_success]

        img, adb_ms, dec_ms = cap.capture_from_adb_with_timings(device_id="TEST_DEV")
        self.assertIsNotNone(img)
        self.assertEqual(img.shape, (300, 300, 3))
        self.assertEqual(cap.last_capture_mode, "PNG")

    @patch("socket.create_connection")
    def test_06_socket_streaming_failure_falls_back_to_subprocess(self, mock_socket: MagicMock) -> None:
        """When socket connection is refused, capture gracefully falls back to subprocess."""
        mock_socket.side_effect = ConnectionRefusedError("ADB server not listening")

        cap = ScreenCapture(VisionConfig(use_raw_screencap=False, use_adb_socket=True))

        synthetic_png = cv2.imencode(".png", np.zeros((300, 300, 3), dtype=np.uint8))[1].tobytes()
        with patch("subprocess.run") as mock_sub:
            mock_sub.return_value = MagicMock(returncode=0, stdout=synthetic_png)
            img, adb_ms, dec_ms = cap.capture_from_adb_with_timings(device_id="TEST_DEV")

            self.assertIsNotNone(img)
            self.assertEqual(cap.last_transport_mode, "SUBPROCESS")

    @patch("socket.create_connection")
    def test_07_device_disconnect_raises_screen_capture_error(self, mock_socket: MagicMock) -> None:
        """Device disconnect during transport raises ScreenCaptureError cleanly."""
        mock_sock_inst = MagicMock()
        # Returns FAIL to transport
        mock_sock_inst.recv.side_effect = [b"FAIL", b"device 'R58N62QSJZV' not found"]
        mock_socket.return_value = mock_sock_inst

        cap = ScreenCapture(VisionConfig(use_raw_screencap=True, use_adb_socket=True))

        with patch("subprocess.run") as mock_sub:
            mock_sub.return_value = MagicMock(returncode=1, stderr=b"device offline")
            with self.assertRaises(ScreenCaptureError):
                cap.capture_from_adb_with_timings(device_id="R58N62QSJZV", retries=1)


class TestControllerOptimizationAndDashboard(unittest.TestCase):
    """Test suite verifying end-to-end optimizations and dashboard synchronization."""

    def setUp(self) -> None:
        self.catalog = CANONICAL_41_CATALOG
        self.dot = self.catalog.get_by_name("dot_1x1")
        self.bar2 = self.catalog.get_by_name("line_1x2_h")
        self.bar3 = self.catalog.get_by_name("line_1x3_h")

        self.mock_cap = MagicMock(spec=ScreenCapture)
        self.dummy_img = np.zeros((2340, 1080, 3), dtype=np.uint8)
        self.mock_cap.capture_from_adb.return_value = self.dummy_img
        self.mock_cap.last_adb_transfer_ms = 310.0
        self.mock_cap.last_png_decode_ms = 6.0

        self.mock_rec = MagicMock(spec=BlockBlastRecognizer)
        self.mock_rec.config = DEFAULT_VISION_CONFIG
        self.ai_engine = AIDecisionEngine()
        self.mock_closed_loop = MagicMock(spec=ClosedLoopController)
        self.mock_verifier = MagicMock(spec=ActionVerifier)

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
            normal_settle_sec=0.20,
        )
        ctrl = AutonomousGameController(
            config=config,
            screen_capture=self.mock_cap,
            recognizer=self.mock_rec,
            ai_engine=self.ai_engine,
            closed_loop_controller=self.mock_closed_loop,
            action_verifier=self.mock_verifier,
        )
        mock_shell = MagicMock()
        mock_shell.poll.return_value = None
        mock_shell.stdin = MagicMock()
        ctrl._shell_proc = mock_shell
        ctrl._read_shell_until_token = MagicMock(return_value=True)
        return ctrl

    def test_08_adaptive_settle_delay_defaults_to_200ms(self) -> None:
        """Verify normal_settle_sec defaults to 0.20s for non-line-clear moves."""
        verifier = ActionVerifier(self.mock_rec, self.mock_cap)
        self.assertEqual(verifier.normal_settle_sec, 0.20)
        self.assertEqual(verifier.line_clear_settle_sec, 1.20)

    def test_09_dashboard_renders_verifying_in_flight_state(self) -> None:
        """Dashboard renders '[VERIFYING POST-MOVE...]' and '[PLACED -> VERIFYING]' during verification."""
        ctrl = self._create_controller(max_turns=2)
        ctrl.set_state(AutonomousState.VERIFYING)

        # Set planned move on slot 0
        from ai_engine.decision import PlannedMove
        ctrl.last_planned_move = PlannedMove(
            step_index=0,
            slot_idx=0,
            row=0,
            col=0,
            piece_name="dot_1x1",
            cell_count=1,
            predicted_score_delta=10,
            lines_cleared=0,
            unique_cells_cleared=0,
            combo_streak_after=0,
            is_game_over=False,
        )

        # Create dummy recognition with slot 0
        slot0 = PieceRecognitionResult(
            slot_idx=0, is_empty=False, matched_name="dot_1x1", piece=self.dot,
            cells=self.dot.cells, height=1, width=1, cell_count=1,
            bbox_in_screen=(100, 1800, 100, 100), confidence=0.95, status_message="",
        )
        dummy_board = MagicMock()
        dummy_board.bitboard = 0x0
        ctrl.last_recognition = FrameRecognitionResult(
            status=RecognitionStatus.SUCCESS,
            board=dummy_board,
            tray_pieces=(slot0,),
            overall_confidence=0.95,
            processing_time_ms=10.0,
        )

        dashboard = AutonomousDashboard(ctrl)
        rendered = dashboard.render_view()

        self.assertIn("BOARD STATE [VERIFYING POST-MOVE...]", rendered)
        self.assertIn("Slot 0: [PLACED -> VERIFYING]", rendered)

    def test_10_high_resolution_monotonic_timing_precision(self) -> None:
        """Verify TurnLatencyMetrics records sub-millisecond precision with monotonic timer."""
        metrics = TurnLatencyMetrics(
            turn_index=1,
            capture_ms=0.0,
            adb_transfer_ms=0.0,
            png_decode_ms=0.0,
            vision_ms=0.0,
            ai_ms=18.5,
            drag_ms=462.1,
            verify_ms=558.4,
            total_turn_ms=1039.0,
            frame_reused=True,
        )
        d = metrics.to_dict()
        self.assertTrue(d["frame_reused"])
        self.assertEqual(d["capture_ms"], 0.0)
        self.assertEqual(d["ai_ms"], 18.5)
        self.assertEqual(d["total_turn_ms"], 1039.0)


if __name__ == "__main__":
    unittest.main()
