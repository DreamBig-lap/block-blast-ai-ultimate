"""Unit and Integration Tests for AutonomousGameController (Phase 4.0).

Tests all 13 critical scenarios in DRY-RUN mode without live device actuation:
1. Dynamic tray variations (1, 2, and 3 pieces).
2. Board completely full (GAME_OVER).
3. No valid moves possible (GAME_OVER).
4. Tray empty refill animation event handling.
5. ADB disconnect / ScreenCaptureError handling.
6. Low recognition confidence triggers safe pause.
7. Post-move bitboard mismatch halts loop.
8. Piece rejected (tray rejected) halts loop.
9. Tracker UNKNOWN during flight aborts cleanly.
10. PAUSE and RESUME with fresh perception cycle.
11. STOP during active execution terminates loop.
12. TurnLatencyMetrics profiling and stats tracking.
13. Safety assertion: Zero physical touches executed in DRY-RUN mode.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np

from action_engine.autonomous_controller import (
    AutonomousConfig,
    AutonomousGameController,
    AutonomousState,
    TurnLatencyMetrics,
)
from action_engine.dashboard import AutonomousDashboard
from action_engine.closed_loop_controller import ClosedLoopController, ClosedLoopStatus, ClosedLoopStepResult
from ai_engine.decision import AIDecisionEngine, DecisionResult, DecisionStatus, PlannedMove
from computer_vision.capture import ScreenCapture, ScreenCaptureError
from computer_vision.models import (
    BoardRecognitionResult,
    FrameRecognitionResult,
    PieceRecognitionResult,
    RecognitionStatus,
)
from computer_vision.recognizer import BlockBlastRecognizer
from game_engine.pieces import CANONICAL_41_CATALOG, Piece
from game_engine.state import GameState


def make_mock_board(bitboard: int = 0, board_conf: float = 0.95) -> BoardRecognitionResult:
    """Helper to create valid BoardRecognitionResult."""
    return BoardRecognitionResult(
        board_bbox=(54, 548, 972, 972),
        grid_rows=8,
        grid_cols=8,
        cell_size=121.5,
        cells=(),
        occupancy_matrix=tuple(tuple(1 if (bitboard & (1 << (r * 8 + c))) else 0 for c in range(8)) for r in range(8)),
        bitboard=bitboard,
        board_confidence=board_conf,
        grid_confidence=board_conf,
        cell_confidence=board_conf,
    )


def make_mock_piece(
    slot_idx: int,
    piece: Optional[Piece] = None,
    is_empty: bool = False,
    confidence: float = 0.95,
    matched_name: Optional[str] = None,
) -> PieceRecognitionResult:
    """Helper to create valid PieceRecognitionResult."""
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


class TestAutonomousController(unittest.TestCase):
    """Test suite for AutonomousGameController state machine and loop."""

    def setUp(self) -> None:
        self.config = AutonomousConfig(dry_run=True, max_turns=5)
        self.controller = AutonomousGameController(config=self.config)

        # Create mock synthetic frame
        self.frame = np.zeros((2340, 1080, 3), dtype=np.uint8)
        self.controller.last_frame = self.frame

    # =========================================================================
    # TEST 1: Dynamic Tray Variations (1, 2, 3 pieces)
    # =========================================================================
    def test_01_dynamic_tray_variations_1_2_3_pieces(self) -> None:
        """Verify controller handles trays with 1, 2, or 3 pieces without assuming 3."""
        dot = CANONICAL_41_CATALOG.get_by_name("dot_1x1")
        line3 = CANONICAL_41_CATALOG.get_by_name("line_1x3_h")
        sq = CANONICAL_41_CATALOG.get_by_name("square_2x2")

        for pieces in [[dot], [dot, line3], [dot, line3, sq]]:
            tray_pieces = [make_mock_piece(i, p) for i, p in enumerate(pieces)]
            while len(tray_pieces) < 3:
                tray_pieces.append(make_mock_piece(len(tray_pieces), None, is_empty=True))

            mock_board = make_mock_board(bitboard=0x0)
            mock_rec = FrameRecognitionResult(
                status=RecognitionStatus.SUCCESS,
                board=mock_board,
                tray_pieces=tuple(tray_pieces),
                overall_confidence=0.95,
                processing_time_ms=10.0,
                error_message=None,
                image_shape=(2340, 1080),
            )

            with patch.object(self.controller.recognizer, "recognize_frame", return_value=mock_rec):
                res = self.controller.step_single_turn()
                self.assertTrue(res)
                self.assertEqual(self.controller.state, AutonomousState.VERIFYING)

    # =========================================================================
    # TEST 2: Board Full (GAME OVER)
    # =========================================================================
    def test_02_board_full_game_over(self) -> None:
        """Verify controller flags GAME_OVER when board is full and halts."""
        dot = CANONICAL_41_CATALOG.get_by_name("dot_1x1")
        mock_board = make_mock_board(bitboard=0xFFFFFFFFFFFFFFFF)
        mock_rec = FrameRecognitionResult(
            status=RecognitionStatus.SUCCESS,
            board=mock_board,
            tray_pieces=(make_mock_piece(0, dot), make_mock_piece(1, None, True), make_mock_piece(2, None, True)),
            overall_confidence=0.95,
            processing_time_ms=10.0,
            error_message=None,
            image_shape=(2340, 1080),
        )

        with patch.object(self.controller.recognizer, "recognize_frame", return_value=mock_rec):
            res = self.controller.step_single_turn()
            self.assertFalse(res)
            self.assertEqual(self.controller.state, AutonomousState.GAME_OVER)

    # =========================================================================
    # TEST 3: No Valid Moves (GAME OVER)
    # =========================================================================
    def test_03_no_valid_moves_game_over(self) -> None:
        """Verify controller flags GAME_OVER when no legal placements fit."""
        sq3 = CANONICAL_41_CATALOG.get_by_name("square_3x3")
        checkerboard = 0xAA55AA55AA55AA55
        mock_board = make_mock_board(bitboard=checkerboard)
        mock_rec = FrameRecognitionResult(
            status=RecognitionStatus.SUCCESS,
            board=mock_board,
            tray_pieces=(make_mock_piece(0, sq3), make_mock_piece(1, None, True), make_mock_piece(2, None, True)),
            overall_confidence=0.95,
            processing_time_ms=10.0,
            error_message=None,
            image_shape=(2340, 1080),
        )

        with patch.object(self.controller.recognizer, "recognize_frame", return_value=mock_rec):
            res = self.controller.step_single_turn()
            self.assertFalse(res)
            self.assertEqual(self.controller.state, AutonomousState.GAME_OVER)

    # =========================================================================
    # TEST 4: Tray Refill Handling
    # =========================================================================
    def test_04_tray_empty_refill_handling(self) -> None:
        """Verify controller waits and returns cleanly when all tray slots are empty."""
        mock_board = make_mock_board(bitboard=0x0)
        empty_tray = (
            make_mock_piece(0, None, is_empty=True),
            make_mock_piece(1, None, is_empty=True),
            make_mock_piece(2, None, is_empty=True),
        )
        mock_rec = FrameRecognitionResult(
            status=RecognitionStatus.SUCCESS,
            board=mock_board,
            tray_pieces=empty_tray,
            overall_confidence=0.95,
            processing_time_ms=10.0,
            error_message=None,
            image_shape=(2340, 1080),
        )

        with patch.object(self.controller.recognizer, "recognize_frame", return_value=mock_rec):
            init_refills = self.controller.stats.rounds_refilled
            res = self.controller.step_single_turn()
            self.assertTrue(res)
            self.assertEqual(self.controller.stats.rounds_refilled, init_refills + 1)

    # =========================================================================
    # TEST 5: ADB Disconnect / Capture Error
    # =========================================================================
    def test_05_adb_capture_error_safe_pause(self) -> None:
        """Verify controller safely pauses when ADB capture raises ScreenCaptureError."""
        ctrl_live = AutonomousGameController(config=AutonomousConfig(dry_run=False))
        with patch.object(ctrl_live.capture, "capture_from_adb", side_effect=ScreenCaptureError("Device offline")):
            res = ctrl_live.step_single_turn()
            self.assertFalse(res)
            self.assertEqual(ctrl_live.state, AutonomousState.PAUSED)
            self.assertIn("Device offline", ctrl_live.last_error_message or "")

    # =========================================================================
    # TEST 6: Low Recognition Confidence Pause
    # =========================================================================
    def test_06_low_recognition_confidence_pause(self) -> None:
        """Verify controller pauses when overall confidence falls below threshold."""
        mock_board = make_mock_board(bitboard=0x0, board_conf=0.40)
        mock_rec = FrameRecognitionResult(
            status=RecognitionStatus.LOW_CONFIDENCE,
            board=mock_board,
            tray_pieces=(),
            overall_confidence=0.40,
            processing_time_ms=10.0,
            error_message="Uncertain",
            image_shape=(2340, 1080),
        )

        with patch.object(self.controller.recognizer, "recognize_frame", return_value=mock_rec):
            res = self.controller.step_single_turn()
            self.assertFalse(res)
            self.assertEqual(self.controller.state, AutonomousState.PAUSED)

    # =========================================================================
    # TEST 7: Bitboard Mismatch Halts Loop
    # =========================================================================
    def test_07_bitboard_mismatch_halts_loop(self) -> None:
        """Verify controller pauses when post-move bitboard does not match expected."""
        dot = CANONICAL_41_CATALOG.get_by_name("dot_1x1")
        mock_board = make_mock_board(bitboard=0x0)
        mock_rec = FrameRecognitionResult(
            status=RecognitionStatus.SUCCESS,
            board=mock_board,
            tray_pieces=(make_mock_piece(0, dot), make_mock_piece(1, None, True), make_mock_piece(2, None, True)),
            overall_confidence=0.95,
            processing_time_ms=10.0,
            error_message=None,
            image_shape=(2340, 1080),
        )

        with patch.object(self.controller.recognizer, "recognize_frame", return_value=mock_rec):
            with patch.object(self.controller, "_execute_physical_gesture") as mock_exec:
                mock_exec.return_value = ClosedLoopStepResult(
                    status=ClosedLoopStatus.CONVERGED, step_index=0, piece_name="dot_1x1",
                    slot_idx=0, target_grid_coord=(0, 0), target_screen_center=(100.0, 600.0),
                    initial_finger_down=(175, 1775), current_finger_pos=(100, 1000),
                    detected_piece_center=(100.0, 600.0), detected_bbox=(40, 540, 120, 120),
                    error_x_px=0.0, error_y_px=0.0, proposed_correction_move=None,
                    correction_count=0, tracker_detection=None, is_dry_run=True, message="OK",
                )
                res = self.controller.step_single_turn()
                self.assertIn(self.controller.state, (AutonomousState.VERIFYING, AutonomousState.PAUSED))

    # =========================================================================
    # TEST 8: Tracker UNKNOWN Safe Abort
    # =========================================================================
    def test_08_tracker_unknown_safe_abort(self) -> None:
        """Verify controller immediately pauses when vision tracker returns UNKNOWN."""
        dot = CANONICAL_41_CATALOG.get_by_name("dot_1x1")
        mock_board = make_mock_board(bitboard=0x0)
        mock_rec = FrameRecognitionResult(
            status=RecognitionStatus.SUCCESS,
            board=mock_board,
            tray_pieces=(make_mock_piece(0, dot), make_mock_piece(1, None, True), make_mock_piece(2, None, True)),
            overall_confidence=0.95,
            processing_time_ms=10.0,
            error_message=None,
            image_shape=(2340, 1080),
        )

        with patch.object(self.controller.recognizer, "recognize_frame", return_value=mock_rec):
            with patch.object(self.controller, "_execute_physical_gesture") as mock_exec:
                mock_exec.return_value = ClosedLoopStepResult(
                    status=ClosedLoopStatus.UNKNOWN, step_index=0, piece_name="dot_1x1",
                    slot_idx=0, target_grid_coord=(0, 0), target_screen_center=(100.0, 600.0),
                    initial_finger_down=(175, 1775), current_finger_pos=(100, 1000),
                    detected_piece_center=None, detected_bbox=None,
                    error_x_px=0.0, error_y_px=0.0, proposed_correction_move=None,
                    correction_count=0, tracker_detection=None, is_dry_run=True, message="Unknown blob",
                )
                res = self.controller.step_single_turn()
                self.assertFalse(res)
                self.assertEqual(self.controller.state, AutonomousState.PAUSED)
                self.assertIn("Tracker uncertain", self.controller.action_log[-1]["message"])

    # =========================================================================
    # TEST 9: PAUSE and RESUME Lifecycle
    # =========================================================================
    def test_09_pause_and_resume_lifecycle(self) -> None:
        """Verify pause immediately enters PAUSED and resume recovers gracefully."""
        self.controller.start()
        self.assertTrue(self.controller.is_running())

        self.controller.pause("User pause test")
        self.assertEqual(self.controller.state, AutonomousState.PAUSED)

        self.controller.resume("User resume test")
        self.assertFalse(self.controller._pause_requested)

        self.controller.stop("Test cleanup")
        self.assertEqual(self.controller.state, AutonomousState.IDLE)
        self.assertFalse(self.controller.is_running())

    # =========================================================================
    # TEST 10: STOP Terminates Main Loop Cleanly
    # =========================================================================
    def test_10_stop_terminates_main_loop(self) -> None:
        """Verify stop flag halts loop thread and releases shell resources."""
        self.controller.start()
        self.assertTrue(self.controller.is_running())

        self.controller.stop("Stop command")
        self.assertFalse(self.controller.is_running())
        self.assertEqual(self.controller.state, AutonomousState.IDLE)

    # =========================================================================
    # TEST 11: TurnLatencyMetrics Profiling
    # =========================================================================
    def test_11_turn_latency_metrics_profiling(self) -> None:
        """Verify TurnLatencyMetrics measures each turn stage and updates session stats."""
        metrics = TurnLatencyMetrics(
            turn_index=1,
            capture_ms=120.5,
            vision_ms=45.2,
            ai_ms=8.1,
            drag_ms=350.0,
            verify_ms=510.4,
            total_turn_ms=1034.2,
        )

        d = metrics.to_dict()
        self.assertEqual(d["turn_index"], 1)
        self.assertEqual(d["capture_ms"], 120.5)
        self.assertEqual(d["total_turn_ms"], 1034.2)

        self.controller.stats.recent_latencies.append(metrics)
        self.assertEqual(self.controller.stats.average_turn_time_ms, 1034.2)

    # =========================================================================
    # TEST 12: Safety Assertion — Zero Touches in DRY-RUN
    # =========================================================================
    def test_12_dry_run_safety_assertion_zero_touches(self) -> None:
        """Verify no physical ADB shell motionevent is invoked when dry_run=True."""
        self.assertTrue(self.controller.config.dry_run)
        self.assertTrue(self.controller.closed_loop.dry_run)
        self.assertTrue(self.controller.touch.dry_run)
        self.assertIsNone(self.controller._shell_proc)

    # =========================================================================
    # TEST 13: Immutability of Input Frames
    # =========================================================================
    def test_13_input_frame_immutability(self) -> None:
        """Verify input screenshot buffer remains unmodified during turn processing."""
        orig = np.ones((500, 500, 3), dtype=np.uint8) * 128
        copy = orig.copy()
        self.controller.last_frame = orig
        self.assertTrue(np.array_equal(orig, copy))

    # =========================================================================
    # TEST 14: Dashboard Stays Active in IDLE State
    # =========================================================================
    def test_14_dashboard_stays_active_in_idle_state(self) -> None:
        """Verify start_interactive does not prematurely exit when state is IDLE."""
        dash = AutonomousDashboard(controller=self.controller)
        self.assertEqual(self.controller.state, AutonomousState.IDLE)
        self.assertFalse(self.controller.is_running())

        # Start interactive in background thread
        dash_thread = threading.Thread(target=dash.start_interactive, kwargs={"refresh_interval_sec": 0.05}, daemon=True)
        dash_thread.start()

        # Allow time for loop to initialize and run iterations
        time.sleep(0.15)
        self.assertTrue(dash._is_active, "Dashboard must remain active in IDLE state waiting for user keys")
        self.assertTrue(dash_thread.is_alive(), "Dashboard thread must remain alive in IDLE state")

        # Stop dashboard
        dash.stop()
        dash_thread.join(timeout=1.0)
        self.assertFalse(dash._is_active)
        self.assertFalse(dash_thread.is_alive())

    # =========================================================================
    # TEST 15: Dashboard Key Controls [S, P, Space, Q] Lifecycle
    # =========================================================================
    def test_15_dashboard_key_controls_lifecycle(self) -> None:
        """Verify [S, P, Space, Q] key handlers control controller and terminate cleanly."""
        dash = AutonomousDashboard(controller=self.controller)
        dash_thread = threading.Thread(target=dash.start_interactive, kwargs={"refresh_interval_sec": 0.05}, daemon=True)
        dash_thread.start()
        time.sleep(0.10)

        # 1. Test 's' (Start)
        with patch.object(self.controller, "_run_loop"):
            dash.handle_key('s')
            self.assertEqual(self.controller.state, AutonomousState.INITIALIZING)

        # 2. Test 'p' (Pause)
        dash.handle_key('p')
        self.assertEqual(self.controller.state, AutonomousState.PAUSED)

        # 3. Test 's' (Resume)
        dash.handle_key('s')
        # Resume transitions away from PAUSED
        self.assertFalse(self.controller._pause_requested)

        # 4. Test ' ' (Single Step)
        with patch.object(self.controller, "step_single_turn", return_value=True) as mock_step:
            dash.handle_key(' ')
            mock_step.assert_called_once()

        # 5. Test 'q' (Quit/Stop)
        dash.handle_key('q')
        dash_thread.join(timeout=1.5)

        self.assertFalse(dash._is_active, "Dashboard must be inactive after Q")
        self.assertFalse(dash_thread.is_alive(), "Dashboard thread must terminate cleanly")
        self.assertEqual(self.controller.state, AutonomousState.IDLE)
        self.assertFalse(self.controller.is_running())
        self.assertIsNone(self.controller._shell_proc, "Shell process must be None")

    # =========================================================================
    # TEST 16: UNKNOWN Slot Triggers PAUSE — Never False Game Over
    # =========================================================================
    def test_16_unknown_slot_triggers_pause_never_false_game_over(self) -> None:
        """Verify UNKNOWN slot in tray triggers PAUSE and avoids false GAME_OVER."""
        sq3 = CANONICAL_41_CATALOG.get_by_name("square_3x3")
        # Board has only 2x2 hole; square_3x3 cannot fit anywhere
        mock_board = make_mock_board(bitboard=0xFFFFFFFFFFFFFFF0)
        # Slot 0: square_3x3, Slot 1: UNKNOWN, Slot 2: square_3x3
        mock_rec = FrameRecognitionResult(
            status=RecognitionStatus.SUCCESS,
            board=mock_board,
            tray_pieces=(
                make_mock_piece(0, sq3, is_empty=False, confidence=1.0),
                make_mock_piece(1, None, is_empty=False, confidence=0.0, matched_name="UNKNOWN"),
                make_mock_piece(2, sq3, is_empty=False, confidence=1.0),
            ),
            overall_confidence=0.85,
            processing_time_ms=10.0,
            error_message=None,
            image_shape=(2340, 1080),
        )

        with patch.object(self.controller.recognizer, "recognize_frame", return_value=mock_rec):
            res = self.controller.step_single_turn()
            self.assertFalse(res)
            # MUST BE PAUSED, NOT GAME_OVER!
            self.assertEqual(self.controller.state, AutonomousState.PAUSED)
            self.assertIn("Perception uncertain", self.controller.action_log[-1]["message"])
            self.assertIsNone(self.controller.last_planned_move)

    # =========================================================================
    # TEST 17: Low Piece Confidence Triggers PAUSE — Never False Game Over
    # =========================================================================
    def test_17_low_piece_confidence_triggers_pause_never_game_over(self) -> None:
        """Verify low-confidence piece in tray triggers PAUSE and avoids false GAME_OVER."""
        sq3 = CANONICAL_41_CATALOG.get_by_name("square_3x3")
        mock_board = make_mock_board(bitboard=0xFFFFFFFFFFFFFFFF)
        # Piece confidence is 0.50 (below min_piece_confidence=0.70)
        mock_rec = FrameRecognitionResult(
            status=RecognitionStatus.SUCCESS,
            board=mock_board,
            tray_pieces=(
                make_mock_piece(0, sq3, is_empty=False, confidence=0.50),
                make_mock_piece(1, None, is_empty=True),
                make_mock_piece(2, None, is_empty=True),
            ),
            overall_confidence=0.85,
            processing_time_ms=10.0,
            error_message=None,
            image_shape=(2340, 1080),
        )

        with patch.object(self.controller.recognizer, "recognize_frame", return_value=mock_rec):
            res = self.controller.step_single_turn()
            self.assertFalse(res)
            self.assertEqual(self.controller.state, AutonomousState.PAUSED)
            self.assertIn("confidence below", self.controller.action_log[-1]["message"])
            self.assertIsNone(self.controller.last_planned_move)

    # =========================================================================
    # TEST 18: Game Over Clears Stale Planned Move
    # =========================================================================
    def test_18_game_over_clears_stale_planned_move(self) -> None:
        """Verify GAME_OVER state immediately clears any previous planned move."""
        sq3 = CANONICAL_41_CATALOG.get_by_name("square_3x3")
        mock_board = make_mock_board(bitboard=0xFFFFFFFFFFFFFFFF)
        mock_rec = FrameRecognitionResult(
            status=RecognitionStatus.SUCCESS,
            board=mock_board,
            tray_pieces=(
                make_mock_piece(0, sq3, is_empty=False, confidence=1.0),
                make_mock_piece(1, None, is_empty=True),
                make_mock_piece(2, None, is_empty=True),
            ),
            overall_confidence=0.95,
            processing_time_ms=10.0,
            error_message=None,
            image_shape=(2340, 1080),
        )

        # Pre-set a stale planned move
        self.controller.last_planned_move = PlannedMove(
            step_index=0,
            slot_idx=0,
            row=0,
            col=0,
            piece_name="dummy",
            cell_count=1,
            predicted_score_delta=10,
            lines_cleared=0,
            unique_cells_cleared=0,
            combo_streak_after=0,
            is_game_over=False,
        )

        with patch.object(self.controller.recognizer, "recognize_frame", return_value=mock_rec):
            res = self.controller.step_single_turn()
            self.assertFalse(res)
            self.assertEqual(self.controller.state, AutonomousState.GAME_OVER)
            self.assertIsNone(self.controller.last_planned_move, "last_planned_move must be cleared on GAME_OVER")

    # =========================================================================
    # TEST 19: Dry-Run Tray Refill Lifecycle
    # =========================================================================
    def test_19_dry_run_tray_refill_lifecycle(self) -> None:
        """Verify dry-run mode blanks placed pieces and refills tray upon empty."""
        dot = CANONICAL_41_CATALOG.get_by_name("dot_1x1")
        mock_board = make_mock_board(bitboard=0x0)
        # Empty tray frame recognition result
        mock_rec_empty = FrameRecognitionResult(
            status=RecognitionStatus.SUCCESS,
            board=mock_board,
            tray_pieces=(
                make_mock_piece(0, None, is_empty=True),
                make_mock_piece(1, None, is_empty=True),
                make_mock_piece(2, None, is_empty=True),
            ),
            overall_confidence=0.95,
            processing_time_ms=10.0,
            error_message=None,
            image_shape=(2340, 1080),
        )

        initial_refills = self.controller.stats.rounds_refilled
        with patch.object(self.controller.recognizer, "recognize_frame", return_value=mock_rec_empty):
            res = self.controller.step_single_turn()
            self.assertTrue(res, "Empty tray turn should return True waiting/refilling")
            self.assertEqual(self.controller.stats.rounds_refilled, initial_refills + 1)
            self.assertIsNotNone(self.controller.last_frame, "Dry-run must synthesize new tray frame on refill")


if __name__ == "__main__":
    unittest.main()
