"""Unit and Integration Tests for Action Engine (Phase 3.4).

Tests TouchController, CoordinateMapper, ActionVerifier, and AutoPlayer.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from action_engine.action_verifier import (
    ActionVerifier,
    CellMismatch,
    VerificationResult,
    VerificationStatus,
)
from action_engine.auto_player import (
    AutoPlayConfig,
    AutoPlayer,
    PlayMode,
)
from action_engine.coordinate_mapper import (
    CoordinateMapper,
    CoordinateMapperError,
    DragTrajectory,
)
from action_engine.touch_controller import TouchController, TouchControllerError
from ai_engine.decision import DecisionResult, DecisionStatus, PlannedMove
from computer_vision.models import (
    BoardRecognitionResult,
    CellInfo,
    CellState,
    FrameRecognitionResult,
    PieceRecognitionResult,
    RecognitionStatus,
)
from game_engine.board import coords_to_bitboard
from game_engine.pieces import CANONICAL_41_CATALOG
from game_engine.simulator import GameSimulator, StepResult
from game_engine.state import GameState


# ---------------------------------------------------------------------------
# Test Helpers: Synthetic Recognition Results
# ---------------------------------------------------------------------------

def create_mock_cell(r: int, c: int, occupied: bool = False) -> CellInfo:
    return CellInfo(
        row=r,
        col=c,
        state=CellState.OCCUPIED if occupied else CellState.EMPTY,
        confidence=0.99,
        center_x=int(54 + (c + 0.5) * 121.5),
        center_y=int(548 + (r + 0.5) * 121.5),
        width=121,
        height=121,
        mean_gray=180.0 if occupied else 80.0,
        mean_sat=10.0 if occupied else 150.0,
    )


def create_synthetic_frame_result(
    bitboard: int = 0,
    tray_piece_names: tuple[str, str, str] = ("line_1x5_h", "l_3x2_v4", "line_1x4_v"),
) -> FrameRecognitionResult:
    # 8x8 cells
    cells = []
    matrix = []
    for r in range(8):
        row_cells = []
        row_mat = []
        for c in range(8):
            is_occ = bool(bitboard & (1 << (r * 8 + c)))
            row_cells.append(create_mock_cell(r, c, is_occ))
            row_mat.append(1 if is_occ else 0)
        cells.append(tuple(row_cells))
        matrix.append(tuple(row_mat))

    board_res = BoardRecognitionResult(
        board_bbox=(54, 548, 972, 972),
        grid_rows=8,
        grid_cols=8,
        cell_size=121.5,
        cells=tuple(cells),
        occupancy_matrix=tuple(matrix),
        bitboard=bitboard,
        board_confidence=0.98,
        grid_confidence=0.99,
        cell_confidence=0.98,
    )

    tray_results = []
    slot_bboxes = [
        (90, 1761, 279, 65),
        (486, 1708, 118, 172),
        (828, 1681, 64, 226),
    ]
    for idx, name in enumerate(tray_piece_names):
        if name == "EMPTY":
            tray_results.append(PieceRecognitionResult(
                slot_idx=idx,
                is_empty=True,
                matched_name=None,
                piece=None,
                cells=(),
                height=0,
                width=0,
                cell_count=0,
                bbox_in_screen=(0, 0, 0, 0),
                confidence=1.0,
                status_message="Slot is empty",
            ))
        else:
            piece = CANONICAL_41_CATALOG.get_by_name(name)
            assert piece is not None
            tray_results.append(PieceRecognitionResult(
                slot_idx=idx,
                is_empty=False,
                matched_name=name,
                piece=piece,
                cells=piece.cells,
                height=piece.height,
                width=piece.width,
                cell_count=piece.cell_count,
                bbox_in_screen=slot_bboxes[idx],
                confidence=1.0,
                status_message=f"Matched: {name}",
            ))

    return FrameRecognitionResult(
        status=RecognitionStatus.SUCCESS,
        board=board_res,
        tray_pieces=tuple(tray_results),
        overall_confidence=0.98,
        processing_time_ms=20.0,
        image_shape=(2340, 1080),
    )


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class TestTouchController(unittest.TestCase):
    """Tests for low-level ADB touch actuation."""

    def test_dry_run_tap(self):
        ctrl = TouchController(dry_run=True)
        res = ctrl.tap(500, 1000)
        self.assertTrue(res)
        self.assertEqual(len(ctrl.history), 1)
        rec = ctrl.history[0]
        self.assertEqual(rec.action_type, "TAP")
        self.assertEqual(rec.params, {"x": 500, "y": 1000})
        self.assertTrue(rec.success)

    def test_dry_run_swipe(self):
        ctrl = TouchController(dry_run=True)
        res = ctrl.swipe(100, 200, 300, 400, duration_ms=450)
        self.assertTrue(res)
        self.assertEqual(len(ctrl.history), 1)
        rec = ctrl.history[0]
        self.assertEqual(rec.action_type, "SWIPE")
        self.assertEqual(rec.params["duration_ms"], 450)

    def test_command_line_formatting(self):
        ctrl = TouchController(adb_path="test_adb.exe", device_id="DEVICE_123", dry_run=True)
        cmd = ctrl._build_adb_cmd(["input", "tap", "10", "20"])
        self.assertEqual(cmd, ["test_adb.exe", "-s", "DEVICE_123", "shell", "input", "tap", "10", "20"])


class TestCoordinateMapper(unittest.TestCase):
    """Tests for physical touch coordinate mapping from vision calibration."""

    def setUp(self):
        self.mapper = CoordinateMapper(vertical_offset_ratio=1.65, default_drag_duration_ms=380)
        self.frame = create_synthetic_frame_result()

    def test_map_move_success(self):
        # Step 0: Slot 1 (l_3x2_v4, 3x2) -> row 2, col 6
        piece = CANONICAL_41_CATALOG.get_by_name("l_3x2_v4")
        self.assertIsNotNone(piece)
        move = PlannedMove(
            step_index=0,
            slot_idx=1,
            row=2,
            col=6,
            piece_name="l_3x2_v4",
            cell_count=piece.cell_count,
            predicted_score_delta=14,
            lines_cleared=1,
            unique_cells_cleared=8,
            combo_streak_after=1,
            is_game_over=False,
        )

        traj = self.mapper.map_move(move, self.frame)
        self.assertEqual(traj.slot_idx, 1)
        self.assertEqual(traj.piece_name, "l_3x2_v4")
        self.assertEqual(traj.target_row, 2)
        self.assertEqual(traj.target_col, 6)
        self.assertEqual(traj.duration_ms, 380)

        # Slot 1 bbox is (486, 1708, 118, 172)
        # Center x = 486 + 59 = 545
        # Center y = 1708 + 86 = 1794
        self.assertEqual(traj.start_x, 545)
        self.assertEqual(traj.start_y, 1794)

        # Destination coordinates:
        # Board bbox: (54, 548), cell_size: 121.5
        # Piece width = 2, height = 3
        # center_grid_x = 54 + (6 + 2/2.0) * 121.5 = 54 + 7 * 121.5 = 904.5 -> 904 or 905
        # center_grid_y = 548 + (2 + 3/2.0) * 121.5 = 548 + 3.5 * 121.5 = 973.25
        # vert_offset = 1.65 * 121.5 = 200.475
        # end_y = 973.25 + 200.475 = 1173.725 -> 1174
        self.assertAlmostEqual(traj.end_x, 904, delta=2)
        self.assertAlmostEqual(traj.end_y, 1174, delta=2)

    def test_map_move_illegal_occupied_target_rejected(self):
        # Create board with cell (2, 6) occupied
        occupied_bitboard = 1 << (2 * 8 + 6)
        frame_blocked = create_synthetic_frame_result(bitboard=occupied_bitboard)
        piece = CANONICAL_41_CATALOG.get_by_name("l_3x2_v4")
        move = PlannedMove(
            step_index=0, slot_idx=1, row=2, col=6, piece_name="l_3x2_v4",
            cell_count=piece.cell_count, predicted_score_delta=14, lines_cleared=0,
            unique_cells_cleared=0, combo_streak_after=0, is_game_over=False
        )
        with self.assertRaises(CoordinateMapperError):
            self.mapper.map_move(move, frame_blocked)

    def test_map_move_invalid_slot(self):
        move = PlannedMove(
            step_index=0, slot_idx=99, row=0, col=0, piece_name="foo",
            cell_count=1, predicted_score_delta=1, lines_cleared=0,
            unique_cells_cleared=0, combo_streak_after=0, is_game_over=False
        )
        with self.assertRaises(CoordinateMapperError):
            self.mapper.map_move(move, self.frame)

    def test_map_move_empty_slot(self):
        frame_with_empty = create_synthetic_frame_result(tray_piece_names=("EMPTY", "l_3x2_v4", "line_1x4_v"))
        move = PlannedMove(
            step_index=0, slot_idx=0, row=0, col=0, piece_name="line_1x5_h",
            cell_count=5, predicted_score_delta=5, lines_cleared=0,
            unique_cells_cleared=0, combo_streak_after=0, is_game_over=False
        )
        with self.assertRaises(CoordinateMapperError):
            self.mapper.map_move(move, frame_with_empty)


class TestActionVerifier(unittest.TestCase):
    """Tests for post-move verification logic."""

    def test_verify_success(self):
        piece = CANONICAL_41_CATALOG.get_by_name("dot_1x1")
        sim = GameSimulator()
        init_state = GameState.create_initial(initial_tray=[piece, None, None])
        step_res = sim.apply_move(init_state, slot_idx=0, row=0, col=0)

        # Mock recognizer and capture
        mock_rec = MagicMock()
        mock_rec.recognize_frame.return_value = create_synthetic_frame_result(
            bitboard=step_res.state_after.board,
            tray_piece_names=("EMPTY", "l_3x2_v4", "line_1x4_v"),
        )
        mock_cap = MagicMock()
        mock_cap.capture_from_adb.return_value = MagicMock()

        verifier = ActionVerifier(
            recognizer=mock_rec,
            capture=mock_cap,
            post_move_settle_sec=0.0,
            max_settle_retries=2,
        )

        res = verifier.verify_move(step_res)
        self.assertTrue(res.is_success)
        self.assertEqual(res.status, VerificationStatus.VERIFIED)
        self.assertEqual(res.mismatch_count, 0)
        self.assertTrue(res.tray_slot_cleared)

    def test_verify_mismatch(self):
        piece = CANONICAL_41_CATALOG.get_by_name("dot_1x1")
        p2 = CANONICAL_41_CATALOG.get_by_name("line_1x5_h")
        p3 = CANONICAL_41_CATALOG.get_by_name("line_1x4_v")
        sim = GameSimulator()
        init_state = GameState.create_initial(initial_tray=[piece, p2, p3])
        step_res = sim.apply_move(init_state, slot_idx=0, row=0, col=0)

        mock_rec = MagicMock()
        # Return wrong bitboard (0x0 instead of 0x1) and slot 0 still has "dot_1x1"
        mock_rec.recognize_frame.return_value = create_synthetic_frame_result(
            bitboard=0x0000000000000000,
            tray_piece_names=("dot_1x1", "l_3x2_v4", "line_1x4_v"),
        )
        mock_cap = MagicMock()
        mock_cap.capture_from_adb.return_value = MagicMock()

        verifier = ActionVerifier(
            recognizer=mock_rec,
            capture=mock_cap,
            post_move_settle_sec=0.0,
            max_settle_retries=1,
        )

        res = verifier.verify_move(step_res)
        self.assertFalse(res.is_success)
        # Piece still in tray slot 0 -> TRAY_REJECTED
        self.assertEqual(res.status, VerificationStatus.TRAY_REJECTED)
        self.assertGreater(res.mismatch_count, 0)

    def test_verify_low_confidence_returns_unknown(self):
        piece = CANONICAL_41_CATALOG.get_by_name("dot_1x1")
        sim = GameSimulator()
        init_state = GameState.create_initial(initial_tray=[piece, None, None])
        step_res = sim.apply_move(init_state, slot_idx=0, row=0, col=0)

        # Mock frame with LOW_CONFIDENCE
        mock_rec = MagicMock()
        mock_rec.recognize_frame.return_value = FrameRecognitionResult(
            status=RecognitionStatus.LOW_CONFIDENCE,
            board=None,
            tray_pieces=(),
            overall_confidence=0.45,
            processing_time_ms=15.0,
            error_message="Uncertain cell state detected",
        )
        mock_cap = MagicMock()
        mock_cap.capture_from_adb.return_value = MagicMock()

        verifier = ActionVerifier(
            recognizer=mock_rec,
            capture=mock_cap,
            post_move_settle_sec=0.0,
            max_settle_retries=1,
        )

        res = verifier.verify_move(step_res)
        self.assertFalse(res.is_success)
        self.assertEqual(res.status, VerificationStatus.UNKNOWN)


class TestAutoPlayer(unittest.TestCase):
    """Tests for AutoPlayer state transitions and loop behavior."""

    def test_mode_transitions(self):
        config = AutoPlayConfig(mode=PlayMode.PAUSE, dry_run=True)
        player = AutoPlayer(config=config)
        self.assertEqual(player.mode, PlayMode.PAUSE)

        player.set_mode(PlayMode.ASSIST, "Switching to assist")
        self.assertEqual(player.mode, PlayMode.ASSIST)

        player.set_mode(PlayMode.AUTO, "Switching to auto")
        self.assertEqual(player.mode, PlayMode.AUTO)

    def test_emergency_stop(self):
        config = AutoPlayConfig(mode=PlayMode.AUTO, dry_run=True)
        player = AutoPlayer(config=config)
        self.assertEqual(player.mode, PlayMode.AUTO)

        player.trigger_emergency_stop("Test emergency stop")
        self.assertEqual(player.mode, PlayMode.PAUSE)
        self.assertTrue(player._emergency_stop)

        player.reset_emergency_stop()
        self.assertFalse(player._emergency_stop)


if __name__ == "__main__":
    unittest.main()
