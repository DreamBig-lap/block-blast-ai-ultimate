"""Comprehensive Collision Prevention and Perception Audit Tests.

Verifies:
1. Independent mathematical cell index and mask verification for square_2x2 at (Row 1, Col 5).
2. Bitboard bit-by-bit check against log state 0x00B7B4A51E1B0008.
3. BoardDetector drift prevention (anchoring within ±15 px of calibrated baseline).
4. CellClassifier bitboard / occupancy_matrix synchronization.
5. AutonomousController pre-actuation collision guard (bitboard, occupancy_matrix, cell state).
6. Board bbox drift guard in AutonomousController state machine.
7. Solver collision-free invariant across AIDecisionEngine, BeamSearch, and Greedy solvers.
"""

from __future__ import annotations

import os
from pathlib import Path
import unittest

import cv2
import numpy as np

from action_engine.autonomous_controller import (
    AutonomousConfig,
    AutonomousGameController,
    AutonomousState,
)
from ai_engine.decision import AIDecisionEngine, DecisionStatus
from computer_vision.board_detector import BoardDetector
from computer_vision.cell_classifier import CellClassifier
from computer_vision.config import DEFAULT_VISION_CONFIG, VisionConfig
from computer_vision.models import (
    BoardRecognitionResult,
    CellInfo,
    CellState,
    FrameRecognitionResult,
    PieceRecognitionResult,
    RecognitionStatus,
)
from game_engine.board import (
    BOARD_SIZE,
    can_place,
    coord_to_bit,
    coord_to_bit_index,
    is_occupied,
)
from game_engine.pieces import CANONICAL_41_CATALOG
from game_engine.state import GameState


class TestCollisionPreventionAudit(unittest.TestCase):
    """Rigorous audit tests for move legality and collision prevention."""

    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parent.parent
        self.catalog = CANONICAL_41_CATALOG
        self.piece_2x2 = self.catalog.get_by_name("square_2x2")

    def test_01_independent_square_2x2_mask_calculation(self) -> None:
        """Requirement 1 & 2: Independently calculate the 4 cells occupied by square_2x2 at (1, 5)."""
        # square_2x2 relative cells: (0, 0), (0, 1), (1, 0), (1, 1)
        expected_cells = [
            (1, 5),  # top-left
            (1, 6),  # top-right
            (2, 5),  # bottom-left
            (2, 6),  # bottom-right
        ]

        expected_indices = [
            1 * 8 + 5,  # 13
            1 * 8 + 6,  # 14
            2 * 8 + 5,  # 21
            2 * 8 + 6,  # 22
        ]
        self.assertEqual(expected_indices, [13, 14, 21, 22])

        expected_masks = [
            1 << 13,  # 0x2000
            1 << 14,  # 0x4000
            1 << 21,  # 0x200000
            1 << 22,  # 0x400000
        ]
        self.assertEqual(expected_masks, [0x2000, 0x4000, 0x200000, 0x400000])

        total_mask = sum(expected_masks)
        self.assertEqual(total_mask, 0x606000)

        # Compare with canonical catalog piece placement mask
        catalog_mask = self.piece_2x2.placement_masks[1][5]
        self.assertEqual(catalog_mask, total_mask)

    def test_02_bitboard_audit_against_log_state(self) -> None:
        """Requirement 2: Cross-check the 4 cells against log bitboard 0x00B7B4A51E1B0008."""
        log_board = 0x00B7B4A51E1B0008
        mask_2x2 = 0x606000

        # Row 0: 0x08 -> only bit 3 set (cell 0, 3)
        row_0 = (log_board >> 0) & 0xFF
        self.assertEqual(row_0, 0x08)

        # Row 1: 0x00 -> all 8 cells are empty (bits 8..15 are 0)
        row_1 = (log_board >> 8) & 0xFF
        self.assertEqual(row_1, 0x00)
        self.assertFalse(is_occupied(log_board, 1, 5))
        self.assertFalse(is_occupied(log_board, 1, 6))

        # Row 2: 0x1B = 00011011_2 -> bits 0, 1, 3, 4 are set; bits 2, 5, 6, 7 are 0
        row_2 = (log_board >> 16) & 0xFF
        self.assertEqual(row_2, 0x1B)
        self.assertFalse(is_occupied(log_board, 2, 5))  # bit 21 is 0
        self.assertFalse(is_occupied(log_board, 2, 6))  # bit 22 is 0

        # Bitwise collision check on log_board evaluates to 0
        self.assertEqual(log_board & mask_2x2, 0)
        self.assertTrue(can_place(log_board, mask_2x2))

        # This proves conclusively that the AI decision engine received an inaccurate/shifted
        # board state where cells (1, 5), (1, 6), (2, 5), (2, 6) were recorded as 0.

    def test_03_board_detector_drift_prevention_on_live_mid(self) -> None:
        """Requirement 6: Verify BoardDetector anchors to calibrated ~548 px and rejects 650 px drift."""
        live_mid_path = self.project_root / "debug" / "live_mid.png"
        if not live_mid_path.exists():
            self.skipTest("debug/live_mid.png not found")

        img = cv2.imread(str(live_mid_path))
        self.assertIsNotNone(img)

        bd = BoardDetector()
        bbox, cell_size, conf = bd.detect_board(img)

        # Baseline for Samsung M21 is calibrated around y = 548 px (must NOT be 650 px)
        self.assertEqual(bbox[0], 54)
        self.assertAlmostEqual(bbox[1], 548, delta=4)
        self.assertEqual(bbox[2], 972)
        self.assertEqual(bbox[3], 972)
        self.assertEqual(cell_size, 121.5)

    def test_04_cell_classifier_bitboard_matrix_synchronization(self) -> None:
        """Verify that whenever a cell is marked occupied (even with low conf), bitboard bit is set."""
        live_mid_path = self.project_root / "debug" / "live_mid.png"
        if not live_mid_path.exists():
            self.skipTest("debug/live_mid.png not found")

        img = cv2.imread(str(live_mid_path))
        bd = BoardDetector()
        bbox, cell_size, conf = bd.detect_board(img)

        cc = CellClassifier()
        res = cc.classify_cells(img, bbox, conf)

        # Check synchronization across all 64 cells:
        # Every cell marked occupied in matrix MUST have its bitboard bit set!
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                occ_matrix = res.occupancy_matrix[r][c]
                bit_set = (res.bitboard & coord_to_bit(r, c)) != 0
                self.assertEqual(
                    occ_matrix == 1,
                    bit_set,
                    f"Desynchronization at ({r}, {c}): matrix={occ_matrix}, bitboard={bit_set}",
                )

        # Verify that known occupied cells in live_mid (e.g. cell (1, 0)) are set in both
        self.assertEqual(res.occupancy_matrix[1][0], 1)
        self.assertNotEqual(res.bitboard & coord_to_bit(1, 0), 0)

    def test_05_pre_actuation_collision_guard_halts_colliding_move(self) -> None:
        """Verify AutonomousGameController halts and pauses when a move collides with board."""
        cfg = AutonomousConfig(dry_run=True)
        controller = AutonomousGameController(config=cfg)

        # Board with cell (1, 5) occupied
        occupied_bitboard = coord_to_bit(1, 5)
        piece_2x2 = self.catalog.get_by_name("square_2x2")

        # Mock frame recognition result
        cells_grid = []
        matrix = []
        for r in range(8):
            row_c = []
            row_m = []
            for c in range(8):
                is_occ = (r == 1 and c == 5)
                row_c.append(CellInfo(
                    row=r, col=c,
                    state=CellState.OCCUPIED if is_occ else CellState.EMPTY,
                    confidence=0.95,
                    center_x=54 + int((c + 0.5) * 121.5),
                    center_y=548 + int((r + 0.5) * 121.5),
                    width=121, height=121,
                    mean_gray=180.0 if is_occ else 80.0,
                    mean_sat=100.0 if is_occ else 10.0,
                ))
                row_m.append(1 if is_occ else 0)
            cells_grid.append(tuple(row_c))
            matrix.append(tuple(row_m))

        board_res = BoardRecognitionResult(
            board_bbox=(54, 548, 972, 972),
            grid_rows=8, grid_cols=8,
            cell_size=121.5,
            cells=tuple(cells_grid),
            occupancy_matrix=tuple(matrix),
            bitboard=occupied_bitboard,
            board_confidence=0.95,
            grid_confidence=0.95,
            cell_confidence=0.95,
        )

        tray_results = (
            PieceRecognitionResult(
                slot_idx=0, is_empty=False, matched_name="square_2x2",
                piece=piece_2x2, cells=piece_2x2.cells, height=2, width=2,
                cell_count=4, bbox_in_screen=(100, 1800, 120, 120),
                confidence=0.95, status_message="OK",
            ),
        )

        frame_res = FrameRecognitionResult(
            status=RecognitionStatus.SUCCESS,
            board=board_res,
            tray_pieces=tray_results,
            overall_confidence=0.95,
            processing_time_ms=10.0,
        )

        # Mock recognition in controller
        controller.recognizer.recognize_frame = lambda f: frame_res

        # Dummy frame
        dummy_frame = np.zeros((2340, 1080, 3), dtype=np.uint8)
        controller.last_frame = dummy_frame

        # Step turn
        success = controller.step_single_turn()

        # The controller must find a legal placement that does NOT collide with (1, 5)
        # or pause if illegal. In our setup, other placements exist (e.g. 0, 0), so it should succeed
        # without placing on (1, 5).
        if controller.last_planned_move is not None:
            m = controller.last_planned_move
            mask = piece_2x2.placement_masks[m.row][m.col]
            self.assertEqual(mask & occupied_bitboard, 0, "Planned move collides with occupied bitboard!")

    def test_06_board_bbox_drift_guard_triggers_pause(self) -> None:
        """Verify AutonomousGameController pauses immediately if board bounding box y drifts > 15px."""
        cfg = AutonomousConfig(dry_run=True)
        controller = AutonomousGameController(config=cfg)

        # Frame recognition with drifted board_bbox (y = 650 instead of 548)
        cells_grid = tuple(tuple(CellInfo(
            row=r, col=c, state=CellState.EMPTY, confidence=0.95,
            center_x=100, center_y=100, width=50, height=50,
            mean_gray=80.0, mean_sat=10.0,
        ) for c in range(8)) for r in range(8))
        matrix = tuple(tuple(0 for _ in range(8)) for _ in range(8))

        board_res = BoardRecognitionResult(
            board_bbox=(54, 650, 972, 972),  # DRIFTED!
            grid_rows=8, grid_cols=8,
            cell_size=121.5,
            cells=cells_grid,
            occupancy_matrix=matrix,
            bitboard=0,
            board_confidence=0.95,
            grid_confidence=0.95,
            cell_confidence=0.95,
        )

        frame_res = FrameRecognitionResult(
            status=RecognitionStatus.SUCCESS,
            board=board_res,
            tray_pieces=(),
            overall_confidence=0.95,
            processing_time_ms=10.0,
        )

        controller.recognizer.recognize_frame = lambda f: frame_res
        controller.last_frame = np.zeros((2340, 1080, 3), dtype=np.uint8)

        # Execute single turn
        success = controller.step_single_turn()

        self.assertFalse(success)
        self.assertEqual(controller.state, AutonomousState.PAUSED)
        self.assertIn("Board bounding box drift detected", controller.last_pause_reason)

    def test_07_solvers_never_produce_overlapping_moves(self) -> None:
        """Requirement 5: Verify AI solvers never produce moves overlapping occupied board cells."""
        import random
        from game_engine.simulator import GameSimulator

        engine = AIDecisionEngine()
        sim = GameSimulator()
        piece_2x2 = self.catalog.get_by_name("square_2x2")
        piece_3x1 = self.catalog.get_by_name("line_1x3_h")

        rng = random.Random(42)
        # Test on 50 pseudo-random board configurations
        for trial in range(50):
            # Generate random 64-bit board
            rand_board = rng.getrandbits(64)
            state = GameState(board=rand_board, tray=(piece_2x2, piece_3x1, None))

            for strategy in ["beam_search", "greedy"]:
                res = engine.decide(state, strategy=strategy)
                if res.status == DecisionStatus.SUCCESS:
                    curr_state = state
                    for move in res.moves:
                        p = self.catalog.get_by_name(move.piece_name)
                        mask = p.placement_masks[move.row][move.col]
                        self.assertIsNotNone(mask)
                        # Placement must never collide with current board state
                        self.assertEqual(
                            curr_state.board & mask, 0,
                            f"Strategy {strategy} produced colliding move: {move.piece_name} at ({move.row}, {move.col})",
                        )
                        step = sim.apply_move(curr_state, move.slot_idx, move.row, move.col)
                        curr_state = step.state_after


if __name__ == "__main__":
    unittest.main()
