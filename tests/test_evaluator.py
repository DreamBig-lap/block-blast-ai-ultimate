"""Unit Tests for ai_engine.evaluator.

Verifies:
- Accurate topological feature extraction (empty cells, occupied cells).
- Trapped / isolated 1x1 hole detection via pure bitwise masks.
- Detection of nearly-full lines (6/8 and 7/8).
- Surface bumpiness / transition counting.
- Key shape fit detection (3x3 square, 2x2 square, 5x1 line).
- Dynamic step evaluation including line clears, combos, and game over.
"""

from __future__ import annotations

import unittest

from ai_engine.evaluator import (
    BoardEvaluator,
    EvaluationWeights,
    MASKS_SQUARE_3X3,
)
from game_engine.board import (
    COL_MASKS,
    FULL_BOARD_MASK,
    ROW_MASKS,
    coord_to_bit,
    popcount,
)


class TestBoardEvaluatorFeatures(unittest.TestCase):
    """Test feature extraction routines of BoardEvaluator."""

    def setUp(self) -> None:
        self.evaluator = BoardEvaluator()

    def test_empty_board_features(self):
        """Empty board must have 64 empty cells, 0 transitions, 0 holes, fits all shapes."""
        board = 0
        feats = self.evaluator.extract_features(board)
        self.assertEqual(feats.empty_cells, 64)
        self.assertEqual(feats.occupied_cells, 0)
        self.assertEqual(feats.nearly_full_7, 0)
        self.assertEqual(feats.nearly_full_6, 0)
        self.assertEqual(feats.isolated_holes, 0)
        self.assertEqual(feats.transitions, 0)
        self.assertTrue(feats.fits_square_3x3)
        self.assertTrue(feats.fits_square_2x2)
        self.assertTrue(feats.fits_line_5)

    def test_full_board_features(self):
        """Full board has 0 empty cells, 0 holes, cannot fit any shapes."""
        board = FULL_BOARD_MASK
        feats = self.evaluator.extract_features(board)
        self.assertEqual(feats.empty_cells, 0)
        self.assertEqual(feats.occupied_cells, 64)
        self.assertEqual(feats.isolated_holes, 0)  # Holes are empty cells; full board has 0 empty cells
        self.assertEqual(feats.transitions, 0)
        self.assertFalse(feats.fits_square_3x3)
        self.assertFalse(feats.fits_square_2x2)
        self.assertFalse(feats.fits_line_5)

    def test_nearly_full_lines_detection(self):
        """Detect rows and columns with 6 and 7 occupied cells."""
        # Row 1 has 7 cells full (col 0 empty)
        row1_7 = ROW_MASKS[1] & ~coord_to_bit(1, 0)
        # Row 3 has 6 cells full (cols 0, 1 empty)
        row3_6 = ROW_MASKS[3] & ~(coord_to_bit(3, 0) | coord_to_bit(3, 1))
        # Col 5 has 7 cells full (row 0 empty)
        col5_7 = COL_MASKS[5] & ~coord_to_bit(0, 5)

        board = row1_7 | row3_6 | col5_7
        feats = self.evaluator.extract_features(board)
        self.assertGreaterEqual(feats.nearly_full_7, 2)
        self.assertGreaterEqual(feats.nearly_full_6, 1)

    def test_isolated_hole_detection(self):
        """Trapped 1x1 empty cell surrounded by 4 occupied neighbors or walls."""
        # Case 1: Interior trapped hole at (3, 3)
        # Neighbors: (2, 3), (4, 3), (3, 2), (3, 4)
        interior_hole_board = (
            coord_to_bit(2, 3) |
            coord_to_bit(4, 3) |
            coord_to_bit(3, 2) |
            coord_to_bit(3, 4)
        )
        feats1 = self.evaluator.extract_features(interior_hole_board)
        self.assertEqual(feats1.isolated_holes, 1)

        # Case 2: Corner trapped hole at (0, 0)
        # Neighbors: (0, 1), (1, 0) - top and left are walls!
        corner_hole_board = coord_to_bit(0, 1) | coord_to_bit(1, 0)
        feats2 = self.evaluator.extract_features(corner_hole_board)
        self.assertEqual(feats2.isolated_holes, 1)

        # Case 3: Open cell (not isolated): (3, 3) has only 3 neighbors filled, 1 open
        not_hole_board = coord_to_bit(2, 3) | coord_to_bit(4, 3) | coord_to_bit(3, 2)
        feats3 = self.evaluator.extract_features(not_hole_board)
        self.assertEqual(feats3.isolated_holes, 0)

    def test_transitions_counting(self):
        """Transitions measure surface bumpiness / fragmentation."""
        # A single 2x2 square: compact
        compact_board = (
            coord_to_bit(1, 1) | coord_to_bit(1, 2) |
            coord_to_bit(2, 1) | coord_to_bit(2, 2)
        )
        feats_compact = self.evaluator.extract_features(compact_board)

        # 4 scattered cells: high fragmentation
        scattered_board = (
            coord_to_bit(0, 0) | coord_to_bit(2, 2) |
            coord_to_bit(4, 4) | coord_to_bit(6, 6)
        )
        feats_scattered = self.evaluator.extract_features(scattered_board)

        # Scattered cells must produce more interior transitions than a compact cluster
        self.assertGreater(feats_scattered.transitions, feats_compact.transitions)

    def test_square_3x3_fit_capacity(self):
        """Verify fits_square_3x3 detects when board can or cannot fit a 3x3 square."""
        # Completely fill all rows except leaving row 7 empty -> 1 row cannot fit 3x3
        blocked_board = FULL_BOARD_MASK & ~ROW_MASKS[7]
        feats_blocked = self.evaluator.extract_features(blocked_board)
        self.assertFalse(feats_blocked.fits_square_3x3)

        # Leave a 3x3 block open at (0, 0) .. (2, 2)
        open_3x3_mask = MASKS_SQUARE_3X3[0]
        board_with_3x3_slot = FULL_BOARD_MASK & ~open_3x3_mask
        feats_open = self.evaluator.extract_features(board_with_3x3_slot)
        self.assertTrue(feats_open.fits_square_3x3)


class TestBoardEvaluatorScoring(unittest.TestCase):
    """Test static board and dynamic step evaluation logic."""

    def setUp(self) -> None:
        self.evaluator = BoardEvaluator()

    def test_clean_board_evaluates_higher_than_clogged_board(self):
        """An open, clean board must have higher static fitness than a clogged board."""
        clean_board = coord_to_bit(0, 0) | coord_to_bit(0, 1)
        clogged_board = FULL_BOARD_MASK & ~(coord_to_bit(0, 0) | coord_to_bit(7, 7))

        eval_clean = self.evaluator.evaluate_board(clean_board)
        eval_clogged = self.evaluator.evaluate_board(clogged_board)
        self.assertGreater(eval_clean, eval_clogged)

    def test_game_over_step_receives_terminal_penalty(self):
        """A move resulting in Game Over must receive a large negative penalty."""
        eval_over = self.evaluator.evaluate_step(
            board_after=0,
            lines_cleared=0,
            unique_cells_cleared=0,
            combo_streak=0,
            score_delta=0,
            is_game_over=True,
        )
        self.assertLessEqual(eval_over, -1_000_000.0)

    def test_line_clear_rewards_in_step_eval(self):
        """Clearing lines increases step evaluation score."""
        eval_no_clear = self.evaluator.evaluate_step(
            board_after=ROW_MASKS[0],
            lines_cleared=0,
            unique_cells_cleared=0,
            combo_streak=0,
            score_delta=5,
            is_game_over=False,
        )
        eval_with_clear = self.evaluator.evaluate_step(
            board_after=0,
            lines_cleared=1,
            unique_cells_cleared=8,
            combo_streak=1,
            score_delta=18,
            is_game_over=False,
        )
        self.assertGreater(eval_with_clear, eval_no_clear)

    def test_custom_evaluation_weights(self):
        """Custom weights override defaults correctly."""
        custom_weights = EvaluationWeights(empty_cells=100.0)
        custom_evaluator = BoardEvaluator(weights=custom_weights)
        score = custom_evaluator.evaluate_board(0)  # 64 empty cells * 100.0 = 6400+
        self.assertGreater(score, 6400.0)


if __name__ == "__main__":
    unittest.main()
