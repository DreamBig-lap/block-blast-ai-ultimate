"""Unit tests for Scoring and Combo Progression (game_engine/scoring.py)."""

import unittest

from game_engine.rules import DEFAULT_RULES, GameRules
from game_engine.scoring import ScoreCalculator


class TestScoreCalculator(unittest.TestCase):
    """Test score calculation, simultaneous clears, and combo grace windows."""

    def setUp(self):
        self.rules = DEFAULT_RULES
        self.calc = ScoreCalculator(self.rules)

    def test_non_clearing_placement_score(self):
        """Placing a piece without clearing lines awards only base cell points."""
        breakdown = self.calc.compute_move_score(
            cells_placed=4,
            lines_cleared=0,
            cleared_rows_count=0,
            cleared_cols_count=0,
            unique_cells_cleared=0,
            streak_before=0,
            non_clearing_before=0,
            is_board_empty_after=False,
        )
        self.assertEqual(breakdown.cells_placed, 4)
        self.assertEqual(breakdown.cell_points, 4)
        self.assertEqual(breakdown.lines_cleared, 0)
        self.assertEqual(breakdown.line_points, 0)
        self.assertEqual(breakdown.combo_points, 0)
        self.assertEqual(breakdown.all_clear_points, 0)
        self.assertEqual(breakdown.total_delta_score, 4)
        self.assertEqual(breakdown.streak_after, 0)
        self.assertEqual(breakdown.non_clearing_after, 1)
        self.assertFalse(breakdown.combo_broken)

    def test_single_line_clear_and_streak_increment(self):
        """Clearing 1 line awards line points and increments combo streak."""
        breakdown = self.calc.compute_move_score(
            cells_placed=3,
            lines_cleared=1,
            cleared_rows_count=1,
            cleared_cols_count=0,
            unique_cells_cleared=8,
            streak_before=0,
            non_clearing_before=2,
            is_board_empty_after=False,
        )
        # 3 cells * 1 = 3 pts; 1 line * 10 = 10 pts; streak_before=0 -> combo_pts = 0
        self.assertEqual(breakdown.cell_points, 3)
        self.assertEqual(breakdown.line_points, 10)
        self.assertEqual(breakdown.combo_points, 0)
        self.assertEqual(breakdown.total_delta_score, 13)
        self.assertEqual(breakdown.streak_after, 1)
        # Non-clearing moves reset to 0 upon line clear!
        self.assertEqual(breakdown.non_clearing_after, 0)

    def test_simultaneous_multi_line_clear_with_active_combo(self):
        """Simultaneous 2-line clear with prior combo streak awards full multiplier."""
        breakdown = self.calc.compute_move_score(
            cells_placed=4,
            lines_cleared=2,
            cleared_rows_count=1,
            cleared_cols_count=1,
            unique_cells_cleared=15,  # Intersection handling
            streak_before=3,
            non_clearing_before=1,
            is_board_empty_after=False,
        )
        # cell_pts = 4 * 1 = 4
        # line_pts = 2 * 10 = 20
        # combo_pts = 2 lines * 10 multiplier * streak(3) = 60
        self.assertEqual(breakdown.cell_points, 4)
        self.assertEqual(breakdown.line_points, 20)
        self.assertEqual(breakdown.combo_points, 60)
        self.assertEqual(breakdown.total_delta_score, 84)
        self.assertEqual(breakdown.streak_after, 4)
        self.assertEqual(breakdown.non_clearing_after, 0)

    def test_android_combo_grace_window(self):
        """CRITICAL: Streak must persist for up to 3 non-clearing moves and reset on the 4th."""
        # Baseline streak of 5
        streak = 5

        # Move 1: NC = 0 -> 1 (within grace window of 3)
        b1 = self.calc.compute_move_score(2, 0, 0, 0, 0, streak_before=streak, non_clearing_before=0, is_board_empty_after=False)
        self.assertEqual(b1.streak_after, 5)
        self.assertEqual(b1.non_clearing_after, 1)
        self.assertFalse(b1.combo_broken)

        # Move 2: NC = 1 -> 2 (within grace window of 3)
        b2 = self.calc.compute_move_score(2, 0, 0, 0, 0, streak_before=b1.streak_after, non_clearing_before=b1.non_clearing_after, is_board_empty_after=False)
        self.assertEqual(b2.streak_after, 5)
        self.assertEqual(b2.non_clearing_after, 2)
        self.assertFalse(b2.combo_broken)

        # Move 3: NC = 2 -> 3 (last move in grace window)
        b3 = self.calc.compute_move_score(2, 0, 0, 0, 0, streak_before=b2.streak_after, non_clearing_before=b2.non_clearing_after, is_board_empty_after=False)
        self.assertEqual(b3.streak_after, 5)
        self.assertEqual(b3.non_clearing_after, 3)
        self.assertFalse(b3.combo_broken)

        # Move 4: NC = 3 -> 4 (grace window expired! Combo breaks)
        b4 = self.calc.compute_move_score(2, 0, 0, 0, 0, streak_before=b3.streak_after, non_clearing_before=b3.non_clearing_after, is_board_empty_after=False)
        self.assertEqual(b4.streak_after, 0)
        self.assertEqual(b4.non_clearing_after, 4)
        self.assertTrue(b4.combo_broken)

    def test_configurable_instant_reset_combo(self):
        """When combo_window_moves=0, streak resets on the very first non-clearing move."""
        strict_rules = self.rules.with_updates(combo_window_moves=0)
        strict_calc = ScoreCalculator(strict_rules)

        b = strict_calc.compute_move_score(3, 0, 0, 0, 0, streak_before=4, non_clearing_before=0, is_board_empty_after=False)
        self.assertEqual(b.streak_after, 0)
        self.assertTrue(b.combo_broken)

    def test_all_clear_bonus(self):
        """Clearing the entire board awards the all-clear bonus."""
        breakdown = self.calc.compute_move_score(
            cells_placed=1,
            lines_cleared=1,
            cleared_rows_count=1,
            cleared_cols_count=0,
            unique_cells_cleared=8,
            streak_before=0,
            non_clearing_before=0,
            is_board_empty_after=True,
        )
        self.assertEqual(breakdown.all_clear_points, 300)
        self.assertEqual(breakdown.total_delta_score, 1 + 10 + 0 + 300)


if __name__ == "__main__":
    unittest.main()
