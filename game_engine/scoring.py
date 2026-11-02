"""Scoring and Combo Progression Engine for BLOCK BLAST AI ULTIMATE.

CRITICAL REQUIREMENTS:
- Scoring is strictly calculated via the active `GameRules` configuration.
- Handles simultaneous row and column clearing without double-counting intersection cells.
- Accurately tracks combo streak and non-clearing grace windows.
- All scoring elements are transparently recorded in a `ScoreBreakdown` audit record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from game_engine.rules import GameRules


# ---------------------------------------------------------------------------
# Score Breakdown Audit Record
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    """Comprehensive breakdown of points earned from a single move placement."""

    cells_placed: int
    cell_points: int
    lines_cleared: int
    cleared_rows_count: int
    cleared_cols_count: int
    unique_cells_cleared: int
    streak_before: int
    streak_after: int
    non_clearing_before: int
    non_clearing_after: int
    combo_broken: bool
    line_points: int
    combo_points: int
    all_clear_points: int
    total_delta_score: int


# ---------------------------------------------------------------------------
# Score Calculator
# ---------------------------------------------------------------------------

class ScoreCalculator:
    """Calculates score deltas and combo streak updates based on active GameRules."""

    def __init__(self, rules: GameRules) -> None:
        self.rules: Final[GameRules] = rules

    def compute_move_score(
        self,
        cells_placed: int,
        lines_cleared: int,
        cleared_rows_count: int,
        cleared_cols_count: int,
        unique_cells_cleared: int,
        streak_before: int,
        non_clearing_before: int,
        is_board_empty_after: bool,
    ) -> ScoreBreakdown:
        """Compute the score delta and combo state transition for a placement.

        Args:
            cells_placed: Number of unit squares occupied by the placed piece.
            lines_cleared: Total number of lines cleared (rows + cols).
            cleared_rows_count: Number of horizontal rows cleared.
            cleared_cols_count: Number of vertical columns cleared.
            unique_cells_cleared: Number of unique cells removed from the board.
            streak_before: Combo streak count before this placement.
            non_clearing_before: Consecutive non-clearing moves before this placement.
            is_board_empty_after: True if board has 0 occupied cells after this placement.

        Returns:
            ScoreBreakdown dataclass with all computed components.
        """
        # 1. Base placement points (points per cell placed)
        cell_pts = cells_placed * self.rules.points_per_cell

        # 2. Line clearing points & combo progression
        if lines_cleared > 0:
            # Clearing move: combo increments, non-clearing counter resets to 0
            streak_after = streak_before + 1
            non_clearing_after = 0
            combo_broken = False

            line_pts = lines_cleared * self.rules.points_per_line

            # Combo bonus: proportional to lines cleared and prior streak
            # Formula: lines_cleared * combo_multiplier_base * streak_before
            combo_pts = lines_cleared * self.rules.combo_multiplier_base * streak_before
        else:
            # Non-clearing move
            line_pts = 0
            combo_pts = 0
            non_clearing_after = non_clearing_before + 1

            # Check if grace window has expired
            if non_clearing_after > self.rules.combo_window_moves:
                streak_after = 0
                combo_broken = (streak_before > 0)
            else:
                # Streak is preserved within grace window!
                streak_after = streak_before
                combo_broken = False

        # 3. All-clear bonus (board wiped clean)
        all_clear_pts = self.rules.all_clear_bonus if is_board_empty_after and lines_cleared > 0 else 0

        # Total delta score
        total_delta = cell_pts + line_pts + combo_pts + all_clear_pts

        return ScoreBreakdown(
            cells_placed=cells_placed,
            cell_points=cell_pts,
            lines_cleared=lines_cleared,
            cleared_rows_count=cleared_rows_count,
            cleared_cols_count=cleared_cols_count,
            unique_cells_cleared=unique_cells_cleared,
            streak_before=streak_before,
            streak_after=streak_after,
            non_clearing_before=non_clearing_before,
            non_clearing_after=non_clearing_after,
            combo_broken=combo_broken,
            line_points=line_pts,
            combo_points=combo_pts,
            all_clear_points=all_clear_pts,
            total_delta_score=total_delta,
        )
