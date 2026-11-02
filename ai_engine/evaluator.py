"""Board Evaluation and Feature Extraction for BLOCK BLAST AI ULTIMATE.

CRITICAL DESIGN PRINCIPLES:
- Evaluates 8x8 bitboard states on long-term survival, piece placement headroom,
  clear opportunity, line congestion, and structural fragmentation.
- Pure-Python bitboard bitwise operations (< 2 us per state evaluation).
- Configurable weights via EvaluationWeights dataclass.
- Balances immediate score gain against catastrophic board clogging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from game_engine.board import (
    BOARD_SIZE,
    COL_MASKS,
    FULL_BOARD_MASK,
    ROW_MASKS,
    popcount,
)
from game_engine.pieces import CANONICAL_41_CATALOG, Piece


# ---------------------------------------------------------------------------
# Precomputed placement masks for key survival-critical shapes
# ---------------------------------------------------------------------------

def _precompute_shape_masks(rows: int, cols: int, coords: list[tuple[int, int]]) -> tuple[int, ...]:
    """Precompute all valid 8x8 bitboard masks for a given polyomino coordinate shape."""
    masks: list[int] = []
    max_r = BOARD_SIZE - rows + 1
    max_c = BOARD_SIZE - cols + 1
    for r in range(max_r):
        for c in range(max_c):
            mask = 0
            for dr, dc in coords:
                bit_idx = (r + dr) * 8 + (c + dc)
                mask |= (1 << bit_idx)
            masks.append(mask)
    return tuple(masks)


# 3x3 square: the most critical game-over shape in Block Blast
MASKS_SQUARE_3X3: tuple[int, ...] = _precompute_shape_masks(
    3, 3, [(r, c) for r in range(3) for c in range(3)]
)

# 2x2 square: staple medium piece
MASKS_SQUARE_2X2: tuple[int, ...] = _precompute_shape_masks(
    2, 2, [(r, c) for r in range(2) for c in range(2)]
)

# 5x1 and 1x5 straight lines: large linear pieces
MASKS_LINE_5_H: tuple[int, ...] = _precompute_shape_masks(1, 5, [(0, c) for c in range(5)])
MASKS_LINE_5_V: tuple[int, ...] = _precompute_shape_masks(5, 1, [(r, 0) for r in range(5)])


# ---------------------------------------------------------------------------
# Configurable Evaluation Weights
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EvaluationWeights:
    """Configurable weights for board state evaluation.

    Positive weights reward desirable states; negative weights penalize hazardous states.
    """

    # Headroom & clearing
    empty_cells: float = 3.5             # Points per open cell
    lines_cleared: float = 40.0          # Immediate reward for clearing lines
    multi_line_bonus: float = 25.0       # Extra reward when 2+ lines clear together
    cleared_unique_cells: float = 1.5    # Reward per unique cell wiped from board

    # Preparation (congestion vs readiness)
    nearly_full_7: float = 10.0          # Lines with 7/8 cells full (1 cell from clear!)
    nearly_full_6: float = 3.0           # Lines with 6/8 cells full (2 cells from clear)

    # Topological quality & fragmentation
    isolated_holes: float = -35.0        # Trapped 1x1 empty pockets with no exit
    transitions: float = -1.2            # Surface bumpiness / fragmented island borders

    # Critical survival capacities
    fit_square_3x3_bonus: float = 25.0   # Bonus if board can accommodate a 3x3 square
    no_square_3x3_penalty: float = -40.0 # Penalty if board CANNOT fit a 3x3 square
    fit_square_2x2_bonus: float = 15.0   # Bonus if board can accommodate a 2x2 square
    no_square_2x2_penalty: float = -20.0 # Penalty if board CANNOT fit a 2x2 square
    fit_line_5_bonus: float = 12.0       # Bonus if board can fit 5-cell straight lines

    # Combo and score progress
    combo_multiplier: float = 12.0       # Bonus for maintaining combo streak
    score_delta: float = 0.5             # Weight of immediate point gain in evaluation


# ---------------------------------------------------------------------------
# Feature Extraction Container
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BoardFeatures:
    """Extracted structural and topological features of an 8x8 bitboard."""

    empty_cells: int
    occupied_cells: int
    nearly_full_7: int
    nearly_full_6: int
    isolated_holes: int
    transitions: int
    fits_square_3x3: bool
    fits_square_2x2: bool
    fits_line_5: bool


# ---------------------------------------------------------------------------
# High-Performance Board Evaluator
# ---------------------------------------------------------------------------

class BoardEvaluator:
    """High-performance heuristic evaluator for Block Blast game boards."""

    def __init__(self, weights: Optional[EvaluationWeights] = None) -> None:
        self.weights = weights or EvaluationWeights()

    def extract_features(self, board: int) -> BoardFeatures:
        """Extract all topological and structural features from a 64-bit board."""
        occ = popcount(board)
        empty = 64 - occ

        # 1. Nearly full rows and columns (6 or 7 occupied cells)
        nearly_7 = 0
        nearly_6 = 0

        for r in range(BOARD_SIZE):
            c_count = (board & ROW_MASKS[r]).bit_count()
            if c_count == 7:
                nearly_7 += 1
            elif c_count == 6:
                nearly_6 += 1

        for c in range(BOARD_SIZE):
            r_count = (board & COL_MASKS[c]).bit_count()
            if r_count == 7:
                nearly_7 += 1
            elif r_count == 6:
                nearly_6 += 1

        # 2. Trapped 1x1 isolated holes
        # An empty cell surrounded on all 4 sides by occupied cells or board boundaries
        empty_mask = FULL_BOARD_MASK & ~board

        # Up neighbor occupied (or row 0 boundary)
        up_occ = (board << 8) | ROW_MASKS[0]
        # Down neighbor occupied (or row 7 boundary)
        down_occ = (board >> 8) | ROW_MASKS[7]
        # Left neighbor occupied (or col 0 boundary)
        left_occ = ((board << 1) & ~COL_MASKS[0]) | COL_MASKS[0]
        # Right neighbor occupied (or col 7 boundary)
        right_occ = ((board >> 1) & ~COL_MASKS[7]) | COL_MASKS[7]

        trapped_mask = empty_mask & up_occ & down_occ & left_occ & right_occ
        isolated_holes = popcount(trapped_mask)

        # 3. Transitions (surface roughness & fragmentation)
        # Interior horizontal transitions (excluding wrap across col 7 to 0)
        h_diff = (board ^ (board >> 1)) & ~COL_MASKS[7]
        # Interior vertical transitions
        v_diff = (board ^ (board >> 8)) & ~ROW_MASKS[7]
        transitions = popcount(h_diff) + popcount(v_diff)

        # 4. Critical shape fit capacity (short-circuiting for speed)
        fits_3x3 = any((board & m) == 0 for m in MASKS_SQUARE_3X3)
        fits_2x2 = any((board & m) == 0 for m in MASKS_SQUARE_2X2)
        fits_l5 = (
            any((board & m) == 0 for m in MASKS_LINE_5_H) or
            any((board & m) == 0 for m in MASKS_LINE_5_V)
        )

        return BoardFeatures(
            empty_cells=empty,
            occupied_cells=occ,
            nearly_full_7=nearly_7,
            nearly_full_6=nearly_6,
            isolated_holes=isolated_holes,
            transitions=transitions,
            fits_square_3x3=fits_3x3,
            fits_square_2x2=fits_2x2,
            fits_line_5=fits_l5,
        )

    def evaluate_board(self, board: int) -> float:
        """Calculate the static structural fitness score of a board."""
        feats = self.extract_features(board)
        w = self.weights

        score = 0.0
        score += feats.empty_cells * w.empty_cells
        score += feats.nearly_full_7 * w.nearly_full_7
        score += feats.nearly_full_6 * w.nearly_full_6
        score += feats.isolated_holes * w.isolated_holes
        score += feats.transitions * w.transitions

        # Critical shape fit bonuses and penalties
        score += w.fit_square_3x3_bonus if feats.fits_square_3x3 else w.no_square_3x3_penalty
        score += w.fit_square_2x2_bonus if feats.fits_square_2x2 else w.no_square_2x2_penalty
        if feats.fits_line_5:
            score += w.fit_line_5_bonus

        return score

    def evaluate_step(
        self,
        board_after: int,
        lines_cleared: int,
        unique_cells_cleared: int,
        combo_streak: int,
        score_delta: int,
        is_game_over: bool = False,
    ) -> float:
        """Comprehensive evaluation of a move outcome, combining static board fitness

        with dynamic clearing rewards, combo multipliers, and game-over penalties.
        """
        if is_game_over:
            return -1_000_000.0  # Terminal catastrophic state

        w = self.weights
        static_score = self.evaluate_board(board_after)

        dynamic_score = 0.0
        # Line clearing rewards
        if lines_cleared > 0:
            dynamic_score += lines_cleared * w.lines_cleared
            if lines_cleared >= 2:
                dynamic_score += (lines_cleared - 1) * w.multi_line_bonus
            dynamic_score += unique_cells_cleared * w.cleared_unique_cells

        # Combo maintenance
        dynamic_score += combo_streak * w.combo_multiplier

        # Score delta
        dynamic_score += score_delta * w.score_delta

        return static_score + dynamic_score
