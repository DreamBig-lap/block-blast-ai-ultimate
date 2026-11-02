"""Deterministic Game Simulation Engine for BLOCK BLAST AI ULTIMATE.

CRITICAL REQUIREMENTS:
- Move legality verification (bounds, overlap, tray availability, game-over status).
- Simultaneous row and column clearing without double-clearing intersections.
- Accurate combo streak progression and grace window accounting.
- Game-over detection: game is over when no available piece in the tray fits anywhere on the board.
- 100% deterministic and reproducible given the same inputs and seeds.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final, Optional

from game_engine.board import (
    BOARD_SIZE,
    InvalidPlacementError,
    can_place,
    check_and_clear_lines,
    place,
)
from game_engine.pieces import (
    CANONICAL_41_CATALOG,
    Piece,
    PieceCatalog,
)
from game_engine.rules import GameRules
from game_engine.scoring import ScoreBreakdown, ScoreCalculator
from game_engine.state import GameState


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class IllegalMoveError(InvalidPlacementError):
    """Raised when an illegal move is executed in the simulator."""


# ---------------------------------------------------------------------------
# Step Result Data Structure
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class StepResult:
    """Detailed result of executing a single move in the game simulator."""

    state_before: GameState
    state_after: GameState
    slot_idx: int
    piece: Piece
    row: int
    col: int
    placement_mask: int
    score_breakdown: ScoreBreakdown
    lines_cleared: int
    unique_cells_cleared: int
    cleared_rows: tuple[int, ...]
    cleared_cols: tuple[int, ...]
    round_completed: bool
    is_game_over: bool


# ---------------------------------------------------------------------------
# Game Simulator
# ---------------------------------------------------------------------------

class GameSimulator:
    """Core simulation engine executing and validating moves deterministically."""

    def __init__(
        self,
        rules: Optional[GameRules] = None,
        catalog: Optional[PieceCatalog] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.rules: Final[GameRules] = rules or GameRules()
        self.catalog: Final[PieceCatalog] = catalog or CANONICAL_41_CATALOG
        self.score_calc: Final[ScoreCalculator] = ScoreCalculator(self.rules)
        self.rng: random.Random = rng or random.Random()

    def is_legal_move(self, state: GameState, slot_idx: int, row: int, col: int) -> bool:
        """Check if placing the piece from `slot_idx` at (row, col) is legal.

        Requirements:
        1. Game is not already over.
        2. `slot_idx` is within valid tray bounds and contains an unplaced piece.
        3. (row, col) allows the piece to fit within the 8x8 grid bounds.
        4. The placement mask does not collide with occupied cells on `state.board`.
        """
        if state.is_game_over:
            return False
        piece = state.get_piece(slot_idx)
        if piece is None:
            return False
        mask = piece.get_placement_mask(row, col)
        if mask is None:
            return False
        return can_place(state.board, mask)

    def get_legal_moves(self, state: GameState) -> list[tuple[int, int, int]]:
        """Return a list of all legal moves (slot_idx, row, col) for the current state."""
        if state.is_game_over:
            return []

        legal_moves: list[tuple[int, int, int]] = []
        for slot_idx, piece in enumerate(state.tray):
            if piece is None:
                continue
            max_r = BOARD_SIZE - piece.height + 1
            max_c = BOARD_SIZE - piece.width + 1
            for r in range(max_r):
                row_table = piece.placement_masks[r]
                for c in range(max_c):
                    mask = row_table[c]
                    if mask is not None and (state.board & mask) == 0:
                        legal_moves.append((slot_idx, r, c))
        return legal_moves

    def has_any_legal_move(self, state: GameState) -> bool:
        """Return True if at least one legal move exists for any piece in the tray."""
        if state.is_game_over:
            return False

        for piece in state.tray:
            if piece is None:
                continue
            max_r = BOARD_SIZE - piece.height + 1
            max_c = BOARD_SIZE - piece.width + 1
            for r in range(max_r):
                row_table = piece.placement_masks[r]
                for c in range(max_c):
                    mask = row_table[c]
                    if mask is not None and (state.board & mask) == 0:
                        return True
        return False

    def check_game_over(self, state: GameState) -> bool:
        """Evaluate if the game is over.

        Game over occurs if there is at least one active piece in the tray,
        but NO active piece can fit anywhere on the current board.
        If the tray is empty, the game is not over (pending next round deal).
        """
        if state.is_tray_empty():
            return False
        return not self.has_any_legal_move(state)

    def apply_move(
        self,
        state: GameState,
        slot_idx: int,
        row: int,
        col: int,
        auto_refill: bool = False,
    ) -> StepResult:
        """Execute a move and return the detailed StepResult.

        Args:
            state: The current GameState before the move.
            slot_idx: Tray index (0..2) of the piece to place.
            row: Board row (0..7) for piece top-left.
            col: Board col (0..7) for piece top-left.
            auto_refill: If True and the round completes, immediately deals a new hand.

        Raises:
            IllegalMoveError: If the move is out-of-bounds, overlaps, or slot is empty.
        """
        if state.is_game_over:
            raise IllegalMoveError("Cannot make a move in a Game Over state.")

        piece = state.get_piece(slot_idx)
        if piece is None:
            raise IllegalMoveError(f"Tray slot {slot_idx} is empty or invalid.")

        mask = piece.get_placement_mask(row, col)
        if mask is None:
            raise IllegalMoveError(
                f"Piece '{piece.name}' at ({row}, {col}) exceeds 8x8 board boundaries "
                f"(height={piece.height}, width={piece.width})."
            )

        if not can_place(state.board, mask):
            overlap = state.board & mask
            raise IllegalMoveError(
                f"Piece '{piece.name}' at ({row}, {col}) overlaps occupied cells (0x{overlap:X})."
            )

        # 1. Place piece onto bitboard
        board_placed = place(state.board, mask)

        # 2. Check and clear full rows and columns simultaneously
        (
            new_board,
            lines_cleared,
            unique_cells_cleared,
            full_r,
            full_c,
            _cleared_mask,
        ) = check_and_clear_lines(board_placed)

        # 3. Calculate score and combo updates
        is_empty = (new_board == 0)
        score_breakdown = self.score_calc.compute_move_score(
            cells_placed=piece.cell_count,
            lines_cleared=lines_cleared,
            cleared_rows_count=len(full_r),
            cleared_cols_count=len(full_c),
            unique_cells_cleared=unique_cells_cleared,
            streak_before=state.combo_streak,
            non_clearing_before=state.non_clearing_moves,
            is_board_empty_after=is_empty,
        )

        # 4. Consume placed piece from tray
        new_tray_list = list(state.tray)
        new_tray_list[slot_idx] = None
        new_tray = tuple(new_tray_list)

        round_completed = all(p is None for p in new_tray)
        new_round_number = state.round_number + (1 if round_completed else 0)

        # 5. Build preliminary post-move state
        provisional_state = state.clone_with(
            board=new_board,
            score=state.score + score_breakdown.total_delta_score,
            combo_streak=score_breakdown.streak_after,
            non_clearing_moves=score_breakdown.non_clearing_after,
            round_number=new_round_number,
            tray=new_tray,
            is_game_over=False,
        )

        # 6. Check game over or auto-refill
        if round_completed and auto_refill:
            final_state = self.refill_tray(provisional_state)
            is_over = final_state.is_game_over
        else:
            # Game is over if remaining unplaced pieces in tray cannot fit anywhere
            is_over = self.check_game_over(provisional_state)
            final_state = provisional_state.clone_with(is_game_over=is_over)

        return StepResult(
            state_before=state,
            state_after=final_state,
            slot_idx=slot_idx,
            piece=piece,
            row=row,
            col=col,
            placement_mask=mask,
            score_breakdown=score_breakdown,
            lines_cleared=lines_cleared,
            unique_cells_cleared=unique_cells_cleared,
            cleared_rows=full_r,
            cleared_cols=full_c,
            round_completed=round_completed,
            is_game_over=is_over,
        )

    def simulate_moves(
        self,
        initial_state: GameState,
        moves: Iterable[tuple[int, int, int]],
        auto_refill: bool = False,
    ) -> list[StepResult]:
        """Execute a sequence of (slot_idx, row, col) moves deterministically.

        Stops and returns results if a move results in Game Over.
        """
        results: list[StepResult] = []
        current_state = initial_state
        for slot_idx, r, c in moves:
            step = self.apply_move(current_state, slot_idx, r, c, auto_refill=auto_refill)
            results.append(step)
            current_state = step.state_after
            if current_state.is_game_over:
                break
        return results

    def deal_hand(
        self,
        count: Optional[int] = None,
        rng: Optional[random.Random] = None,
    ) -> tuple[Piece, ...]:
        """Draw a fresh hand of pieces from the catalog.

        NOTE: Deal distribution is currently configured via GameRules.deal_strategy.
        Default is uniform i.i.d. with replacement (PROVISIONAL/UNKNOWN).
        """
        n = count or self.rules.tray_capacity
        picker = rng or self.rng
        all_pieces = self.catalog.all_pieces
        return tuple(picker.choice(all_pieces) for _ in range(n))

    def refill_tray(
        self,
        state: GameState,
        new_pieces: Optional[Sequence[Piece]] = None,
        rng: Optional[random.Random] = None,
        force: bool = False,
    ) -> GameState:
        """Refill an empty tray with a new hand of pieces and evaluate game-over status.

        Raises:
            ValueError: If tray still contains unplaced pieces and force is False.
        """
        if not state.is_tray_empty() and not force:
            raise ValueError(
                f"Cannot refill tray: current tray still contains {state.active_piece_count()} unplaced piece(s)."
            )
        hand = tuple(new_pieces) if new_pieces is not None else self.deal_hand(rng=rng)
        new_state = state.clone_with(tray=hand, is_game_over=False)
        is_over = self.check_game_over(new_state)
        return new_state.clone_with(is_game_over=is_over)
