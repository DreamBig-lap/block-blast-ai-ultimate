"""Immutable Game State Representation for BLOCK BLAST AI ULTIMATE.

CRITICAL REQUIREMENTS:
- GameState is strictly immutable (frozen) to prevent subtle state-leak bugs during search/simulation.
- Holds bitboard, score, combo metrics, round number, active tray slots, and rule config.
- Clearly distinguishes between initial state, post-placement state, and post-round state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from game_engine.board import (
    BOARD_CELLS,
    BOARD_SIZE,
    board_to_string,
    is_occupied,
    popcount,
    validate_bitboard,
)
from game_engine.pieces import Piece
from game_engine.rules import DEFAULT_RULES, GameRules


# ---------------------------------------------------------------------------
# Immutable GameState
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class GameState:
    """Immutable representation of a complete Block Blast game state.

    Attributes:
        board: 64-bit unsigned integer representing the 8x8 grid occupancy.
        score: Cumulative points scored in this game.
        combo_streak: Current active combo multiplier streak.
        non_clearing_moves: Consecutive moves placed without clearing a line.
        round_number: Current round index (increments when tray is refilled).
        tray: Tuple of tray slots (typically length 3). Each slot holds
              a Piece or None (if that slot's piece has already been placed).
        is_game_over: True if no legal moves remain for any available tray piece.
        rules: Active GameRules configuration governing this game.
    """

    board: int = 0
    score: int = 0
    combo_streak: int = 0
    non_clearing_moves: int = 0
    round_number: int = 1
    tray: tuple[Optional[Piece], ...] = field(default_factory=tuple)
    is_game_over: bool = False
    rules: GameRules = field(default_factory=lambda: DEFAULT_RULES)

    def __post_init__(self) -> None:
        """Validate state invariants on construction."""
        validate_bitboard(self.board)
        if self.score < 0:
            raise ValueError(f"Score cannot be negative: {self.score}")
        if self.combo_streak < 0:
            raise ValueError(f"Combo streak cannot be negative: {self.combo_streak}")
        if self.non_clearing_moves < 0:
            raise ValueError(f"non_clearing_moves cannot be negative: {self.non_clearing_moves}")
        if self.round_number < 1:
            raise ValueError(f"round_number must be >= 1: {self.round_number}")

    @classmethod
    def create_initial(
        cls,
        initial_tray: Sequence[Optional[Piece]] = (),
        rules: Optional[GameRules] = None,
    ) -> GameState:
        """Construct a fresh initial GameState with an empty board and score 0."""
        active_rules = rules or DEFAULT_RULES
        return cls(
            board=0,
            score=0,
            combo_streak=0,
            non_clearing_moves=0,
            round_number=1,
            tray=tuple(initial_tray),
            is_game_over=False,
            rules=active_rules,
        )

    def is_tray_empty(self) -> bool:
        """Return True if all tray slots are empty (None)."""
        return all(p is None for p in self.tray)

    def active_piece_count(self) -> int:
        """Return the number of unplaced pieces currently remaining in the tray."""
        return sum(1 for p in self.tray if p is not None)

    def has_piece_in_slot(self, slot_idx: int) -> bool:
        """Return True if the specified tray slot holds an unplaced piece."""
        return 0 <= slot_idx < len(self.tray) and self.tray[slot_idx] is not None

    def get_piece(self, slot_idx: int) -> Optional[Piece]:
        """Return piece at tray slot, or None if empty or invalid index."""
        if 0 <= slot_idx < len(self.tray):
            return self.tray[slot_idx]
        return None

    @property
    def occupied_cells_count(self) -> int:
        """Return the number of filled cells on the board (0..64)."""
        return popcount(self.board)

    @property
    def empty_cells_count(self) -> int:
        """Return the number of empty cells on the board (0..64)."""
        return BOARD_CELLS - self.occupied_cells_count

    def clone_with(
        self,
        board: Optional[int] = None,
        score: Optional[int] = None,
        combo_streak: Optional[int] = None,
        non_clearing_moves: Optional[int] = None,
        round_number: Optional[int] = None,
        tray: Optional[Sequence[Optional[Piece]]] = None,
        is_game_over: Optional[bool] = None,
        rules: Optional[GameRules] = None,
    ) -> GameState:
        """Return a new GameState with specified fields updated."""
        return GameState(
            board=self.board if board is None else board,
            score=self.score if score is None else score,
            combo_streak=self.combo_streak if combo_streak is None else combo_streak,
            non_clearing_moves=(
                self.non_clearing_moves if non_clearing_moves is None else non_clearing_moves
            ),
            round_number=self.round_number if round_number is None else round_number,
            tray=self.tray if tray is None else tuple(tray),
            is_game_over=self.is_game_over if is_game_over is None else is_game_over,
            rules=self.rules if rules is None else rules,
        )

    def render(self) -> str:
        """Return a formatted string representing the board, score, and tray state."""
        header = (
            f"=== Round {self.round_number} | Score: {self.score} | "
            f"Combo: x{self.combo_streak} (NC: {self.non_clearing_moves}) | "
            f"Game Over: {self.is_game_over} ==="
        )
        grid_str = board_to_string(self.board)
        tray_str = "Tray: " + ", ".join(
            f"[{i}]: {p.name if p else 'EMPTY'}" for i, p in enumerate(self.tray)
        )
        return f"{header}\n{grid_str}\n{tray_str}"
