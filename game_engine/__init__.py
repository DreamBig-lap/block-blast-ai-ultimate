"""BLOCK BLAST AI ULTIMATE — Core Game Engine Package.

A modular, zero-dependency, 64-bit pure-Python bitboard simulation engine
designed for high-throughput CPU-first performance.
"""

from game_engine.board import (
    BOARD_CELLS,
    BOARD_SIZE,
    COL_MASKS,
    FULL_BOARD_MASK,
    ROW_MASKS,
    BitboardError,
    InvalidBitboardError,
    InvalidCoordinateError,
    InvalidPlacementError,
    bit_index_to_coord,
    bitboard_to_coords,
    board_to_string,
    can_place,
    check_and_clear_lines,
    clear_lines,
    coord_to_bit,
    coord_to_bit_index,
    coords_to_bitboard,
    find_full_cols,
    find_full_rows,
    is_occupied,
    is_valid_coord,
    place,
    popcount,
    string_to_board,
    validate_bitboard,
)
from game_engine.pieces import (
    CANONICAL_41_CATALOG,
    Piece,
    PieceCatalog,
    create_piece,
)
from game_engine.rules import (
    DEFAULT_RULES,
    GameRules,
    RuleStatus,
)
from game_engine.scoring import (
    ScoreBreakdown,
    ScoreCalculator,
)
from game_engine.simulator import (
    GameSimulator,
    IllegalMoveError,
    StepResult,
)
from game_engine.state import (
    GameState,
)

__all__ = [
    # Board Constants & Primitives
    "BOARD_SIZE",
    "BOARD_CELLS",
    "FULL_BOARD_MASK",
    "ROW_MASKS",
    "COL_MASKS",
    "is_valid_coord",
    "coord_to_bit_index",
    "coord_to_bit",
    "bit_index_to_coord",
    "validate_bitboard",
    "popcount",
    "is_occupied",
    "can_place",
    "place",
    "find_full_rows",
    "find_full_cols",
    "clear_lines",
    "check_and_clear_lines",
    "coords_to_bitboard",
    "bitboard_to_coords",
    "board_to_string",
    "string_to_board",
    # Exceptions
    "BitboardError",
    "InvalidCoordinateError",
    "InvalidBitboardError",
    "InvalidPlacementError",
    "IllegalMoveError",
    # Pieces
    "Piece",
    "create_piece",
    "PieceCatalog",
    "CANONICAL_41_CATALOG",
    # Rules
    "RuleStatus",
    "GameRules",
    "DEFAULT_RULES",
    # Scoring
    "ScoreBreakdown",
    "ScoreCalculator",
    # State
    "GameState",
    # Simulator
    "GameSimulator",
    "StepResult",
]
