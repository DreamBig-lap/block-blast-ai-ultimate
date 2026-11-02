"""64-bit Bitboard Engine for BLOCK BLAST AI ULTIMATE.

This module implements low-level bitboard representations and bitwise operations
for the 8x8 Block Blast grid.

Coordinate and Bitboard Mapping:
- The board has 8 rows (0..7, top to bottom) and 8 columns (0..7, left to right).
- Bit index = row * 8 + col.
- Bit 0  is (row=0, col=0) [top-left].
- Bit 7  is (row=0, col=7) [top-right].
- Bit 56 is (row=7, col=0) [bottom-left].
- Bit 63 is (row=7, col=7) [bottom-right].
- Value range: 0 <= board <= 0xFFFFFFFFFFFFFFFF (strictly 64-bit integer).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Final

# ---------------------------------------------------------------------------
# Board Constants
# ---------------------------------------------------------------------------

BOARD_SIZE: Final[int] = 8
BOARD_CELLS: Final[int] = 64
FULL_BOARD_MASK: Final[int] = (1 << BOARD_CELLS) - 1  # 0xFFFFFFFFFFFFFFFF

# Single row template (row 0: bits 0..7)
_ROW_TEMPLATE: Final[int] = (1 << BOARD_SIZE) - 1  # 0xFF

# Precomputed row bitmasks (rows 0..7)
ROW_MASKS: Final[tuple[int, ...]] = tuple(
    _ROW_TEMPLATE << (r * BOARD_SIZE) for r in range(BOARD_SIZE)
)

# Single column template (col 0: bits 0, 8, 16, 24, 32, 40, 48, 56 = 0x0101010101010101)
_COL_TEMPLATE: Final[int] = sum(1 << (r * BOARD_SIZE) for r in range(BOARD_SIZE))

# Precomputed column bitmasks (cols 0..7)
COL_MASKS: Final[tuple[int, ...]] = tuple(
    _COL_TEMPLATE << c for c in range(BOARD_SIZE)
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class BitboardError(Exception):
    """Base exception for bitboard errors."""


class InvalidCoordinateError(BitboardError, ValueError):
    """Raised when a coordinate is outside the 8x8 board boundaries."""


class InvalidBitboardError(BitboardError, ValueError):
    """Raised when a bitboard value contains bits outside the 64-bit board."""


class InvalidPlacementError(BitboardError, ValueError):
    """Raised when an illegal placement is attempted (overlap or out-of-bounds)."""


# ---------------------------------------------------------------------------
# Coordinate & Bit Conversion Primitives
# ---------------------------------------------------------------------------

def is_valid_coord(row: int, col: int) -> bool:
    """Return True if (row, col) is within the 8x8 board boundaries."""
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def coord_to_bit_index(row: int, col: int) -> int:
    """Convert (row, col) to a bit index in 0..63.

    Raises:
        InvalidCoordinateError: If row or col is out of 0..7 range.
    """
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise InvalidCoordinateError(
            f"Coordinate ({row}, {col}) is out of board bounds (0..7, 0..7)"
        )
    return row * BOARD_SIZE + col


def coord_to_bit(row: int, col: int) -> int:
    """Convert (row, col) to a 1-hot 64-bit integer mask.

    Raises:
        InvalidCoordinateError: If row or col is out of 0..7 range.
    """
    return 1 << coord_to_bit_index(row, col)


def bit_index_to_coord(bit_index: int) -> tuple[int, int]:
    """Convert bit index (0..63) to (row, col).

    Raises:
        InvalidCoordinateError: If bit_index is out of 0..63 range.
    """
    if not (0 <= bit_index < BOARD_CELLS):
        raise InvalidCoordinateError(
            f"Bit index {bit_index} is out of bounds (0..63)"
        )
    return bit_index // BOARD_SIZE, bit_index % BOARD_SIZE


def validate_bitboard(board: int) -> None:
    """Ensure bitboard is within the valid 64-bit unsigned integer range.

    Raises:
        InvalidBitboardError: If any bit outside bits 0..63 is set or if negative.
    """
    if board < 0 or (board & ~FULL_BOARD_MASK) != 0:
        raise InvalidBitboardError(
            f"Bitboard value 0x{board:X} exceeds valid 64-bit range (0x0..0x{FULL_BOARD_MASK:X})"
        )


# ---------------------------------------------------------------------------
# State Query & Bit Inspection
# ---------------------------------------------------------------------------

def popcount(board: int) -> int:
    """Return the number of set bits (occupied cells) in the bitboard.

    Uses Python 3.10+ int.bit_count() which maps to CPU POPCNT instruction.
    """
    validate_bitboard(board)
    return board.bit_count()


def is_occupied(board: int, row: int, col: int) -> bool:
    """Return True if cell (row, col) is occupied."""
    bit = coord_to_bit(row, col)
    return (board & bit) != 0


# ---------------------------------------------------------------------------
# Placement & Collision Logic
# ---------------------------------------------------------------------------

def can_place(board: int, mask: int) -> bool:
    """Return True if `mask` can be legally placed on `board`.

    A placement is legal if and only if:
    1. Mask is non-zero.
    2. Mask has no bits outside the 64-bit board.
    3. Mask has no overlapping bits with currently occupied cells in `board`.
    """
    if mask <= 0 or (mask & ~FULL_BOARD_MASK) != 0:
        return False
    if board < 0 or (board & ~FULL_BOARD_MASK) != 0:
        return False
    return (board & mask) == 0


def place(board: int, mask: int) -> int:
    """Place `mask` on `board` and return the new 64-bit bitboard.

    Raises:
        InvalidPlacementError: If the placement is illegal (overlap or out-of-bounds).
    """
    if not can_place(board, mask):
        overlap = board & mask
        if overlap != 0:
            raise InvalidPlacementError(
                f"Cannot place mask 0x{mask:X}: overlaps occupied cells at 0x{overlap:X}"
            )
        raise InvalidPlacementError(
            f"Cannot place mask 0x{mask:X}: invalid mask or out-of-bounds"
        )
    return board | mask


# ---------------------------------------------------------------------------
# Line Detection & Simultaneous Clearing
# ---------------------------------------------------------------------------

def find_full_rows(board: int) -> tuple[int, ...]:
    """Return tuple of row indices (0..7) that are completely filled."""
    validate_bitboard(board)
    return tuple(r for r in range(BOARD_SIZE) if (board & ROW_MASKS[r]) == ROW_MASKS[r])


def find_full_cols(board: int) -> tuple[int, ...]:
    """Return tuple of column indices (0..7) that are completely filled."""
    validate_bitboard(board)
    return tuple(c for c in range(BOARD_SIZE) if (board & COL_MASKS[c]) == COL_MASKS[c])


def clear_lines(
    board: int,
    rows: Sequence[int],
    cols: Sequence[int],
) -> tuple[int, int]:
    """Simultaneously clear the specified rows and columns from the board.

    Intersection Handling:
    If a cell is at the intersection of a cleared row and a cleared column,
    its bit is cleared once via bitwise NOT. Intersection cells are not
    double-cleared, preserving accurate unique cell counts.

    Returns:
        (new_board, unique_cells_cleared):
            new_board: The updated 64-bit bitboard.
            unique_cells_cleared: The exact number of occupied cells cleared (1..64).
    """
    validate_bitboard(board)
    cleared_mask = 0
    for r in rows:
        if 0 <= r < BOARD_SIZE:
            cleared_mask |= ROW_MASKS[r]
    for c in cols:
        if 0 <= c < BOARD_SIZE:
            cleared_mask |= COL_MASKS[c]

    unique_cells_cleared = (board & cleared_mask).bit_count()
    new_board = (board & ~cleared_mask) & FULL_BOARD_MASK
    return new_board, unique_cells_cleared


def check_and_clear_lines(
    board: int,
) -> tuple[int, int, int, tuple[int, ...], tuple[int, ...], int]:
    """Detect and simultaneously clear all completely filled rows and columns.

    Returns:
        (new_board, total_lines_cleared, unique_cells_cleared, cleared_rows, cleared_cols, cleared_mask)
        - new_board: The updated 64-bit bitboard after clearing.
        - total_lines_cleared: Number of lines cleared (len(cleared_rows) + len(cleared_cols)).
        - unique_cells_cleared: Number of unique cells removed from the board.
        - cleared_rows: Tuple of cleared row indices (0..7).
        - cleared_cols: Tuple of cleared column indices (0..7).
        - cleared_mask: 64-bit integer mask of all cleared cells.
    """
    validate_bitboard(board)
    full_r = find_full_rows(board)
    full_c = find_full_cols(board)

    if not full_r and not full_c:
        return board, 0, 0, (), (), 0

    cleared_mask = 0
    for r in full_r:
        cleared_mask |= ROW_MASKS[r]
    for c in full_c:
        cleared_mask |= COL_MASKS[c]

    unique_cells_cleared = (board & cleared_mask).bit_count()
    new_board = (board & ~cleared_mask) & FULL_BOARD_MASK
    total_lines = len(full_r) + len(full_c)

    return new_board, total_lines, unique_cells_cleared, full_r, full_c, cleared_mask


# ---------------------------------------------------------------------------
# Conversion & Debug Utilities
# ---------------------------------------------------------------------------

def coords_to_bitboard(coords: Iterable[tuple[int, int]]) -> int:
    """Convert an iterable of (row, col) coordinates to a 64-bit bitboard."""
    board = 0
    for r, c in coords:
        board |= coord_to_bit(r, c)
    return board


def bitboard_to_coords(board: int) -> set[tuple[int, int]]:
    """Convert a 64-bit bitboard to a set of (row, col) coordinates."""
    validate_bitboard(board)
    coords: set[tuple[int, int]] = set()
    b = board
    while b:
        lsb = b & -b
        bit_idx = lsb.bit_length() - 1
        coords.add(bit_index_to_coord(bit_idx))
        b ^= lsb
    return coords


def board_to_string(board: int, filled_char: str = "#", empty_char: str = ".") -> str:
    """Render the 64-bit bitboard as an 8x8 ASCII grid string."""
    validate_bitboard(board)
    lines: list[str] = []
    for r in range(BOARD_SIZE):
        row_chars = [
            filled_char if is_occupied(board, r, c) else empty_char
            for c in range(BOARD_SIZE)
        ]
        lines.append(" ".join(row_chars))
    return "\n".join(lines)


def string_to_board(s: str, filled_chars: str = "#1X*") -> int:
    """Parse an 8x8 ASCII representation into a 64-bit bitboard.

    Whitespace is stripped. Exactly 64 cell tokens are expected.
    """
    tokens = [ch for ch in s if not ch.isspace()]
    if len(tokens) != BOARD_CELLS:
        raise ValueError(f"Expected exactly 64 tokens, found {len(tokens)}")

    board = 0
    for bit_idx, ch in enumerate(tokens):
        if ch in filled_chars:
            board |= (1 << bit_idx)
    return board
