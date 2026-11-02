"""Independent Piece Representation and Extensible Catalog for BLOCK BLAST AI ULTIMATE.

CRITICAL ARCHITECTURAL RULE:
- Each piece is an immutable polyomino with a canonical bitmask anchored at (0, 0).
- Pieces never rotate or mutate during gameplay.
- The 41-piece canonical catalog provided here is an INITIAL PROVISIONAL HYPOTHESIS
  derived from repository audits (PKpKDdEa, Gurstel, Mathr81). It is explicitly
  NOT verified against official Google Play Block Blast telemetry.
- Custom catalogs can be registered and configured via `PieceCatalog`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Optional, Sequence

from game_engine.board import BOARD_SIZE, FULL_BOARD_MASK


# ---------------------------------------------------------------------------
# Piece Data Structure
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Piece:
    """Immutable representation of a Block Blast piece.

    Attributes:
        piece_id: Unique integer identifier.
        name: Human-readable identifier (e.g. 'square_2x2', 'line_1x3_h').
        category: Shape family category (e.g. 'dots', 'lines', 'squares', etc.).
        cells: Sorted tuple of relative coordinate offsets: ((dr, dc), ...).
               Anchored such that min(dr) == 0 and min(dc) == 0.
        height: Height in cells (max(dr) + 1).
        width: Width in cells (max(dc) + 1).
        cell_count: Number of unit squares occupied by this piece (len(cells)).
        canonical_mask: 64-bit integer mask when piece is placed at (0, 0).
        is_provisional: Flag indicating this piece is a provisional hypothesis.
        placement_masks: Precomputed 8x8 lookup table where
                         table[row][col] is the 64-bit placement mask, or None
                         if placing at (row, col) extends out of bounds.
    """

    piece_id: int
    name: str
    category: str
    cells: tuple[tuple[int, int], ...]
    height: int
    width: int
    cell_count: int
    canonical_mask: int
    placement_masks: tuple[tuple[Optional[int], ...], ...] = field(
        repr=False, hash=False, compare=False
    )
    is_provisional: bool = True

    def get_placement_mask(self, row: int, col: int) -> Optional[int]:
        """Return the 64-bit bitboard mask for this piece placed at (row, col).

        Returns:
            64-bit integer mask, or None if the piece would exceed board boundaries.
        """
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            return self.placement_masks[row][col]
        return None

    def fits_at(self, board: int, row: int, col: int) -> bool:
        """Return True if this piece fits at (row, col) without overlap or boundary violation."""
        mask = self.get_placement_mask(row, col)
        if mask is None:
            return False
        return (board & mask) == 0

    def get_all_legal_placements(self, board: int) -> list[tuple[int, int, int]]:
        """Return all legal (row, col, mask) placements for this piece on the given board."""
        legal: list[tuple[int, int, int]] = []
        max_r = BOARD_SIZE - self.height + 1
        max_c = BOARD_SIZE - self.width + 1
        for r in range(max_r):
            row_table = self.placement_masks[r]
            for c in range(max_c):
                mask = row_table[c]
                if mask is not None and (board & mask) == 0:
                    legal.append((r, c, mask))
        return legal


def create_piece(
    piece_id: int,
    name: str,
    category: str,
    cells: Sequence[tuple[int, int]],
    is_provisional: bool = True,
) -> Piece:
    """Factory to construct an immutable Piece with precomputed canonical and placement masks."""
    sorted_cells = tuple(sorted(set(cells)))
    if not sorted_cells:
        raise ValueError(f"Piece '{name}' must have at least one cell.")

    min_r = min(r for r, _ in sorted_cells)
    min_c = min(c for _, c in sorted_cells)
    if min_r != 0 or min_c != 0:
        # Re-anchor to (0, 0)
        sorted_cells = tuple((r - min_r, c - min_c) for r, c in sorted_cells)

    h = max(r for r, _ in sorted_cells) + 1
    w = max(c for _, c in sorted_cells) + 1
    cell_count = len(sorted_cells)

    # Compute canonical mask anchored at (0, 0)
    canonical = 0
    for dr, dc in sorted_cells:
        canonical |= (1 << (dr * BOARD_SIZE + dc))

    # Precompute 8x8 placement mask table
    placement_table: list[list[Optional[int]]] = []
    for r in range(BOARD_SIZE):
        row_masks: list[Optional[int]] = []
        for c in range(BOARD_SIZE):
            if r + h <= BOARD_SIZE and c + w <= BOARD_SIZE:
                shift = r * BOARD_SIZE + c
                row_masks.append(canonical << shift)
            else:
                row_masks.append(None)
        placement_table.append(row_masks)

    frozen_table = tuple(tuple(row) for row in placement_table)

    return Piece(
        piece_id=piece_id,
        name=name,
        category=category,
        cells=sorted_cells,
        height=h,
        width=w,
        cell_count=cell_count,
        canonical_mask=canonical,
        placement_masks=frozen_table,
        is_provisional=is_provisional,
    )


# ---------------------------------------------------------------------------
# Extensible Piece Catalog
# ---------------------------------------------------------------------------

class PieceCatalog:
    """Registry and lookup container for game pieces."""

    def __init__(self, pieces: Optional[Sequence[Piece]] = None) -> None:
        self._pieces: list[Piece] = []
        self._by_id: dict[int, Piece] = {}
        self._by_name: dict[str, Piece] = {}
        self._by_category: dict[str, list[Piece]] = {}
        if pieces:
            for p in pieces:
                self.register(p)

    def register(self, piece: Piece) -> None:
        """Register a piece in the catalog.

        Raises:
            ValueError: If piece_id or name already exists in this catalog.
        """
        if piece.piece_id in self._by_id:
            raise ValueError(f"Duplicate piece_id {piece.piece_id}")
        if piece.name in self._by_name:
            raise ValueError(f"Duplicate piece name '{piece.name}'")

        self._pieces.append(piece)
        self._by_id[piece.piece_id] = piece
        self._by_name[piece.name] = piece
        self._by_category.setdefault(piece.category, []).append(piece)

    def get_by_id(self, piece_id: int) -> Piece:
        """Lookup piece by integer ID."""
        return self._by_id[piece_id]

    def get_by_name(self, name: str) -> Piece:
        """Lookup piece by string name."""
        return self._by_name[name]

    def get_by_category(self, category: str) -> list[Piece]:
        """Return all pieces in a category."""
        return list(self._by_category.get(category, []))

    @property
    def all_pieces(self) -> tuple[Piece, ...]:
        """Return immutable tuple of all registered pieces."""
        return tuple(self._pieces)

    def __len__(self) -> int:
        return len(self._pieces)

    def __iter__(self):
        return iter(self._pieces)

    def __getitem__(self, idx: int) -> Piece:
        return self._pieces[idx]


# ---------------------------------------------------------------------------
# 41 Provisional Canonical Pieces (Initial Hypothesis)
# ---------------------------------------------------------------------------

def _build_provisional_41_catalog() -> PieceCatalog:
    """Build the 41-piece provisional catalog."""
    catalog = PieceCatalog()
    pid = 0

    # 1. Dots (1)
    catalog.register(create_piece(pid, "dot_1x1", "dots", [(0, 0)])); pid += 1

    # 2. Lines (8)
    catalog.register(create_piece(pid, "line_1x2_h", "lines", [(0, 0), (0, 1)])); pid += 1
    catalog.register(create_piece(pid, "line_1x2_v", "lines", [(0, 0), (1, 0)])); pid += 1
    catalog.register(create_piece(pid, "line_1x3_h", "lines", [(0, 0), (0, 1), (0, 2)])); pid += 1
    catalog.register(create_piece(pid, "line_1x3_v", "lines", [(0, 0), (1, 0), (2, 0)])); pid += 1
    catalog.register(create_piece(pid, "line_1x4_h", "lines", [(0, 0), (0, 1), (0, 2), (0, 3)])); pid += 1
    catalog.register(create_piece(pid, "line_1x4_v", "lines", [(0, 0), (1, 0), (2, 0), (3, 0)])); pid += 1
    catalog.register(create_piece(pid, "line_1x5_h", "lines", [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)])); pid += 1
    catalog.register(create_piece(pid, "line_1x5_v", "lines", [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])); pid += 1

    # 3. Squares (2)
    catalog.register(create_piece(pid, "square_2x2", "squares", [(0, 0), (0, 1), (1, 0), (1, 1)])); pid += 1
    catalog.register(create_piece(pid, "square_3x3", "squares", [
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 1), (2, 2)
    ])); pid += 1

    # 4. Rectangles (2)
    catalog.register(create_piece(pid, "rect_2x3_h", "rectangles", [
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 2)
    ])); pid += 1
    catalog.register(create_piece(pid, "rect_3x2_v", "rectangles", [
        (0, 0), (0, 1),
        (1, 0), (1, 1),
        (2, 0), (2, 1)
    ])); pid += 1

    # 5. Small Corners 2x2 (4)
    catalog.register(create_piece(pid, "corner_2x2_tl", "corners", [(0, 1), (1, 0), (1, 1)])); pid += 1
    catalog.register(create_piece(pid, "corner_2x2_tr", "corners", [(0, 0), (1, 0), (1, 1)])); pid += 1
    catalog.register(create_piece(pid, "corner_2x2_bl", "corners", [(0, 0), (0, 1), (1, 1)])); pid += 1
    catalog.register(create_piece(pid, "corner_2x2_br", "corners", [(0, 0), (0, 1), (1, 0)])); pid += 1

    # 6. Big Corners 3x3 (4)
    catalog.register(create_piece(pid, "corner_3x3_tl", "corners", [
        (0, 0), (0, 1), (0, 2),
        (1, 0),
        (2, 0)
    ])); pid += 1
    catalog.register(create_piece(pid, "corner_3x3_tr", "corners", [
        (0, 0), (0, 1), (0, 2),
        (1, 2),
        (2, 2)
    ])); pid += 1
    catalog.register(create_piece(pid, "corner_3x3_bl", "corners", [
        (0, 0),
        (1, 0),
        (2, 0), (2, 1), (2, 2)
    ])); pid += 1
    catalog.register(create_piece(pid, "corner_3x3_br", "corners", [
        (0, 2),
        (1, 2),
        (2, 0), (2, 1), (2, 2)
    ])); pid += 1

    # 7. L-shapes 3x2 (8)
    # Vertical L-shapes (height 3, width 2)
    catalog.register(create_piece(pid, "l_3x2_v1", "l_shapes", [(0, 0), (1, 0), (2, 0), (2, 1)])); pid += 1
    catalog.register(create_piece(pid, "l_3x2_v2", "l_shapes", [(0, 1), (1, 1), (2, 1), (2, 0)])); pid += 1
    catalog.register(create_piece(pid, "l_3x2_v3", "l_shapes", [(0, 0), (0, 1), (1, 0), (2, 0)])); pid += 1
    catalog.register(create_piece(pid, "l_3x2_v4", "l_shapes", [(0, 0), (0, 1), (1, 1), (2, 1)])); pid += 1
    # Horizontal L-shapes (height 2, width 3)
    catalog.register(create_piece(pid, "l_2x3_h1", "l_shapes", [(0, 0), (1, 0), (1, 1), (1, 2)])); pid += 1
    catalog.register(create_piece(pid, "l_2x3_h2", "l_shapes", [(0, 2), (1, 0), (1, 1), (1, 2)])); pid += 1
    catalog.register(create_piece(pid, "l_2x3_h3", "l_shapes", [(0, 0), (0, 1), (0, 2), (1, 0)])); pid += 1
    catalog.register(create_piece(pid, "l_2x3_h4", "l_shapes", [(0, 0), (0, 1), (0, 2), (1, 2)])); pid += 1

    # 8. Specials (12)
    # T-shapes (4)
    catalog.register(create_piece(pid, "t_shape_up", "specials", [(0, 1), (1, 0), (1, 1), (1, 2)])); pid += 1
    catalog.register(create_piece(pid, "t_shape_down", "specials", [(0, 0), (0, 1), (0, 2), (1, 1)])); pid += 1
    catalog.register(create_piece(pid, "t_shape_left", "specials", [(0, 1), (1, 0), (1, 1), (2, 1)])); pid += 1
    catalog.register(create_piece(pid, "t_shape_right", "specials", [(0, 0), (1, 0), (1, 1), (2, 0)])); pid += 1
    # Z-shapes (2)
    catalog.register(create_piece(pid, "z_shape_h", "specials", [(0, 0), (0, 1), (1, 1), (1, 2)])); pid += 1
    catalog.register(create_piece(pid, "z_shape_v", "specials", [(0, 1), (1, 0), (1, 1), (2, 0)])); pid += 1
    # S-shapes (2)
    catalog.register(create_piece(pid, "s_shape_h", "specials", [(0, 1), (0, 2), (1, 0), (1, 1)])); pid += 1
    catalog.register(create_piece(pid, "s_shape_v", "specials", [(0, 0), (1, 0), (1, 1), (2, 1)])); pid += 1
    # Diagonals (4)
    catalog.register(create_piece(pid, "diagonal_2_tr", "specials", [(0, 1), (1, 0)])); pid += 1
    catalog.register(create_piece(pid, "diagonal_2_tl", "specials", [(0, 0), (1, 1)])); pid += 1
    catalog.register(create_piece(pid, "diagonal_3_tr", "specials", [(0, 2), (1, 1), (2, 0)])); pid += 1
    catalog.register(create_piece(pid, "diagonal_3_tl", "specials", [(0, 0), (1, 1), (2, 2)])); pid += 1

    return catalog


CANONICAL_41_CATALOG: Final[PieceCatalog] = _build_provisional_41_catalog()
