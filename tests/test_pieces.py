"""Unit tests for Piece representations and Catalog (game_engine/pieces.py)."""

import unittest
from dataclasses import FrozenInstanceError

from game_engine.board import BOARD_SIZE, popcount
from game_engine.pieces import (
    CANONICAL_41_CATALOG,
    Piece,
    PieceCatalog,
    create_piece,
)


class TestPieceRepresentation(unittest.TestCase):
    """Test piece mask generation, boundaries, and immutability."""

    def test_canonical_catalog_integrity(self):
        """Verify the 41 canonical pieces satisfy all geometric invariants."""
        self.assertEqual(len(CANONICAL_41_CATALOG), 41)
        seen_ids = set()
        seen_names = set()

        for piece in CANONICAL_41_CATALOG:
            # Check unique ID and Name
            self.assertNotIn(piece.piece_id, seen_ids)
            self.assertNotIn(piece.name, seen_names)
            seen_ids.add(piece.piece_id)
            seen_names.add(piece.name)

            # Epistemic requirement: Must be marked provisional
            self.assertTrue(piece.is_provisional)

            # Dimension checks
            min_r = min(r for r, _ in piece.cells)
            min_c = min(c for _, c in piece.cells)
            max_r = max(r for r, _ in piece.cells)
            max_c = max(c for _, c in piece.cells)

            self.assertEqual(min_r, 0, f"Piece {piece.name} dr not anchored at 0")
            self.assertEqual(min_c, 0, f"Piece {piece.name} dc not anchored at 0")
            self.assertEqual(piece.height, max_r + 1)
            self.assertEqual(piece.width, max_c + 1)
            self.assertEqual(piece.cell_count, len(piece.cells))

            # Canonical mask popcount must match cell_count
            self.assertEqual(piece.canonical_mask.bit_count(), piece.cell_count)

    def test_piece_immutability(self):
        """Piece instances must be frozen against attribute mutation."""
        piece = CANONICAL_41_CATALOG[0]
        with self.assertRaises(FrozenInstanceError):
            piece.name = "mutated_name"  # type: ignore
        with self.assertRaises(FrozenInstanceError):
            piece.piece_id = 999  # type: ignore

    def test_boundary_clipping_and_wrapping_prevention(self):
        """CRITICAL: Pieces placed near boundaries must return None without row-wrapping."""
        # 1x3 horizontal piece (height 1, width 3)
        piece_h3 = CANONICAL_41_CATALOG.get_by_name("line_1x3_h")
        self.assertEqual(piece_h3.height, 1)
        self.assertEqual(piece_h3.width, 3)

        # Valid placements: col 0..5
        for c in range(6):
            mask = piece_h3.get_placement_mask(0, c)
            self.assertIsNotNone(mask)
            self.assertEqual(popcount(mask), 3)

        # Invalid placements: col 6, 7 (would wrap into next row)
        self.assertIsNone(piece_h3.get_placement_mask(0, 6))
        self.assertIsNone(piece_h3.get_placement_mask(0, 7))

        # Negative rows/cols
        self.assertIsNone(piece_h3.get_placement_mask(-1, 0))
        self.assertIsNone(piece_h3.get_placement_mask(0, -1))

        # 3x1 vertical piece (height 3, width 1)
        piece_v3 = CANONICAL_41_CATALOG.get_by_name("line_1x3_v")
        # Valid: row 0..5; Invalid: row 6, 7
        for r in range(6):
            self.assertIsNotNone(piece_v3.get_placement_mask(r, 0))
        self.assertIsNone(piece_v3.get_placement_mask(6, 0))
        self.assertIsNone(piece_v3.get_placement_mask(7, 0))

    def test_fits_at(self):
        """Test fits_at collision check against an occupied board."""
        piece_2x2 = CANONICAL_41_CATALOG.get_by_name("square_2x2")
        board = 0

        # Fits on empty board
        self.assertTrue(piece_2x2.fits_at(board, 0, 0))
        self.assertTrue(piece_2x2.fits_at(board, 6, 6))
        # Does not fit at row 7 (extends to row 8)
        self.assertFalse(piece_2x2.fits_at(board, 7, 0))

        # Add single block at (1, 1)
        board |= (1 << (1 * 8 + 1))
        # Now 2x2 placed at (0, 0) overlaps (1, 1)!
        self.assertFalse(piece_2x2.fits_at(board, 0, 0))
        # But placed at (0, 2) it does not overlap
        self.assertTrue(piece_2x2.fits_at(board, 0, 2))


class TestPieceCatalog(unittest.TestCase):
    """Test PieceCatalog registration and querying."""

    def test_custom_catalog_registration(self):
        """Custom pieces can be registered in a new catalog."""
        cat = PieceCatalog()
        p1 = create_piece(0, "custom_mono", "custom", [(0, 0)], is_provisional=False)
        cat.register(p1)
        self.assertEqual(len(cat), 1)
        self.assertEqual(cat.get_by_id(0), p1)
        self.assertEqual(cat.get_by_name("custom_mono"), p1)
        self.assertEqual(cat.get_by_category("custom"), [p1])

    def test_duplicate_registration_error(self):
        """Registering duplicate ID or name raises ValueError."""
        cat = PieceCatalog()
        p1 = create_piece(0, "p1", "test", [(0, 0)])
        cat.register(p1)

        p_dup_id = create_piece(0, "p2", "test", [(0, 0)])
        with self.assertRaises(ValueError):
            cat.register(p_dup_id)

        p_dup_name = create_piece(1, "p1", "test", [(0, 0)])
        with self.assertRaises(ValueError):
            cat.register(p_dup_name)


if __name__ == "__main__":
    unittest.main()
