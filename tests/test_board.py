"""Unit tests for 64-bit Bitboard Engine (game_engine/board.py)."""

import unittest

from game_engine.board import (
    BOARD_CELLS,
    BOARD_SIZE,
    COL_MASKS,
    FULL_BOARD_MASK,
    ROW_MASKS,
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


class TestBitboardCoordinates(unittest.TestCase):
    """Test coordinate to bit mapping and boundary enforcement."""

    def test_coordinate_bijection(self):
        """Every (r, c) in 0..7 must map to bit r*8+c and invert back identically."""
        seen_indices = set()
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                self.assertTrue(is_valid_coord(r, c))
                idx = coord_to_bit_index(r, c)
                expected_idx = r * 8 + c
                self.assertEqual(idx, expected_idx)
                self.assertNotIn(idx, seen_indices)
                seen_indices.add(idx)

                # Invert
                inv_r, inv_c = bit_index_to_coord(idx)
                self.assertEqual((inv_r, inv_c), (r, c))

                # Bit mask
                mask = coord_to_bit(r, c)
                self.assertEqual(mask, 1 << expected_idx)
                self.assertEqual(mask.bit_count(), 1)

        self.assertEqual(len(seen_indices), 64)

    def test_out_of_bounds_coordinates(self):
        """Coordinates outside 0..7 must raise InvalidCoordinateError."""
        invalid_coords = [
            (-1, 0), (0, -1), (8, 0), (0, 8),
            (-5, -5), (10, 10), (100, 0), (0, 100)
        ]
        for r, c in invalid_coords:
            self.assertFalse(is_valid_coord(r, c))
            with self.assertRaises(InvalidCoordinateError):
                coord_to_bit_index(r, c)
            with self.assertRaises(InvalidCoordinateError):
                coord_to_bit(r, c)

    def test_out_of_bounds_bit_indices(self):
        """Bit indices outside 0..63 must raise InvalidCoordinateError."""
        for invalid_idx in [-1, -10, 64, 65, 128]:
            with self.assertRaises(InvalidCoordinateError):
                bit_index_to_coord(invalid_idx)

    def test_validate_bitboard(self):
        """Bitboard validation ensures values are within 64-bit unsigned bounds."""
        validate_bitboard(0)
        validate_bitboard(1)
        validate_bitboard(FULL_BOARD_MASK)

        # Invalid: negative or bits > 63
        with self.assertRaises(InvalidBitboardError):
            validate_bitboard(-1)
        with self.assertRaises(InvalidBitboardError):
            validate_bitboard(FULL_BOARD_MASK + 1)
        with self.assertRaises(InvalidBitboardError):
            validate_bitboard(1 << 64)


class TestBitboardPlacements(unittest.TestCase):
    """Test occupancy, popcount, and placement collision detection."""

    def test_popcount_and_is_occupied(self):
        """Popcount and is_occupied must report exact cell counts."""
        board = 0
        self.assertEqual(popcount(board), 0)

        # Set specific cells
        test_cells = [(0, 0), (1, 3), (7, 7), (4, 4)]
        for r, c in test_cells:
            self.assertFalse(is_occupied(board, r, c))
            board |= coord_to_bit(r, c)
            self.assertTrue(is_occupied(board, r, c))

        self.assertEqual(popcount(board), len(test_cells))
        self.assertEqual(popcount(FULL_BOARD_MASK), 64)

    def test_can_place_and_place(self):
        """Test legal placement and overlap prevention."""
        board = coord_to_bit(0, 0) | coord_to_bit(0, 1)

        # Legal placement (no overlap)
        mask_legal = coord_to_bit(1, 0) | coord_to_bit(1, 1)
        self.assertTrue(can_place(board, mask_legal))
        new_board = place(board, mask_legal)
        self.assertEqual(popcount(new_board), 4)

        # Illegal placement (overlaps cell (0, 1))
        mask_overlap = coord_to_bit(0, 1) | coord_to_bit(1, 1)
        self.assertFalse(can_place(board, mask_overlap))
        with self.assertRaises(InvalidPlacementError):
            place(board, mask_overlap)

        # Illegal: zero mask or out-of-bounds mask
        self.assertFalse(can_place(board, 0))
        self.assertFalse(can_place(board, 1 << 64))
        with self.assertRaises(InvalidPlacementError):
            place(board, 0)


class TestLineClearing(unittest.TestCase):
    """Test single line, multi-line, and simultaneous row/column line clearing."""

    def test_row_clearing(self):
        """Filling row 3 should detect and clear row 3."""
        board = ROW_MASKS[3] | coord_to_bit(2, 5)
        full_rows = find_full_rows(board)
        full_cols = find_full_cols(board)
        self.assertEqual(full_rows, (3,))
        self.assertEqual(full_cols, ())

        new_board, cleared_cells = clear_lines(board, [3], [])
        self.assertEqual(new_board, coord_to_bit(2, 5))
        self.assertEqual(cleared_cells, 8)
        self.assertEqual(popcount(new_board), 1)

    def test_col_clearing(self):
        """Filling column 5 should detect and clear column 5."""
        board = COL_MASKS[5] | coord_to_bit(1, 2)
        full_rows = find_full_rows(board)
        full_cols = find_full_cols(board)
        self.assertEqual(full_rows, ())
        self.assertEqual(full_cols, (5,))

        new_board, cleared_cells = clear_lines(board, [], [5])
        self.assertEqual(new_board, coord_to_bit(1, 2))
        self.assertEqual(cleared_cells, 8)
        self.assertEqual(popcount(new_board), 1)

    def test_simultaneous_row_and_column_clearing(self):
        """CRITICAL: Intersection of row r and col c must NOT be double-cleared or double-counted."""
        r_idx = 2
        c_idx = 4
        # Board with full row 2 and full col 4, plus one extra block at (0, 0)
        board = ROW_MASKS[r_idx] | COL_MASKS[c_idx] | coord_to_bit(0, 0)

        # Row 2 has 8 cells, Col 4 has 8 cells.
        # Intersection cell (2, 4) belongs to both.
        # Total occupied cells = 8 + 8 - 1 + 1 (extra) = 16 cells.
        self.assertEqual(popcount(board), 16)

        new_board, total_lines, unique_cells, full_r, full_c, _mask = check_and_clear_lines(board)

        self.assertEqual(total_lines, 2)
        self.assertEqual(full_r, (r_idx,))
        self.assertEqual(full_c, (c_idx,))

        # Unique cells cleared must be EXACTLY 15 (8 + 8 - 1), NOT 16!
        self.assertEqual(unique_cells, 15)

        # Only the extra cell (0, 0) should remain!
        self.assertEqual(new_board, coord_to_bit(0, 0))
        self.assertEqual(popcount(new_board), 1)

        # Intersection cell (2, 4) must be empty (0)
        self.assertFalse(is_occupied(new_board, r_idx, c_idx))

    def test_multi_row_multi_col_clearing(self):
        """Simultaneous 2 rows (1, 3) and 2 cols (2, 5) clearing."""
        board = ROW_MASKS[1] | ROW_MASKS[3] | COL_MASKS[2] | COL_MASKS[5]
        # 2 rows = 16 cells, 2 cols = 16 cells, 4 intersections (1,2), (1,5), (3,2), (3,5)
        # Unique cells = 16 + 16 - 4 = 28 cells.
        self.assertEqual(popcount(board), 28)

        new_board, total_lines, unique_cells, full_r, full_c, _ = check_and_clear_lines(board)
        self.assertEqual(total_lines, 4)
        self.assertEqual(full_r, (1, 3))
        self.assertEqual(full_c, (2, 5))
        self.assertEqual(unique_cells, 28)
        self.assertEqual(new_board, 0)


class TestBoardConversions(unittest.TestCase):
    """Test ASCII and coordinate set conversion utilities."""

    def test_coords_to_bitboard_and_back(self):
        """Converting coords to bitboard and back must be a perfect round-trip."""
        original_coords = {(0, 0), (1, 2), (3, 4), (7, 6), (7, 7)}
        board = coords_to_bitboard(original_coords)
        self.assertEqual(popcount(board), len(original_coords))

        recovered_coords = bitboard_to_coords(board)
        self.assertEqual(recovered_coords, original_coords)

    def test_string_round_trip(self):
        """board_to_string and string_to_board must round-trip cleanly."""
        board = coord_to_bit(0, 0) | coord_to_bit(3, 3) | coord_to_bit(7, 7)
        s = board_to_string(board)
        recovered_board = string_to_board(s)
        self.assertEqual(recovered_board, board)

    def test_string_to_board_invalid_length(self):
        """string_to_board must raise ValueError if token count is not exactly 64."""
        # Empty string
        with self.assertRaises(ValueError):
            string_to_board("")

        # 63 tokens (short by 1)
        with self.assertRaises(ValueError):
            string_to_board("." * 63)

        # 65 tokens (too long by 1)
        with self.assertRaises(ValueError):
            string_to_board("." * 65)


if __name__ == "__main__":
    unittest.main()
