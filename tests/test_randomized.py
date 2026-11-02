"""Property-based randomized stress testing for BLOCK BLAST AI ULTIMATE.

CRITICAL REQUIREMENT:
- Executes at least 50,000 randomized test cases.
- Validates 64-bit board invariants (zero bits beyond bit 63).
- Validates coordinate-to-bitboard mathematical bijection.
- Validates placement collision detection and popcount conservation.
- Validates simultaneous line detection and clear invariants.
"""

import random
import time
import unittest

from game_engine.board import (
    BOARD_SIZE,
    COL_MASKS,
    FULL_BOARD_MASK,
    ROW_MASKS,
    bitboard_to_coords,
    can_place,
    check_and_clear_lines,
    find_full_cols,
    find_full_rows,
    place,
    popcount,
)
from game_engine.pieces import CANONICAL_41_CATALOG


class TestRandomizedEngineProperties(unittest.TestCase):
    """50,000-case randomized fuzzing and property validation suite."""

    def test_randomized_50k_cases(self):
        """Execute 50,000 randomized board and placement tests."""
        num_cases = 50_000
        rng = random.Random(0xC0FFEE)
        catalog = CANONICAL_41_CATALOG
        pieces = catalog.all_pieces

        t0 = time.perf_counter()

        for case_idx in range(num_cases):
            # 1. Generate random 64-bit board
            board = rng.getrandbits(64)

            # Invariant: No bits outside 64-bit board mask
            self.assertEqual(board & ~FULL_BOARD_MASK, 0, f"Case {case_idx}: Board overflow")

            # 2. Pick a random piece and random coordinates (including out-of-bounds)
            piece = pieces[rng.randrange(len(pieces))]
            r = rng.randint(-3, 10)
            c = rng.randint(-3, 10)

            mask = piece.get_placement_mask(r, c)
            in_bounds = (0 <= r and r + piece.height <= BOARD_SIZE and
                         0 <= c and c + piece.width <= BOARD_SIZE)

            # Invariant: Boundary check consistency
            if in_bounds:
                self.assertIsNotNone(mask, f"Case {case_idx}: Mask should not be None for in-bounds")
                self.assertEqual(mask & ~FULL_BOARD_MASK, 0, f"Case {case_idx}: Mask bits exceed 64-bit")
                self.assertEqual(mask.bit_count(), piece.cell_count)

                # Invariant: Exact cell coordinate set match
                expected_cells = {(r + dr, c + dc) for dr, dc in piece.cells}
                actual_cells = bitboard_to_coords(mask)
                self.assertEqual(actual_cells, expected_cells, f"Case {case_idx}: Coords mismatch")

                # Invariant: can_place strictly matches overlap check
                overlap = (board & mask) != 0
                self.assertEqual(can_place(board, mask), not overlap)

                # If legal placement:
                if not overlap:
                    placed = place(board, mask)
                    self.assertEqual(placed & ~FULL_BOARD_MASK, 0)
                    self.assertEqual(
                        popcount(placed),
                        popcount(board) + piece.cell_count,
                        f"Case {case_idx}: Popcount conservation failed on placement",
                    )

                    # Invariant: Line clearing post-conditions
                    (
                        cleared_board,
                        lines_cleared,
                        unique_cells_cleared,
                        full_r,
                        full_c,
                        cmask,
                    ) = check_and_clear_lines(placed)

                    self.assertEqual(cleared_board & ~FULL_BOARD_MASK, 0)

                    # Verify all cleared rows are now empty in cleared_board
                    for fr in full_r:
                        self.assertEqual(cleared_board & ROW_MASKS[fr], 0)
                    # Verify all cleared cols are now empty in cleared_board
                    for fc in full_c:
                        self.assertEqual(cleared_board & COL_MASKS[fc], 0)

                    # Verify no full lines remain
                    self.assertEqual(find_full_rows(cleared_board), ())
                    self.assertEqual(find_full_cols(cleared_board), ())

                    # Verify cell conservation during clear
                    self.assertEqual(
                        unique_cells_cleared,
                        popcount(placed) - popcount(cleared_board),
                        f"Case {case_idx}: Cleared cells conservation failed",
                    )
            else:
                self.assertIsNone(mask, f"Case {case_idx}: Out of bounds mask must be None")

        t1 = time.perf_counter()
        elapsed = t1 - t0
        cases_per_sec = num_cases / elapsed if elapsed > 0 else 0
        print(f"\n[Randomized Property Test] Successfully verified {num_cases:,} cases in {elapsed:.2f}s ({cases_per_sec:,.0f} cases/sec).")


if __name__ == "__main__":
    unittest.main()
