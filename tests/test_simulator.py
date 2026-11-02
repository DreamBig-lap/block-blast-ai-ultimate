"""Unit tests for Game Simulator (game_engine/simulator.py)."""

import random
import unittest

from game_engine.board import FULL_BOARD_MASK, ROW_MASKS, coord_to_bit, popcount
from game_engine.pieces import CANONICAL_41_CATALOG
from game_engine.rules import DEFAULT_RULES
from game_engine.simulator import GameSimulator, IllegalMoveError
from game_engine.state import GameState


class TestGameSimulator(unittest.TestCase):
    """Test game simulation, sequential moves, round transitions, and determinism."""

    def setUp(self):
        self.rules = DEFAULT_RULES
        self.catalog = CANONICAL_41_CATALOG
        self.sim = GameSimulator(rules=self.rules, catalog=self.catalog)

    def test_move_legality_checks(self):
        """Simulator correctly verifies move legality before execution."""
        p_mono = self.catalog.get_by_name("dot_1x1")
        p_square = self.catalog.get_by_name("square_2x2")
        state = GameState.create_initial(initial_tray=[p_mono, p_square, None])

        # Legal placements
        self.assertTrue(self.sim.is_legal_move(state, slot_idx=0, row=0, col=0))
        self.assertTrue(self.sim.is_legal_move(state, slot_idx=1, row=2, col=2))

        # Illegal: empty slot
        self.assertFalse(self.sim.is_legal_move(state, slot_idx=2, row=0, col=0))
        # Illegal: out of range slot
        self.assertFalse(self.sim.is_legal_move(state, slot_idx=5, row=0, col=0))
        # Illegal: out of bounds coordinate for 2x2
        self.assertFalse(self.sim.is_legal_move(state, slot_idx=1, row=7, col=0))

        # Illegal: cell collision
        state_blocked = state.clone_with(board=coord_to_bit(0, 0))
        self.assertFalse(self.sim.is_legal_move(state_blocked, slot_idx=0, row=0, col=0))

    def test_apply_move_state_transition(self):
        """Applying a move updates board, score, tray, and round metadata."""
        p1 = self.catalog.get_by_name("line_1x2_h")  # 2 cells
        p2 = self.catalog.get_by_name("dot_1x1")     # 1 cell
        state = GameState.create_initial(initial_tray=[p1, p2, None])

        step = self.sim.apply_move(state, slot_idx=0, row=0, col=0)
        self.assertEqual(step.lines_cleared, 0)
        self.assertEqual(step.score_breakdown.total_delta_score, 2)
        self.assertEqual(step.state_after.score, 2)
        self.assertEqual(step.state_after.tray[0], None)
        self.assertEqual(step.state_after.tray[1], p2)
        self.assertEqual(popcount(step.state_after.board), 2)
        self.assertFalse(step.round_completed)
        self.assertFalse(step.is_game_over)

    def test_illegal_move_raises_error(self):
        """Attempting to execute an illegal move raises IllegalMoveError."""
        p1 = self.catalog.get_by_name("dot_1x1")
        state = GameState.create_initial(initial_tray=[p1, None, None])

        # Out of bounds
        with self.assertRaises(IllegalMoveError):
            self.sim.apply_move(state, slot_idx=0, row=8, col=0)

        # Empty slot
        with self.assertRaises(IllegalMoveError):
            self.sim.apply_move(state, slot_idx=1, row=0, col=0)

    def test_apply_move_on_game_over_rejected(self):
        """apply_move() on a terminal Game Over state must be rejected immediately."""
        p1 = self.catalog.get_by_name("dot_1x1")
        state = GameState.create_initial(initial_tray=[p1, None, None])
        state_game_over = state.clone_with(is_game_over=True)

        with self.assertRaises(IllegalMoveError):
            self.sim.apply_move(state_game_over, slot_idx=0, row=0, col=0)

    def test_refill_tray_non_empty_rejected(self):
        """refill_tray() must raise ValueError if current tray still holds unplaced pieces."""
        p_dot = self.catalog.get_by_name("dot_1x1")
        # State with active piece in tray
        state_non_empty = GameState.create_initial(initial_tray=[p_dot, None, None])
        self.assertFalse(state_non_empty.is_tray_empty())

        with self.assertRaises(ValueError):
            self.sim.refill_tray(state_non_empty)

        # Forced refill with force=True must succeed
        state_forced = self.sim.refill_tray(state_non_empty, force=True)
        self.assertEqual(len(state_forced.tray), 3)

    def test_round_completion_and_round_number(self):
        """When all pieces in the tray are placed, round_completed is True and round increments."""
        p_dot = self.catalog.get_by_name("dot_1x1")
        state = GameState.create_initial(initial_tray=[p_dot, p_dot, p_dot])
        self.assertEqual(state.round_number, 1)

        # Move 1
        s1 = self.sim.apply_move(state, slot_idx=0, row=0, col=0)
        self.assertFalse(s1.round_completed)
        self.assertEqual(s1.state_after.round_number, 1)

        # Move 2
        s2 = self.sim.apply_move(s1.state_after, slot_idx=1, row=1, col=1)
        self.assertFalse(s2.round_completed)
        self.assertEqual(s2.state_after.round_number, 1)

        # Move 3: Last piece in tray!
        s3 = self.sim.apply_move(s2.state_after, slot_idx=2, row=2, col=2)
        self.assertTrue(s3.round_completed)
        self.assertEqual(s3.state_after.round_number, 2)
        self.assertTrue(s3.state_after.is_tray_empty())

    def test_game_over_detection(self):
        """When no available tray piece fits anywhere on the board, game over is triggered."""
        # Diagonal empty cells pattern: no row or col is fully empty or fully full
        diag_mask = sum(coord_to_bit(i, i) for i in range(8))
        near_full_board = FULL_BOARD_MASK & ~diag_mask
        p_3x3 = self.catalog.get_by_name("square_3x3")
        p_dot = self.catalog.get_by_name("dot_1x1")

        # Case A: Tray has 3x3 square. It cannot fit on scattered diagonal cells. Game is OVER.
        state_over = GameState.create_initial(
            initial_tray=[p_3x3, None, None]
        ).clone_with(board=near_full_board)
        self.assertTrue(self.sim.check_game_over(state_over))

        # Case B: Tray has 1x1 dot. It fits at (0, 0). Game is NOT over.
        state_alive = GameState.create_initial(
            initial_tray=[p_dot, None, None]
        ).clone_with(board=near_full_board)
        self.assertFalse(self.sim.check_game_over(state_alive))

        # Case C: Move that leads to Game Over
        # Diagonal empty + (0, 1) open so dot can be placed at (0, 0)
        # Even after clearing row 0, 3x3 square needs 3 consecutive empty rows and cannot fit!
        board_diagonal_open = FULL_BOARD_MASK & ~(diag_mask | coord_to_bit(0, 1))
        state_turn = GameState.create_initial(
            initial_tray=[p_dot, p_3x3, None]
        ).clone_with(board=board_diagonal_open)
        self.assertFalse(state_turn.is_game_over)

        step = self.sim.apply_move(state_turn, slot_idx=0, row=0, col=0)
        self.assertTrue(step.is_game_over)
        self.assertTrue(step.state_after.is_game_over)

    def test_deterministic_simulation_reproducibility(self):
        """CRITICAL: Same initial state + same moves must yield bitwise identical outcomes."""
        rng1 = random.Random(42)
        rng2 = random.Random(42)

        sim1 = GameSimulator(rules=self.rules, catalog=self.catalog, rng=rng1)
        sim2 = GameSimulator(rules=self.rules, catalog=self.catalog, rng=rng2)

        hand1 = sim1.deal_hand(count=3)
        hand2 = sim2.deal_hand(count=3)
        self.assertEqual([p.name for p in hand1], [p.name for p in hand2])

        state1 = GameState.create_initial(initial_tray=hand1)
        state2 = GameState.create_initial(initial_tray=hand2)

        # Generate a sequence of moves
        moves = [(0, 0, 0), (1, 3, 3)]
        steps1 = sim1.simulate_moves(state1, moves)
        steps2 = sim2.simulate_moves(state2, moves)

        self.assertEqual(len(steps1), len(steps2))
        for st1, st2 in zip(steps1, steps2):
            self.assertEqual(st1.state_after.board, st2.state_after.board)
            self.assertEqual(st1.state_after.score, st2.state_after.score)
            self.assertEqual(st1.state_after.combo_streak, st2.state_after.combo_streak)
            self.assertEqual(st1.lines_cleared, st2.lines_cleared)
            self.assertEqual(st1.is_game_over, st2.is_game_over)


if __name__ == "__main__":
    unittest.main()
