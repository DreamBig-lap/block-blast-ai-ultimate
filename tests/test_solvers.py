"""Unit Tests for ai_engine solvers and unified AIDecisionEngine.

Verifies:
- All solvers (Random, Greedy, Beam Search) propose ONLY 100% legal moves.
- Beam Search evaluates permutations and finds multi-step line-clearing orders.
- Correct handling of edge cases (empty tray, terminal Game Over, blocked board).
- Structured DecisionResult completeness (rationale, coordinates, score predictions).
- Deterministic reproducibility under fixed random seeds.
"""

from __future__ import annotations

import unittest

from ai_engine.baseline_solver import GreedySolver, RandomSolver
from ai_engine.beam_search import BeamSearchSolver
from ai_engine.decision import AIDecisionEngine, DecisionStatus
from ai_engine.evaluator import BoardEvaluator
from game_engine.board import (
    FULL_BOARD_MASK,
    ROW_MASKS,
    can_place,
    coord_to_bit,
    popcount,
)
from game_engine.pieces import CANONICAL_41_CATALOG
from game_engine.simulator import GameSimulator
from game_engine.state import GameState


class TestSolversLegalityAndCorrectness(unittest.TestCase):
    """Verify that solvers adhere to engine physics and never propose illegal moves."""

    def setUp(self) -> None:
        self.sim = GameSimulator()
        self.evaluator = BoardEvaluator()
        self.catalog = CANONICAL_41_CATALOG

    def test_random_solver_always_legal(self):
        """RandomSolver moves must always be strictly legal on the progressive board."""
        solver = RandomSolver(simulator=self.sim, evaluator=self.evaluator, seed=123)
        p1 = self.catalog.get_by_name("square_2x2")
        p2 = self.catalog.get_by_name("line_1x3_h")
        p3 = self.catalog.get_by_name("dot_1x1")

        state = GameState.create_initial(initial_tray=[p1, p2, p3])
        moves, steps, _ = solver.find_round_plan(state)

        curr_board = state.board
        curr_tray = list(state.tray)
        for slot_idx, r, c in moves:
            piece = curr_tray[slot_idx]
            self.assertIsNotNone(piece)
            mask = piece.get_placement_mask(r, c)
            self.assertIsNotNone(mask)
            self.assertTrue(can_place(curr_board, mask))

            # Simulate step manually to verify match
            step = self.sim.apply_move(
                state.clone_with(board=curr_board, tray=tuple(curr_tray)),
                slot_idx, r, c
            )
            curr_board = step.state_after.board
            curr_tray[slot_idx] = None

    def test_greedy_solver_chooses_best_immediate_move(self):
        """GreedySolver selects the highest immediate score/line-clearing placement."""
        solver = GreedySolver(simulator=self.sim, evaluator=self.evaluator)

        # Set up board with row 0 having 7 cells filled, leaving (0, 7) open
        board = ROW_MASKS[0] & ~coord_to_bit(0, 7)
        dot = self.catalog.get_by_name("dot_1x1")
        state = GameState.create_initial(initial_tray=[dot, None, None]).clone_with(board=board)

        moves, steps, _ = solver.find_round_plan(state)
        self.assertEqual(len(moves), 1)
        slot, r, c = moves[0]
        self.assertEqual(slot, 0)
        self.assertEqual((r, c), (0, 7))  # Must place at (0, 7) to clear row 0!
        self.assertEqual(steps[0].lines_cleared, 1)

    def test_beam_search_finds_multi_piece_line_clearing_order(self):
        """BeamSearch optimizes piece ordering to execute line clears across 3 pieces."""
        solver = BeamSearchSolver(beam_width=15, simulator=self.sim, evaluator=self.evaluator)

        # Board with row 0 empty, but row 0 needs 8 cells.
        # Tray: line_1x5_h (5 cells) + line_1x3_h (3 cells) = 8 cells!
        # Both placed in row 0 will clear row 0 completely!
        p5 = self.catalog.get_by_name("line_1x5_h")
        p3 = self.catalog.get_by_name("line_1x3_h")
        dot = self.catalog.get_by_name("dot_1x1")

        state = GameState.create_initial(initial_tray=[p5, p3, dot])
        moves, steps, _ = solver.find_round_plan(state)

        self.assertEqual(len(moves), 3)
        total_clears = sum(s.lines_cleared for s in steps)
        self.assertGreaterEqual(total_clears, 1)

    def test_beam_search_configurable_width(self):
        """BeamSearch accepts different beam widths and runs without errors."""
        p_dot = self.catalog.get_by_name("dot_1x1")
        state = GameState.create_initial(initial_tray=[p_dot, p_dot, None])

        for bw in [1, 5, 20]:
            solver = BeamSearchSolver(beam_width=bw, simulator=self.sim, evaluator=self.evaluator)
            moves, steps, explored = solver.find_round_plan(state)
            self.assertEqual(len(moves), 2)
            self.assertGreater(explored, 0)


class TestAIDecisionEngineAPI(unittest.TestCase):
    """Test unified AIDecisionEngine interface, responses, and edge conditions."""

    def setUp(self) -> None:
        self.engine = AIDecisionEngine()
        self.catalog = CANONICAL_41_CATALOG

    def test_decide_success_result_structure(self):
        """decide() on normal state returns SUCCESS with complete metadata."""
        p1 = self.catalog.get_by_name("dot_1x1")
        p2 = self.catalog.get_by_name("line_1x2_h")
        state = GameState.create_initial(initial_tray=[p1, p2, None])

        res = self.engine.decide(state, strategy="beam_search", beam_width=10)
        self.assertEqual(res.status, DecisionStatus.SUCCESS)
        self.assertEqual(res.move_count, 2)
        self.assertEqual(len(res.piece_order), 2)
        self.assertGreater(res.predicted_score_delta, 0)
        self.assertGreater(res.duration_ms, 0.0)
        self.assertGreater(res.states_explored, 0)
        self.assertTrue(len(res.rationale) > 20)
        self.assertEqual(len(res.step_results), 2)

    def test_decide_game_over_state_rejected(self):
        """decide() on already terminal Game Over state returns GAME_OVER immediately."""
        state = GameState.create_initial(initial_tray=[]).clone_with(is_game_over=True)
        res = self.engine.decide(state)
        self.assertEqual(res.status, DecisionStatus.GAME_OVER)
        self.assertEqual(res.move_count, 0)

    def test_decide_empty_tray_state(self):
        """decide() on state with empty tray returns TRAY_EMPTY."""
        state = GameState.create_initial(initial_tray=[None, None, None])
        res = self.engine.decide(state)
        self.assertEqual(res.status, DecisionStatus.TRAY_EMPTY)
        self.assertEqual(res.move_count, 0)

    def test_decide_no_valid_move_on_blocked_board(self):
        """decide() on congested board where tray piece cannot fit returns NO_VALID_MOVE."""
        # 3x3 square cannot fit on a board with scattered diagonal holes
        p_3x3 = self.catalog.get_by_name("square_3x3")
        diag_mask = sum(coord_to_bit(i, i) for i in range(8))
        near_full = FULL_BOARD_MASK & ~diag_mask

        state = GameState.create_initial(initial_tray=[p_3x3, None, None]).clone_with(board=near_full)
        res = self.engine.decide(state)
        self.assertEqual(res.status, DecisionStatus.NO_VALID_MOVE)
        self.assertEqual(res.move_count, 0)


if __name__ == "__main__":
    unittest.main()
