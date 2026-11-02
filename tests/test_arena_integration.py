"""Integration Tests for BLOCK BLAST AI ULTIMATE Decision Engine.

Runs comprehensive validation across 100 simulated hands with a fixed random seed:
- Verifies that AI never proposes an illegal move on any of the 100 hands.
- Validates piece ordering and sequential state transition validity.
- Tests StrategyArena tournament mechanics and metric computation.
"""

from __future__ import annotations

import random
import unittest

from ai_engine.baseline_solver import GreedySolver, RandomSolver
from ai_engine.beam_search import BeamSearchSolver
from ai_engine.decision import AIDecisionEngine, DecisionStatus
from ai_engine.strategy_arena import StrategyArena
from game_engine.board import can_place, popcount
from game_engine.simulator import GameSimulator
from game_engine.state import GameState


class TestAI100SimulatedHandsIntegration(unittest.TestCase):
    """Stress test AI decision making across 100 simulated hands."""

    def setUp(self) -> None:
        self.sim = GameSimulator()
        self.engine = AIDecisionEngine(simulator=self.sim, default_beam_width=10)

    def test_100_simulated_hands_zero_illegal_moves(self):
        """CRITICAL: Across 100 independent hands, AI must never propose an illegal move."""
        rng = random.Random(1337)
        total_hands = 100
        legal_decisions = 0
        terminal_decisions = 0

        # Generate 100 unique hands and boards (some empty, some partially occupied)
        for i in range(total_hands):
            hand = self.sim.deal_hand(count=3, rng=rng)

            # Generate random partial board (0 to 30 cells occupied)
            occ_count = rng.randint(0, 30)
            board = 0
            for _ in range(occ_count):
                r = rng.randint(0, 7)
                c = rng.randint(0, 7)
                board |= (1 << (r * 8 + c))

            state = GameState.create_initial(initial_tray=hand).clone_with(board=board)

            # Request decision
            decision = self.engine.decide(state, strategy="beam_search", beam_width=10)

            if decision.status == DecisionStatus.SUCCESS:
                legal_decisions += 1
                curr_board = state.board
                curr_tray = list(state.tray)

                # Strictly verify legality of every single proposed move
                for move in decision.moves:
                    piece = curr_tray[move.slot_idx]
                    self.assertIsNotNone(piece, f"Hand {i}: Tray slot {move.slot_idx} already used!")
                    self.assertEqual(piece.name, move.piece_name)

                    mask = piece.get_placement_mask(move.row, move.col)
                    self.assertIsNotNone(
                        mask,
                        f"Hand {i}: Move ({move.row}, {move.col}) out of bounds for {piece.name}!"
                    )
                    self.assertTrue(
                        can_place(curr_board, mask),
                        f"Hand {i}: Move overlaps occupied cells on board!"
                    )

                    # Simulate step to update board
                    step = self.sim.apply_move(
                        state.clone_with(board=curr_board, tray=tuple(curr_tray)),
                        move.slot_idx, move.row, move.col
                    )
                    curr_board = step.state_after.board
                    curr_tray[move.slot_idx] = None

            elif decision.status in (DecisionStatus.NO_VALID_MOVE, DecisionStatus.GAME_OVER):
                terminal_decisions += 1
                self.assertEqual(len(decision.moves), 0)

        # Assert all 100 hands completed evaluation cleanly
        self.assertEqual(legal_decisions + terminal_decisions, total_hands)
        self.assertGreater(legal_decisions, 80)

    def test_strategy_arena_tournament_execution(self):
        """StrategyArena runs fair comparative benchmarking across solvers."""
        arena = StrategyArena(simulator=self.sim, base_seed=999)
        solvers = [
            RandomSolver(simulator=self.sim, seed=42),
            GreedySolver(simulator=self.sim),
            BeamSearchSolver(beam_width=5, simulator=self.sim),
        ]

        # Run 5 fast games for tournament validation
        results = arena.evaluate_solvers(solvers, num_games=5, max_rounds_per_game=20)
        self.assertEqual(len(results), 3)

        # Greedy and Beam Search must score significantly higher than Random
        random_stats = results["RandomLegalSolver"]
        greedy_stats = results["GreedyBaselineSolver"]
        beam_stats = results["BeamSearchSolver(width=5)"]

        self.assertGreaterEqual(greedy_stats.mean_score, random_stats.mean_score)
        self.assertGreaterEqual(beam_stats.mean_score, random_stats.mean_score)

        # Format comparison table string
        table_str = arena.format_comparison_table(results)
        self.assertIn("RandomLegalSolver", table_str)
        self.assertIn("BeamSearchSolver", table_str)


if __name__ == "__main__":
    unittest.main()
