"""Baseline Move Solvers for BLOCK BLAST AI ULTIMATE.

Contains reference implementations for comparative evaluation:
1. BaseSolver: Abstract interface for all decision strategies.
2. RandomSolver: Picks uniformly at random among all legal placements.
3. GreedySolver: Selects the move that maximizes immediate 1-step heuristic score.
"""

from __future__ import annotations

import abc
import random
from typing import Optional

from ai_engine.evaluator import BoardEvaluator
from game_engine.board import popcount
from game_engine.simulator import GameSimulator, StepResult
from game_engine.state import GameState


# ---------------------------------------------------------------------------
# Base Solver Interface
# ---------------------------------------------------------------------------

class BaseSolver(abc.ABC):
    """Abstract interface for all Block Blast decision solvers."""

    def __init__(
        self,
        simulator: Optional[GameSimulator] = None,
        evaluator: Optional[BoardEvaluator] = None,
    ) -> None:
        self.sim = simulator or GameSimulator()
        self.evaluator = evaluator or BoardEvaluator()

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable identifier for the strategy."""
        pass

    @abc.abstractmethod
    def find_round_plan(
        self,
        state: GameState,
    ) -> tuple[list[tuple[int, int, int]], list[StepResult], int]:
        """Find a sequence of (slot_idx, row, col) placements for pieces currently in the tray.

        Returns:
            moves: List of (slot_idx, row, col) tuples.
            step_results: StepResult for each simulated step.
            states_explored: Total candidate states generated/evaluated.
        """
        pass


# ---------------------------------------------------------------------------
# Random Legal Solver
# ---------------------------------------------------------------------------

class RandomSolver(BaseSolver):
    """Baseline solver that picks uniformly among available legal moves."""

    def __init__(
        self,
        simulator: Optional[GameSimulator] = None,
        evaluator: Optional[BoardEvaluator] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(simulator, evaluator)
        self.rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "RandomLegalSolver"

    def find_round_plan(
        self,
        state: GameState,
    ) -> tuple[list[tuple[int, int, int]], list[StepResult], int]:
        current_state = state
        moves: list[tuple[int, int, int]] = []
        steps: list[StepResult] = []
        states_explored = 0

        while not current_state.is_tray_empty() and not current_state.is_game_over:
            legal_moves = self.sim.get_legal_moves(current_state)
            states_explored += len(legal_moves)
            if not legal_moves:
                break

            chosen_move = self.rng.choice(legal_moves)
            slot_idx, r, c = chosen_move
            step = self.sim.apply_move(current_state, slot_idx, r, c)
            moves.append(chosen_move)
            steps.append(step)
            current_state = step.state_after

        return moves, steps, states_explored


# ---------------------------------------------------------------------------
# Greedy Baseline Solver (1-Step Lookahead)
# ---------------------------------------------------------------------------

class GreedySolver(BaseSolver):
    """Greedy baseline solver that chooses the move maximizing immediate 1-step score."""

    def __init__(
        self,
        simulator: Optional[GameSimulator] = None,
        evaluator: Optional[BoardEvaluator] = None,
    ) -> None:
        super().__init__(simulator, evaluator)

    @property
    def name(self) -> str:
        return "GreedyBaselineSolver"

    def find_round_plan(
        self,
        state: GameState,
    ) -> tuple[list[tuple[int, int, int]], list[StepResult], int]:
        current_state = state
        moves: list[tuple[int, int, int]] = []
        steps: list[StepResult] = []
        states_explored = 0

        while not current_state.is_tray_empty() and not current_state.is_game_over:
            legal_moves = self.sim.get_legal_moves(current_state)
            states_explored += len(legal_moves)
            if not legal_moves:
                break

            best_move: Optional[tuple[int, int, int]] = None
            best_step: Optional[StepResult] = None
            best_eval: float = float("-inf")

            for slot_idx, r, c in legal_moves:
                step = self.sim.apply_move(current_state, slot_idx, r, c)
                eval_val = self.evaluator.evaluate_step(
                    board_after=step.state_after.board,
                    lines_cleared=step.lines_cleared,
                    unique_cells_cleared=step.unique_cells_cleared,
                    combo_streak=step.state_after.combo_streak,
                    score_delta=step.score_breakdown.total_delta_score,
                    is_game_over=step.is_game_over,
                )

                if eval_val > best_eval:
                    best_eval = eval_val
                    best_move = (slot_idx, r, c)
                    best_step = step

            if best_move is None or best_step is None:
                break

            moves.append(best_move)
            steps.append(best_step)
            current_state = best_step.state_after

        return moves, steps, states_explored
