"""High-Performance Beam Search Solver for BLOCK BLAST AI ULTIMATE.

CRITICAL DESIGN PRINCIPLES:
- Multi-step lookahead across all active pieces in the tray (typically 3 pieces).
- Explores permutations of piece placement orders deterministically.
- Prunes unviable branches and retains top K (beam_width) candidate states at each depth.
- Balances high-scoring line clears with long-term board topological health.
- Extremely lightweight and CPU-friendly (< 50 ms per decision on Intel i5-5300U).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ai_engine.baseline_solver import BaseSolver
from ai_engine.evaluator import BoardEvaluator
from game_engine.simulator import GameSimulator, StepResult
from game_engine.state import GameState


# ---------------------------------------------------------------------------
# Search Path Representation
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SearchNode:
    """A partial or complete search path explored by Beam Search."""

    state: GameState
    moves: list[tuple[int, int, int]] = field(default_factory=list)
    steps: list[StepResult] = field(default_factory=list)
    cumulative_eval: float = 0.0

    @property
    def depth(self) -> int:
        return len(self.moves)


# ---------------------------------------------------------------------------
# Beam Search Solver
# ---------------------------------------------------------------------------

class BeamSearchSolver(BaseSolver):
    """Multi-depth Beam Search solver exploring permutations of tray placements."""

    def __init__(
        self,
        beam_width: int = 15,
        max_depth: Optional[int] = None,
        simulator: Optional[GameSimulator] = None,
        evaluator: Optional[BoardEvaluator] = None,
    ) -> None:
        super().__init__(simulator, evaluator)
        if beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, got {beam_width}")
        self.beam_width = beam_width
        self.max_depth = max_depth

    @property
    def name(self) -> str:
        return f"BeamSearchSolver(width={self.beam_width})"

    def find_round_plan(
        self,
        state: GameState,
    ) -> tuple[list[tuple[int, int, int]], list[StepResult], int]:
        """Find the optimal sequence of moves for current tray pieces via Beam Search.

        Returns:
            moves: List of (slot_idx, row, col) tuples.
            step_results: StepResult for each simulated step.
            states_explored: Total candidate states generated/evaluated.
        """
        if state.is_tray_empty() or state.is_game_over:
            return [], [], 0

        target_depth = self.max_depth or state.active_piece_count()
        states_explored = 0

        # Beam initialized with root state
        current_beam: list[SearchNode] = [
            SearchNode(state=state, moves=[], steps=[], cumulative_eval=0.0)
        ]

        completed_paths: list[SearchNode] = []

        for current_depth in range(target_depth):
            candidates: list[SearchNode] = []

            for node in current_beam:
                # If node has reached terminal state or tray is empty, save to completed
                if node.state.is_tray_empty() or node.state.is_game_over:
                    completed_paths.append(node)
                    continue

                legal_moves = self.sim.get_legal_moves(node.state)
                states_explored += len(legal_moves)

                if not legal_moves:
                    # Dead end: cannot place any remaining pieces from this state
                    completed_paths.append(node)
                    continue

                for slot_idx, r, c in legal_moves:
                    step = self.sim.apply_move(node.state, slot_idx, r, c)
                    step_eval = self.evaluator.evaluate_step(
                        board_after=step.state_after.board,
                        lines_cleared=step.lines_cleared,
                        unique_cells_cleared=step.unique_cells_cleared,
                        combo_streak=step.state_after.combo_streak,
                        score_delta=step.score_breakdown.total_delta_score,
                        is_game_over=step.is_game_over,
                    )

                    new_eval = node.cumulative_eval + step_eval
                    child_node = SearchNode(
                        state=step.state_after,
                        moves=node.moves + [(slot_idx, r, c)],
                        steps=node.steps + [step],
                        cumulative_eval=new_eval,
                    )
                    candidates.append(child_node)

            if not candidates:
                # No further expansion possible
                break

            # Sort descending by cumulative evaluation score and retain top K
            candidates.sort(key=lambda n: n.cumulative_eval, reverse=True)
            current_beam = candidates[: self.beam_width]

        # Combine surviving beam candidates with completed paths
        all_candidates = completed_paths + current_beam
        if not all_candidates:
            return [], [], states_explored

        # Selection criteria:
        # 1. Prefer paths that placed the maximum number of pieces (completed round)
        # 2. Break ties with highest cumulative evaluation score
        # 3. Ensure no Game Over in final state if non-game-over alternatives exist
        def path_sort_key(n: SearchNode) -> tuple[int, bool, float]:
            placed_count = len(n.moves)
            not_over = not n.state.is_game_over
            return (placed_count, not_over, n.cumulative_eval)

        best_node = max(all_candidates, key=path_sort_key)
        return best_node.moves, best_node.steps, states_explored
