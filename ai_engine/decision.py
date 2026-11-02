"""Unified AI Decision Engine API for BLOCK BLAST AI ULTIMATE.

Provides the single entry point for deciding move sequences from a given GameState.
Translates solver search output into structured, verifiable DecisionResult objects
containing full placement coordinates, predicted scores, board evaluations, and rationales.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import enum
import time
from typing import Optional, Sequence

from ai_engine.baseline_solver import BaseSolver, GreedySolver, RandomSolver
from ai_engine.beam_search import BeamSearchSolver
from ai_engine.evaluator import BoardEvaluator
from game_engine.board import popcount
from game_engine.simulator import GameSimulator, StepResult
from game_engine.state import GameState


# ---------------------------------------------------------------------------
# Decision Status
# ---------------------------------------------------------------------------

class DecisionStatus(enum.Enum):
    """Status outcome of an AI decision query."""

    SUCCESS = "SUCCESS"                # Found a legal plan for 1 or more pieces
    NO_VALID_MOVE = "NO_VALID_MOVE"    # Tray has pieces, but none can legally fit
    GAME_OVER = "GAME_OVER"            # Game is already in a terminal Game Over state
    TRAY_EMPTY = "TRAY_EMPTY"          # Tray has 0 pieces (pending next deal)


# ---------------------------------------------------------------------------
# Planned Move Detail
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PlannedMove:
    """Detailed specifications for an individual piece placement within a round."""

    step_index: int
    slot_idx: int
    row: int
    col: int
    piece_name: str
    cell_count: int
    predicted_score_delta: int
    lines_cleared: int
    unique_cells_cleared: int
    combo_streak_after: int
    is_game_over: bool


# ---------------------------------------------------------------------------
# Unified Decision Result
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DecisionResult:
    """Comprehensive output container for an AI decision."""

    status: DecisionStatus
    strategy_name: str
    moves: tuple[PlannedMove, ...]
    predicted_score_delta: int
    final_board_evaluation: float
    rationale: str
    states_explored: int
    duration_ms: float
    initial_board: int
    final_board: int
    step_results: tuple[StepResult, ...] = field(default_factory=tuple)

    @property
    def move_count(self) -> int:
        return len(self.moves)

    @property
    def piece_order(self) -> list[int]:
        """Ordered list of tray slot indices selected."""
        return [m.slot_idx for m in self.moves]

    @property
    def total_lines_cleared(self) -> int:
        return sum(m.lines_cleared for m in self.moves)


# ---------------------------------------------------------------------------
# Unified AI Decision Engine
# ---------------------------------------------------------------------------

class AIDecisionEngine:
    """Unified controller that directs state evaluation and move planning."""

    def __init__(
        self,
        simulator: Optional[GameSimulator] = None,
        evaluator: Optional[BoardEvaluator] = None,
        default_strategy: str = "beam_search",
        default_beam_width: int = 15,
    ) -> None:
        self.sim = simulator or GameSimulator()
        self.evaluator = evaluator or BoardEvaluator()
        self.default_strategy = default_strategy
        self.default_beam_width = default_beam_width

    def _get_solver(self, strategy: str, **kwargs) -> BaseSolver:
        strat = strategy.lower().strip()
        if strat in ("beam", "beam_search"):
            bw = kwargs.get("beam_width", self.default_beam_width)
            depth = kwargs.get("max_depth", None)
            return BeamSearchSolver(
                beam_width=bw,
                max_depth=depth,
                simulator=self.sim,
                evaluator=self.evaluator,
            )
        elif strat in ("greedy", "greedy_baseline"):
            return GreedySolver(simulator=self.sim, evaluator=self.evaluator)
        elif strat in ("random", "random_legal"):
            seed = kwargs.get("seed", None)
            return RandomSolver(simulator=self.sim, evaluator=self.evaluator, seed=seed)
        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Supported: 'beam_search', 'greedy', 'random'."
            )

    def decide(
        self,
        state: GameState,
        strategy: Optional[str] = None,
        **kwargs,
    ) -> DecisionResult:
        """Analyze the given GameState and return the recommended move sequence.

        Args:
            state: The current GameState containing board bitboard and tray pieces.
            strategy: Strategy identifier ('beam_search', 'greedy', 'random').
            **kwargs: Extra parameters (beam_width, max_depth, seed).
        """
        start_time = time.perf_counter()

        # 1. Edge Case: Already Game Over
        if state.is_game_over:
            duration = (time.perf_counter() - start_time) * 1000.0
            return DecisionResult(
                status=DecisionStatus.GAME_OVER,
                strategy_name="None",
                moves=(),
                predicted_score_delta=0,
                final_board_evaluation=-1_000_000.0,
                rationale="State is already marked as Game Over; no legal moves exist.",
                states_explored=0,
                duration_ms=duration,
                initial_board=state.board,
                final_board=state.board,
            )

        # 2. Edge Case: Tray is empty
        if state.is_tray_empty():
            duration = (time.perf_counter() - start_time) * 1000.0
            return DecisionResult(
                status=DecisionStatus.TRAY_EMPTY,
                strategy_name="None",
                moves=(),
                predicted_score_delta=0,
                final_board_evaluation=self.evaluator.evaluate_board(state.board),
                rationale="Tray is currently empty; awaiting next round deal.",
                states_explored=0,
                duration_ms=duration,
                initial_board=state.board,
                final_board=state.board,
            )

        # 3. Edge Case: Has pieces in tray, but zero legal placements exist anywhere
        if not self.sim.has_any_legal_move(state):
            duration = (time.perf_counter() - start_time) * 1000.0
            return DecisionResult(
                status=DecisionStatus.NO_VALID_MOVE,
                strategy_name="None",
                moves=(),
                predicted_score_delta=0,
                final_board_evaluation=-1_000_000.0,
                rationale="No piece currently in the tray can fit anywhere on the 8x8 board.",
                states_explored=0,
                duration_ms=duration,
                initial_board=state.board,
                final_board=state.board,
            )

        # 4. Execute chosen solver
        chosen_strat = strategy or self.default_strategy
        solver = self._get_solver(chosen_strat, **kwargs)

        moves_tuples, step_results, states_explored = solver.find_round_plan(state)
        duration = (time.perf_counter() - start_time) * 1000.0

        if not moves_tuples or not step_results:
            return DecisionResult(
                status=DecisionStatus.NO_VALID_MOVE,
                strategy_name=solver.name,
                moves=(),
                predicted_score_delta=0,
                final_board_evaluation=-1_000_000.0,
                rationale="Solver explored moves but could not establish a viable placement.",
                states_explored=states_explored,
                duration_ms=duration,
                initial_board=state.board,
                final_board=state.board,
            )

        # 5. Build structured PlannedMove sequence
        planned_moves: list[PlannedMove] = []
        total_score_delta = 0
        total_lines_cleared = 0

        for i, step in enumerate(step_results):
            delta = step.score_breakdown.total_delta_score
            total_score_delta += delta
            total_lines_cleared += step.lines_cleared

            planned_moves.append(
                PlannedMove(
                    step_index=i,
                    slot_idx=step.slot_idx,
                    row=step.row,
                    col=step.col,
                    piece_name=step.piece.name,
                    cell_count=step.piece.cell_count,
                    predicted_score_delta=delta,
                    lines_cleared=step.lines_cleared,
                    unique_cells_cleared=step.unique_cells_cleared,
                    combo_streak_after=step.state_after.combo_streak,
                    is_game_over=step.is_game_over,
                )
            )

        final_board = step_results[-1].state_after.board
        final_eval = self.evaluator.evaluate_board(final_board)
        feats = self.evaluator.extract_features(final_board)

        # 6. Compose detailed rationale
        pieces_str = " -> ".join(
            f"slot {m.slot_idx} ({m.piece_name}) at ({m.row}, {m.col})" for m in planned_moves
        )
        rationale_parts = [
            f"Strategy [{solver.name}] planned {len(planned_moves)} move(s): {pieces_str}.",
            f"Earns +{total_score_delta} points, clearing {total_lines_cleared} line(s).",
            f"Post-turn board has {feats.empty_cells} empty cells (occ={feats.occupied_cells}/64).",
        ]
        if feats.fits_square_3x3:
            rationale_parts.append("Preserves headroom for 3x3 square.")
        else:
            rationale_parts.append("Caution: 3x3 square cannot currently fit.")

        if feats.isolated_holes > 0:
            rationale_parts.append(f"Contains {feats.isolated_holes} isolated hole(s).")

        return DecisionResult(
            status=DecisionStatus.SUCCESS,
            strategy_name=solver.name,
            moves=tuple(planned_moves),
            predicted_score_delta=total_score_delta,
            final_board_evaluation=final_eval,
            rationale=" ".join(rationale_parts),
            states_explored=states_explored,
            duration_ms=duration,
            initial_board=state.board,
            final_board=final_board,
            step_results=tuple(step_results),
        )
