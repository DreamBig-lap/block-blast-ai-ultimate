"""Strategy Arena for Comparative Evaluation of AI Solvers.

Evaluates multiple solvers (Random, Greedy, Beam Search) across identical
pre-generated hands and seeds to ensure rigorous, scientifically fair benchmarking.
Tracks survival rounds, total score, lines cleared, decision latency, and states explored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
import statistics
import time
from typing import Optional, Sequence

from ai_engine.baseline_solver import BaseSolver, GreedySolver, RandomSolver
from ai_engine.beam_search import BeamSearchSolver
from ai_engine.evaluator import BoardEvaluator
from game_engine.pieces import Piece
from game_engine.simulator import GameSimulator
from game_engine.state import GameState


# ---------------------------------------------------------------------------
# Individual Strategy Game Record
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class GameRecord:
    """Record of a single game run by a solver."""

    game_index: int
    final_score: int
    rounds_survived: int
    moves_placed: int
    lines_cleared: int
    total_latency_ms: float
    total_states_explored: int
    round_latencies: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Strategy Aggregate Statistics
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class StrategyStats:
    """Aggregated statistics across multiple games for a given strategy."""

    strategy_name: str
    games_played: int
    mean_score: float
    std_score: float
    max_score: int
    min_score: int
    mean_rounds: float
    mean_moves: float
    mean_lines_cleared: float
    mean_latency_per_round_ms: float
    mean_states_per_round: float


# ---------------------------------------------------------------------------
# Strategy Arena Runner
# ---------------------------------------------------------------------------

class StrategyArena:
    """Runs head-to-head evaluation of multiple AI solvers on identical hands."""

    def __init__(
        self,
        simulator: Optional[GameSimulator] = None,
        evaluator: Optional[BoardEvaluator] = None,
        base_seed: int = 42,
    ) -> None:
        self.sim = simulator or GameSimulator()
        self.evaluator = evaluator or BoardEvaluator()
        self.base_seed = base_seed

    def _generate_game_hands(self, seed: int, max_rounds: int) -> list[tuple[Piece, ...]]:
        """Pre-generate a deterministic sequence of hands for a game."""
        rng = random.Random(seed)
        return [self.sim.deal_hand(count=3, rng=rng) for _ in range(max_rounds)]

    def play_single_game(
        self,
        solver: BaseSolver,
        hands: Sequence[tuple[Piece, ...]],
        game_index: int = 0,
        max_rounds: int = 200,
    ) -> GameRecord:
        """Run a single game using the provided solver and fixed sequence of hands."""
        state = GameState.create_initial(initial_tray=hands[0])
        round_idx = 0
        total_moves = 0
        total_lines = 0
        total_states = 0
        round_latencies: list[float] = []

        while round_idx < len(hands) and round_idx < max_rounds:
            t0 = time.perf_counter()
            moves, steps, states_explored = solver.find_round_plan(state)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            round_latencies.append(dt_ms)
            total_states += states_explored

            if not moves or not steps:
                # No legal moves possible for current hand -> Game Over
                break

            total_moves += len(moves)
            total_lines += sum(s.lines_cleared for s in steps)

            # Move state forward
            final_step = steps[-1]
            state = final_step.state_after

            if state.is_game_over:
                break

            # If all 3 pieces were placed, deal next hand from fixed sequence
            if state.is_tray_empty():
                round_idx += 1
                if round_idx < len(hands):
                    state = self.sim.refill_tray(state, new_pieces=hands[round_idx])
                    if state.is_game_over:
                        break
            else:
                # Partial placement that could not finish the hand
                break

        return GameRecord(
            game_index=game_index,
            final_score=state.score,
            rounds_survived=round_idx,
            moves_placed=total_moves,
            lines_cleared=total_lines,
            total_latency_ms=sum(round_latencies),
            total_states_explored=total_states,
            round_latencies=round_latencies,
        )

    def evaluate_solvers(
        self,
        solvers: Sequence[BaseSolver],
        num_games: int = 20,
        max_rounds_per_game: int = 150,
    ) -> dict[str, StrategyStats]:
        """Run head-to-head tournament across all solvers using identical seeds and hands."""
        # Pre-generate hand sequences for all games
        all_game_hands = [
            self._generate_game_hands(self.base_seed + g * 1000, max_rounds_per_game)
            for g in range(num_games)
        ]

        results: dict[str, StrategyStats] = {}

        for solver in solvers:
            records: list[GameRecord] = []
            for g_idx in range(num_games):
                rec = self.play_single_game(
                    solver=solver,
                    hands=all_game_hands[g_idx],
                    game_index=g_idx,
                    max_rounds=max_rounds_per_game,
                )
                records.append(rec)

            scores = [r.final_score for r in records]
            rounds = [r.rounds_survived for r in records]
            moves = [r.moves_placed for r in records]
            lines = [r.lines_cleared for r in records]

            all_round_lats = [lat for r in records for lat in r.round_latencies]
            mean_lat = statistics.mean(all_round_lats) if all_round_lats else 0.0
            total_rounds_all = sum(rounds) or 1
            mean_states = sum(r.total_states_explored for r in records) / total_rounds_all

            mean_s = statistics.mean(scores)
            std_s = statistics.stdev(scores) if len(scores) > 1 else 0.0

            stats = StrategyStats(
                strategy_name=solver.name,
                games_played=num_games,
                mean_score=mean_s,
                std_score=std_s,
                max_score=max(scores),
                min_score=min(scores),
                mean_rounds=statistics.mean(rounds),
                mean_moves=statistics.mean(moves),
                mean_lines_cleared=statistics.mean(lines),
                mean_latency_per_round_ms=mean_lat,
                mean_states_per_round=mean_states,
            )
            results[solver.name] = stats

        return results

    @staticmethod
    def format_comparison_table(results: dict[str, StrategyStats]) -> str:
        """Format arena evaluation results into a Markdown table."""
        header = (
            "| Chiến thuật (Strategy) | Điểm TB (Mean) | Điểm Max | Vòng TB | Nước đặt TB | Dòng xóa TB | Độ trễ/Vòng (ms) | Trạng thái/Vòng |\n"
            "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
        )
        rows: list[str] = [header]
        for name, s in results.items():
            row = (
                f"| **{s.strategy_name}** | {s.mean_score:.1f} ± {s.std_score:.1f} | "
                f"{s.max_score} | {s.mean_rounds:.1f} | {s.mean_moves:.1f} | "
                f"{s.mean_lines_cleared:.1f} | {s.mean_latency_per_round_ms:.2f} ms | "
                f"{s.mean_states_per_round:.0f} |"
            )
            rows.append(row)
        return "\n".join(rows)
