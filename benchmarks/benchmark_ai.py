"""Comprehensive Benchmark Suite for BLOCK BLAST AI ULTIMATE Decision Engine.

Measures:
1. BoardEvaluator throughput and feature extraction latency.
2. Solver decision latency and states/sec across different beam widths.
3. Head-to-head Strategy Arena tournament results comparing Random, Greedy, and Beam Search.
"""

from __future__ import annotations

import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ai_engine.baseline_solver import GreedySolver, RandomSolver
from ai_engine.beam_search import BeamSearchSolver
from ai_engine.decision import AIDecisionEngine
from ai_engine.evaluator import BoardEvaluator
from ai_engine.strategy_arena import StrategyArena
from game_engine.board import ROW_MASKS, coord_to_bit
from game_engine.pieces import CANONICAL_41_CATALOG
from game_engine.simulator import GameSimulator
from game_engine.state import GameState


def benchmark_evaluator(evaluator: BoardEvaluator, iterations: int = 200_000) -> None:
    """Benchmark raw board evaluation and feature extraction throughput."""
    print("\n[Benchmark 1] BoardEvaluator Throughput & Latency...")
    # Representative non-trivial board
    test_board = (
        ROW_MASKS[0] & ~coord_to_bit(0, 7) |
        ROW_MASKS[2] |
        coord_to_bit(4, 4) |
        coord_to_bit(5, 5)
    )

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = evaluator.evaluate_board(test_board)
    dt = time.perf_counter() - t0

    ops_sec = iterations / dt
    lat_us = (dt / iterations) * 1_000_000.0
    print(f"  {iterations:,} evaluate_board calls in {dt:.3f}s -> {ops_sec:,.0f} ops/sec ({lat_us:.2f} us/eval)")


def benchmark_solvers_latency(sim: GameSimulator, evaluator: BoardEvaluator, rounds: int = 50) -> None:
    """Benchmark average decision latency across 50 representative hands."""
    print(f"\n[Benchmark 2] Solver Decision Latency across {rounds} random hands...")

    hands = [sim.deal_hand(count=3) for _ in range(rounds)]
    states = [GameState.create_initial(initial_tray=h) for h in hands]

    solvers = [
        RandomSolver(simulator=sim, seed=42),
        GreedySolver(simulator=sim, evaluator=evaluator),
        BeamSearchSolver(beam_width=5, simulator=sim, evaluator=evaluator),
        BeamSearchSolver(beam_width=10, simulator=sim, evaluator=evaluator),
        BeamSearchSolver(beam_width=15, simulator=sim, evaluator=evaluator),
    ]

    for solver in solvers:
        t0 = time.perf_counter()
        total_states = 0
        total_moves = 0
        for st in states:
            moves, steps, explored = solver.find_round_plan(st)
            total_states += explored
            total_moves += len(moves)
        dt = time.perf_counter() - t0

        avg_latency_ms = (dt / rounds) * 1000.0
        states_per_sec = total_states / dt if dt > 0 else 0
        print(
            f"  {solver.name:<28}: {avg_latency_ms:6.2f} ms/hand | "
            f"{states_per_sec:9,.0f} states/sec | {total_moves/rounds:.1f} moves/hand"
        )


def benchmark_strategy_arena(arena: StrategyArena, num_games: int = 20) -> None:
    """Run full tournament in StrategyArena across identical seeds."""
    print(f"\n[Benchmark 3] Head-to-Head Strategy Arena Tournament ({num_games} identical games)...")
    solvers = [
        RandomSolver(simulator=arena.sim, seed=42),
        GreedySolver(simulator=arena.sim, evaluator=arena.evaluator),
        BeamSearchSolver(beam_width=10, simulator=arena.sim, evaluator=arena.evaluator),
    ]

    t0 = time.perf_counter()
    results = arena.evaluate_solvers(solvers, num_games=num_games, max_rounds_per_game=100)
    total_time = time.perf_counter() - t0

    print(f"Tournament completed in {total_time:.2f}s.\n")
    table = arena.format_comparison_table(results)
    print(table)


def main() -> None:
    print("=" * 70)
    print("BLOCK BLAST AI ULTIMATE — AI ENGINE BENCHMARK (Dell Latitude E7250)")
    print(f"Python: {sys.version.split()[0]} | Platform: {sys.platform}")
    print("=" * 70)

    sim = GameSimulator()
    evaluator = BoardEvaluator()
    arena = StrategyArena(simulator=sim, evaluator=evaluator, base_seed=42)

    benchmark_evaluator(evaluator, iterations=200_000)
    benchmark_solvers_latency(sim, evaluator, rounds=50)
    benchmark_strategy_arena(arena, num_games=15)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
