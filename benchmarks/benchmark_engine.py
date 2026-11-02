"""Empirical Benchmark Suite for BLOCK BLAST AI ULTIMATE Game Engine.

Measures actual, reproducible performance directly on the host development machine
(Dell Latitude E7250, Intel Core i5-5300U, Windows 10 Pro 64-bit).

Metrics Evaluated:
1. Cold-start / first-call initialization time vs. Steady-state execution time.
2. Placement validation throughput (can_place ops/sec).
3. Bitboard piece placement throughput (place ops/sec).
4. Line detection and simultaneous clearing throughput (check_and_clear ops/sec).
5. Full simulator state step throughput (apply_move steps/sec).
6. Simulated states per second in full sequential game play.
7. Peak memory consumption and heap allocation rate (tracemalloc).
"""

from __future__ import annotations

import gc
import random
import sys
import time
import tracemalloc
from typing import Final

from game_engine.board import (
    BOARD_SIZE,
    ROW_MASKS,
    can_place,
    check_and_clear_lines,
    coord_to_bit,
    place,
)
from game_engine.pieces import CANONICAL_41_CATALOG
from game_engine.rules import DEFAULT_RULES
from game_engine.simulator import GameSimulator
from game_engine.state import GameState

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def benchmark_all() -> dict[str, str]:
    """Run full empirical benchmark suite and report exact timings."""
    results: dict[str, str] = {}

    print("=" * 70)
    print("BLOCK BLAST AI ULTIMATE — ENGINE BENCHMARK (Dell Latitude E7250)")
    print("Python:", sys.version.split()[0], "| Platform:", sys.platform)
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Benchmark 1: Cold-Start vs. Steady-State First-Call Latency
    # -----------------------------------------------------------------------
    print("\n[Benchmark 1] Cold-Start vs. Steady-State Latency...")
    # Measure first-call latency
    t_start = time.perf_counter_ns()
    dummy_board = 0
    dummy_mask = 0x0F
    _ = can_place(dummy_board, dummy_mask)
    t_first_call_ns = time.perf_counter_ns() - t_start

    # Steady-state sample (average of 100,000 calls)
    n_warm = 100_000
    t_warm_start = time.perf_counter_ns()
    for _ in range(n_warm):
        _ = can_place(dummy_board, dummy_mask)
    t_warm_total_ns = time.perf_counter_ns() - t_warm_start
    steady_call_ns = t_warm_total_ns / n_warm

    print(f"  First-call cold latency : {t_first_call_ns / 1000:.3f} microseconds")
    print(f"  Steady-state call latency: {steady_call_ns:.2f} nanoseconds")
    results["cold_call_us"] = f"{t_first_call_ns / 1000:.3f} us"
    results["steady_call_ns"] = f"{steady_call_ns:.2f} ns"

    # -----------------------------------------------------------------------
    # Benchmark 2: Placement Validation Throughput (can_place)
    # -----------------------------------------------------------------------
    print("\n[Benchmark 2] Placement Validation Throughput (can_place)...")
    n_validate = 1_000_000
    test_masks = [CANONICAL_41_CATALOG[i % 41].canonical_mask for i in range(100)]
    test_board = 0x5555555555555555  # Alternating occupied checkerboard

    gc.disable()
    t0 = time.perf_counter()
    for i in range(n_validate):
        _ = can_place(test_board, test_masks[i % 100])
    t1 = time.perf_counter()
    gc.enable()

    elapsed = t1 - t0
    ops_sec = n_validate / elapsed
    print(f"  {n_validate:,} can_place calls in {elapsed:.3f}s -> {ops_sec:,.0f} ops/sec")
    results["can_place_ops_sec"] = f"{ops_sec:,.0f}"

    # -----------------------------------------------------------------------
    # Benchmark 3: Bitboard Placement Throughput (place)
    # -----------------------------------------------------------------------
    print("\n[Benchmark 3] Piece Placement Throughput (place)...")
    n_place = 1_000_000
    board_empty = 0

    gc.disable()
    t0 = time.perf_counter()
    for i in range(n_place):
        _ = place(board_empty, test_masks[i % 100])
    t1 = time.perf_counter()
    gc.enable()

    elapsed = t1 - t0
    ops_sec = n_place / elapsed
    print(f"  {n_place:,} place calls in {elapsed:.3f}s -> {ops_sec:,.0f} ops/sec")
    results["place_ops_sec"] = f"{ops_sec:,.0f}"

    # -----------------------------------------------------------------------
    # Benchmark 4: Line Detection and Simultaneous Clear
    # -----------------------------------------------------------------------
    print("\n[Benchmark 4] Line Detection & Clear Throughput (check_and_clear_lines)...")
    n_clear = 500_000
    # Board with full row 2 and full col 4
    board_full_lines = ROW_MASKS[2] | 0x0101010101010101 << 4 | 0x1234

    gc.disable()
    t0 = time.perf_counter()
    for _ in range(n_clear):
        _ = check_and_clear_lines(board_full_lines)
    t1 = time.perf_counter()
    gc.enable()

    elapsed = t1 - t0
    ops_sec = n_clear / elapsed
    print(f"  {n_clear:,} line clear detections in {elapsed:.3f}s -> {ops_sec:,.0f} ops/sec")
    results["clear_lines_ops_sec"] = f"{ops_sec:,.0f}"

    # -----------------------------------------------------------------------
    # Benchmark 5: Full Simulator Step Throughput (apply_move)
    # -----------------------------------------------------------------------
    print("\n[Benchmark 5] Full Simulator Step Throughput (apply_move)...")
    n_sim_steps = 100_000
    sim = GameSimulator(rules=DEFAULT_RULES, catalog=CANONICAL_41_CATALOG)
    p_dot = CANONICAL_41_CATALOG.get_by_name("dot_1x1")
    state_step = GameState.create_initial(initial_tray=[p_dot, p_dot, p_dot])

    gc.disable()
    t0 = time.perf_counter()
    current_state = state_step
    for i in range(n_sim_steps):
        slot = i % 3
        if current_state.is_tray_empty():
            current_state = current_state.clone_with(tray=[p_dot, p_dot, p_dot], board=0)
        # Place at rotating (r, c)
        r = (i // 3) % 8
        c = i % 8
        step_res = sim.apply_move(current_state, slot_idx=slot, row=r, col=c)
        current_state = step_res.state_after
    t1 = time.perf_counter()
    gc.enable()

    elapsed = t1 - t0
    steps_sec = n_sim_steps / elapsed
    print(f"  {n_sim_steps:,} full apply_move steps in {elapsed:.3f}s -> {steps_sec:,.0f} steps/sec")
    results["apply_move_steps_sec"] = f"{steps_sec:,.0f}"

    # -----------------------------------------------------------------------
    # Benchmark 6: Full Sequential Gameplay Simulation
    # -----------------------------------------------------------------------
    print("\n[Benchmark 6] Simulated Game States per Second (Legal Move Exploration)...")
    n_games = 1_000
    rng = random.Random(12345)
    sim_play = GameSimulator(rules=DEFAULT_RULES, catalog=CANONICAL_41_CATALOG, rng=rng)

    total_states_simulated = 0
    t0 = time.perf_counter()

    for _ in range(n_games):
        hand = sim_play.deal_hand(count=3)
        state_play = GameState.create_initial(initial_tray=hand)

        while not state_play.is_game_over:
            moves = sim_play.get_legal_moves(state_play)
            total_states_simulated += len(moves)
            if not moves:
                break
            # Pick a greedy or first move
            slot, r, c = moves[0]
            step = sim_play.apply_move(state_play, slot, r, c, auto_refill=True)
            state_play = step.state_after
            if state_play.round_number > 20:  # Cap episode length
                break

    t1 = time.perf_counter()
    elapsed = t1 - t0
    states_sec = total_states_simulated / elapsed if elapsed > 0 else 0
    print(f"  Explored {total_states_simulated:,} states in {elapsed:.3f}s across {n_games:,} games -> {states_sec:,.0f} states/sec")
    results["states_explored_per_sec"] = f"{states_sec:,.0f}"

    # -----------------------------------------------------------------------
    # Benchmark 7: Memory Consumption & Allocation Rate
    # -----------------------------------------------------------------------
    print("\n[Benchmark 7] Memory Consumption Profiling (tracemalloc)...")
    tracemalloc.start()
    t_mem0 = tracemalloc.take_snapshot()

    # Allocate and run 10,000 steps
    mem_sim = GameSimulator()
    mem_state = GameState.create_initial(initial_tray=[p_dot, p_dot, p_dot])
    for i in range(10_000):
        if mem_state.is_tray_empty():
            mem_state = mem_state.clone_with(tray=[p_dot, p_dot, p_dot], board=0)
        s = mem_sim.apply_move(mem_state, i % 3, 0, (i % 8))
        mem_state = s.state_after

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  Current memory allocated : {current_mem / 1024:.2f} KB")
    print(f"  Peak memory allocated    : {peak_mem / 1024:.2f} KB")
    results["peak_memory_kb"] = f"{peak_mem / 1024:.2f} KB"

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY FOR PHASE 1 REPORT")
    print("=" * 70)
    for k, v in results.items():
        print(f"  {k:28s}: {v}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    benchmark_all()
