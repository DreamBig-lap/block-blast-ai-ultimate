"""Interactive CLI Demo for BLOCK BLAST AI ULTIMATE.

Displays 8x8 ASCII board, available tray pieces, runs the AI decision engine,
and visualizes the step-by-step execution of the planned moves.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from typing import Optional

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ai_engine.decision import AIDecisionEngine, DecisionStatus
from game_engine.board import (
    BOARD_SIZE,
    coord_to_bit,
    is_occupied,
    popcount,
)
from game_engine.pieces import Piece
from game_engine.simulator import GameSimulator
from game_engine.state import GameState


# ---------------------------------------------------------------------------
# Visual Formatting Helpers
# ---------------------------------------------------------------------------

def render_board(board: int, title: str = "BOARD (8x8)", highlight_mask: int = 0) -> str:
    """Render 8x8 bitboard with optional highlight of newly placed cells."""
    lines: list[str] = [f"=== {title} (Occupied: {popcount(board)}/64) ==="]
    lines.append("    0 1 2 3 4 5 6 7")
    lines.append("  +-----------------+")
    for r in range(BOARD_SIZE):
        row_str = f"{r} | "
        for c in range(BOARD_SIZE):
            bit = coord_to_bit(r, c)
            if highlight_mask & bit:
                row_str += "* "  # Newly placed cell
            elif board & bit:
                row_str += "# "  # Occupied cell
            else:
                row_str += ". "  # Empty cell
        row_str += "|"
        lines.append(row_str)
    lines.append("  +-----------------+")
    return "\n".join(lines)


def render_piece_shape(piece: Optional[Piece]) -> list[str]:
    """Render a single piece as ASCII lines."""
    if piece is None:
        return ["  [EMPTY]"]
    h, w = piece.height, piece.width
    coords_set = set(piece.cells)
    lines: list[str] = []
    for r in range(h):
        line = ""
        for c in range(w):
            line += "# " if (r, c) in coords_set else "  "
        lines.append(line.rstrip())
    return lines


def render_tray(tray: tuple[Optional[Piece], ...]) -> str:
    """Format all pieces currently in the tray side-by-side."""
    lines: list[str] = ["=== TRAY PIECES ==="]
    header_parts = []
    for idx, piece in enumerate(tray):
        if piece is not None:
            header_parts.append(f"[Slot {idx}]: {piece.name} ({piece.cell_count}c, {piece.height}x{piece.width})")
        else:
            header_parts.append(f"[Slot {idx}]: (EMPTY)")
    lines.append(" | ".join(header_parts))
    lines.append("-" * 65)

    # Render ASCII shapes side-by-side
    shapes = [render_piece_shape(p) for p in tray]
    max_h = max(len(s) for s in shapes)

    for row_idx in range(max_h):
        row_parts = []
        for s in shapes:
            part = s[row_idx] if row_idx < len(s) else ""
            row_parts.append(f"{part:<20}")
        lines.append("   ".join(row_parts))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI Runner
# ---------------------------------------------------------------------------

def run_demo(
    rounds: int = 1,
    strategy: str = "beam_search",
    beam_width: int = 15,
    seed: Optional[int] = None,
    delay: float = 0.0,
) -> None:
    """Execute the interactive CLI demonstration."""
    print("=" * 70)
    print("BLOCK BLAST AI ULTIMATE — INTERACTIVE DECISION DEMO")
    print(f"Strategy: {strategy} (beam_width={beam_width}) | Seed: {seed}")
    print("=" * 70)

    sim = GameSimulator()
    engine = AIDecisionEngine(
        simulator=sim,
        default_strategy=strategy,
        default_beam_width=beam_width,
    )

    rng = random.Random(seed) if seed is not None else random.Random(42)

    # Initialize game state with first hand
    hand = sim.deal_hand(count=3, rng=rng)
    state = GameState.create_initial(initial_tray=hand)

    round_num = 1
    while round_num <= rounds:
        print(f"\n>>>>>>>>>>>>>>> ROUND {round_num} (Score: {state.score}, Combo: {state.combo_streak}) <<<<<<<<<<<<<<<")
        print(render_board(state.board, title=f"ROUND {round_num} - INITIAL BOARD"))
        print()
        print(render_tray(state.tray))
        print()

        print(f"[*] Running AI Decision Engine ({strategy})...")
        decision = engine.decide(
            state,
            strategy=strategy,
            beam_width=beam_width,
            seed=seed,
        )

        print(f"[+] Decision Status   : {decision.status.value}")
        print(f"[+] Duration          : {decision.duration_ms:.2f} ms")
        print(f"[+] States Explored   : {decision.states_explored}")
        print(f"[+] Predicted Gain    : +{decision.predicted_score_delta} pts")
        print(f"[+] Final Board Eval  : {decision.final_board_evaluation:.2f}")
        print(f"[+] Selected Moves    : {len(decision.moves)} step(s)")
        print(f"[+] Rationale         :\n    {decision.rationale}\n")

        if decision.status != DecisionStatus.SUCCESS:
            print(f"[!] AI cannot continue: {decision.status.value}. Game Over.")
            break

        # Step-by-step playback
        current_board = state.board
        for i, move in enumerate(decision.moves):
            step = decision.step_results[i]
            print(f"--- [Step {i+1}/{len(decision.moves)}] Place Slot {move.slot_idx} ({move.piece_name}) at Row {move.row}, Col {move.col} ---")
            print(f"    Gained +{move.predicted_score_delta} pts | Cleared {move.lines_cleared} line(s) ({move.unique_cells_cleared} cells)")
            print(render_board(step.state_after.board, title=f"AFTER STEP {i+1}", highlight_mask=step.placement_mask))
            print()
            current_board = step.state_after.board
            if delay > 0:
                time.sleep(delay)

        # Advance state to end of round
        last_step = decision.step_results[-1]
        state = last_step.state_after

        if state.is_game_over:
            print("[!] Terminal Game Over reached after completing round.")
            break

        # Deal next hand if round completed
        if state.is_tray_empty():
            round_num += 1
            if round_num <= rounds:
                next_hand = sim.deal_hand(count=3, rng=rng)
                state = sim.refill_tray(state, new_pieces=next_hand)
                if state.is_game_over:
                    print("[!] Game Over upon new deal: No piece in new hand can fit.")
                    break

    print("=" * 70)
    print(f"DEMO COMPLETE | Final Score: {state.score} | Rounds Played: {min(round_num, rounds)}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="BLOCK BLAST AI ULTIMATE CLI Demo")
    parser.add_argument("--rounds", type=int, default=1, help="Number of rounds to play (default: 1)")
    parser.add_argument(
        "--strategy",
        type=str,
        default="beam_search",
        choices=["beam_search", "greedy", "random"],
        help="Strategy algorithm (default: beam_search)",
    )
    parser.add_argument("--beam-width", type=int, default=15, help="Beam width for beam_search (default: 15)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility (default: 42)")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay in seconds between steps (default: 0)")
    args = parser.parse_args()

    run_demo(
        rounds=args.rounds,
        strategy=args.strategy,
        beam_width=args.beam_width,
        seed=args.seed,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
