"""CLI Entrypoint for BLOCK BLAST AI ULTIMATE Computer Vision.

Usage:
    python -m computer_vision [--image PATH] [--adb] [--output-dir DIR] [--no-ai]
"""

from __future__ import annotations

import argparse
import os
import sys

# Windows UTF-8 console output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ai_engine.decision import AIDecisionEngine
from computer_vision.config import DEFAULT_VISION_CONFIG, VisionConfig
from computer_vision.models import RecognitionStatus
from computer_vision.recognizer import BlockBlastRecognizer
from game_engine.board import board_to_string


def render_ascii_piece(cells: tuple[tuple[int, int], ...], h: int, w: int) -> list[str]:
    """Render piece cells as ASCII lines."""
    if not cells:
        return ["  (empty)"]
    cell_set = set(cells)
    lines: list[str] = []
    for r in range(h):
        line = ""
        for c in range(w):
            line += "# " if (r, c) in cell_set else ". "
        lines.append(line.rstrip())
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BLOCK BLAST AI ULTIMATE — Computer Vision Board Recognition",
    )
    default_test_img = r"C:\platform-tools\blockblast_test.png"
    initial_img = default_test_img if os.path.isfile(default_test_img) else None

    parser.add_argument(
        "--image",
        type=str,
        default=initial_img,
        help="Path to screenshot image (default: C:\\platform-tools\\blockblast_test.png)",
    )
    parser.add_argument(
        "--adb",
        action="store_true",
        help="Capture live screenshot from connected Android phone via ADB",
    )
    parser.add_argument(
        "--device-id",
        type=str,
        default=DEFAULT_VISION_CONFIG.device_id,
        help="ADB Device ID serial (default: auto-detected)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="debug",
        help="Output directory for debug visualizer images and JSON metadata (default: debug)",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Skip running AI Decision Engine after recognition",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="beam_search",
        help="AI solver strategy to plan moves (default: beam_search)",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=15,
        help="Beam width for beam_search solver (default: 15)",
    )

    args = parser.parse_args()

    cfg = VisionConfig(
        device_id=args.device_id,
        debug_dir=args.output_dir,
    )
    recognizer = BlockBlastRecognizer(config=cfg)

    print("=" * 72)
    print("BLOCK BLAST AI ULTIMATE — COMPUTER VISION PIPELINE")
    print("=" * 72)

    # 1. Acquire Image
    if args.adb:
        print(f"[*] Capturing live frame from Android phone via ADB (device={args.device_id})...")
        try:
            image = recognizer.capture.capture_from_adb(device_id=args.device_id)
        except Exception as exc:
            print(f"[!] ADB capture failed: {exc}", file=sys.stderr)
            sys.exit(1)
    elif args.image:
        print(f"[*] Loading screenshot from file: {args.image}...")
        try:
            image = recognizer.capture.load_from_file(args.image)
        except Exception as exc:
            print(f"[!] Failed to load image: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        print("[!] No input image specified. Provide --image PATH or --adb.", file=sys.stderr)
        sys.exit(1)

    h_img, w_img = image.shape[:2]
    print(f"[+] Image loaded successfully: {w_img}x{h_img} pixels (3 channels)\n")

    # 2. Run Vision Recognition
    print("[*] Running recognition pipeline...")
    result = recognizer.recognize_frame(image)

    print(f"[+] Recognition Status  : {result.status.value}")
    print(f"[+] Processing Time     : {result.processing_time_ms:.2f} ms")
    print(f"[+] Overall Confidence  : {result.overall_confidence:.4f}")

    if result.board is not None:
        b = result.board
        print(f"[+] Board Bounding Box  : x={b.board_bbox[0]}, y={b.board_bbox[1]}, w={b.board_bbox[2]}, h={b.board_bbox[3]}")
        print(f"[+] Cell Size           : {b.cell_size:.2f} px")
        print(f"[+] Board Confidence    : {b.board_confidence:.4f}")
        print(f"[+] Cell Confidence     : {b.cell_confidence:.4f}")
        print(f"[+] Bitboard Hex        : 0x{b.bitboard:016X}")
        print(f"[+] Occupied / Empty    : {b.occupied_count} occupied / {b.empty_count} empty")

        print("\n--- 8x8 BOARD OCCUPANCY MATRIX (0=Empty, 1=Occupied) ---")
        print("    0 1 2 3 4 5 6 7")
        print("  +-----------------+")
        for r_idx, row in enumerate(b.occupancy_matrix):
            row_str = " ".join(str(val) for val in row)
            print(f"{r_idx} | {row_str} |")
        print("  +-----------------+")

    # 3. Print Tray Pieces
    print("\n--- TRAY PIECES RECOGNITION (3 SLOTS) ---")
    for pr in result.tray_pieces:
        if pr.is_empty:
            print(f"[Slot {pr.slot_idx}]: (EMPTY) [Conf: {pr.confidence:.2f}]")
        else:
            print(
                f"[Slot {pr.slot_idx}]: {pr.matched_name} | "
                f"Conf: {pr.confidence*100:.1f}% | "
                f"Dim: {pr.height}x{pr.width} ({pr.cell_count} cells) | "
                f"BBox: {pr.bbox_in_screen}"
            )
            print("  Piece Mask / Shape:")
            for line in render_ascii_piece(pr.cells, pr.height, pr.width):
                print("    " + line)
        print()

    # 4. Save Debug Visualizer Artifacts
    img_path, json_path = recognizer.visualizer.save_debug_artifacts(
        image=image,
        result=result,
        output_dir=args.output_dir,
    )
    print("--- DEBUG ARTIFACTS SAVED ---")
    print(f"[+] Annotated Image : {os.path.abspath(img_path)}")
    print(f"[+] Metadata JSON   : {os.path.abspath(json_path)}\n")

    # 5. Hand-off to AI Decision Engine
    if not args.no_ai:
        if result.is_usable_for_ai():
            print("=" * 72)
            print("[*] HANDING OFF RECOGNIZED STATE TO AI DECISION ENGINE...")
            print("=" * 72)
            try:
                decision = recognizer.plan_best_moves(
                    result,
                    strategy=args.strategy,
                    beam_width=args.beam_width,
                )
                print(f"[+] AI Strategy         : {decision.strategy_name}")
                print(f"[+] AI Decision Status  : {decision.status.value}")
                print(f"[+] Search Time         : {decision.duration_ms:.2f} ms")
                print(f"[+] States Explored     : {decision.states_explored}")
                print(f"[+] Predicted Score Gain: +{decision.predicted_score_delta} pts")
                print(f"[+] Final Board Eval    : {decision.final_board_evaluation:.2f}")
                print(f"[+] Total Lines Cleared : {decision.total_lines_cleared}")
                print(f"[+] Piece Order Planned : {decision.piece_order}")
                print(f"\n[+] AI Rationale:\n    {decision.rationale}\n")

                print("--- PLANNED MOVES SEQUENCE ---")
                for m in decision.moves:
                    print(
                        f"  Step {m.step_index + 1}: Place Slot {m.slot_idx} ({m.piece_name}) "
                        f"at Row {m.row}, Col {m.col} -> Gained +{m.predicted_score_delta} pts "
                        f"| Cleared {m.lines_cleared} line(s)"
                    )
            except Exception as exc:
                print(f"[!] AI Decision planning failed: {exc}", file=sys.stderr)
        else:
            print("[!] Skipping AI Decision: Recognition status is not usable for AI:")
            print(f"    Status: {result.status.value} | Error: {result.error_message}")

    print("=" * 72)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
