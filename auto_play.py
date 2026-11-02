"""Main CLI Entrypoint for BLOCK BLAST AI ULTIMATE Auto Play System.

Usage:
    python auto_play.py --mode assist               # AI recommendations only (no touch)
    python auto_play.py --mode auto --max-rounds 5  # Autonomous verified execution
    python auto_play.py --dry-run                   # Simulation without device touch
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import threading
from typing import Optional

from action_engine.auto_player import AutoPlayConfig, AutoPlayer, PlayMode
from action_engine.dashboard import ConsoleDashboard


def get_default_adb_path() -> str:
    which_adb = shutil.which("adb")
    if which_adb:
        return which_adb
    if os.path.isfile(r"C:\platform-tools\adb.exe"):
        return r"C:\platform-tools\adb.exe"
    return "adb"


def get_default_device_id(adb_path: str = "adb") -> Optional[str]:
    if os.environ.get("ADB_DEVICE_SERIAL"):
        return os.environ.get("ADB_DEVICE_SERIAL")
    try:
        res = subprocess.run([adb_path, "devices"], capture_output=True, text=True, timeout=2.0)
        lines = [line.split()[0] for line in res.stdout.strip().splitlines()[1:] if "\tdevice" in line]
        if len(lines) == 1:
            return lines[0]
    except Exception:
        pass
    return None


def setup_logging(verbose: bool = False) -> None:
    # Ensure stdout handles UTF-8 correctly
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BLOCK BLAST AI ULTIMATE — Automated Play System (Phase 3.4)"
    )
    parser.add_argument(
        "--mode",
        choices=["assist", "auto", "pause"],
        default="assist",
        help="Initial operating mode (default: assist)",
    )
    parser.add_argument(
        "--strategy",
        choices=["beam_search", "greedy"],
        default="beam_search",
        help="AI search strategy (default: beam_search)",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=15,
        help="Beam width for beam_search (default: 15)",
    )
    default_adb = get_default_adb_path()
    default_dev = get_default_device_id(default_adb)

    parser.add_argument(
        "--device-id",
        type=str,
        default=default_dev,
        help="Target ADB serial (default: auto-detected)",
    )
    parser.add_argument(
        "--adb-path",
        type=str,
        default=default_adb,
        help="Path to adb executable",
    )
    parser.add_argument(
        "--offset-ratio",
        type=float,
        default=1.65,
        help="Touch-to-piece vertical offset ratio in grid units (default: 1.65)",
    )
    parser.add_argument(
        "--drag-duration",
        type=int,
        default=380,
        help="Duration of drag swipe in ms (default: 380)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Maximum rounds to play before stopping (default: unlimited)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate moves without injecting physical touch commands",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch full interactive keyboard-driven console dashboard",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable detailed debug logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    mode_map = {
        "assist": PlayMode.ASSIST,
        "auto": PlayMode.AUTO,
        "pause": PlayMode.PAUSE,
    }
    initial_mode = mode_map[args.mode]

    config = AutoPlayConfig(
        mode=initial_mode,
        strategy=args.strategy,
        beam_width=args.beam_width,
        device_id=args.device_id,
        adb_path=args.adb_path,
        vertical_offset_ratio=args.offset_ratio,
        drag_duration_ms=args.drag_duration,
        max_rounds=args.max_rounds,
        dry_run=args.dry_run,
    )

    player = AutoPlayer(config=config)
    dashboard = ConsoleDashboard(player)

    print("\n" + "=" * 72)
    print("  BLOCK BLAST AI ULTIMATE — AUTO PLAY ENGINE (PHASE 3.4)")
    print(f"  Mode: {initial_mode.value} | Strategy: {args.strategy} (BW={args.beam_width})")
    print(f"  Target Device: {args.device_id} | Dry Run: {args.dry_run}")
    print("=" * 72 + "\n")

    if args.interactive:
        # Run player loop in background thread and dashboard in foreground
        player_thread = threading.Thread(target=player.run_loop, daemon=True)
        player_thread.start()
        dashboard.start_interactive()
    else:
        # Single cycle or run loop
        if initial_mode == PlayMode.ASSIST and args.max_rounds == 1:
            decision = player.step_single_cycle()
            dashboard.print_update()
        else:
            player.run_loop()


if __name__ == "__main__":
    main()
