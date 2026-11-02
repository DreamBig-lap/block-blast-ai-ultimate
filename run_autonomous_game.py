"""Main Entrypoint for BLOCK BLAST AI ULTIMATE — Full Autonomous Game Loop.

Usage:
    python run_autonomous_game.py [--dry-run] [--device SERIAL] [--max-turns N] [--auto-start]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from typing import Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from action_engine.autonomous_controller import (
    AutonomousConfig,
    AutonomousGameController,
    AutonomousState,
)
from action_engine.dashboard import AutonomousDashboard


def get_default_adb_path() -> str:
    """Find adb in system PATH or standard fallback location."""
    which_adb = shutil.which("adb")
    if which_adb:
        return which_adb
    if os.path.isfile(r"C:\platform-tools\adb.exe"):
        return r"C:\platform-tools\adb.exe"
    return "adb"


def get_default_device_id(adb_path: str = "adb") -> Optional[str]:
    """Auto-detect connected ADB device serial if exactly one device is attached."""
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


def parse_args() -> argparse.Namespace:
    default_adb = get_default_adb_path()
    default_device = get_default_device_id(default_adb)

    parser = argparse.ArgumentParser(
        description="BLOCK BLAST AI ULTIMATE — Full Autonomous Game Loop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in simulation mode without sending physical touch events to device.",
    )
    parser.add_argument(
        "--device",
        default=default_device,
        help="Target ADB device serial identifier (auto-detected if single device connected).",
    )
    parser.add_argument(
        "--adb",
        default=default_adb,
        help="Path to ADB executable.",
    )
    parser.add_argument(
        "--strategy",
        default="beam_search",
        choices=["beam_search", "greedy", "random"],
        help="AI search strategy.",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=15,
        help="Beam width for beam search solver.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Maximum number of turns before stopping.",
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Start autonomous loop immediately on startup.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 74)
    print("  STARTING BLOCK BLAST AI ULTIMATE — AUTONOMOUS CONTROLLER")
    print("=" * 74)
    print(f"Device: {args.device} | Dry-Run: {args.dry_run} | Strategy: {args.strategy}")

    config = AutonomousConfig(
        device_id=args.device,
        adb_path=args.adb,
        strategy=args.strategy,
        beam_width=args.beam_width,
        dry_run=args.dry_run,
        max_turns=args.max_turns,
    )

    controller = AutonomousGameController(config=config)
    dashboard = AutonomousDashboard(controller=controller)

    if args.auto_start:
        controller.start()

    dashboard.start_interactive(refresh_interval_sec=1.0)


if __name__ == "__main__":
    main()
