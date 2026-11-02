"""Configuration and Thresholds for Computer Vision in BLOCK BLAST AI ULTIMATE.

Defines all geometric heuristics, color/contrast thresholds, confidence minimums,
and device settings for reliable board and piece recognition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class VisionConfig:
    """Configuration parameters for Block Blast computer vision recognition."""

    # -------------------------------------------------------------------------
    # ADB & Device Settings
    # -------------------------------------------------------------------------
    adb_path: str = "adb"
    device_id: Optional[str] = None
    use_raw_screencap: bool = True
    fallback_to_png_on_raw_error: bool = True
    use_adb_socket: bool = True
    adb_server_host: str = "127.0.0.1"
    adb_server_port: int = 5037

    # -------------------------------------------------------------------------
    # Board Geometry Heuristics (Relative to screen dimensions W x H)
    # -------------------------------------------------------------------------
    # Expected relative vertical span of the 8x8 board on standard 20:9 / 19.5:9 screens
    board_expected_y_min_rel: float = 0.20
    board_expected_y_max_rel: float = 0.70
    board_expected_width_rel_min: float = 0.85
    board_expected_width_rel_max: float = 0.95
    grid_rows: int = 8
    grid_cols: int = 8

    # Fallback calibrated relative coordinates for 1080x2340 / standard full-HD+ phones
    # (x_rel, y_rel, w_rel, h_rel)
    calibrated_board_rel: tuple[float, float, float, float] = (0.05, 0.234, 0.90, 0.415)

    # -------------------------------------------------------------------------
    # Cell Classification Thresholds
    # -------------------------------------------------------------------------
    # Central patch sampling ratio (0.4 means middle 40% of cell width & height)
    cell_sample_ratio: float = 0.45
    # Empty cell brightness threshold (grayscale 0..255)
    # Empty cells on device have mean gray ~45-85.
    # Occupied blocks have mean gray >= 125, or vivid colors (val >= 160, or color-dominant channels).
    cell_occupancy_gray_threshold: float = 125.0
    # Minimum saturation threshold in HSV (0..255) for vivid colored blocks (dark blue, purple, red)
    cell_occupancy_sat_threshold: float = 60.0
    # Minimum brightness value in HSV (0..255) for vivid colored blocks
    cell_occupancy_val_threshold: float = 130.0
    # Margin around threshold for 100% confidence
    cell_confidence_margin: float = 25.0

    # -------------------------------------------------------------------------
    # Tray & Piece Detection Settings (Relative to screen dimensions W x H)
    # -------------------------------------------------------------------------
    # Expected vertical span where the 3 tray pieces reside
    tray_expected_y_min_rel: float = 0.70
    tray_expected_y_max_rel: float = 0.86
    # Relative horizontal partitions for the 3 slots [Slot 0, Slot 1, Slot 2]
    tray_slot_x_splits: tuple[tuple[float, float], ...] = (
        (0.04, 0.36),   # Slot 0 (Left)
        (0.36, 0.64),   # Slot 1 (Middle)
        (0.64, 0.96),   # Slot 2 (Right)
    )
    # Background color difference threshold to isolate piece pixels (Euclidean BGR distance)
    tray_bg_diff_threshold: float = 25.0
    tray_color_diff_threshold: float = 25.0
    # Minimum contour area in pixels for a valid tray piece
    tray_min_contour_area: float = 800.0
    # Expected unit cell size range in tray (scaled smaller than board cells)
    tray_unit_size_min: float = 45.0
    tray_unit_size_max: float = 68.0

    # -------------------------------------------------------------------------
    # Piece Catalog Matching & Confidence Thresholds
    # -------------------------------------------------------------------------
    min_piece_match_confidence: float = 0.85
    min_success_overall_confidence: float = 0.75
    min_cell_confidence: float = 0.70

    # -------------------------------------------------------------------------
    # Output Directories
    # -------------------------------------------------------------------------
    debug_dir: str = "debug"


DEFAULT_VISION_CONFIG = VisionConfig()
