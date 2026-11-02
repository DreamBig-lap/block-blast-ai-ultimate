"""Cell Classification and Bitboard Conversion for BLOCK BLAST AI ULTIMATE.

Analyzes central cell patches on the 8x8 grid to classify empty vs occupied blocks,
computes cell-level confidences, and produces the 64-bit bitboard for the Game Engine.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from computer_vision.board_detector import BoardDetector
from computer_vision.config import DEFAULT_VISION_CONFIG, VisionConfig
from computer_vision.models import (
    BoardRecognitionResult,
    CellInfo,
    CellState,
)
from game_engine.board import coord_to_bit, validate_bitboard


class CellClassifier:
    """Classifies occupancy states across the 8x8 grid cells."""

    def __init__(
        self,
        config: Optional[VisionConfig] = None,
        detector: Optional[BoardDetector] = None,
    ) -> None:
        self.config = config or DEFAULT_VISION_CONFIG
        self.detector = detector or BoardDetector(self.config)

    def classify_cells(
        self,
        image: np.ndarray,
        board_bbox: Tuple[int, int, int, int],
        board_confidence: float = 0.90,
    ) -> BoardRecognitionResult:
        """Classify all 64 cells on the 8x8 grid into EMPTY or OCCUPIED.

        Args:
            image: Full-frame BGR numpy array.
            board_bbox: (x, y, w, h) bounding box of the 8x8 board.
            board_confidence: Confidence score of the board localization.

        Returns:
            BoardRecognitionResult with 64-bit bitboard and structured CellInfo.
        """
        cfg = self.config
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        rows = cfg.grid_rows
        cols = cfg.grid_cols
        sample_ratio = cfg.cell_sample_ratio
        gray_thresh = cfg.cell_occupancy_gray_threshold
        sat_thresh = cfg.cell_occupancy_sat_threshold
        val_thresh = getattr(cfg, "cell_occupancy_val_threshold", 125.0)
        margin = cfg.cell_confidence_margin

        cells_grid: list[list[CellInfo]] = []
        matrix: list[list[int]] = []
        bitboard = 0
        total_cell_conf = 0.0

        for r in range(rows):
            row_cells: list[CellInfo] = []
            row_matrix: list[int] = []

            for c in range(cols):
                cx, cy, cw, ch = self.detector.get_cell_bounding_box(board_bbox, r, c)
                center_x, center_y = self.detector.get_cell_center(board_bbox, r, c)

                # Sample central patch avoiding cell borders
                pw = max(4, int(cw * sample_ratio))
                ph = max(4, int(ch * sample_ratio))
                x0 = max(0, center_x - pw // 2)
                x1 = min(image.shape[1], center_x + pw // 2)
                y0 = max(0, center_y - ph // 2)
                y1 = min(image.shape[0], center_y + ph // 2)

                patch_bgr = image[y0:y1, x0:x1]
                patch_gray = gray[y0:y1, x0:x1]
                patch_sat = sat[y0:y1, x0:x1]
                patch_val = val[y0:y1, x0:x1]

                mean_b = float(np.mean(patch_bgr[:, :, 0])) if patch_bgr.size > 0 else 0.0
                mean_green = float(np.mean(patch_bgr[:, :, 1])) if patch_bgr.size > 0 else 0.0
                mean_r = float(np.mean(patch_bgr[:, :, 2])) if patch_bgr.size > 0 else 0.0
                mean_g = float(np.mean(patch_gray)) if patch_gray.size > 0 else 0.0
                mean_s = float(np.mean(patch_sat)) if patch_sat.size > 0 else 0.0
                mean_v = float(np.mean(patch_val)) if patch_val.size > 0 else 0.0

                # In Block Blast, empty cells have dark/teal theme background (mean_gray ~45-85, mean_val ~50-105).
                # Crucially, empty teal tiles have Green > Blue (e.g. G ~ 102, B ~ 96, R ~ 45, Sat ~ 143).
                # Occupied blocks have:
                # 1. Bright colors (Yellow, Orange, Cyan, Light Green, Pink): mean_gray >= 125.0 or mean_val >= 160.0.
                # 2. Vivid Blue / Purple blocks: Blue dominant over Green and Red (mean_b >= 130, mean_b >= mean_green + 12).
                # 3. Vivid Red / Deep Red blocks: Red dominant over Green and Blue (mean_r >= 130, mean_r >= mean_green + 15).
                # 4. General vivid blocks with high brightness and saturation: mean_v >= 150.0, mean_s >= 70.0, mean_g >= 100.0.
                is_bright = (mean_g >= gray_thresh or mean_v >= 160.0)
                is_blue_purple = (
                    mean_b >= 130.0
                    and mean_b >= mean_green + 12.0
                    and mean_v >= val_thresh
                    and mean_s >= sat_thresh
                )
                is_red = (
                    mean_r >= 130.0
                    and mean_r >= mean_green + 15.0
                    and mean_v >= val_thresh
                    and mean_s >= sat_thresh
                )
                is_vivid = (
                    mean_v >= 150.0
                    and mean_s >= 70.0
                    and mean_g >= 100.0
                )
                is_occupied = is_bright or is_blue_purple or is_red or is_vivid

                # Confidence calculation
                if is_occupied:
                    if is_bright:
                        dist = max(mean_g - 110.0, mean_v - 135.0)
                    elif is_blue_purple:
                        dist = max(mean_b - mean_green - 12.0, mean_b - 120.0)
                    elif is_red:
                        dist = max(mean_r - mean_green - 15.0, mean_r - 120.0)
                    else:
                        dist = min(mean_v - 135.0, mean_s - 60.0)
                else:
                    dist = min(gray_thresh - mean_g, 135.0 - mean_v)

                conf = min(1.0, 0.5 + 0.5 * (max(0.0, dist) / margin))

                if conf < 0.60:
                    state = CellState.UNCERTAIN
                    occ_val = 1 if is_occupied else 0
                    if is_occupied:
                        bitboard |= coord_to_bit(r, c)
                elif is_occupied:
                    state = CellState.OCCUPIED
                    occ_val = 1
                    bitboard |= coord_to_bit(r, c)
                else:
                    state = CellState.EMPTY
                    occ_val = 0

                total_cell_conf += conf

                info = CellInfo(
                    row=r,
                    col=c,
                    state=state,
                    confidence=conf,
                    center_x=center_x,
                    center_y=center_y,
                    width=cw,
                    height=ch,
                    mean_gray=mean_g,
                    mean_sat=mean_s,
                )
                row_cells.append(info)
                row_matrix.append(occ_val)

            cells_grid.append(row_cells)
            matrix.append(row_matrix)

        avg_cell_conf = total_cell_conf / 64.0
        grid_conf = min(board_confidence, avg_cell_conf)
        cell_size = board_bbox[2] / float(cols)

        validate_bitboard(bitboard)

        return BoardRecognitionResult(
            board_bbox=board_bbox,
            grid_rows=rows,
            grid_cols=cols,
            cell_size=cell_size,
            cells=tuple(tuple(r) for r in cells_grid),
            occupancy_matrix=tuple(tuple(r) for r in matrix),
            bitboard=bitboard,
            board_confidence=board_confidence,
            grid_confidence=grid_conf,
            cell_confidence=avg_cell_conf,
        )
