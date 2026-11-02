"""Board Detection and Grid Geometry Localization for BLOCK BLAST AI ULTIMATE.

Detects the 8x8 square game board bounding box from screen frames using contrast segmentation
and gradient edge localization, with calibrated geometric fallback.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from computer_vision.config import DEFAULT_VISION_CONFIG, VisionConfig


class BoardDetector:
    """Detects the 8x8 game board boundary and computes grid cell geometries."""

    def __init__(self, config: Optional[VisionConfig] = None) -> None:
        self.config = config or DEFAULT_VISION_CONFIG

    def detect_board(
        self,
        image: np.ndarray,
    ) -> Tuple[Tuple[int, int, int, int], float, float]:
        """Detect the (x, y, w, h) bounding box of the 8x8 board in the image.

        Returns:
            board_bbox: (x, y, w, h) in pixels.
            cell_size: Float width/height of a single grid cell.
            confidence: Confidence score of the detection in [0.0, 1.0].
        """
        h_img, w_img = image.shape[:2]
        cfg = self.config

        # Restrict search to expected vertical band
        y_min = int(h_img * cfg.board_expected_y_min_rel)
        y_max = int(h_img * cfg.board_expected_y_max_rel)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # ---------------------------------------------------------------------
        # Approach 1: Gradient Edge & Contrast Profile Localization
        # ---------------------------------------------------------------------
        # The 8x8 board has distinct contrast boundaries against the outer background.
        # Anchor the search tightly around calibrated expected_y_top to prevent
        # false triggers on block edges in lower rows (which cause 1-row drift).
        expected_y_top = int(round(h_img * cfg.calibrated_board_rel[1]))
        search_top_min = max(0, expected_y_top - 25)
        search_top_max = min(h_img, expected_y_top + 25)
        central_strip = gray[search_top_min:search_top_max, int(w_img * 0.25) : int(w_img * 0.75)]
        row_means = np.mean(central_strip, axis=1)

        detected_y_top = None
        for idx in range(len(row_means) - 1):
            # Look for the sharp drop from frame level (> 118) to inner grid (< 112)
            if row_means[idx] >= 118 and row_means[idx + 1] < 112:
                candidate_y = search_top_min + idx
                if abs(candidate_y - expected_y_top) <= 15:
                    detected_y_top = candidate_y
                    break

        cand_w = int(round(w_img * cfg.calibrated_board_rel[2]))  # Typically 0.90 * W
        cand_h = cand_w  # Strictly square 8x8 grid
        cand_x = int(round((w_img - cand_w) / 2))

        if detected_y_top is not None:
            cand_y = detected_y_top
            best_bbox = (cand_x, cand_y, cand_w, cand_h)
            best_conf = 0.95
        else:
            # Fallback to calibrated relative coordinates
            cand_y = expected_y_top
            best_bbox = (cand_x, cand_y, cand_w, cand_h)
            best_conf = 0.88

        cell_size = best_bbox[2] / float(cfg.grid_cols)
        return best_bbox, cell_size, best_conf

    def get_cell_bounding_box(
        self,
        board_bbox: Tuple[int, int, int, int],
        row: int,
        col: int,
    ) -> Tuple[int, int, int, int]:
        """Compute the pixel bounding box (x, y, w, h) of cell (row, col)."""
        bx, by, bw, bh = board_bbox
        cell_w = bw / float(self.config.grid_cols)
        cell_h = bh / float(self.config.grid_rows)

        cx = int(round(bx + col * cell_w))
        cy = int(round(by + row * cell_h))
        cw = int(round(cell_w))
        ch = int(round(cell_h))
        return cx, cy, cw, ch

    def get_cell_center(
        self,
        board_bbox: Tuple[int, int, int, int],
        row: int,
        col: int,
    ) -> Tuple[int, int]:
        """Compute the center pixel coordinate (cx, cy) of cell (row, col)."""
        bx, by, bw, bh = board_bbox
        cell_w = bw / float(self.config.grid_cols)
        cell_h = bh / float(self.config.grid_rows)

        cx = int(round(bx + (col + 0.5) * cell_w))
        cy = int(round(by + (row + 0.5) * cell_h))
        return cx, cy
