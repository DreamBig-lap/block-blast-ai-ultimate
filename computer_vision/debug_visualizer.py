"""Debug Visualization and Diagnostic Exporter for BLOCK BLAST AI ULTIMATE Computer Vision.

Generates annotated debug images displaying board boundaries, 8x8 grid lines,
row/column indexes, cell occupancy markers, tray piece bounding boxes, and piece labels,
and exports comprehensive JSON telemetry.
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

from computer_vision.config import DEFAULT_VISION_CONFIG, VisionConfig
from computer_vision.models import (
    CellState,
    FrameRecognitionResult,
    RecognitionStatus,
)


class DebugVisualizer:
    """Generates visual debug overlays and serializes frame recognition metadata."""

    def __init__(self, config: Optional[VisionConfig] = None) -> None:
        self.config = config or DEFAULT_VISION_CONFIG

    def draw_debug_overlay(
        self,
        image: np.ndarray,
        result: FrameRecognitionResult,
    ) -> np.ndarray:
        """Render diagnostic annotations onto a copy of the input screenshot."""
        vis = image.copy()
        h_img, w_img = vis.shape[:2]

        # ---------------------------------------------------------------------
        # 1. Status Banner Header
        # ---------------------------------------------------------------------
        banner_h = 90
        cv2.rectangle(vis, (0, 0), (w_img, banner_h), (25, 25, 25), -1)

        status_colors = {
            RecognitionStatus.SUCCESS: (0, 200, 0),        # Green
            RecognitionStatus.LOW_CONFIDENCE: (0, 180, 255), # Orange
            RecognitionStatus.FAILED: (0, 0, 230),         # Red
        }
        color = status_colors.get(result.status, (255, 255, 255))

        title = f"BLOCK BLAST CV | Status: {result.status.value}"
        cv2.putText(vis, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)

        metrics = (
            f"Overall Conf: {result.overall_confidence:.2f} | "
            f"Time: {result.processing_time_ms:.1f}ms | "
            f"Shape: {w_img}x{h_img}"
        )
        cv2.putText(vis, metrics, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2, cv2.LINE_AA)

        # ---------------------------------------------------------------------
        # 2. Board and 8x8 Grid Annotations
        # ---------------------------------------------------------------------
        if result.board is not None:
            bx, by, bw, bh = result.board.board_bbox
            cell_size = result.board.cell_size

            # Draw outer board bounding box
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 240, 255), 4)  # Cyan/Yellow
            cv2.putText(
                vis,
                f"Board (8x8) [Conf: {result.board.board_confidence:.2f}]",
                (bx, by - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 240, 255),
                2,
                cv2.LINE_AA,
            )

            # Draw grid lines and row/col indices
            for i in range(9):
                # Horizontal lines
                y_line = int(round(by + i * cell_size))
                cv2.line(vis, (bx, y_line), (bx + bw, y_line), (180, 180, 180), 1)
                # Vertical lines
                x_line = int(round(bx + i * cell_size))
                cv2.line(vis, (x_line, by), (x_line, by + bh), (180, 180, 180), 1)

            # Draw Row (0..7) and Col (0..7) headers
            for c in range(8):
                cx = int(round(bx + (c + 0.5) * cell_size))
                cv2.putText(vis, str(c), (cx - 6, by - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            for r in range(8):
                cy = int(round(by + (r + 0.5) * cell_size))
                cv2.putText(vis, str(r), (bx - 26, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # Draw cell markers (Occupied = Red circle, Empty = Green dot)
            for r in range(8):
                for c in range(8):
                    cell = result.board.cells[r][c]
                    center = (cell.center_x, cell.center_y)

                    if cell.state == CellState.OCCUPIED:
                        # Red circle for occupied block
                        cv2.circle(vis, center, int(cell_size * 0.28), (0, 0, 230), -1)
                        cv2.circle(vis, center, int(cell_size * 0.28), (255, 255, 255), 2)
                        cv2.putText(vis, "1", (center[0] - 6, center[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    elif cell.state == CellState.EMPTY:
                        # Green dot for empty cell
                        cv2.circle(vis, center, 8, (0, 200, 0), -1)
                    else:
                        # Yellow triangle / question mark for uncertain
                        cv2.circle(vis, center, 14, (0, 215, 255), -1)
                        cv2.putText(vis, "?", (center[0] - 5, center[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # ---------------------------------------------------------------------
        # 3. Tray Pieces Annotations
        # ---------------------------------------------------------------------
        for p in result.tray_pieces:
            px, py, pw, ph = p.bbox_in_screen

            if p.is_empty:
                # Draw subtle gray box for empty slot
                cv2.rectangle(vis, (px, py), (px + pw, py + ph), (100, 100, 100), 2)
                cv2.putText(
                    vis,
                    f"Slot {p.slot_idx}: (EMPTY)",
                    (px + 10, py + ph // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (150, 150, 150),
                    2,
                )
            elif p.matched_name == "UNKNOWN":
                # Draw red box for unknown piece
                cv2.rectangle(vis, (px, py), (px + pw, py + ph), (0, 0, 255), 3)
                label = f"Slot {p.slot_idx}: UNKNOWN ({p.confidence*100:.0f}%)"
                cv2.putText(vis, label, (px, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            else:
                # Draw green box for successfully recognized piece
                cv2.rectangle(vis, (px, py), (px + pw, py + ph), (0, 220, 0), 3)
                label = f"Slot {p.slot_idx}: {p.matched_name} ({p.confidence*100:.0f}%)"
                cv2.putText(
                    vis,
                    label,
                    (px, max(20, py - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        return vis

    def save_debug_artifacts(
        self,
        image: np.ndarray,
        result: FrameRecognitionResult,
        output_dir: Optional[str] = None,
        prefix: str = "recognition",
    ) -> tuple[str, str]:
        """Save both annotated debug image and JSON metadata to output directory.

        Returns:
            (image_path, json_path)
        """
        out_dir = output_dir or self.config.debug_dir
        os.makedirs(out_dir, exist_ok=True)

        img_path = os.path.join(out_dir, f"{prefix}_debug.png")
        json_path = os.path.join(out_dir, f"{prefix}_result.json")

        # 1. Save annotated image
        overlay = self.draw_debug_overlay(image, result)
        cv2.imwrite(img_path, overlay)

        # 2. Save JSON file
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(result.to_json(indent=2))

        return img_path, json_path
