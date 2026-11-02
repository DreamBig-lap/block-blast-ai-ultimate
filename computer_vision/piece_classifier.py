"""Piece Shape Classification and Canonical Catalog Matching for BLOCK BLAST AI ULTIMATE.

Discretizes raw binary piece masks into polyomino grid coordinates and matches them
against the canonical 41-piece catalog with rigorous similarity scoring and UNKNOWN fallbacks.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from computer_vision.config import DEFAULT_VISION_CONFIG, VisionConfig
from computer_vision.models import PieceRecognitionResult
from computer_vision.piece_detector import RawPieceBlob
from game_engine.pieces import CANONICAL_41_CATALOG, Piece, PieceCatalog


class PieceClassifier:
    """Discretizes segmented piece blobs and maps them to canonical Game Engine pieces."""

    def __init__(
        self,
        config: Optional[VisionConfig] = None,
        catalog: Optional[PieceCatalog] = None,
    ) -> None:
        self.config = config or DEFAULT_VISION_CONFIG
        self.catalog = catalog or CANONICAL_41_CATALOG

    def _discretize_mask(
        self,
        binary_mask: np.ndarray,
        rows: int,
        cols: int,
    ) -> tuple[tuple[int, int], ...]:
        """Sample an R x C grid across the binary mask and return occupied (dr, dc) cells."""
        bh, bw = binary_mask.shape[:2]
        cell_w = bw / float(cols)
        cell_h = bh / float(rows)

        occupied_cells: list[tuple[int, int]] = []
        for r in range(rows):
            for c in range(cols):
                # Sample central 50% patch of this grid cell
                x0 = int(round(c * cell_w + 0.25 * cell_w))
                x1 = int(round(c * cell_w + 0.75 * cell_w))
                y0 = int(round(r * cell_h + 0.25 * cell_h))
                y1 = int(round(r * cell_h + 0.75 * cell_h))

                patch = binary_mask[y0:y1, x0:x1]
                if patch.size > 0:
                    fill_ratio = np.count_nonzero(patch) / float(patch.size)
                    if fill_ratio >= 0.40:
                        occupied_cells.append((r, c))

        return tuple(occupied_cells)

    def classify_blob(self, blob: RawPieceBlob) -> PieceRecognitionResult:
        """Classify a single RawPieceBlob into a PieceRecognitionResult."""
        if blob.is_empty or blob.binary_mask is None:
            return PieceRecognitionResult(
                slot_idx=blob.slot_idx,
                is_empty=True,
                matched_name=None,
                piece=None,
                cells=(),
                height=0,
                width=0,
                cell_count=0,
                bbox_in_screen=blob.bbox_in_screen,
                confidence=blob.confidence,
                status_message=blob.message,
            )

        mask = blob.binary_mask
        bh, bw = mask.shape[:2]
        cfg = self.config

        # ---------------------------------------------------------------------
        # Grid Size Search: Try candidate (R, C) dimensions from 1 to 5
        # ---------------------------------------------------------------------
        best_piece: Optional[Piece] = None
        best_cells: tuple[tuple[int, int], ...] = ()
        best_iou: float = 0.0
        best_composite_score: float = -1.0
        best_r, best_c = 0, 0

        # Typical tray unit block size on 1080p full HD screens is ~56.5 px
        target_u = 56.5

        # Estimate plausible unit cell size based on aspect ratio
        for r in range(1, 6):
            for c in range(1, 6):
                u_w = bw / float(c)
                u_h = bh / float(r)

                # Blocks must be roughly square
                aspect = u_w / u_h if u_h > 0 else 0
                if not (0.70 <= aspect <= 1.40):
                    continue

                avg_u = (u_w + u_h) / 2.0
                if not (cfg.tray_unit_size_min <= avg_u <= cfg.tray_unit_size_max):
                    continue

                extracted_cells = self._discretize_mask(mask, r, c)
                if not extracted_cells:
                    continue

                # Must span the bounding rows and columns
                min_r = min(dr for dr, _ in extracted_cells)
                max_r = max(dr for dr, _ in extracted_cells)
                min_c = min(dc for _, dc in extracted_cells)
                max_c = max(dc for _, dc in extracted_cells)

                if (max_r - min_r + 1) != r or (max_c - min_c + 1) != c:
                    continue

                # Normalize so min_r == 0, min_c == 0
                normalized = tuple(
                    sorted((dr - min_r, dc - min_c) for dr, dc in extracted_cells)
                )

                # Match against catalog pieces with height=r and width=c
                for cat_piece in self.catalog.all_pieces:
                    if cat_piece.height == r and cat_piece.width == c:
                        # Compute IoU between normalized cells and cat_piece.cells
                        set_a = set(normalized)
                        set_b = set(cat_piece.cells)
                        intersection = len(set_a & set_b)
                        union = len(set_a | set_b)
                        iou = intersection / float(union) if union > 0 else 0.0

                        if iou >= cfg.min_piece_match_confidence:
                            # Composite score balances shape match with unit cell scale consistency
                            composite_score = (
                                iou
                                - 0.005 * abs(avg_u - target_u)
                                - 0.05 * abs(aspect - 1.0)
                            )
                            if composite_score > best_composite_score:
                                best_composite_score = composite_score
                                best_iou = iou
                                best_piece = cat_piece
                                best_cells = cat_piece.cells
                                best_r, best_c = r, c

        # ---------------------------------------------------------------------
        # Threshold Check: Must meet min_piece_match_confidence
        # ---------------------------------------------------------------------
        if best_piece is not None and best_iou >= cfg.min_piece_match_confidence:
            return PieceRecognitionResult(
                slot_idx=blob.slot_idx,
                is_empty=False,
                matched_name=best_piece.name,
                piece=best_piece,
                cells=best_cells,
                height=best_piece.height,
                width=best_piece.width,
                cell_count=best_piece.cell_count,
                bbox_in_screen=blob.bbox_in_screen,
                confidence=best_iou,
                status_message=f"Matched canonical piece '{best_piece.name}' (IoU={best_iou:.2f}).",
            )
        else:
            # Fallback to UNKNOWN: Prompt rule: "Nếu không khớp catalog, trả về UNKNOWN thay vì đoán."
            return PieceRecognitionResult(
                slot_idx=blob.slot_idx,
                is_empty=False,
                matched_name="UNKNOWN",
                piece=None,
                cells=best_cells,
                height=best_r,
                width=best_c,
                cell_count=len(best_cells),
                bbox_in_screen=blob.bbox_in_screen,
                confidence=best_iou,
                status_message="Piece shape does not match any canonical catalog piece with sufficient confidence.",
            )

    def classify_all_blobs(
        self,
        blobs: Sequence[RawPieceBlob],
    ) -> list[PieceRecognitionResult]:
        """Classify a sequence of detected piece blobs."""
        return [self.classify_blob(b) for b in blobs]
