"""Tray Piece Detection and Region Segmentation for BLOCK BLAST AI ULTIMATE.

Extracts the 3 candidate piece regions from the lower tray area of the screen,
segments them from the background, and isolates binary piece masks for classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from computer_vision.config import DEFAULT_VISION_CONFIG, VisionConfig


@dataclass(frozen=True, slots=True)
class RawPieceBlob:
    """Segmented candidate piece blob extracted from a tray slot."""

    slot_idx: int
    is_empty: bool
    bbox_in_screen: Tuple[int, int, int, int]  # (x, y, w, h)
    binary_mask: Optional[np.ndarray]           # 2D uint8 mask (255 for piece, 0 for bg)
    confidence: float
    message: str


@dataclass
class SlotDetectionDetail:
    """Intermediate detection details for a single tray slot."""

    slot_idx: int
    crop_coords: Tuple[int, int, int, int]      # (x_min, x_max, y_tray_start, y_tray_end)
    slot_bgr: np.ndarray                        # exact BGR crop passed to detector
    slot_mask: np.ndarray                       # binary mask produced by segmentation
    all_contours: list[np.ndarray]              # all external contours found in slot_mask
    contour_areas: list[float]                  # areas of all contours
    valid_contours: list[np.ndarray]            # contours with area >= min_area
    valid_contour_areas: list[float]
    bbox_in_slot: Optional[Tuple[int, int, int, int]]  # (lx, ly, lw, lh) in slot coords
    bbox_in_screen: Tuple[int, int, int, int]          # (gx, gy, lw, lh) in screen coords
    is_empty: bool
    blob: RawPieceBlob


@dataclass
class TrayDetectionDetails:
    """Intermediate artifacts for the full tray detection pass."""

    tray_band: np.ndarray
    tray_y_start: int
    tray_y_end: int
    bg_gray_est: float
    closed_mask: np.ndarray
    slot_details: list[SlotDetectionDetail]
    bg_bgr_est: Optional[Tuple[float, float, float]] = None


class PieceDetector:
    """Detects and isolates the 3 piece candidates in the tray."""

    def __init__(self, config: Optional[VisionConfig] = None) -> None:
        self.config = config or DEFAULT_VISION_CONFIG
        self.last_detection_details: Optional[TrayDetectionDetails] = None

    def detect_tray_pieces(
        self,
        image: np.ndarray,
    ) -> list[RawPieceBlob]:
        """Detect and segment all 3 piece slots from the tray area.

        Args:
            image: Full-frame BGR numpy array.

        Returns:
            List of 3 RawPieceBlob instances (for slot 0, 1, 2).
        """
        h_img, w_img = image.shape[:2]
        cfg = self.config

        # 1. Isolate the vertical tray band
        y_tray_start = int(h_img * cfg.tray_expected_y_min_rel)
        y_tray_end = int(h_img * cfg.tray_expected_y_max_rel)
        tray_band = image[y_tray_start:y_tray_end, :]

        # 2. Estimate background color vector dynamically from tray borders
        # Supports vertical gradient backgrounds common across Block Blast themes
        bg_top = np.median(tray_band[:10, :].reshape(-1, 3), axis=0).astype(np.float32)
        bg_bot = np.median(tray_band[-10:, :].reshape(-1, 3), axis=0).astype(np.float32)
        alphas = np.linspace(0.0, 1.0, tray_band.shape[0], dtype=np.float32).reshape((-1, 1, 1))
        bg_grad = (1.0 - alphas) * bg_top + alphas * bg_bot

        # Sample top and bottom rows of the tray band in 3-channel BGR for legacy diagnostics
        border_samples = np.concatenate(
            [tray_band[:10, :].reshape(-1, 3), tray_band[-10:, :].reshape(-1, 3)],
            axis=0,
        )
        bg_bgr_vec = np.median(border_samples, axis=0).astype(np.float32)
        bg_bgr_est = (float(bg_bgr_vec[0]), float(bg_bgr_vec[1]), float(bg_bgr_vec[2]))

        # Grayscale estimate for diagnostic logging and backward compatibility
        bg_gray_est = float(0.114 * bg_bgr_vec[0] + 0.587 * bg_bgr_vec[1] + 0.299 * bg_bgr_vec[2])

        # Color-aware per-pixel Euclidean distance from gradient background model
        color_diff = tray_band.astype(np.float32) - bg_grad
        dist_map = np.sqrt(np.sum(color_diff ** 2, axis=2))

        # Configurable color difference threshold (default 25.0)
        threshold_val = float(getattr(cfg, "tray_color_diff_threshold", cfg.tray_bg_diff_threshold))
        _, thresh = cv2.threshold(dist_map, threshold_val, 255, cv2.THRESH_BINARY)
        thresh = thresh.astype(np.uint8)

        # Morphological closing to bridge small gaps between blocks in a polyomino
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        blobs: list[RawPieceBlob] = []
        slot_details: list[SlotDetectionDetail] = []

        # 3. Analyze each of the 3 horizontal slot zones
        for slot_idx, (x_rel_min, x_rel_max) in enumerate(cfg.tray_slot_x_splits):
            x_min = int(w_img * x_rel_min)
            x_max = int(w_img * x_rel_max)

            slot_bgr = tray_band[:, x_min:x_max]
            slot_mask = closed[:, x_min:x_max]

            # Find external contours within this slot's partition
            contours, _ = cv2.findContours(slot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_contour_list = list(contours)
            all_contour_areas = [float(cv2.contourArea(c)) for c in all_contour_list]

            valid_contours = [
                c for c in all_contour_list
                if cv2.contourArea(c) >= cfg.tray_min_contour_area
            ]
            valid_contour_areas = [float(cv2.contourArea(c)) for c in valid_contours]

            if not valid_contours:
                # Slot is empty
                empty_blob = RawPieceBlob(
                    slot_idx=slot_idx,
                    is_empty=True,
                    bbox_in_screen=(x_min, y_tray_start, x_max - x_min, y_tray_end - y_tray_start),
                    binary_mask=None,
                    confidence=0.95,
                    message="Tray slot is empty.",
                )
                blobs.append(empty_blob)
                slot_details.append(
                    SlotDetectionDetail(
                        slot_idx=slot_idx,
                        crop_coords=(x_min, x_max, y_tray_start, y_tray_end),
                        slot_bgr=slot_bgr.copy(),
                        slot_mask=slot_mask.copy(),
                        all_contours=all_contour_list,
                        contour_areas=all_contour_areas,
                        valid_contours=valid_contours,
                        valid_contour_areas=valid_contour_areas,
                        bbox_in_slot=None,
                        bbox_in_screen=(x_min, y_tray_start, x_max - x_min, y_tray_end - y_tray_start),
                        is_empty=True,
                        blob=empty_blob,
                    )
                )
                continue

            # If multiple small contours exist in the slot, take the union of their bounding boxes
            # or the largest contour
            all_pts = np.concatenate(valid_contours)
            lx, ly, lw, lh = cv2.boundingRect(all_pts)

            # Global screen coordinates
            gx = x_min + lx
            gy = y_tray_start + ly

            # Crop the piece's tight binary mask
            piece_mask = slot_mask[ly : ly + lh, lx : lx + lw]

            # Validate reasonable polyomino aspect ratio and dimensions
            if lw < 20 or lh < 20:
                small_blob = RawPieceBlob(
                    slot_idx=slot_idx,
                    is_empty=True,
                    bbox_in_screen=(gx, gy, lw, lh),
                    binary_mask=None,
                    confidence=0.95,
                    message="Detected contour too small to be a piece; confirmed empty slot.",
                )
                blobs.append(small_blob)
                slot_details.append(
                    SlotDetectionDetail(
                        slot_idx=slot_idx,
                        crop_coords=(x_min, x_max, y_tray_start, y_tray_end),
                        slot_bgr=slot_bgr.copy(),
                        slot_mask=slot_mask.copy(),
                        all_contours=all_contour_list,
                        contour_areas=all_contour_areas,
                        valid_contours=valid_contours,
                        valid_contour_areas=valid_contour_areas,
                        bbox_in_slot=(int(lx), int(ly), int(lw), int(lh)),
                        bbox_in_screen=(int(gx), int(gy), int(lw), int(lh)),
                        is_empty=True,
                        blob=small_blob,
                    )
                )
                continue

            piece_blob = RawPieceBlob(
                slot_idx=slot_idx,
                is_empty=False,
                bbox_in_screen=(gx, gy, lw, lh),
                binary_mask=piece_mask,
                confidence=0.92,
                message="Piece blob extracted successfully.",
            )
            blobs.append(piece_blob)
            slot_details.append(
                SlotDetectionDetail(
                    slot_idx=slot_idx,
                    crop_coords=(x_min, x_max, y_tray_start, y_tray_end),
                    slot_bgr=slot_bgr.copy(),
                    slot_mask=slot_mask.copy(),
                    all_contours=all_contour_list,
                    contour_areas=all_contour_areas,
                    valid_contours=valid_contours,
                    valid_contour_areas=valid_contour_areas,
                    bbox_in_slot=(int(lx), int(ly), int(lw), int(lh)),
                    bbox_in_screen=(int(gx), int(gy), int(lw), int(lh)),
                    is_empty=False,
                    blob=piece_blob,
                )
            )

        self.last_detection_details = TrayDetectionDetails(
            tray_band=tray_band.copy(),
            tray_y_start=y_tray_start,
            tray_y_end=y_tray_end,
            bg_gray_est=bg_gray_est,
            closed_mask=closed.copy(),
            slot_details=slot_details,
            bg_bgr_est=bg_bgr_est,
        )

        return blobs
