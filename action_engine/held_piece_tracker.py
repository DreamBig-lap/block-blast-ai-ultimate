"""Held Piece Tracker for BLOCK BLAST AI ULTIMATE.

Provides closed-loop in-flight vision tracking of polyomino pieces held on screen:
- Ultra-fast SIMD differential color frame differencing (hold_frame - base_frame)
- Morphological closing and contour segmentation in flight ROI
- Shape, dimension, and aspect ratio matching against canonical pieces
- Color saturation and brightness gating to separate solid blocks from drop shadows and drop previews
- Moment-based spatial centroid calculation
- Detection of abnormal bounding boxes and ambiguous multi-contour scenarios
- Strict UNKNOWN / LOW_CONFIDENCE fallbacks when confidence is insufficient
"""

from __future__ import annotations

from dataclasses import dataclass, field
import enum
import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from game_engine.pieces import CANONICAL_41_CATALOG, Piece

logger = logging.getLogger(__name__)


class TrackingStatus(enum.Enum):
    """Outcome status of held piece tracking."""

    SUCCESS = "SUCCESS"                      # Piece accurately localized with high confidence
    UNKNOWN = "UNKNOWN"                      # Insufficient evidence to classify
    NO_CONTOUR_FOUND = "NO_CONTOUR_FOUND"    # Zero candidate contours in flight region
    AMBIGUOUS_CONTOURS = "AMBIGUOUS"          # Multiple competing candidates with similar confidence
    ABNORMAL_BOUNDS = "ABNORMAL_BOUNDS"      # Bounding box or area abnormally oversized (e.g. shadow merged)
    LOW_CONFIDENCE = "LOW_CONFIDENCE"        # Best candidate does not meet minimum confidence threshold
    INVALID_INPUT = "INVALID_INPUT"          # Images missing, corrupt, or shape mismatched


@dataclass(frozen=True, slots=True)
class CandidateBlob:
    """Individual segmented blob candidate evaluated during tracking."""

    bbox: Tuple[int, int, int, int]          # (x, y, w, h) in screen coordinates
    centroid: Tuple[float, float]            # (cx, cy) from spatial moments
    area: float
    aspect_ratio: float
    mean_saturation: float
    mean_value: float
    dim_score: float
    sat_score: float
    composite_score: float
    is_abnormal_bbox: bool
    rejection_reason: Optional[str] = None


@dataclass(frozen=True, slots=True)
class HeldPieceDetection:
    """Structured result of held piece tracking."""

    status: TrackingStatus
    piece_name: Optional[str]
    bbox: Optional[Tuple[int, int, int, int]]  # (x, y, w, h) in screen pixels
    centroid: Optional[Tuple[float, float]]    # (cx, cy) in screen pixels
    contour_area: float
    aspect_ratio: float
    mean_saturation: float
    mean_value: float
    confidence: float
    is_abnormal_bbox: bool
    shadow_detected: bool
    message: str
    candidates_evaluated: int
    top_candidates: tuple[CandidateBlob, ...]
    elapsed_ms: float

    @property
    def is_success(self) -> bool:
        return self.status == TrackingStatus.SUCCESS


@dataclass
class HeldPieceTrackerConfig:
    """Configuration thresholds for held piece tracking."""

    # Difference threshold (Max channel difference across B, G, R)
    diff_threshold: float = 25.0

    # Minimum and maximum area ratios relative to theoretical piece area
    min_area_ratio: float = 0.25
    max_area_ratio: float = 2.40

    # Minimum absolute area (px^2) to ignore tiny particle noise
    min_absolute_area: float = 3000.0

    # Maximum dimension multipliers before flagging abnormal bounding box
    max_width_ratio: float = 1.65
    max_height_ratio: float = 2.20

    # Color saturation thresholds (HSV S-channel 0..255)
    # Real solid pieces in Block Blast have S > 100, shadows have S < 40
    min_piece_saturation: float = 60.0
    shadow_saturation_ceiling: float = 40.0

    # Minimum confidence to accept a detection as SUCCESS
    confidence_success_threshold: float = 0.60

    # Ambiguity threshold: if top 2 candidates are within this delta, flag AMBIGUOUS
    ambiguity_delta_threshold: float = 0.08

    # Morphological closing kernel (W, H) to bridge sub-block seams
    closing_kernel_size: Tuple[int, int] = (9, 9)

    # Vertical spatial bounds to ignore tray emptying and top system UI
    tray_mask_y_start: int = 1650
    top_mask_y_end: int = 250


class HeldPieceTracker:
    """Tracks and localizes in-flight polyomino pieces using differential vision."""

    def __init__(self, config: Optional[HeldPieceTrackerConfig] = None) -> None:
        self.config = config or HeldPieceTrackerConfig()

    def track_held_piece(
        self,
        base_image: np.ndarray,
        hold_image: np.ndarray,
        expected_piece_name: str,
        cell_size_px: float = 121.5,
        board_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> HeldPieceDetection:
        """Track and locate an expected piece held on screen.

        Args:
            base_image: Frame captured immediately before touch initiation (BGR).
            hold_image: Frame captured while touch is held DOWN in flight (BGR).
            expected_piece_name: Canonical identifier of the expected piece (e.g. 'line_1x5_h').
            cell_size_px: Size of a single grid cell on screen in pixels.
            board_bbox: Optional (x, y, w, h) bounding box of the active 8x8 board.

        Returns:
            HeldPieceDetection detailing location, confidence, and status.
        """
        t0 = time.perf_counter()

        # ---------------------------------------------------------------------
        # 1. Input Validation (Ensure immutability and valid types)
        # ---------------------------------------------------------------------
        validation_error = self._validate_inputs(base_image, hold_image)
        if validation_error:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return HeldPieceDetection(
                status=TrackingStatus.INVALID_INPUT,
                piece_name=expected_piece_name,
                bbox=None,
                centroid=None,
                contour_area=0.0,
                aspect_ratio=0.0,
                mean_saturation=0.0,
                mean_value=0.0,
                confidence=0.0,
                is_abnormal_bbox=False,
                shadow_detected=False,
                message=validation_error,
                candidates_evaluated=0,
                top_candidates=(),
                elapsed_ms=elapsed,
            )

        # ---------------------------------------------------------------------
        # 2. Check for Zero Difference (Identical base and hold frames)
        # ---------------------------------------------------------------------
        if np.array_equal(base_image, hold_image):
            elapsed = (time.perf_counter() - t0) * 1000.0
            return HeldPieceDetection(
                status=TrackingStatus.NO_CONTOUR_FOUND,
                piece_name=expected_piece_name,
                bbox=None,
                centroid=None,
                contour_area=0.0,
                aspect_ratio=0.0,
                mean_saturation=0.0,
                mean_value=0.0,
                confidence=0.0,
                is_abnormal_bbox=False,
                shadow_detected=False,
                message="Base and hold images are identical; zero differential change detected.",
                candidates_evaluated=0,
                top_candidates=(),
                elapsed_ms=elapsed,
            )

        # ---------------------------------------------------------------------
        # 3. Retrieve Expected Piece Model from Catalog
        # ---------------------------------------------------------------------
        expected_piece = CANONICAL_41_CATALOG.get_by_name(expected_piece_name)
        if expected_piece is None:
            w_exp_px = cell_size_px
            h_exp_px = cell_size_px
            area_exp_px = cell_size_px * cell_size_px
            aspect_exp = 1.0
        else:
            w_exp_px = expected_piece.width * cell_size_px
            h_exp_px = expected_piece.height * cell_size_px
            area_exp_px = expected_piece.cell_count * (cell_size_px * cell_size_px)
            aspect_exp = expected_piece.width / float(expected_piece.height)

        # ---------------------------------------------------------------------
        # 4. Ultra-Fast Differential Color Masking in Flight ROI
        # ---------------------------------------------------------------------
        cfg = self.config
        h_img, w_img = hold_image.shape[:2]
        y_tray_cut = min(h_img, cfg.tray_mask_y_start)
        y_top_cut = max(0, cfg.top_mask_y_end)

        # Process exclusively the vertical flight band
        base_roi = base_image[y_top_cut:y_tray_cut, :]
        hold_roi = hold_image[y_top_cut:y_tray_cut, :]

        # SIMD C++ accelerated absolute difference
        abs_diff = cv2.absdiff(hold_roi, base_roi)
        b_ch, g_ch, r_ch = cv2.split(abs_diff)
        max_diff = cv2.max(cv2.max(b_ch, g_ch), r_ch)
        _, roi_bin = cv2.threshold(max_diff, int(cfg.diff_threshold), 255, cv2.THRESH_BINARY)

        # Morphological Closing to bridge sub-block joints
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, cfg.closing_kernel_size)
        closed_roi = cv2.morphologyEx(roi_bin, cv2.MORPH_CLOSE, kernel)

        # ---------------------------------------------------------------------
        # 5. Extract and Filter Candidate Contours
        # ---------------------------------------------------------------------
        contours, _ = cv2.findContours(closed_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return HeldPieceDetection(
                status=TrackingStatus.NO_CONTOUR_FOUND,
                piece_name=expected_piece_name,
                bbox=None,
                centroid=None,
                contour_area=0.0,
                aspect_ratio=0.0,
                mean_saturation=0.0,
                mean_value=0.0,
                confidence=0.0,
                is_abnormal_bbox=False,
                shadow_detected=False,
                message="No candidate contours detected in flight region.",
                candidates_evaluated=0,
                top_candidates=(),
                elapsed_ms=elapsed,
            )

        evaluated_candidates: list[CandidateBlob] = []
        min_allowed_area = max(cfg.min_absolute_area, area_exp_px * cfg.min_area_ratio)

        for c in contours:
            area = float(cv2.contourArea(c))
            if area < min_allowed_area:
                continue

            bx, by_roi, bw, bh = cv2.boundingRect(c)
            by = by_roi + y_top_cut  # Remap to full screen Y coordinate
            aspect = bw / float(bh) if bh > 0 else 0.0

            # Moments Centroid (remapped to full screen)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"]) + y_top_cut
            else:
                cx = float(bx + bw / 2.0)
                cy = float(by + bh / 2.0)

            # Analyze patch color saturation and value from hold_image
            patch = hold_image[by:by+bh, bx:bx+bw]
            if patch.size > 0:
                hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                mean_s = float(np.mean(hsv_patch[:, :, 1]))
                mean_v = float(np.mean(hsv_patch[:, :, 2]))
            else:
                mean_s = 0.0
                mean_v = 0.0

            # Detect drop shadow or dark gray noise
            is_pure_shadow = (mean_s < cfg.shadow_saturation_ceiling and mean_v < 90.0)

            # Check for abnormally large bounding box
            is_abnormal_w = (bw > w_exp_px * cfg.max_width_ratio)
            is_abnormal_h = (bh > h_exp_px * cfg.max_height_ratio)
            is_abnormal_area = (area > area_exp_px * cfg.max_area_ratio)
            is_abnormal = is_abnormal_w or is_abnormal_h or is_abnormal_area

            # Dimension similarity scoring
            w_err = abs(bw - w_exp_px) / w_exp_px if w_exp_px > 0 else 1.0
            h_err = abs(bh - h_exp_px) / h_exp_px if h_exp_px > 0 else 1.0
            aspect_err = abs(aspect - aspect_exp) / aspect_exp if aspect_exp > 0 else 1.0
            dim_score = max(0.0, 1.0 - (0.40 * w_err + 0.30 * h_err + 0.30 * aspect_err))

            # Saturation gating:
            # If the blob lacks vivid saturation, it CANNOT be a genuine piece in Block Blast
            if is_pure_shadow or mean_s < cfg.min_piece_saturation:
                sat_score = 0.0
                # Heavy penalty prevents pure shadow from ever exceeding success threshold
                composite = 0.30 * dim_score
            else:
                sat_score = min(1.0, mean_s / 120.0)
                composite = 0.65 * dim_score + 0.35 * sat_score

            if is_abnormal:
                composite *= 0.50  # Penalize abnormal blobs

            evaluated_candidates.append(
                CandidateBlob(
                    bbox=(bx, by, bw, bh),
                    centroid=(cx, cy),
                    area=area,
                    aspect_ratio=aspect,
                    mean_saturation=mean_s,
                    mean_value=mean_v,
                    dim_score=dim_score,
                    sat_score=sat_score,
                    composite_score=composite,
                    is_abnormal_bbox=is_abnormal,
                    rejection_reason="PURE_SHADOW" if is_pure_shadow else None,
                )
            )

        if not evaluated_candidates:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return HeldPieceDetection(
                status=TrackingStatus.NO_CONTOUR_FOUND,
                piece_name=expected_piece_name,
                bbox=None,
                centroid=None,
                contour_area=0.0,
                aspect_ratio=0.0,
                mean_saturation=0.0,
                mean_value=0.0,
                confidence=0.0,
                is_abnormal_bbox=False,
                shadow_detected=False,
                message=f"All {len(contours)} contours fell below minimum area ({min_allowed_area:.0f} px^2).",
                candidates_evaluated=len(contours),
                top_candidates=(),
                elapsed_ms=elapsed,
            )

        # Sort candidates descending by composite confidence
        evaluated_candidates.sort(key=lambda c: c.composite_score, reverse=True)
        top = evaluated_candidates[0]

        # ---------------------------------------------------------------------
        # 6. Ambiguity Check: Multiple Competing Candidates
        # ---------------------------------------------------------------------
        if len(evaluated_candidates) >= 2:
            second = evaluated_candidates[1]
            if (
                top.composite_score >= 0.45
                and second.composite_score >= 0.45
                and abs(top.composite_score - second.composite_score) < cfg.ambiguity_delta_threshold
            ):
                elapsed = (time.perf_counter() - t0) * 1000.0
                return HeldPieceDetection(
                    status=TrackingStatus.AMBIGUOUS_CONTOURS,
                    piece_name=expected_piece_name,
                    bbox=top.bbox,
                    centroid=top.centroid,
                    contour_area=top.area,
                    aspect_ratio=top.aspect_ratio,
                    mean_saturation=top.mean_saturation,
                    mean_value=top.mean_value,
                    confidence=top.composite_score,
                    is_abnormal_bbox=top.is_abnormal_bbox,
                    shadow_detected=False,
                    message=f"Ambiguous candidates: Top 2 blobs score {top.composite_score:.3f} and {second.composite_score:.3f} (delta < {cfg.ambiguity_delta_threshold}).",
                    candidates_evaluated=len(evaluated_candidates),
                    top_candidates=tuple(evaluated_candidates[:5]),
                    elapsed_ms=elapsed,
                )

        # ---------------------------------------------------------------------
        # 7. Abnormal Bounding Box Check (with line-clear preview decomposition)
        # ---------------------------------------------------------------------
        if top.is_abnormal_bbox:
            # Check if this abnormal bounding box is caused by a thin line-clear preview border
            # that artificially connected the piece to other board cells or row/col borders.
            decomposed_cand = self._decompose_line_clear_preview(
                max_diff=max_diff,
                hold_image=hold_image,
                y_top_cut=y_top_cut,
                w_exp_px=w_exp_px,
                h_exp_px=h_exp_px,
                area_exp_px=area_exp_px,
                aspect_exp=aspect_exp,
                is_abnormal_w=top.bbox[2] > w_exp_px * cfg.max_width_ratio,
                is_abnormal_h=top.bbox[3] > h_exp_px * cfg.max_height_ratio,
            )
            if decomposed_cand is not None and not decomposed_cand.is_abnormal_bbox:
                top = decomposed_cand

        if top.is_abnormal_bbox:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return HeldPieceDetection(
                status=TrackingStatus.ABNORMAL_BOUNDS,
                piece_name=expected_piece_name,
                bbox=top.bbox,
                centroid=top.centroid,
                contour_area=top.area,
                aspect_ratio=top.aspect_ratio,
                mean_saturation=top.mean_saturation,
                mean_value=top.mean_value,
                confidence=top.composite_score,
                is_abnormal_bbox=True,
                shadow_detected=top.bbox[3] > h_exp_px * 1.25,
                message=f"Abnormally large bounding box: {top.bbox} (expected W~{w_exp_px:.0f}, H~{h_exp_px:.0f}).",
                candidates_evaluated=len(evaluated_candidates),
                top_candidates=tuple(evaluated_candidates[:5]),
                elapsed_ms=elapsed,
            )

        # ---------------------------------------------------------------------
        # 8. Low Confidence Check -> UNKNOWN / LOW_CONFIDENCE
        # ---------------------------------------------------------------------
        if top.composite_score < cfg.confidence_success_threshold:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return HeldPieceDetection(
                status=TrackingStatus.LOW_CONFIDENCE,
                piece_name=expected_piece_name,
                bbox=top.bbox,
                centroid=top.centroid,
                contour_area=top.area,
                aspect_ratio=top.aspect_ratio,
                mean_saturation=top.mean_saturation,
                mean_value=top.mean_value,
                confidence=top.composite_score,
                is_abnormal_bbox=False,
                shadow_detected=False,
                message=f"Low confidence ({top.composite_score:.3f} < {cfg.confidence_success_threshold}). Insufficient evidence to classify as {expected_piece_name}.",
                candidates_evaluated=len(evaluated_candidates),
                top_candidates=tuple(evaluated_candidates[:5]),
                elapsed_ms=elapsed,
            )

        # ---------------------------------------------------------------------
        # 9. SUCCESS Localization
        # ---------------------------------------------------------------------
        shadow_present = (top.bbox[3] > h_exp_px * 1.15)
        elapsed = (time.perf_counter() - t0) * 1000.0

        return HeldPieceDetection(
            status=TrackingStatus.SUCCESS,
            piece_name=expected_piece_name,
            bbox=top.bbox,
            centroid=top.centroid,
            contour_area=top.area,
            aspect_ratio=top.aspect_ratio,
            mean_saturation=top.mean_saturation,
            mean_value=top.mean_value,
            confidence=top.composite_score,
            is_abnormal_bbox=False,
            shadow_detected=shadow_present,
            message=f"Piece {expected_piece_name} localized at ({top.centroid[0]:.1f}, {top.centroid[1]:.1f}) with confidence {top.composite_score:.3f}.",
            candidates_evaluated=len(evaluated_candidates),
            top_candidates=tuple(evaluated_candidates[:5]),
            elapsed_ms=elapsed,
        )

    def _decompose_line_clear_preview(
        self,
        max_diff: np.ndarray,
        hold_image: np.ndarray,
        y_top_cut: int,
        w_exp_px: float,
        h_exp_px: float,
        area_exp_px: float,
        aspect_exp: float,
        is_abnormal_w: bool,
        is_abnormal_h: bool,
    ) -> Optional[CandidateBlob]:
        """Attempt to isolate solid held piece blocks when merged with a thin line-clear preview beam."""
        cfg = self.config
        # High-difference threshold to suppress subtle glow and isolate solid piece blocks
        _, bin_high = cv2.threshold(max_diff, 70, 255, cv2.THRESH_BINARY)

        # Apply directional opening to filter thin highlight strips (height <= 25px for horizontal, width <= 25px for vertical)
        k_open_h = 25 if is_abnormal_w else 1
        k_open_w = 25 if is_abnormal_h else 1
        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (k_open_w, k_open_h))
        opened = cv2.morphologyEx(bin_high, cv2.MORPH_OPEN, k_open)

        # Close with block-bridging kernel to reconnect polyomino sub-blocks
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        min_allowed_area = max(cfg.min_absolute_area, area_exp_px * cfg.min_area_ratio)
        best_cand: Optional[CandidateBlob] = None

        for c in contours:
            area = float(cv2.contourArea(c))
            if area < min_allowed_area:
                continue

            bx, by_roi, bw, bh = cv2.boundingRect(c)
            by = by_roi + y_top_cut
            aspect = bw / float(bh) if bh > 0 else 0.0

            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"]) + y_top_cut
            else:
                cx = float(bx + bw / 2.0)
                cy = float(by + bh / 2.0)

            patch = hold_image[by:by+bh, bx:bx+bw]
            if patch.size > 0:
                hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                mean_s = float(np.mean(hsv_patch[:, :, 1]))
                mean_v = float(np.mean(hsv_patch[:, :, 2]))
            else:
                mean_s, mean_v = 0.0, 0.0

            abnormal_w = (bw > w_exp_px * cfg.max_width_ratio)
            abnormal_h = (bh > h_exp_px * cfg.max_height_ratio)
            abnormal_area = (area > area_exp_px * cfg.max_area_ratio)
            is_abnormal = abnormal_w or abnormal_h or abnormal_area
            if is_abnormal:
                continue

            w_err = abs(bw - w_exp_px) / w_exp_px if w_exp_px > 0 else 1.0
            h_err = abs(bh - h_exp_px) / h_exp_px if h_exp_px > 0 else 1.0
            aspect_err = abs(aspect - aspect_exp) / aspect_exp if aspect_exp > 0 else 1.0
            dim_score = max(0.0, 1.0 - (0.40 * w_err + 0.30 * h_err + 0.30 * aspect_err))

            if mean_s < cfg.min_piece_saturation:
                continue
            sat_score = min(1.0, mean_s / 120.0)
            composite = 0.65 * dim_score + 0.35 * sat_score

            cand = CandidateBlob(
                bbox=(bx, by, bw, bh),
                centroid=(cx, cy),
                area=area,
                aspect_ratio=aspect,
                mean_saturation=mean_s,
                mean_value=mean_v,
                dim_score=dim_score,
                sat_score=sat_score,
                composite_score=composite,
                is_abnormal_bbox=False,
                rejection_reason=None,
            )

            if best_cand is None or cand.composite_score > best_cand.composite_score:
                best_cand = cand

        return best_cand

    def _validate_inputs(self, base_image: Any, hold_image: Any) -> Optional[str]:
        """Validate input arrays for image integrity and shape match."""
        if base_image is None or hold_image is None:
            return "Input image cannot be None."

        if not isinstance(base_image, np.ndarray) or not isinstance(hold_image, np.ndarray):
            return "Input images must be numpy.ndarray instances."

        if base_image.ndim != 3 or hold_image.ndim != 3:
            return f"Input images must have 3 dimensions (got base={base_image.ndim}, hold={hold_image.ndim})."

        if base_image.shape[2] != 3 or hold_image.shape[2] != 3:
            return f"Input images must be 3-channel BGR (got base channels={base_image.shape[2]}, hold channels={hold_image.shape[2]})."

        if base_image.shape != hold_image.shape:
            return f"Dimension mismatch: base shape {base_image.shape} != hold shape {hold_image.shape}."

        h, w = base_image.shape[:2]
        if h < 300 or w < 300:
            return f"Image dimensions too small ({w}x{h}); minimum is 300x300."

        return None
