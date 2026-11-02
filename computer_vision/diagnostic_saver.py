"""Perception Diagnostic Failure Capture for BLOCK BLAST AI ULTIMATE.

When perception returns LOW_CONFIDENCE, FAILED, or contains UNKNOWN / uncertain
tray pieces, automatically saves a uniquely named forensic diagnostic folder
containing:
1. full_screen.png — original full-resolution BGR screenshot.
2. tray_band.png — complete tray region before segmentation.
3. slot_0.png, slot_1.png, slot_2.png — exact crops passed to detector.
4. slot_0_mask.png, slot_1_mask.png, slot_2_mask.png — binary masks produced by segmentation.
5. tray_contours.png — tray image with detected contours, bounding boxes, and contour areas drawn.
6. recognition_report.json — comprehensive diagnostic metrics and thresholds.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import cv2
import numpy as np

from computer_vision.config import DEFAULT_VISION_CONFIG, VisionConfig
from computer_vision.models import FrameRecognitionResult, RecognitionStatus
from computer_vision.piece_detector import PieceDetector, TrayDetectionDetails

logger = logging.getLogger(__name__)


class PerceptionDiagnosticSaver:
    """Saves forensic diagnostic packages on perception failure / low confidence."""

    def __init__(
        self,
        config: Optional[VisionConfig] = None,
        base_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.config = config or DEFAULT_VISION_CONFIG
        self.base_dir = Path(base_dir or (Path(self.config.debug_dir) / "perception_failures"))
        self.last_saved_dir: Optional[Path] = None

    def should_save(
        self,
        recognition_result: FrameRecognitionResult,
        min_overall_confidence: Optional[float] = None,
        min_piece_confidence: Optional[float] = None,
    ) -> bool:
        """Determine whether the recognition result warrants diagnostic capture."""
        min_overall = (
            min_overall_confidence
            if min_overall_confidence is not None
            else self.config.min_success_overall_confidence
        )
        min_piece = (
            min_piece_confidence
            if min_piece_confidence is not None
            else getattr(self.config, "min_piece_confidence", 0.70)
        )

        if recognition_result.status in (RecognitionStatus.FAILED, RecognitionStatus.LOW_CONFIDENCE):
            return True

        if recognition_result.board is None:
            return True

        if recognition_result.overall_confidence < min_overall:
            return True

        for p in recognition_result.tray_pieces:
            if not p.is_empty and (p.piece is None or p.matched_name == "UNKNOWN" or p.confidence < min_piece):
                return True

        return False

    def create_unique_failure_dir(self, timestamp: Optional[datetime] = None) -> Path:
        """Create a uniquely named folder ensuring no previous folder is overwritten.

        Format: debug/perception_failures/YYYYMMDD_HHMMSS/
        If collision occurs, appends _1, _2, etc.
        """
        now = timestamp or datetime.now()
        folder_base = now.strftime("%Y%m%d_%H%M%S")
        target_dir = self.base_dir / folder_base

        counter = 1
        while target_dir.exists():
            target_dir = self.base_dir / f"{folder_base}_{counter}"
            counter += 1

        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    def render_tray_contours_visualization(
        self,
        tray_details: TrayDetectionDetails,
    ) -> np.ndarray:
        """Draw tray image with detected contours, bounding boxes, and contour areas."""
        vis = tray_details.tray_band.copy()
        h_vis, w_vis = vis.shape[:2]

        for slot_idx, slot in enumerate(tray_details.slot_details):
            x_min, x_max, _, _ = slot.crop_coords

            # Slot partition divider lines
            cv2.line(vis, (x_min, 0), (x_min, h_vis), (255, 255, 0), 1)
            cv2.line(vis, (x_max - 1, 0), (x_max - 1, h_vis), (255, 255, 0), 1)

            # Slot header label
            status_lbl = "EMPTY" if slot.is_empty else f"CONTOURS: {len(slot.valid_contours)}"
            cv2.putText(
                vis,
                f"Slot {slot_idx} ({status_lbl})",
                (x_min + 6, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

            # Draw contours shifted by slot x_min
            for c in slot.all_contours:
                c_shifted = c + np.array([[[x_min, 0]]], dtype=c.dtype)
                area = float(cv2.contourArea(c))
                is_valid = area >= self.config.tray_min_contour_area

                color = (0, 255, 0) if is_valid else (120, 120, 120)
                thickness = 2 if is_valid else 1
                cv2.drawContours(vis, [c_shifted], -1, color, thickness)

                # Label contour area near center of mass or first point
                moments = cv2.moments(c)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"]) + x_min
                    cy = int(moments["m01"] / moments["m00"])
                else:
                    cx = int(c_shifted[0][0][0])
                    cy = int(c_shifted[0][0][1])

                cx = max(x_min + 5, min(x_max - 50, cx))
                cy = max(15, min(h_vis - 10, cy))

                cv2.putText(
                    vis,
                    f"A:{int(area)}",
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (0, 255, 255) if is_valid else (170, 170, 170),
                    1,
                    cv2.LINE_AA,
                )

            # Draw detected bounding box
            if slot.bbox_in_slot is not None and not slot.is_empty:
                lx, ly, lw, lh = slot.bbox_in_slot
                gx = x_min + lx
                cv2.rectangle(vis, (gx, ly), (gx + lw, ly + lh), (0, 0, 255), 2)
                cv2.putText(
                    vis,
                    f"BBox: {lw}x{lh}",
                    (gx, max(15, ly - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

        return vis

    def save_diagnostic_failure(
        self,
        frame: np.ndarray,
        recognition_result: FrameRecognitionResult,
        pause_reason: Optional[str] = None,
        piece_detector: Optional[PieceDetector] = None,
        timestamp: Optional[datetime] = None,
    ) -> Optional[Path]:
        """Save complete diagnostic artifacts for the failed recognition attempt.

        Artifacts:
        1. full_screen.png
        2. tray_band.png
        3. slot_0.png, slot_1.png, slot_2.png
        4. slot_0_mask.png, slot_1_mask.png, slot_2_mask.png
        5. tray_contours.png
        6. recognition_report.json

        Returns:
            The Path to the created folder, or None if saving failed.
        """
        try:
            folder = self.create_unique_failure_dir(timestamp=timestamp)
            h_img, w_img = frame.shape[:2]
            c_img = frame.shape[2] if frame.ndim > 2 else 1

            # 1. full_screen.png
            cv2.imwrite(str(folder / "full_screen.png"), frame)

            # 2. Extract or retrieve TrayDetectionDetails from the same attempt
            tray_details: Optional[TrayDetectionDetails] = None
            if piece_detector is not None and getattr(piece_detector, "last_detection_details", None) is not None:
                tray_details = piece_detector.last_detection_details
            else:
                # If detector didn't record details yet, run detection on frame
                det = piece_detector or PieceDetector(self.config)
                det.detect_tray_pieces(frame)
                tray_details = getattr(det, "last_detection_details", None)

            # Fallback if tray detection could not run
            y_start = int(h_img * self.config.tray_expected_y_min_rel)
            y_end = int(h_img * self.config.tray_expected_y_max_rel)
            if tray_details is None or tray_details.tray_band is None or tray_details.tray_band.size == 0:
                raw_tray = frame[y_start:y_end, :] if y_end > y_start else np.zeros((10, 10, 3), dtype=np.uint8)
            else:
                raw_tray = tray_details.tray_band

            # 2. tray_band.png
            cv2.imwrite(str(folder / "tray_band.png"), raw_tray)

            # 3 & 4. slot crops and slot masks
            slots_report_data: list[dict[str, Any]] = []

            for slot_idx, (x_rel_min, x_rel_max) in enumerate(self.config.tray_slot_x_splits):
                x_min = int(w_img * x_rel_min)
                x_max = int(w_img * x_rel_max)
                w_slot = x_max - x_min
                h_slot = y_end - y_start

                # Find slot detail from intermediate detector details if available
                slot_detail = None
                if tray_details is not None and slot_idx < len(tray_details.slot_details):
                    slot_detail = tray_details.slot_details[slot_idx]

                # Slot BGR crop
                if slot_detail is not None and slot_detail.slot_bgr is not None and slot_detail.slot_bgr.size > 0:
                    slot_bgr = slot_detail.slot_bgr
                else:
                    slot_bgr = raw_tray[:, x_min:x_max] if raw_tray.size > 0 else np.zeros((10, 10, 3), dtype=np.uint8)
                cv2.imwrite(str(folder / f"slot_{slot_idx}.png"), slot_bgr)

                # Slot binary mask
                if slot_detail is not None and slot_detail.slot_mask is not None and slot_detail.slot_mask.size > 0:
                    slot_mask = slot_detail.slot_mask
                else:
                    slot_mask = np.zeros((max(1, h_slot), max(1, w_slot)), dtype=np.uint8)
                cv2.imwrite(str(folder / f"slot_{slot_idx}_mask.png"), slot_mask)

                # Matched piece recognition result from pipeline
                matched_p = None
                for p in recognition_result.tray_pieces:
                    if p.slot_idx == slot_idx:
                        matched_p = p
                        break

                matched_name = getattr(matched_p, "matched_name", None)
                p_conf = float(getattr(matched_p, "confidence", 0.0))
                p_cell_count = getattr(matched_p, "cell_count", 0)
                p_cells = getattr(matched_p, "cells", ())
                p_height = getattr(matched_p, "height", 0)
                p_width = getattr(matched_p, "width", 0)
                p_status_msg = getattr(matched_p, "status_message", "No piece data")
                is_p_empty = getattr(matched_p, "is_empty", (slot_detail.is_empty if slot_detail else True))

                detected_bbox_slot = list(slot_detail.bbox_in_slot) if (slot_detail and slot_detail.bbox_in_slot) else None
                detected_bbox_screen = list(slot_detail.bbox_in_screen) if slot_detail else (
                    list(getattr(matched_p, "bbox_in_screen", ())) if matched_p else None
                )

                slot_dict = {
                    "slot_idx": slot_idx,
                    "crop_coordinates": {
                        "x_min": x_min,
                        "x_max": x_max,
                        "y_min": y_start,
                        "y_max": y_end,
                        "width": w_slot,
                        "height": h_slot,
                    },
                    "detected_contour_count": len(slot_detail.all_contours) if slot_detail else 0,
                    "valid_contour_count": len(slot_detail.valid_contours) if slot_detail else 0,
                    "contour_areas": slot_detail.contour_areas if slot_detail else [],
                    "detected_bounding_box_slot": detected_bbox_slot,
                    "detected_bounding_box_screen": detected_bbox_screen,
                    "is_empty": is_p_empty,
                    "classification_result": {
                        "matched_name": matched_name,
                        "confidence": p_conf,
                        "cell_count": p_cell_count,
                        "cells": [list(c) for c in p_cells],
                        "height": p_height,
                        "width": p_width,
                    },
                    "classifier_rejection_reason": p_status_msg,
                }
                slots_report_data.append(slot_dict)

            # 5. tray_contours.png
            if tray_details is not None and tray_details.slot_details:
                tray_vis = self.render_tray_contours_visualization(tray_details)
            else:
                tray_vis = raw_tray.copy()
            cv2.imwrite(str(folder / "tray_contours.png"), tray_vis)

            # 6. recognition_report.json
            b = recognition_result.board
            board_dict: dict[str, Any] = {
                "detected": b is not None,
                "board_bbox": list(b.board_bbox) if b else None,
                "board_confidence": float(b.board_confidence) if b else None,
                "cell_confidence": float(b.cell_confidence) if b else None,
                "occupied_count": b.occupied_count if b else 0,
                "empty_count": b.empty_count if b else 0,
                "bitboard_hex": f"0x{b.bitboard:016X}" if b else None,
            }

            report = {
                "timestamp": (timestamp or datetime.now()).isoformat(),
                "folder_name": folder.name,
                "recognition_status": recognition_result.status.value if hasattr(recognition_result.status, "value") else str(recognition_result.status),
                "overall_confidence": float(recognition_result.overall_confidence),
                "pause_reason": pause_reason or recognition_result.error_message or "Unknown",
                "screenshot": {
                    "height": int(h_img),
                    "width": int(w_img),
                    "channels": int(c_img),
                },
                "board": board_dict,
                "tray_roi": {
                    "y_min": y_start,
                    "y_max": y_end,
                    "y_min_rel": float(self.config.tray_expected_y_min_rel),
                    "y_max_rel": float(self.config.tray_expected_y_max_rel),
                    "width": int(w_img),
                    "height": int(y_end - y_start),
                    "bg_gray_estimate": float(tray_details.bg_gray_est) if tray_details else None,
                    "bg_bgr_estimate": [round(float(c), 1) for c in tray_details.bg_bgr_est] if (tray_details and getattr(tray_details, "bg_bgr_est", None) is not None) else None,
                },
                "slots": slots_report_data,
                "segmentation_thresholds_and_config": {
                    "tray_bg_diff_threshold": float(self.config.tray_bg_diff_threshold),
                    "tray_color_diff_threshold": float(getattr(self.config, "tray_color_diff_threshold", self.config.tray_bg_diff_threshold)),
                    "tray_min_contour_area": float(self.config.tray_min_contour_area),
                    "tray_unit_size_min": float(self.config.tray_unit_size_min),
                    "tray_unit_size_max": float(self.config.tray_unit_size_max),
                    "min_piece_match_confidence": float(self.config.min_piece_match_confidence),
                    "min_success_overall_confidence": float(self.config.min_success_overall_confidence),
                    "min_cell_confidence": float(self.config.min_cell_confidence),
                    "min_overall_confidence": float(getattr(self.config, "min_overall_confidence", 0.75)),
                    "min_piece_confidence": float(getattr(self.config, "min_piece_confidence", 0.70)),
                    "cell_occupancy_gray_threshold": float(self.config.cell_occupancy_gray_threshold),
                    "cell_occupancy_sat_threshold": float(self.config.cell_occupancy_sat_threshold),
                },
            }

            with open(folder / "recognition_report.json", "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

            self.last_saved_dir = folder
            logger.info("Perception diagnostic artifacts successfully saved to: %s", folder)
            return folder

        except Exception as e:
            logger.exception("Failed to save perception diagnostic artifacts: %s", e)
            return None
