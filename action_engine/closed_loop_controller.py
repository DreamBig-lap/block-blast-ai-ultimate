"""Closed-Loop Drag Controller for BLOCK BLAST AI ULTIMATE (Dry-Run Mode).

Provides an independent feedback control layer that coordinates:
- AI Decision moves (PlannedMove / GameState)
- Vision-calibrated touch trajectory planning (with modular, non-hardcoded calibration)
- In-flight held-piece localization via HeldPieceTracker
- Spatial error measurement (e_X, e_Y) relative to target grid snap centers
- Single-step micro-correction proposals (NEEDS_CORRECTION)
- Strict safety aborts (UNKNOWN, AMBIGUOUS, CORRECTION_LIMIT_EXCEEDED)
- Strict DRY-RUN guarantees (NO physical ADB touch commands or device actuation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import enum
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from action_engine.held_piece_tracker import (
    HeldPieceDetection,
    HeldPieceTracker,
    HeldPieceTrackerConfig,
    TrackingStatus,
)
from ai_engine.decision import PlannedMove
from game_engine.pieces import CANONICAL_41_CATALOG, Piece
from game_engine.state import GameState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibration Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DragCalibrationConfig:
    """Configurable touch-to-piece motion transformation parameters.

    CRITICAL ENGINEERING PRINCIPLE:
    Hằng số Gain X, Gain Y và Lift Y là số liệu thực nghiệm đo được trên thiết bị mục tiêu,
    KHÔNG PHẢI hằng số toán học phổ quát. Module này hỗ trợ cấu hình động 2D độc lập
    giữa các trục và phân biệt nguồn gốc hiệu chuẩn (calibration_source).
    """

    drag_gain_x: float = 1.0
    drag_gain_y: float = 1.0
    vertical_lift_px: float = 200.0
    calibration_source: str = "DEFAULT_UNCHECKED"
    calibration_confidence: float = 0.50
    convergence_tolerance_px: float = 25.0
    max_corrections_allowed: int = 1
    cell_size_px: float = 121.5
    board_bbox: Tuple[int, int, int, int] = (54, 548, 972, 972)  # (x, y, w, h)

    # Per-piece specific gains and lifts if calibrated
    piece_gains: dict[str, float] = field(default_factory=dict)       # Legacy alias for piece_gains_x
    piece_gains_x: dict[str, float] = field(default_factory=dict)
    piece_gains_y: dict[str, float] = field(default_factory=dict)
    piece_lifts: dict[str, float] = field(default_factory=dict)
    piece_provenance: dict[str, str] = field(default_factory=dict)

    @classmethod
    def create_default(cls) -> DragCalibrationConfig:
        """Create baseline default uncalibrated configuration."""
        return cls(
            drag_gain_x=1.0,
            drag_gain_y=1.0,
            vertical_lift_px=200.0,
            calibration_source="DEFAULT_OPEN_LOOP_HYPOTHESIS",
            calibration_confidence=0.30,
            convergence_tolerance_px=25.0,
            max_corrections_allowed=1,
            cell_size_px=121.5,
            board_bbox=(54, 548, 972, 972),
        )

    @classmethod
    def create_samsung_m21_calibrated(cls) -> DragCalibrationConfig:
        """Create empirical configuration calibrated on Samsung Galaxy M21 (Phase 3.5.3 & 3.5.7)."""
        return cls(
            drag_gain_x=1.428,
            drag_gain_y=1.438,
            vertical_lift_px=314.0,
            calibration_source="EMPIRICAL_M21_PHASE_3_5_7",
            calibration_confidence=0.95,
            convergence_tolerance_px=25.0,
            max_corrections_allowed=1,
            cell_size_px=121.5,
            board_bbox=(54, 548, 972, 972),
            piece_gains_x={"line_1x5_h": 1.428, "line_1x3_h": 1.413},
            piece_gains_y={"line_1x5_h": 1.380, "line_1x3_h": 1.518},
            piece_lifts={"line_1x5_h": 314.0, "line_1x3_h": 314.0},
            piece_provenance={
                "line_1x5_h": "M21_PHASE_3_5_3_MEASURED",
                "line_1x3_h": "M21_PHASE_3_5_7_MEASURED",
            },
        )

    def get_gain_x_for_piece(self, piece_name: str) -> float:
        """Retrieve piece-specific Gain X, falling back to piece_gains or general drag_gain_x."""
        if piece_name in self.piece_gains_x:
            return self.piece_gains_x[piece_name]
        return self.piece_gains.get(piece_name, self.drag_gain_x)

    def get_gain_y_for_piece(self, piece_name: str) -> float:
        """Retrieve piece-specific Gain Y, falling back to general drag_gain_y."""
        return self.piece_gains_y.get(piece_name, self.drag_gain_y)

    def get_gain_for_piece(self, piece_name: str) -> float:
        """Legacy helper for Gain X."""
        return self.get_gain_x_for_piece(piece_name)

    def get_lift_for_piece(self, piece_name: str) -> float:
        """Retrieve piece-specific Lift Y, falling back to general vertical_lift_px."""
        return self.piece_lifts.get(piece_name, self.vertical_lift_px)

    def is_valid(self) -> bool:
        """Validate calibration sanity across all axes and piece tables."""
        if self.drag_gain_x <= 0.1 or np.isnan(self.drag_gain_x) or np.isinf(self.drag_gain_x):
            return False
        if self.drag_gain_y <= 0.1 or np.isnan(self.drag_gain_y) or np.isinf(self.drag_gain_y):
            return False
        if self.vertical_lift_px <= 0.0 or np.isnan(self.vertical_lift_px) or np.isinf(self.vertical_lift_px):
            return False
        if self.cell_size_px <= 10.0 or np.isnan(self.cell_size_px) or np.isinf(self.cell_size_px):
            return False
        if self.convergence_tolerance_px <= 0.0 or np.isnan(self.convergence_tolerance_px):
            return False

        # Validate piece-specific dictionaries
        for gx in self.piece_gains_x.values():
            if gx <= 0.1 or np.isnan(gx) or np.isinf(gx):
                return False
        for gy in self.piece_gains_y.values():
            if gy <= 0.1 or np.isnan(gy) or np.isinf(gy):
                return False
        for ly in self.piece_lifts.values():
            if ly <= 0.0 or np.isnan(ly) or np.isinf(ly):
                return False
        for g in self.piece_gains.values():
            if g <= 0.1 or np.isnan(g) or np.isinf(g):
                return False

        return True


# ---------------------------------------------------------------------------
# Control Status & Result Models
# ---------------------------------------------------------------------------

class ClosedLoopStatus(enum.Enum):
    """Outcome status of closed-loop evaluation."""

    CONVERGED = "CONVERGED"                              # Piece within snap tolerance; safe to release UP
    NEEDS_CORRECTION = "NEEDS_CORRECTION"                # Deviation > tolerance; correction move proposed
    CORRECTION_LIMIT_EXCEEDED = "LIMIT_EXCEEDED"        # Max corrections budget exhausted
    UNKNOWN = "UNKNOWN"                                  # Vision tracker uncertain, ambiguous, or failed
    INVALID_CALIBRATION = "INVALID_CALIBRATION"          # Calibration config is non-physical or invalid


@dataclass(frozen=True, slots=True)
class ClosedLoopStepResult:
    """Comprehensive diagnostic container for a closed-loop evaluation step."""

    status: ClosedLoopStatus
    step_index: int
    piece_name: str
    slot_idx: int
    target_grid_coord: Tuple[int, int]                  # (row, col)
    target_screen_center: Tuple[float, float]           # (X_target, Y_target) visual grid snap center
    initial_finger_down: Tuple[int, int]                # (start_x, start_y) on tray
    current_finger_pos: Tuple[int, int]                 # (end_x, end_y) current touch target
    detected_piece_center: Optional[Tuple[float, float]] # (X_seen, Y_seen) from HeldPieceTracker
    detected_bbox: Optional[Tuple[int, int, int, int]]  # (x, y, w, h) from HeldPieceTracker
    error_x_px: float                                    # X_target - X_seen
    error_y_px: float                                    # Y_target - Y_seen
    proposed_correction_move: Optional[Tuple[int, int]]  # Next proposed finger MOVE target
    correction_count: int
    tracker_detection: Optional[HeldPieceDetection]
    is_dry_run: bool
    message: str

    @property
    def is_converged(self) -> bool:
        return self.status == ClosedLoopStatus.CONVERGED

    @property
    def requires_failsafe_release(self) -> bool:
        """True if touch must be released immediately and host must PAUSE."""
        return self.status in (
            ClosedLoopStatus.CORRECTION_LIMIT_EXCEEDED,
            ClosedLoopStatus.UNKNOWN,
            ClosedLoopStatus.INVALID_CALIBRATION,
        )

    @property
    def euclidean_error_px(self) -> float:
        """Euclidean distance between detected piece center and target center."""
        return float(np.hypot(self.error_x_px, self.error_y_px))


# ---------------------------------------------------------------------------
# Independent Closed-Loop Controller
# ---------------------------------------------------------------------------

class ClosedLoopController:
    """Manages closed-loop trajectory generation, held-piece feedback, and error correction."""

    def __init__(
        self,
        config: Optional[DragCalibrationConfig] = None,
        tracker: Optional[HeldPieceTracker] = None,
        dry_run: bool = True,
    ) -> None:
        self.config = config or DragCalibrationConfig.create_default()
        self.tracker = tracker or HeldPieceTracker()
        self.dry_run = dry_run

    def plan_initial_trajectory(
        self,
        move: PlannedMove,
        slot_center_x: int,
        slot_center_y: int,
        piece_width_cells: Optional[int] = None,
        piece_height_cells: Optional[int] = None,
    ) -> Tuple[int, int, int, int]:
        """Compute the initial (start_x, start_y, end_x, end_y) finger coordinates.

        Args:
            move: Move proposed by AI Decision Engine.
            slot_center_x: Center X of tray slot on screen.
            slot_center_y: Center Y of tray slot on screen.
            piece_width_cells: Optional width in cells (retrieved from catalog if None).
            piece_height_cells: Optional height in cells.

        Returns:
            Tuple (start_x, start_y, end_x, end_y) for finger actuation.
        """
        cfg = self.config
        board_x, board_y, _, _ = cfg.board_bbox
        cell_size = cfg.cell_size_px

        # Determine piece dimensions
        if piece_width_cells is None or piece_height_cells is None:
            piece = CANONICAL_41_CATALOG.get_by_name(move.piece_name)
            pw = piece.width if piece else 1
            ph = piece.height if piece else 1
        else:
            pw = piece_width_cells
            ph = piece_height_cells

        # Target geometric visual center on the 8x8 grid
        x_target_grid = board_x + (move.col + pw / 2.0) * cell_size
        y_target_grid = board_y + (move.row + ph / 2.0) * cell_size

        # Retrieve calibration parameters
        gain_x = cfg.get_gain_x_for_piece(move.piece_name)
        gain_y = cfg.get_gain_y_for_piece(move.piece_name)
        lift_y = cfg.get_lift_for_piece(move.piece_name)

        # Invert horizontal gain: delta_finger_x = delta_piece_x / gain_x
        delta_piece_x = x_target_grid - slot_center_x
        delta_finger_x = delta_piece_x / gain_x if gain_x != 0 else delta_piece_x
        end_x = int(round(slot_center_x + delta_finger_x))

        # Invert vertical trajectory with Gain Y and initial pickup lift H0 (lift_y):
        # Piece visual center: Y_piece = (slot_center_y - lift_y) + gain_y * (end_y - slot_center_y)
        # When gain_y == 1.0 (classic uncalibrated model):
        #   end_y = y_target_grid + lift_y
        if abs(gain_y - 1.0) < 1e-6:
            end_y = int(round(y_target_grid + lift_y))
        else:
            delta_piece_y = y_target_grid - (slot_center_y - lift_y)
            delta_finger_y = delta_piece_y / gain_y if gain_y != 0 else delta_piece_y
            end_y = int(round(slot_center_y + delta_finger_y))

        return (int(slot_center_x), int(slot_center_y), end_x, end_y)

    def evaluate_held_frame(
        self,
        move: PlannedMove,
        slot_center: Tuple[int, int],
        current_finger_pos: Tuple[int, int],
        base_frame: np.ndarray,
        mid_hold_frame: np.ndarray,
        correction_count: int = 0,
        piece_width_cells: Optional[int] = None,
        piece_height_cells: Optional[int] = None,
    ) -> ClosedLoopStepResult:
        """Evaluate in-flight hold feedback, compute spatial error, and propose corrections.

        Args:
            move: Move proposed by AI Decision Engine.
            slot_center: Initial (x, y) coordinates of the pickup slot.
            current_finger_pos: Current (end_x, end_y) touch position held on screen.
            base_frame: Image captured before touch began.
            mid_hold_frame: Image captured while touch is held DOWN.
            correction_count: How many corrections have already been performed for this move.
            piece_width_cells: Width of piece in cells.
            piece_height_cells: Height of piece in cells.

        Returns:
            ClosedLoopStepResult with convergence state and proposed next action.
        """
        cfg = self.config

        # 1. Calibration Sanity Check
        if not cfg.is_valid():
            return ClosedLoopStepResult(
                status=ClosedLoopStatus.INVALID_CALIBRATION,
                step_index=move.step_index,
                piece_name=move.piece_name,
                slot_idx=move.slot_idx,
                target_grid_coord=(move.row, move.col),
                target_screen_center=(0.0, 0.0),
                initial_finger_down=slot_center,
                current_finger_pos=current_finger_pos,
                detected_piece_center=None,
                detected_bbox=None,
                error_x_px=0.0,
                error_y_px=0.0,
                proposed_correction_move=None,
                correction_count=correction_count,
                tracker_detection=None,
                is_dry_run=self.dry_run,
                message=f"Invalid calibration parameters: Gain_X={cfg.drag_gain_x}, Lift_Y={cfg.vertical_lift_px}",
            )

        # 2. Determine Theoretical Target Center on Grid
        board_x, board_y, _, _ = cfg.board_bbox
        cell_size = cfg.cell_size_px
        if piece_width_cells is None or piece_height_cells is None:
            piece = CANONICAL_41_CATALOG.get_by_name(move.piece_name)
            pw = piece.width if piece else 1
            ph = piece.height if piece else 1
        else:
            pw = piece_width_cells
            ph = piece_height_cells

        x_target_grid = float(board_x + (move.col + pw / 2.0) * cell_size)
        y_target_grid = float(board_y + (move.row + ph / 2.0) * cell_size)

        # 3. Invoke Vision Tracker
        detection = self.tracker.track_held_piece(
            base_image=base_frame,
            hold_image=mid_hold_frame,
            expected_piece_name=move.piece_name,
            cell_size_px=cell_size,
        )

        # 4. Handle Vision Tracker Non-Success Outcomes or Low Confidence
        if detection.status != TrackingStatus.SUCCESS or detection.centroid is None or detection.confidence < 0.60:
            err_x = float(x_target_grid - detection.centroid[0]) if detection.centroid is not None else 0.0
            err_y = float(y_target_grid - detection.centroid[1]) if detection.centroid is not None else 0.0
            return ClosedLoopStepResult(
                status=ClosedLoopStatus.UNKNOWN,
                step_index=move.step_index,
                piece_name=move.piece_name,
                slot_idx=move.slot_idx,
                target_grid_coord=(move.row, move.col),
                target_screen_center=(x_target_grid, y_target_grid),
                initial_finger_down=slot_center,
                current_finger_pos=current_finger_pos,
                detected_piece_center=detection.centroid,
                detected_bbox=detection.bbox,
                error_x_px=err_x,
                error_y_px=err_y,
                proposed_correction_move=None,
                correction_count=correction_count,
                tracker_detection=detection,
                is_dry_run=self.dry_run,
                message=f"Vision tracker did not achieve SUCCESS (confidence={detection.confidence:.2f}): status={detection.status.value} ({detection.message})",
            )

        # 5. Spatial Error Calculation
        assert detection.centroid is not None
        seen_x, seen_y = detection.centroid
        error_x = x_target_grid - seen_x
        error_y = y_target_grid - seen_y
        dist_2d = float(np.hypot(error_x, error_y))

        tol = cfg.convergence_tolerance_px
        gain_x = cfg.get_gain_for_piece(move.piece_name)

        # 6. Convergence Evaluation (Strict 2D criteria: both axes and Euclidean distance <= tol)
        if abs(error_x) <= tol and abs(error_y) <= tol and dist_2d <= tol:
            return ClosedLoopStepResult(
                status=ClosedLoopStatus.CONVERGED,
                step_index=move.step_index,
                piece_name=move.piece_name,
                slot_idx=move.slot_idx,
                target_grid_coord=(move.row, move.col),
                target_screen_center=(x_target_grid, y_target_grid),
                initial_finger_down=slot_center,
                current_finger_pos=current_finger_pos,
                detected_piece_center=(seen_x, seen_y),
                detected_bbox=detection.bbox,
                error_x_px=error_x,
                error_y_px=error_y,
                proposed_correction_move=None,
                correction_count=correction_count,
                tracker_detection=detection,
                is_dry_run=self.dry_run,
                message=f"Piece {move.piece_name} converged within tolerance ({abs(error_x):.1f}px, {abs(error_y):.1f}px <= {tol:.1f}px).",
            )

        # 7. Correction Limit Check
        if correction_count >= cfg.max_corrections_allowed:
            return ClosedLoopStepResult(
                status=ClosedLoopStatus.CORRECTION_LIMIT_EXCEEDED,
                step_index=move.step_index,
                piece_name=move.piece_name,
                slot_idx=move.slot_idx,
                target_grid_coord=(move.row, move.col),
                target_screen_center=(x_target_grid, y_target_grid),
                initial_finger_down=slot_center,
                current_finger_pos=current_finger_pos,
                detected_piece_center=(seen_x, seen_y),
                detected_bbox=detection.bbox,
                error_x_px=error_x,
                error_y_px=error_y,
                proposed_correction_move=None,
                correction_count=correction_count,
                tracker_detection=detection,
                is_dry_run=self.dry_run,
                message=f"Correction limit reached ({correction_count}/{cfg.max_corrections_allowed}). Error persists: dX={error_x:.1f}px, dY={error_y:.1f}px.",
            )

        # 8. Micro-Correction Calculation
        # Delta finger to compensate piece deviation along both calibrated axes:
        gain_x = cfg.get_gain_x_for_piece(move.piece_name)
        gain_y = cfg.get_gain_y_for_piece(move.piece_name)

        delta_finger_corr_x = error_x / gain_x if gain_x != 0 else error_x
        delta_finger_corr_y = error_y / gain_y if gain_y != 0 else error_y

        proposed_x = int(round(current_finger_pos[0] + delta_finger_corr_x))
        proposed_y = int(round(current_finger_pos[1] + delta_finger_corr_y))

        return ClosedLoopStepResult(
            status=ClosedLoopStatus.NEEDS_CORRECTION,
            step_index=move.step_index,
            piece_name=move.piece_name,
            slot_idx=move.slot_idx,
            target_grid_coord=(move.row, move.col),
            target_screen_center=(x_target_grid, y_target_grid),
            initial_finger_down=slot_center,
            current_finger_pos=current_finger_pos,
            detected_piece_center=(seen_x, seen_y),
            detected_bbox=detection.bbox,
            error_x_px=error_x,
            error_y_px=error_y,
            proposed_correction_move=(proposed_x, proposed_y),
            correction_count=correction_count,
            tracker_detection=detection,
            is_dry_run=self.dry_run,
            message=f"Needs micro-correction: dX={error_x:.1f}px, dY={error_y:.1f}px. Proposed finger adjustment -> ({proposed_x}, {proposed_y}).",
        )

    def should_pause_after_evaluation(self, result: ClosedLoopStepResult) -> bool:
        """Determines whether the closed-loop evaluation mandates a host PAUSE."""
        return result.status in (
            ClosedLoopStatus.CORRECTION_LIMIT_EXCEEDED,
            ClosedLoopStatus.UNKNOWN,
            ClosedLoopStatus.INVALID_CALIBRATION,
        )

    def verify_game_state_immutability(self, state_before: GameState, state_after: GameState) -> bool:
        """Verify that GameState bitboard, score, combo, and tray were not modified."""
        if state_before.board != state_after.board:
            return False
        if state_before.score != state_after.score:
            return False
        if state_before.combo_streak != state_after.combo_streak:
            return False
        if state_before.tray != state_after.tray:
            return False
        return True
