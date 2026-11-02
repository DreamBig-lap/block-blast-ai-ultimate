"""Physical Screen Coordinate Mapper for BLOCK BLAST AI ULTIMATE.

Translates abstract AI planned moves (slot_idx, row, col) into precise
screen pixel trajectories (start_x, start_y, end_x, end_y, duration_ms)
calibrated dynamically from Computer Vision detection.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional

from ai_engine.decision import PlannedMove
from computer_vision.models import FrameRecognitionResult, PieceRecognitionResult
from game_engine.board import can_place
from game_engine.pieces import Piece

logger = logging.getLogger(__name__)


class CoordinateMapperError(Exception):
    """Raised when coordinate mapping fails or parameters are invalid."""
    pass


@dataclass(frozen=True, slots=True)
class DragTrajectory:
    """Complete physical actuation path for dragging a piece from tray to board."""

    slot_idx: int
    piece_name: str
    target_row: int
    target_col: int
    start_x: int
    start_y: int
    end_x: int
    end_y: int
    duration_ms: int
    vertical_offset_px: float

    def to_dict(self) -> dict:
        return {
            "slot_idx": self.slot_idx,
            "piece_name": self.piece_name,
            "target": [self.target_row, self.target_col],
            "start": [self.start_x, self.start_y],
            "end": [self.end_x, self.end_y],
            "duration_ms": self.duration_ms,
            "vertical_offset_px": round(self.vertical_offset_px, 1),
        }


class CoordinateMapper:
    """Calculates screen-space drag trajectories using dynamic vision calibration."""

    def __init__(
        self,
        vertical_offset_ratio: float = 1.65,
        default_drag_duration_ms: int = 380,
        use_cell_centroid: bool = False,
    ) -> None:
        """
        Args:
            vertical_offset_ratio: Multiplier of cell_size to offset the touch point below
                the piece center so the visual piece aligns with the board grid.
                Default 1.65 is derived from the official Android Block Blast mechanics.
            default_drag_duration_ms: Duration of the swipe in milliseconds (300-500ms recommended).
            use_cell_centroid: If True, uses the center of mass of occupied cells instead of bbox center.
        """
        self.vertical_offset_ratio = vertical_offset_ratio
        self.default_drag_duration_ms = default_drag_duration_ms
        self.use_cell_centroid = use_cell_centroid

    def map_move(
        self,
        move: PlannedMove,
        frame_result: FrameRecognitionResult,
        drag_duration_ms: Optional[int] = None,
    ) -> DragTrajectory:
        """Translate a PlannedMove into a physical DragTrajectory.

        Args:
            move: Move proposed by AI Decision Engine.
            frame_result: Recognition result containing board_bbox, cell_size, and tray pieces.
            drag_duration_ms: Optional custom swipe duration.

        Returns:
            DragTrajectory ready for injection via TouchController.

        Raises:
            CoordinateMapperError: If board is missing, slot is invalid/empty, or coordinates out of bounds.
        """
        if frame_result.board is None:
            raise CoordinateMapperError("Cannot map move: frame recognition has no board detected.")

        board_x, board_y, board_w, board_h = frame_result.board.board_bbox
        cell_size = frame_result.board.cell_size
        img_h, img_w = frame_result.image_shape

        # 1. Validate slot index
        if move.slot_idx < 0 or move.slot_idx >= len(frame_result.tray_pieces):
            raise CoordinateMapperError(
                f"Invalid tray slot {move.slot_idx}; frame has {len(frame_result.tray_pieces)} slots."
            )

        slot_res: PieceRecognitionResult = frame_result.tray_pieces[move.slot_idx]
        if slot_res.is_empty:
            raise CoordinateMapperError(f"Tray slot {move.slot_idx} is reported as empty.")

        if slot_res.piece is None:
            raise CoordinateMapperError(f"Tray slot {move.slot_idx} has no recognized piece model.")

        piece: Piece = slot_res.piece

        # 2. Destination Legality Validation
        if not (0 <= move.row <= 8 - piece.height and 0 <= move.col <= 8 - piece.width):
            raise CoordinateMapperError(
                f"Target position ({move.row}, {move.col}) out of grid bounds for "
                f"{piece.name} ({piece.height}x{piece.width})."
            )

        placement_mask = piece.placement_masks[move.row][move.col]
        if not can_place(frame_result.board.bitboard, placement_mask):
            raise CoordinateMapperError(
                f"Target position ({move.row}, {move.col}) is occupied / illegal on active board."
            )

        # 3. Compute Start Coordinate (Center of tray piece bounding box or centroid)
        bx, by, bw, bh = slot_res.bbox_in_screen
        if self.use_cell_centroid and piece.cells and piece.width > 0 and piece.height > 0:
            c_mean = sum(c for _, c in piece.cells) / float(len(piece.cells))
            r_mean = sum(r for r, _ in piece.cells) / float(len(piece.cells))
            start_x = int(round(bx + (c_mean + 0.5) / piece.width * bw))
            start_y = int(round(by + (r_mean + 0.5) / piece.height * bh))
        else:
            start_x = int(round(bx + bw / 2.0))
            start_y = int(round(by + bh / 2.0))

        # 4. Compute Destination Coordinate on Board
        # Piece bounding box spans rows [move.row, move.row + piece.height - 1]
        # and cols [move.col, move.col + piece.width - 1].
        # The visual piece geometric center on the 8x8 grid:
        center_grid_x = board_x + (move.col + piece.width / 2.0) * cell_size
        center_grid_y = board_y + (move.row + piece.height / 2.0) * cell_size

        # Compensate for finger touch offset:
        # Since the piece hovers `vertical_offset` pixels ABOVE the finger,
        # the finger must end `vertical_offset` pixels BELOW the piece center.
        vert_offset_px = self.vertical_offset_ratio * cell_size
        end_x = int(round(center_grid_x))
        end_y = int(round(center_grid_y + vert_offset_px))

        # 5. Screen Bounds Safety Check (Without rigid bottom clamping)
        duration = drag_duration_ms or self.default_drag_duration_ms

        if img_w > 0 and img_h > 0:
            if not (0 <= start_x < img_w and 0 <= start_y < img_h):
                raise CoordinateMapperError(
                    f"Start coordinate ({start_x}, {start_y}) outside screen boundaries ({img_w}x{img_h})"
                )
            # Safe physical margin to prevent bezel/system edge gesture interference
            margin = 5
            end_x = max(margin, min(img_w - margin, end_x))
            end_y = max(margin, min(img_h - margin, end_y))

        trajectory = DragTrajectory(
            slot_idx=move.slot_idx,
            piece_name=move.piece_name,
            target_row=move.row,
            target_col=move.col,
            start_x=start_x,
            start_y=start_y,
            end_x=end_x,
            end_y=end_y,
            duration_ms=duration,
            vertical_offset_px=vert_offset_px,
        )

        logger.info(
            "Mapped Move: Slot %d (%s) -> (%d, %d) | Drag: (%d, %d) -> (%d, %d) [%d ms]",
            trajectory.slot_idx,
            trajectory.piece_name,
            trajectory.target_row,
            trajectory.target_col,
            trajectory.start_x,
            trajectory.start_y,
            trajectory.end_x,
            trajectory.end_y,
            trajectory.duration_ms,
        )

        return trajectory
