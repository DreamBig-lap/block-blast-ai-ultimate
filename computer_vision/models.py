"""Data Models and Results for BLOCK BLAST AI ULTIMATE Computer Vision.

Defines structured containers for cell classification, board geometry,
tray piece detection, and comprehensive frame recognition outcomes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import enum
import json
from typing import Any, Optional

from game_engine.board import coords_to_bitboard, popcount
from game_engine.pieces import CANONICAL_41_CATALOG, Piece
from game_engine.rules import DEFAULT_RULES, GameRules
from game_engine.state import GameState


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CellState(enum.IntEnum):
    """Occupancy state of a single grid cell on the 8x8 board."""
    EMPTY = 0
    OCCUPIED = 1
    UNCERTAIN = 2


class RecognitionStatus(enum.Enum):
    """Overall status of a frame recognition pass."""
    SUCCESS = "SUCCESS"                # Board and all 3 tray slots recognized with high confidence
    LOW_CONFIDENCE = "LOW_CONFIDENCE"  # Board or pieces recognized with sub-threshold confidence
    FAILED = "FAILED"                  # Critical failure: board not found or corrupted screenshot


# ---------------------------------------------------------------------------
# Cell Information
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CellInfo:
    """Detailed visual metrics and classification for an individual 8x8 cell."""

    row: int
    col: int
    state: CellState
    confidence: float
    center_x: int
    center_y: int
    width: int
    height: int
    mean_gray: float
    mean_sat: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "col": self.col,
            "state": self.state.name,
            "confidence": round(self.confidence, 4),
            "center": [self.center_x, self.center_y],
            "size": [self.width, self.height],
            "mean_gray": round(self.mean_gray, 2),
            "mean_sat": round(self.mean_sat, 2),
        }


# ---------------------------------------------------------------------------
# Board Recognition Result
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BoardRecognitionResult:
    """Recognized geometry and 8x8 occupancy state of the game board."""

    board_bbox: tuple[int, int, int, int]  # (x, y, w, h) in screen pixels
    grid_rows: int
    grid_cols: int
    cell_size: float
    cells: tuple[tuple[CellInfo, ...], ...]
    occupancy_matrix: tuple[tuple[int, ...], ...]
    bitboard: int
    board_confidence: float
    grid_confidence: float
    cell_confidence: float

    @property
    def occupied_count(self) -> int:
        return popcount(self.bitboard)

    @property
    def empty_count(self) -> int:
        return 64 - self.occupied_count

    @property
    def has_uncertain_cells(self) -> bool:
        return any(c.state == CellState.UNCERTAIN for row in self.cells for c in row)

    def to_dict(self) -> dict[str, Any]:
        return {
            "board_bbox": list(self.board_bbox),
            "grid_dimensions": [self.grid_rows, self.grid_cols],
            "cell_size": round(self.cell_size, 2),
            "bitboard_hex": f"0x{self.bitboard:016X}",
            "occupied_count": self.occupied_count,
            "empty_count": self.empty_count,
            "confidences": {
                "board": round(self.board_confidence, 4),
                "grid": round(self.grid_confidence, 4),
                "cells": round(self.cell_confidence, 4),
            },
            "occupancy_matrix": [list(row) for row in self.occupancy_matrix],
        }


# ---------------------------------------------------------------------------
# Piece Recognition Result
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PieceRecognitionResult:
    """Recognized shape and catalog mapping for an individual tray slot."""

    slot_idx: int
    is_empty: bool
    matched_name: Optional[str]
    piece: Optional[Piece]
    cells: tuple[tuple[int, int], ...]
    height: int
    width: int
    cell_count: int
    bbox_in_screen: tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    status_message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_idx": self.slot_idx,
            "is_empty": self.is_empty,
            "matched_name": self.matched_name,
            "piece_id": self.piece.piece_id if self.piece else None,
            "dimensions": [self.height, self.width],
            "cell_count": self.cell_count,
            "cells": [list(c) for c in self.cells],
            "bbox_in_screen": list(self.bbox_in_screen),
            "confidence": round(self.confidence, 4),
            "status_message": self.status_message,
        }


# ---------------------------------------------------------------------------
# Full Frame Recognition Result
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FrameRecognitionResult:
    """Comprehensive output of the complete vision recognition pipeline."""

    status: RecognitionStatus
    board: Optional[BoardRecognitionResult]
    tray_pieces: tuple[PieceRecognitionResult, ...]
    overall_confidence: float
    processing_time_ms: float
    error_message: Optional[str] = None
    image_shape: tuple[int, int] = (0, 0)  # (height, width)

    def is_usable_for_ai(self) -> bool:
        """Return True if results are sufficiently reliable to feed to AI Decision Engine."""
        if self.status != RecognitionStatus.SUCCESS:
            return False
        if self.board is None:
            return False
        # Every non-empty tray slot must have a valid piece from the catalog
        for pr in self.tray_pieces:
            if not pr.is_empty and (pr.piece is None or pr.matched_name == "UNKNOWN"):
                return False
        return True

    def to_game_state(self, rules: Optional[GameRules] = None) -> GameState:
        """Convert the recognized board and tray into a verified GameState object.

        Raises:
            ValueError: If recognition status is not SUCCESS or pieces are unmapped.
        """
        if not self.is_usable_for_ai():
            raise ValueError(
                f"Cannot build GameState: frame recognition status is {self.status.value} "
                f"(error: {self.error_message})"
            )

        active_rules = rules or DEFAULT_RULES
        assert self.board is not None

        tray_pieces: list[Optional[Piece]] = []
        for pr in self.tray_pieces:
            if pr.is_empty:
                tray_pieces.append(None)
            else:
                tray_pieces.append(pr.piece)

        # Pad to active_rules.tray_capacity (usually 3)
        while len(tray_pieces) < active_rules.tray_capacity:
            tray_pieces.append(None)

        return GameState.create_initial(
            initial_tray=tray_pieces,
            rules=active_rules,
        ).clone_with(board=self.board.bitboard)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "overall_confidence": round(self.overall_confidence, 4),
            "processing_time_ms": round(self.processing_time_ms, 2),
            "image_shape": list(self.image_shape),
            "error_message": self.error_message,
            "board": self.board.to_dict() if self.board else None,
            "tray": [p.to_dict() for p in self.tray_pieces],
        }

    def to_json(self, indent: int = 2) -> str:
        def _json_default(obj: Any) -> Any:
            if hasattr(obj, "item"):
                return obj.item()
            if isinstance(obj, (int, float, str, bool)):
                return obj
            return str(obj)

        return json.dumps(self.to_dict(), indent=indent, default=_json_default)
