"""Complete Computer Vision Pipeline for BLOCK BLAST AI ULTIMATE.

Integrates BoardDetector, CellClassifier, PieceDetector, PieceClassifier, and DebugVisualizer
into a single, high-performance, CPU-first frame recognition engine.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ai_engine.decision import AIDecisionEngine, DecisionResult
from computer_vision.board_detector import BoardDetector
from computer_vision.capture import ScreenCapture
from computer_vision.cell_classifier import CellClassifier
from computer_vision.config import DEFAULT_VISION_CONFIG, VisionConfig
from computer_vision.debug_visualizer import DebugVisualizer
from computer_vision.diagnostic_saver import PerceptionDiagnosticSaver
from computer_vision.models import (
    FrameRecognitionResult,
    PieceRecognitionResult,
    RecognitionStatus,
)
from computer_vision.piece_classifier import PieceClassifier
from computer_vision.piece_detector import PieceDetector
from game_engine.pieces import CANONICAL_41_CATALOG, PieceCatalog


class BlockBlastRecognizer:
    """Unified Computer Vision Recognizer for Block Blast."""

    def __init__(
        self,
        config: Optional[VisionConfig] = None,
        catalog: Optional[PieceCatalog] = None,
        ai_engine: Optional[AIDecisionEngine] = None,
    ) -> None:
        self.config = config or DEFAULT_VISION_CONFIG
        self.catalog = catalog or CANONICAL_41_CATALOG
        self.capture = ScreenCapture(self.config)
        self.board_detector = BoardDetector(self.config)
        self.cell_classifier = CellClassifier(self.config, self.board_detector)
        self.piece_detector = PieceDetector(self.config)
        self.piece_classifier = PieceClassifier(self.config, self.catalog)
        self.visualizer = DebugVisualizer(self.config)
        self.diagnostic_saver = PerceptionDiagnosticSaver(self.config)
        self.ai_engine = ai_engine or AIDecisionEngine()

    def recognize_frame(self, image: np.ndarray) -> FrameRecognitionResult:
        """Execute the full recognition pipeline on an input BGR image.

        Args:
            image: Full-frame BGR numpy array from device or file.

        Returns:
            FrameRecognitionResult with board state, piece mappings, and confidence metrics.
        """
        t0 = time.perf_counter()
        h_img, w_img = image.shape[:2]

        # Guard against corrupted, empty, or sub-resolution images
        if h_img < 600 or w_img < 350:
            duration = (time.perf_counter() - t0) * 1000.0
            return FrameRecognitionResult(
                status=RecognitionStatus.FAILED,
                board=None,
                tray_pieces=(),
                overall_confidence=0.0,
                processing_time_ms=duration,
                error_message=f"Image dimensions ({w_img}x{h_img}) too small for valid mobile screen.",
                image_shape=(h_img, w_img),
            )

        # ---------------------------------------------------------------------
        # 1. Board Detection
        # ---------------------------------------------------------------------
        try:
            board_bbox, cell_size, board_conf = self.board_detector.detect_board(image)
        except Exception as exc:
            duration = (time.perf_counter() - t0) * 1000.0
            return FrameRecognitionResult(
                status=RecognitionStatus.FAILED,
                board=None,
                tray_pieces=(),
                overall_confidence=0.0,
                processing_time_ms=duration,
                error_message=f"Board detection raised error: {exc}",
                image_shape=(h_img, w_img),
            )

        if board_conf < 0.50:
            duration = (time.perf_counter() - t0) * 1000.0
            return FrameRecognitionResult(
                status=RecognitionStatus.FAILED,
                board=None,
                tray_pieces=(),
                overall_confidence=board_conf,
                processing_time_ms=duration,
                error_message=f"Board detection confidence too low ({board_conf:.2f} < 0.50)",
                image_shape=(h_img, w_img),
            )

        # ---------------------------------------------------------------------
        # 2. 8x8 Cell Classification
        # ---------------------------------------------------------------------
        board_res = self.cell_classifier.classify_cells(
            image=image,
            board_bbox=board_bbox,
            board_confidence=board_conf,
        )

        # ---------------------------------------------------------------------
        # 3. Tray Piece Segmentation & Classification
        # ---------------------------------------------------------------------
        piece_blobs = self.piece_detector.detect_tray_pieces(image)
        tray_results = self.piece_classifier.classify_all_blobs(piece_blobs)

        # ---------------------------------------------------------------------
        # 4. Overall Confidence & Status Determination
        # ---------------------------------------------------------------------
        # Compute piece confidence average
        piece_confs = [p.confidence for p in tray_results]
        avg_piece_conf = sum(piece_confs) / float(len(piece_confs)) if piece_confs else 0.0

        # Weighted overall confidence
        overall_conf = (
            0.35 * board_res.board_confidence +
            0.35 * board_res.cell_confidence +
            0.30 * avg_piece_conf
        )

        # Status rules:
        # Prompt rule: "Nếu không chắc chắn: Không chuyển dữ liệu không đáng tin cậy sang AI Decision Engine."
        has_unknown_piece = any(
            not p.is_empty and p.matched_name == "UNKNOWN" for p in tray_results
        )
        has_uncertain_cell = any(
            c.state.name == "UNCERTAIN" for row in board_res.cells for c in row
        )

        if has_unknown_piece and has_uncertain_cell:
            status = RecognitionStatus.LOW_CONFIDENCE
            error_msg = "Contains unmapped tray pieces (UNKNOWN) and uncertain board cells."
        elif has_unknown_piece:
            status = RecognitionStatus.LOW_CONFIDENCE
            error_msg = "Contains unmapped tray pieces (UNKNOWN)."
        elif has_uncertain_cell:
            status = RecognitionStatus.LOW_CONFIDENCE
            error_msg = "Perception uncertain: Board contains UNCERTAIN cells."
        elif overall_conf >= self.config.min_success_overall_confidence:
            status = RecognitionStatus.SUCCESS
            error_msg = None
        else:
            status = RecognitionStatus.LOW_CONFIDENCE
            error_msg = f"Overall confidence ({overall_conf:.2f}) below threshold ({self.config.min_success_overall_confidence:.2f})."

        duration = (time.perf_counter() - t0) * 1000.0

        return FrameRecognitionResult(
            status=status,
            board=board_res,
            tray_pieces=tuple(tray_results),
            overall_confidence=overall_conf,
            processing_time_ms=duration,
            error_message=error_msg,
            image_shape=(h_img, w_img),
        )

    def recognize_file(self, image_path: str) -> FrameRecognitionResult:
        """Load an image file and execute the recognition pipeline."""
        image = self.capture.load_from_file(image_path)
        return self.recognize_frame(image)

    def recognize_adb(self, device_id: Optional[str] = None) -> FrameRecognitionResult:
        """Capture frame from Android via ADB and execute the recognition pipeline."""
        image = self.capture.capture_from_adb(device_id=device_id)
        return self.recognize_frame(image)

    def save_diagnostic_failure(
        self,
        image: np.ndarray,
        recognition_result: FrameRecognitionResult,
        pause_reason: Optional[str] = None,
    ) -> Optional[Any]:
        """Save forensic diagnostic package for this recognition attempt."""
        return self.diagnostic_saver.save_diagnostic_failure(
            frame=image,
            recognition_result=recognition_result,
            pause_reason=pause_reason,
            piece_detector=self.piece_detector,
        )

    def plan_best_moves(
        self,
        frame_result: FrameRecognitionResult,
        strategy: str = "beam_search",
        **kwargs,
    ) -> DecisionResult:
        """Feed recognized frame directly into AI Decision Engine to plan the optimal moves.

        Args:
            frame_result: Output from recognize_frame()
            strategy: 'beam_search', 'greedy', or 'random'
            **kwargs: Extra parameters (beam_width, etc.)

        Raises:
            ValueError: If frame is not usable for AI (e.g. LOW_CONFIDENCE, UNKNOWN piece).
        """
        state = frame_result.to_game_state()
        return self.ai_engine.decide(state, strategy=strategy, **kwargs)
