"""BLOCK BLAST AI ULTIMATE — Computer Vision Package.

A modular, lightweight, CPU-first vision system for Block Blast board detection,
8x8 cell classification, tray piece recognition, and debug visualization.
"""

from computer_vision.board_detector import BoardDetector
from computer_vision.capture import ScreenCapture, ScreenCaptureError
from computer_vision.cell_classifier import CellClassifier
from computer_vision.config import DEFAULT_VISION_CONFIG, VisionConfig
from computer_vision.debug_visualizer import DebugVisualizer
from computer_vision.models import (
    BoardRecognitionResult,
    CellInfo,
    CellState,
    FrameRecognitionResult,
    PieceRecognitionResult,
    RecognitionStatus,
)
from computer_vision.piece_classifier import PieceClassifier
from computer_vision.piece_detector import PieceDetector, RawPieceBlob
from computer_vision.recognizer import BlockBlastRecognizer

__all__ = [
    # Config
    "VisionConfig",
    "DEFAULT_VISION_CONFIG",
    # Models
    "CellState",
    "RecognitionStatus",
    "CellInfo",
    "BoardRecognitionResult",
    "PieceRecognitionResult",
    "FrameRecognitionResult",
    # Pipeline components
    "ScreenCapture",
    "ScreenCaptureError",
    "BoardDetector",
    "CellClassifier",
    "PieceDetector",
    "RawPieceBlob",
    "PieceClassifier",
    "DebugVisualizer",
    "BlockBlastRecognizer",
]
