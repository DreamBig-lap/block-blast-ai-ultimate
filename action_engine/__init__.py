"""Action Engine for BLOCK BLAST AI ULTIMATE.

Provides ADB touch actuation, dynamic coordinate mapping, post-move verification,
auto-play state management, and real-time dashboard monitoring.
"""

from action_engine.action_verifier import ActionVerifier, VerificationResult, VerificationStatus
from action_engine.auto_player import AutoPlayConfig, AutoPlayer, PlayMode
from action_engine.coordinate_mapper import CoordinateMapper, DragTrajectory
from action_engine.touch_controller import TouchController

__all__ = [
    "ActionVerifier",
    "AutoPlayConfig",
    "AutoPlayer",
    "CoordinateMapper",
    "DragTrajectory",
    "PlayMode",
    "TouchController",
    "VerificationResult",
    "VerificationStatus",
]
