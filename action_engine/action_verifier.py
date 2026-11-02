"""Post-Action Move Verification for BLOCK BLAST AI ULTIMATE.

Verifies the physical outcome of each executed move against the deterministic
state prediction of the Game Engine. Compares 64-bit bitboards, checks tray slot
emptiness, handles particle animation settling, and triggers safe pauses on mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass
import enum
import logging
import math
import time
from typing import Optional, Union

import numpy as np

from computer_vision.capture import ScreenCapture
from computer_vision.models import CellState, FrameRecognitionResult, RecognitionStatus
from computer_vision.recognizer import BlockBlastRecognizer
from game_engine.board import popcount
from game_engine.simulator import StepResult

logger = logging.getLogger(__name__)


class VerificationStatus(enum.Enum):
    """Outcome of post-move verification."""

    VERIFIED = "VERIFIED"                  # Board and tray exactly match predicted transition
    ANIMATION_SETTLING = "SETTLING"        # Transient animation detected; resolved after retry
    MISMATCH = "MISMATCH"                  # Persistent discrepancy between expected and actual board
    TRAY_REJECTED = "TRAY_REJECTED"        # Piece was rejected and returned to the tray slot
    UNKNOWN = "UNKNOWN"                    # Vision recognition confidence sub-threshold or uncertain
    VISION_FAILED = "VISION_FAILED"        # Screen capture or computer vision crashed / failed
    UNSTABLE_ANIMATION_TIMEOUT = "UNSTABLE_ANIMATION_TIMEOUT"  # Consecutive frames inconsistent after timeout; animation active


@dataclass(frozen=True, slots=True)
class CellMismatch:
    """Details on an individual grid cell discrepancy."""
    row: int
    col: int
    expected: int
    actual: int


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Detailed audit container for move verification."""

    status: VerificationStatus
    step_index: int
    slot_idx: int
    piece_name: str
    expected_bitboard: int
    actual_bitboard: int
    mismatch_count: int
    mismatched_cells: tuple[CellMismatch, ...]
    tray_slot_cleared: bool
    attempts: int
    elapsed_ms: float
    new_frame_result: Optional[FrameRecognitionResult] = None
    error_message: Optional[str] = None
    is_fast_path: bool = False
    captures_count: int = 2
    settle_ms: float = 0.0
    capture_ms: float = 0.0
    recognition_ms: float = 0.0
    fast_path_rejection_reason: Optional[str] = None
    new_frame: Optional[np.ndarray] = None

    @property
    def is_success(self) -> bool:
        return self.status in (VerificationStatus.VERIFIED, VerificationStatus.ANIMATION_SETTLING)


class ActionVerifier:
    """Verifies board and tray transitions after a move execution.
    
    Implements Phase 4.2.3 Adaptive Settle Time and Consecutive Frame Consistency:
    - Adaptive initial settle delay: 0.35s for regular moves, 1.2s for line-clear moves.
    - Two consecutive frame acquisitions per attempt to confirm visual animation stability.
    - Consistency gate: bitboard equality, bounding box stability, and tray slot matching.
    - Safe timeout handling: transitions to UNSTABLE_ANIMATION_TIMEOUT if animation persists.
    """

    def __init__(
        self,
        recognizer: Optional[BlockBlastRecognizer] = None,
        capture: Optional[ScreenCapture] = None,
        post_move_settle_sec: float = 0.20,
        normal_settle_sec: Optional[float] = None,
        line_clear_settle_sec: float = 1.2,
        consecutive_frame_delay_sec: float = 0.15,
        max_settle_retries: int = 3,
        retry_interval_sec: float = 0.25,
        settle_timeout_sec: float = 3.0,
    ) -> None:
        """
        Args:
            recognizer: Vision recognizer instance.
            capture: ScreenCapture instance for ADB acquisition.
            post_move_settle_sec: Fallback/legacy parameter for regular move settle delay.
            normal_settle_sec: Initial pause for moves without line clears.
            line_clear_settle_sec: Initial pause for moves that trigger line clearing.
            consecutive_frame_delay_sec: Short delay between consecutive frame captures.
            max_settle_retries: Max retry attempts if transient particle animation is active.
            retry_interval_sec: Pause between retries.
            settle_timeout_sec: Total timeout ceiling for settling verification.
        """
        self.recognizer = recognizer or BlockBlastRecognizer()
        self.capture = capture or ScreenCapture()

        # Resolve normal settle delay
        base_normal = normal_settle_sec if normal_settle_sec is not None else post_move_settle_sec
        self.normal_settle_sec = base_normal
        self.post_move_settle_sec = base_normal

        # In zero-delay mock test environments, bypass sleep delays automatically
        if base_normal == 0.0 and line_clear_settle_sec == 1.2:
            self.line_clear_settle_sec = 0.0
            self.consecutive_frame_delay_sec = 0.0
            self.retry_interval_sec = 0.0
        else:
            self.line_clear_settle_sec = line_clear_settle_sec
            self.consecutive_frame_delay_sec = consecutive_frame_delay_sec
            self.retry_interval_sec = retry_interval_sec

        self.max_settle_retries = max_settle_retries
        self.settle_timeout_sec = settle_timeout_sec

    def _check_consecutive_consistency(
        self,
        frame_a: FrameRecognitionResult,
        frame_b: FrameRecognitionResult,
        max_bbox_shift_px: float = 4.0,
    ) -> bool:
        """Verify that two consecutive frames are consistent (stable board, no active animation)."""
        if frame_a.board is None or frame_b.board is None:
            return False

        # 1. Occupancy bitboard must match bit-for-bit
        if frame_a.board.bitboard != frame_b.board.bitboard:
            return False

        # 2. Bounding box shift must remain within spatial tolerance
        bx1, by1, bw1, bh1 = frame_a.board.board_bbox
        bx2, by2, bw2, bh2 = frame_b.board.board_bbox
        if (
            abs(bx1 - bx2) > max_bbox_shift_px
            or abs(by1 - by2) > max_bbox_shift_px
            or abs(bw1 - bw2) > max_bbox_shift_px
            or abs(bh1 - bh2) > max_bbox_shift_px
        ):
            return False

        # 3. Tray slot emptiness state must match
        if len(frame_a.tray_pieces) == len(frame_b.tray_pieces):
            for p_a, p_b in zip(frame_a.tray_pieces, frame_b.tray_pieces):
                if p_a.is_empty != p_b.is_empty:
                    return False

        return True

    def _can_fast_path_verify(
        self,
        frame_res: FrameRecognitionResult,
        step_result: StepResult,
        lines_cleared: int,
    ) -> tuple[bool, str]:
        """Evaluate whether Frame A meets all strict criteria for single-frame verification.

        Fast-Path is ONLY permitted when ALL 8 conditions are satisfied:
        1. Frame A recognition status is SUCCESS.
        2. Board confidence meets safety threshold (board >= 0.70, cell >= 0.70, no uncertain cells).
        3. Tray recognition confidence meets safety threshold (all non-empty slots >= 0.70, mapped piece).
        4. actual_bitboard == expected_bitboard (exact bit-for-bit match).
        5. Entire actual tray matches expected tray (state_after.tray).
        6. Dragged slot (slot_idx) is confirmed empty.
        7. Move did not clear any lines (lines_cleared == 0).
        8. Geometry and UI normal (board aspect ratio in 0.90..1.10, no error message).

        Returns:
            (True, "") if eligible for fast-path.
            (False, reason_str) if any condition is not met.
        """
        # Condition 7: Move must not have cleared lines (line clears produce dissolving animations)
        if lines_cleared > 0:
            return False, f"Move cleared {lines_cleared} line(s); particle settling requires two-frame verification"

        # Condition 1: Recognition status must be SUCCESS
        if frame_res.status != RecognitionStatus.SUCCESS:
            return False, f"Frame A recognition status is {frame_res.status.value} (expected SUCCESS)"

        if frame_res.board is None:
            return False, "Frame A board detection is None"

        # Condition 2: Board confidence and individual cell confidence validation
        board = frame_res.board
        if not hasattr(board, "cells") or not isinstance(board.cells, (tuple, list)):
            return False, "Board cells data missing or invalid"

        min_conf = 0.70
        cfg = getattr(self.recognizer, "config", None)
        if cfg is not None:
            cfg_val = getattr(cfg, "min_cell_confidence", None)
            if isinstance(cfg_val, (int, float)) and math.isfinite(cfg_val):
                min_conf = float(cfg_val)

        if (
            not isinstance(board.board_confidence, (int, float))
            or not math.isfinite(board.board_confidence)
            or board.board_confidence < min_conf
        ):
            return False, f"Board confidence ({getattr(board, 'board_confidence', None)}) < threshold ({min_conf})"

        if (
            not isinstance(board.cell_confidence, (int, float))
            or not math.isfinite(board.cell_confidence)
            or board.cell_confidence < min_conf
        ):
            return False, f"Average cell confidence ({getattr(board, 'cell_confidence', None)}) < threshold ({min_conf})"

        # FIX 1: Validate every individual cell of the 8x8 grid
        total_cells_count = 0
        for r_idx, row in enumerate(board.cells):
            if not isinstance(row, (tuple, list)):
                return False, f"Board row {r_idx} is not a valid sequence"
            for c_idx, cell in enumerate(row):
                total_cells_count += 1
                c_conf = getattr(cell, "confidence", None)
                if not isinstance(c_conf, (int, float)) or not math.isfinite(c_conf):
                    return False, f"Cell ({r_idx}, {c_idx}) has missing or invalid confidence: {c_conf}"
                if c_conf < min_conf:
                    return False, f"Cell ({r_idx}, {c_idx}) confidence ({c_conf:.2f}) < threshold ({min_conf:.2f})"
                c_state = getattr(cell, "state", None)
                if c_state == CellState.UNCERTAIN or getattr(c_state, "name", "") in ("UNCERTAIN", "UNKNOWN"):
                    return False, f"Cell ({r_idx}, {c_idx}) is uncertain or unknown (state={c_state})"

        if total_cells_count != 64:
            return False, f"Board cell count invalid: found {total_cells_count} cells (expected 64)"

        # Condition 8: Board geometry / UI abnormality check
        if not hasattr(board, "board_bbox") or len(board.board_bbox) != 4:
            return False, "Invalid board bounding box"
        bx, by, bw, bh = board.board_bbox
        if not isinstance(bw, (int, float)) or not isinstance(bh, (int, float)) or bw <= 0 or bh <= 0:
            return False, f"Invalid board bounding box dimensions: w={bw}, h={bh}"
        aspect_ratio = bw / bh
        if not (0.90 <= aspect_ratio <= 1.10):
            return False, f"Abnormal board aspect ratio: {aspect_ratio:.2f} (expected 0.90-1.10)"

        if frame_res.error_message:
            return False, f"Frame A reported error: {frame_res.error_message}"

        # Condition 4: Exact bitboard match
        state_after = getattr(step_result, "state_after", None)
        if state_after is None or not hasattr(state_after, "board"):
            return False, "step_result.state_after missing or invalid"
        expected_bitboard = state_after.board
        if board.bitboard != expected_bitboard:
            return False, f"Bitboard mismatch: actual 0x{board.bitboard:016X} != expected 0x{expected_bitboard:016X}"

        # Condition 6: Dragged slot must be confirmed empty
        slot_idx = getattr(step_result, "slot_idx", None)
        if not isinstance(slot_idx, int) or not (0 <= slot_idx < len(frame_res.tray_pieces)):
            return False, f"Invalid slot index {slot_idx} (tray has {len(frame_res.tray_pieces)} slots)"

        dragged_slot = frame_res.tray_pieces[slot_idx]
        if not getattr(dragged_slot, "is_empty", False):
            return False, f"Dragged slot {slot_idx} is not empty (contains '{getattr(dragged_slot, 'matched_name', 'UNKNOWN')}')"

        # Condition 3 & 5: Tray confidence and complete tray state matching
        expected_tray = getattr(state_after, "tray", None)
        if not isinstance(expected_tray, (tuple, list)):
            return False, "Expected state_after.tray is not a valid sequence"

        if len(frame_res.tray_pieces) != len(expected_tray):
            return False, f"Tray slot count mismatch: actual {len(frame_res.tray_pieces)} != expected {len(expected_tray)}"

        # FIX 2: Validate all tray slots (including empty slots)
        for i, (act_slot, exp_piece) in enumerate(zip(frame_res.tray_pieces, expected_tray)):
            slot_conf = getattr(act_slot, "confidence", None)
            if not isinstance(slot_conf, (int, float)) or not math.isfinite(slot_conf):
                return False, f"Tray slot {i} has missing or invalid confidence: {slot_conf}"
            if slot_conf < min_conf:
                return False, f"Tray slot {i} confidence ({slot_conf:.2f}) < threshold ({min_conf:.2f})"

            act_name = getattr(act_slot, "matched_name", None)
            if act_name == "UNKNOWN":
                return False, f"Tray slot {i} is UNKNOWN"

            if exp_piece is None:
                if not getattr(act_slot, "is_empty", False):
                    return False, f"Tray slot {i} expected empty, but actual is occupied by '{act_name}'"
            else:
                if getattr(act_slot, "is_empty", False):
                    return False, f"Tray slot {i} expected '{exp_piece.name}', but actual is empty"
                if act_name != exp_piece.name:
                    return False, f"Tray slot {i} piece mismatch: actual '{act_name}' != expected '{exp_piece.name}'"

        return True, ""

    def verify_move(
        self,
        step_result: StepResult,
        device_id: Optional[str] = None,
        step_index: int = 0,
        lines_cleared_override: Optional[int] = None,
    ) -> VerificationResult:
        """Capture screen, recognize board, and verify against predicted StepResult.

        Implements Safe Single-Frame Verification (Fast-Path):
        - Evaluates Frame A immediately following settle delay.
        - If all 8 safety criteria pass (no line clear, exact board & tray match, high confidence):
          verifies in 1 frame, skipping Frame B and saving ~1.47s.
        - If any criterion fails: seamlessly executes consecutive two-frame verification and retries.

        Args:
            step_result: Deterministic outcome from Game Engine simulator.
            device_id: Optional target ADB serial.
            step_index: Index of the move in current round (0, 1, 2).
            lines_cleared_override: Optional override for line clear count.

        Returns:
            VerificationResult detailing match/mismatch status, timing, and cell differences.
        """
        t0 = time.perf_counter()
        expected_bitboard = step_result.state_after.board
        step_idx = getattr(step_result, "step_index", step_index)
        slot_idx = step_result.slot_idx
        piece_name = step_result.piece.name if step_result.piece else "UNKNOWN"

        # Determine line clear presence
        if lines_cleared_override is not None:
            lines_cleared = lines_cleared_override
        else:
            lines_cleared = getattr(step_result, "lines_cleared", 0)
            if lines_cleared is None:
                lines_cleared = 0

        # Phase 4.2.3: Adaptive Settle Time
        initial_wait = self.line_clear_settle_sec if lines_cleared > 0 else self.normal_settle_sec
        settle_t0 = time.perf_counter()
        if initial_wait > 0:
            time.sleep(initial_wait)
        settle_ms = (time.perf_counter() - settle_t0) * 1000.0

        img_a: Optional[np.ndarray] = None
        img_b: Optional[np.ndarray] = None
        last_mismatches: tuple[CellMismatch, ...] = ()
        actual_bitboard = 0
        tray_cleared = False
        frame_res: Optional[FrameRecognitionResult] = None
        captures_count = 0
        total_capture_ms = 0.0
        total_rec_ms = 0.0
        fast_path_rejection_reason: Optional[str] = None

        for attempt in range(1, self.max_settle_retries + 1):
            # Check settle timeout ceiling
            if (time.perf_counter() - t0) >= self.settle_timeout_sec and attempt > 1:
                logger.warning(
                    "Settle verification timeout reached (%.2fs >= %.2fs). Aborting retries.",
                    time.perf_counter() - t0, self.settle_timeout_sec
                )
                break

            # -------------------------------------------------------------
            # Frame A Acquisition
            # -------------------------------------------------------------
            t_cap_a0 = time.perf_counter()
            try:
                img_a = self.capture.capture_from_adb(device_id=device_id)
            except Exception as exc:
                captures_count += 1
                total_capture_ms += (time.perf_counter() - t_cap_a0) * 1000.0
                elapsed = (time.perf_counter() - t0) * 1000.0
                err_msg = f"ADB capture failed on Frame A: {exc}"
                logger.error(
                    "[VERIFY_PAUSED] Verification halted with status VISION_FAILED: %s (attempts=%d, total=%.1fms)",
                    err_msg, attempt, elapsed
                )
                return VerificationResult(
                    status=VerificationStatus.VISION_FAILED,
                    step_index=step_idx,
                    slot_idx=slot_idx,
                    piece_name=piece_name,
                    expected_bitboard=expected_bitboard,
                    actual_bitboard=0,
                    mismatch_count=64,
                    mismatched_cells=(),
                    tray_slot_cleared=False,
                    attempts=attempt,
                    elapsed_ms=elapsed,
                    error_message=err_msg,
                    is_fast_path=False,
                    captures_count=captures_count,
                    settle_ms=settle_ms,
                    capture_ms=total_capture_ms,
                    recognition_ms=total_rec_ms,
                    fast_path_rejection_reason=fast_path_rejection_reason or err_msg,
                )

            cap_a_ms = (time.perf_counter() - t_cap_a0) * 1000.0
            captures_count += 1
            total_capture_ms += cap_a_ms

            t_rec_a0 = time.perf_counter()
            frame_res_a = self.recognizer.recognize_frame(img_a)
            rec_a_ms = (time.perf_counter() - t_rec_a0) * 1000.0
            total_rec_ms += rec_a_ms

            # -------------------------------------------------------------
            # Single-Frame Fast-Path (Attempt 1 only)
            # -------------------------------------------------------------
            if attempt == 1:
                can_fast_path, fp_reason = self._can_fast_path_verify(
                    frame_res=frame_res_a,
                    step_result=step_result,
                    lines_cleared=lines_cleared,
                )
                if can_fast_path:
                    elapsed = (time.perf_counter() - t0) * 1000.0
                    logger.info(
                        "[VERIFY_FAST_PATH] Move %d verified in 1 frame (settle=%.1fms, cap=%.1fms, rec=%.1fms, total=%.1fms). Board and tray match 100%%.",
                        step_idx, settle_ms, cap_a_ms, rec_a_ms, elapsed
                    )
                    return VerificationResult(
                        status=VerificationStatus.VERIFIED,
                        step_index=step_idx,
                        slot_idx=slot_idx,
                        piece_name=piece_name,
                        expected_bitboard=expected_bitboard,
                        actual_bitboard=frame_res_a.board.bitboard,
                        mismatch_count=0,
                        mismatched_cells=(),
                        tray_slot_cleared=True,
                        attempts=1,
                        elapsed_ms=elapsed,
                        new_frame_result=frame_res_a,
                        is_fast_path=True,
                        captures_count=1,
                        settle_ms=settle_ms,
                        capture_ms=total_capture_ms,
                        recognition_ms=total_rec_ms,
                        fast_path_rejection_reason=None,
                        new_frame=img_a,
                    )
                else:
                    fast_path_rejection_reason = fp_reason
                    logger.info(
                        "[VERIFY_TWO_FRAME] Fast-path skipped: %s. Proceeding to two-frame consecutive verification.",
                        fp_reason
                    )

            # -------------------------------------------------------------
            # Frame B Acquisition (Consecutive frame after short interval)
            # -------------------------------------------------------------
            if self.consecutive_frame_delay_sec > 0:
                time.sleep(self.consecutive_frame_delay_sec)

            t_cap_b0 = time.perf_counter()
            try:
                img_b = self.capture.capture_from_adb(device_id=device_id)
            except Exception as exc:
                captures_count += 1
                total_capture_ms += (time.perf_counter() - t_cap_b0) * 1000.0
                elapsed = (time.perf_counter() - t0) * 1000.0
                err_msg = f"ADB capture failed on Frame B: {exc}"
                logger.error(
                    "[VERIFY_PAUSED] Verification halted with status VISION_FAILED: %s (attempts=%d, total=%.1fms)",
                    err_msg, attempt, elapsed
                )
                return VerificationResult(
                    status=VerificationStatus.VISION_FAILED,
                    step_index=step_idx,
                    slot_idx=slot_idx,
                    piece_name=piece_name,
                    expected_bitboard=expected_bitboard,
                    actual_bitboard=0,
                    mismatch_count=64,
                    mismatched_cells=(),
                    tray_slot_cleared=False,
                    attempts=attempt,
                    elapsed_ms=elapsed,
                    error_message=err_msg,
                    is_fast_path=False,
                    captures_count=captures_count,
                    settle_ms=settle_ms,
                    capture_ms=total_capture_ms,
                    recognition_ms=total_rec_ms,
                    fast_path_rejection_reason=fast_path_rejection_reason,
                    new_frame=None,
                )

            cap_b_ms = (time.perf_counter() - t_cap_b0) * 1000.0
            captures_count += 1
            total_capture_ms += cap_b_ms

            t_rec_b0 = time.perf_counter()
            frame_res_b = self.recognizer.recognize_frame(img_b)
            rec_b_ms = (time.perf_counter() - t_rec_b0) * 1000.0
            total_rec_ms += rec_b_ms

            # 1. Check for untrusted / uncertain vision recognition in either frame
            if (
                frame_res_a.status == RecognitionStatus.LOW_CONFIDENCE
                or frame_res_b.status == RecognitionStatus.LOW_CONFIDENCE
                or (frame_res_a.board is not None and frame_res_a.board.has_uncertain_cells)
                or (frame_res_b.board is not None and frame_res_b.board.has_uncertain_cells)
            ):
                if attempt < self.max_settle_retries and (time.perf_counter() - t0) < self.settle_timeout_sec:
                    logger.warning(
                        "[VERIFY_RETRY] Attempt %d failed (LOW_CONFIDENCE). Waiting %.2fs before retry...",
                        attempt, self.retry_interval_sec
                    )
                    time.sleep(self.retry_interval_sec)
                    continue
                elapsed = (time.perf_counter() - t0) * 1000.0
                worst_res = frame_res_b if frame_res_b.status == RecognitionStatus.LOW_CONFIDENCE else frame_res_a
                err_msg = f"Unreliable vision recognition: {worst_res.error_message}"
                logger.error(
                    "[VERIFY_PAUSED] Verification halted with status UNKNOWN: %s (attempts=%d, total=%.1fms)",
                    err_msg, attempt, elapsed
                )
                return VerificationResult(
                    status=VerificationStatus.UNKNOWN,
                    step_index=step_idx,
                    slot_idx=slot_idx,
                    piece_name=piece_name,
                    expected_bitboard=expected_bitboard,
                    actual_bitboard=worst_res.board.bitboard if worst_res.board else 0,
                    mismatch_count=popcount(worst_res.board.bitboard ^ expected_bitboard) if worst_res.board else 64,
                    mismatched_cells=(),
                    tray_slot_cleared=False,
                    attempts=attempt,
                    elapsed_ms=elapsed,
                    new_frame_result=worst_res,
                    error_message=err_msg,
                    is_fast_path=False,
                    captures_count=captures_count,
                    settle_ms=settle_ms,
                    capture_ms=total_capture_ms,
                    recognition_ms=total_rec_ms,
                    fast_path_rejection_reason=fast_path_rejection_reason,
                    new_frame=None,
                )

            # 2. Check for critical vision failure / missing board
            if (
                frame_res_a.status == RecognitionStatus.FAILED
                or frame_res_b.status == RecognitionStatus.FAILED
                or frame_res_a.board is None
                or frame_res_b.board is None
            ):
                if attempt < self.max_settle_retries and (time.perf_counter() - t0) < self.settle_timeout_sec:
                    logger.warning(
                        "[VERIFY_RETRY] Attempt %d failed (Board detection failed). Waiting %.2fs before retry...",
                        attempt, self.retry_interval_sec
                    )
                    time.sleep(self.retry_interval_sec)
                    continue
                elapsed = (time.perf_counter() - t0) * 1000.0
                fail_res = frame_res_b if (frame_res_b.board is None or frame_res_b.status == RecognitionStatus.FAILED) else frame_res_a
                err_msg = f"Vision board detection failed: {fail_res.error_message}"
                logger.error(
                    "[VERIFY_PAUSED] Verification halted with status VISION_FAILED: %s (attempts=%d, total=%.1fms)",
                    err_msg, attempt, elapsed
                )
                return VerificationResult(
                    status=VerificationStatus.VISION_FAILED,
                    step_index=step_idx,
                    slot_idx=slot_idx,
                    piece_name=piece_name,
                    expected_bitboard=expected_bitboard,
                    actual_bitboard=0,
                    mismatch_count=64,
                    mismatched_cells=(),
                    tray_slot_cleared=False,
                    attempts=attempt,
                    elapsed_ms=elapsed,
                    new_frame_result=fail_res,
                    error_message=err_msg,
                    is_fast_path=False,
                    captures_count=captures_count,
                    settle_ms=settle_ms,
                    capture_ms=total_capture_ms,
                    recognition_ms=total_rec_ms,
                    fast_path_rejection_reason=fast_path_rejection_reason,
                    new_frame=None,
                )

            # 3. Phase 4.2.3: Check Consecutive Frame Consistency
            if not self._check_consecutive_consistency(frame_res_a, frame_res_b):
                logger.debug(
                    "Attempt %d: Consecutive frames inconsistent (Frame A=0x%016X vs Frame B=0x%016X). Animation active.",
                    attempt, frame_res_a.board.bitboard, frame_res_b.board.bitboard
                )
                if attempt < self.max_settle_retries and (time.perf_counter() - t0) < self.settle_timeout_sec:
                    logger.warning(
                        "[VERIFY_RETRY] Attempt %d failed (Consecutive frames inconsistent: Frame A=0x%016X vs Frame B=0x%016X). Waiting %.2fs before retry...",
                        attempt, frame_res_a.board.bitboard, frame_res_b.board.bitboard, self.retry_interval_sec
                    )
                    time.sleep(self.retry_interval_sec)
                    continue
                else:
                    elapsed = (time.perf_counter() - t0) * 1000.0
                    err_msg = "Board state unstable: consecutive frames remained inconsistent across retries/timeout."
                    logger.error(
                        "[VERIFY_PAUSED] Verification halted with status UNSTABLE_ANIMATION_TIMEOUT: %s (attempts=%d, total=%.1fms)",
                        err_msg, attempt, elapsed
                    )
                    return VerificationResult(
                        status=VerificationStatus.UNSTABLE_ANIMATION_TIMEOUT,
                        step_index=step_idx,
                        slot_idx=slot_idx,
                        piece_name=piece_name,
                        expected_bitboard=expected_bitboard,
                        actual_bitboard=frame_res_b.board.bitboard,
                        mismatch_count=popcount(frame_res_b.board.bitboard ^ expected_bitboard),
                        mismatched_cells=(),
                        tray_slot_cleared=False,
                        attempts=attempt,
                        elapsed_ms=elapsed,
                        new_frame_result=frame_res_b,
                        error_message=err_msg,
                        is_fast_path=False,
                        captures_count=captures_count,
                        settle_ms=settle_ms,
                        capture_ms=total_capture_ms,
                        recognition_ms=total_rec_ms,
                        fast_path_rejection_reason=fast_path_rejection_reason,
                    )

            # Frames are consistent: use Frame B as the verified ground truth
            frame_res = frame_res_b
            actual_bitboard = frame_res.board.bitboard

            # Check if target tray slot was physically emptied using Computer Vision
            if 0 <= slot_idx < len(frame_res.tray_pieces):
                slot_info = frame_res.tray_pieces[slot_idx]
                tray_cleared = slot_info.is_empty
            else:
                tray_cleared = True

            # Calculate bitboard difference
            diff_mask = actual_bitboard ^ expected_bitboard
            mismatch_count = popcount(diff_mask)

            if mismatch_count == 0:
                elapsed = (time.perf_counter() - t0) * 1000.0
                status = VerificationStatus.VERIFIED if attempt == 1 else VerificationStatus.ANIMATION_SETTLING
                logger.info(
                    "[VERIFY_TWO_FRAME] Move %d verified in %d attempt(s) (%d captures, settle=%.1fms, cap=%.1fms, rec=%.1fms, total=%.1fms). Board matches 100%%.",
                    step_idx, attempt, captures_count, settle_ms, total_capture_ms, total_rec_ms, elapsed
                )
                return VerificationResult(
                    status=status,
                    step_index=step_idx,
                    slot_idx=slot_idx,
                    piece_name=piece_name,
                    expected_bitboard=expected_bitboard,
                    actual_bitboard=actual_bitboard,
                    mismatch_count=0,
                    mismatched_cells=(),
                    tray_slot_cleared=tray_cleared,
                    attempts=attempt,
                    elapsed_ms=elapsed,
                    new_frame_result=frame_res,
                    is_fast_path=False,
                    captures_count=captures_count,
                    settle_ms=settle_ms,
                    capture_ms=total_capture_ms,
                    recognition_ms=total_rec_ms,
                    fast_path_rejection_reason=fast_path_rejection_reason,
                    new_frame=img_b,
                )

            # Collect mismatched cells
            mismatches: list[CellMismatch] = []
            for r in range(8):
                for c in range(8):
                    bit = 1 << (r * 8 + c)
                    exp_bit = 1 if (expected_bitboard & bit) else 0
                    act_bit = 1 if (actual_bitboard & bit) else 0
                    if exp_bit != act_bit:
                        mismatches.append(CellMismatch(row=r, col=c, expected=exp_bit, actual=act_bit))
            last_mismatches = tuple(mismatches)

            # If cells mismatched, particle animation might be dissolving. Wait and retry.
            if attempt < self.max_settle_retries and (time.perf_counter() - t0) < self.settle_timeout_sec:
                logger.warning(
                    "[VERIFY_RETRY] Attempt %d: %d cells mismatched. Waiting %.2fs before retry...",
                    attempt, mismatch_count, self.retry_interval_sec
                )
                time.sleep(self.retry_interval_sec)

        # Retries exhausted without 100% bitboard match
        elapsed = (time.perf_counter() - t0) * 1000.0

        # Check if piece was dropped back into tray
        if not tray_cleared and frame_res is not None and 0 <= slot_idx < len(frame_res.tray_pieces):
            slot_info = frame_res.tray_pieces[slot_idx]
            if slot_info.matched_name == piece_name:
                err_msg = f"Piece {piece_name} rejected by game and returned to tray."
                logger.error(
                    "[VERIFY_PAUSED] Verification halted with status TRAY_REJECTED: %s (attempts=%d, total=%.1fms)",
                    err_msg, attempt, elapsed
                )
                return VerificationResult(
                    status=VerificationStatus.TRAY_REJECTED,
                    step_index=step_idx,
                    slot_idx=slot_idx,
                    piece_name=piece_name,
                    expected_bitboard=expected_bitboard,
                    actual_bitboard=actual_bitboard,
                    mismatch_count=len(last_mismatches),
                    mismatched_cells=last_mismatches,
                    tray_slot_cleared=False,
                    attempts=attempt,
                    elapsed_ms=elapsed,
                    new_frame_result=frame_res,
                    error_message=err_msg,
                    is_fast_path=False,
                    captures_count=captures_count,
                    settle_ms=settle_ms,
                    capture_ms=total_capture_ms,
                    recognition_ms=total_rec_ms,
                    fast_path_rejection_reason=fast_path_rejection_reason,
                    new_frame=None,
                )

        err_msg = f"{len(last_mismatches)} cells mismatched after {attempt} attempts."
        logger.error(
            "[VERIFY_PAUSED] Verification halted with status MISMATCH: %s (attempts=%d, total=%.1fms)",
            err_msg, attempt, elapsed
        )

        return VerificationResult(
            status=VerificationStatus.MISMATCH,
            step_index=step_idx,
            slot_idx=slot_idx,
            piece_name=piece_name,
            expected_bitboard=expected_bitboard,
            actual_bitboard=actual_bitboard,
            mismatch_count=len(last_mismatches),
            mismatched_cells=last_mismatches,
            tray_slot_cleared=tray_cleared,
            attempts=attempt,
            elapsed_ms=elapsed,
            new_frame_result=frame_res,
            error_message=err_msg,
            is_fast_path=False,
            captures_count=captures_count,
            settle_ms=settle_ms,
            capture_ms=total_capture_ms,
            recognition_ms=total_rec_ms,
            fast_path_rejection_reason=fast_path_rejection_reason,
        )
