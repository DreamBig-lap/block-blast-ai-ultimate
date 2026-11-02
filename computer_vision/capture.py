"""Screen Capture via ADB Stream and Local Image File Loading.

Provides robust, zero-copy, in-memory decoding of live Android frames via ADB exec-out,
as well as file loading for testing and offline analysis.
"""

from __future__ import annotations

import logging
import os
import shutil
import socket
import struct
import subprocess
import time
from typing import Optional

import cv2
import numpy as np

from computer_vision.config import DEFAULT_VISION_CONFIG, VisionConfig

logger = logging.getLogger(__name__)


def _recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
    """Read exactly num_bytes from socket unless EOF is reached."""
    buf = bytearray(num_bytes)
    view = memoryview(buf)
    pos = 0
    while pos < num_bytes:
        n = sock.recv_into(view[pos:])
        if n == 0:
            break
        pos += n
    return bytes(buf[:pos])


def parse_raw_screencap(raw_bytes: bytes) -> Optional[np.ndarray]:
    """Parse Android raw framebuffer binary stream from 'adb exec-out screencap'.

    Supports:
    - 16-byte header (Android 10+): width (4B), height (4B), format (4B), color_space (4B)
    - 12-byte header (Android 9 and older): width (4B), height (4B), format (4B)

    Tolerant to trailing bytes / newlines by slicing exactly the expected payload bytes.

    Returns:
        BGR numpy ndarray of shape (height, width, 3), or None if buffer is invalid.
    """
    total_len = len(raw_bytes)
    if total_len < 16:
        return None

    # Check 16-byte header first (Android 10+)
    w, h, fmt, _ = struct.unpack("<IIII", raw_bytes[:16])
    expected_16 = 16 + w * h * 4
    if w >= 200 and h >= 200 and 0 <= total_len - expected_16 <= 64:
        pixel_bytes = raw_bytes[16:expected_16]
        raw_pixels = np.frombuffer(pixel_bytes, dtype=np.uint8).reshape((h, w, 4))
        if fmt == 5:  # BGRA_8888
            return cv2.cvtColor(raw_pixels, cv2.COLOR_BGRA2BGR)
        # Default RGBA_8888 (1) or RGBX_8888 (2)
        return cv2.cvtColor(raw_pixels, cv2.COLOR_RGBA2BGR)

    # Check 12-byte header (Android 9 and older)
    w_12, h_12, fmt_12 = struct.unpack("<III", raw_bytes[:12])
    expected_12 = 12 + w_12 * h_12 * 4
    if w_12 >= 200 and h_12 >= 200 and 0 <= total_len - expected_12 <= 64:
        pixel_bytes = raw_bytes[12:expected_12]
        raw_pixels = np.frombuffer(pixel_bytes, dtype=np.uint8).reshape((h_12, w_12, 4))
        if fmt_12 == 5:
            return cv2.cvtColor(raw_pixels, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(raw_pixels, cv2.COLOR_RGBA2BGR)

    return None


class ScreenCaptureError(Exception):
    """Raised when screenshot capture or loading fails."""
    pass


class ScreenCapture:
    """Manages screenshot acquisition from USB/ADB or disk."""

    def __init__(self, config: Optional[VisionConfig] = None) -> None:
        self.config = config or DEFAULT_VISION_CONFIG
        self.last_adb_transfer_ms: float = 0.0
        self.last_png_decode_ms: float = 0.0
        self.last_capture_mode: str = "RAW"
        self.last_transport_mode: str = "SOCKET"
        self.last_bytes_count: int = 0
        self.last_connect_ms: float = 0.0
        self.last_fallback_reason: Optional[str] = None
        self.last_telemetry: dict = {}

    def load_from_file(self, image_path: str) -> np.ndarray:
        """Load an image file from disk and return as a BGR numpy array.

        Args:
            image_path: Absolute or relative file path to PNG/JPEG image.

        Raises:
            ScreenCaptureError: If file does not exist or cannot be decoded.
        """
        if not os.path.isfile(image_path):
            raise ScreenCaptureError(f"Image file not found: {image_path}")

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ScreenCaptureError(f"Failed to decode image file: {image_path}")

        h, w = img.shape[:2]
        if h < 200 or w < 200:
            raise ScreenCaptureError(f"Image dimensions too small: {w}x{h}")

        return img

    def _stream_via_adb_socket(
        self,
        device_id: Optional[str] = None,
        raw: bool = True,
        timeout: float = 8.0,
    ) -> tuple[bytes, float, float]:
        """Stream screencap bytes directly from ADB server daemon via TCP socket.

        Bypasses Windows subprocess creation overhead for maximum throughput.
        Returns:
            Tuple of (payload_bytes, connect_ms, transfer_ms).
        """
        host = getattr(self.config, "adb_server_host", "127.0.0.1")
        port = getattr(self.config, "adb_server_port", 5037)
        dev = device_id or self.config.device_id

        t0 = time.perf_counter()
        s = socket.create_connection((host, port), timeout=timeout)
        try:
            s.settimeout(timeout)
            try:
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except Exception:
                pass
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)
            except Exception:
                pass

            t_conn = time.perf_counter()
            connect_ms = (t_conn - t0) * 1000.0

            # 1. Transport to device
            if dev:
                req = f"host:transport:{dev}".encode("ascii")
            else:
                req = b"host:transport-any"
            s.sendall(f"{len(req):04x}".encode("ascii") + req)
            reply = _recv_exact(s, 4)
            if reply != b"OKAY":
                err = s.recv(1024).decode("utf-8", errors="ignore")
                raise ScreenCaptureError(f"ADB device transport failed ({reply.decode('ascii', errors='ignore')}): {err}")

            # 2. Exec screencap command
            cmd = b"exec:screencap" if raw else b"exec:screencap -p"
            s.sendall(f"{len(cmd):04x}".encode("ascii") + cmd)
            reply = _recv_exact(s, 4)
            if reply != b"OKAY":
                err = s.recv(1024).decode("utf-8", errors="ignore")
                raise ScreenCaptureError(f"ADB exec screencap failed ({reply.decode('ascii', errors='ignore')}): {err}")

            # 3. Read stream until EOF
            t_trans_start = time.perf_counter()
            chunks: list[bytes] = []
            while True:
                chunk = s.recv(131072)
                if not chunk:
                    break
                chunks.append(chunk)
            transfer_ms = (time.perf_counter() - t_trans_start) * 1000.0
            return b"".join(chunks), connect_ms, transfer_ms
        finally:
            try:
                s.close()
            except Exception:
                pass

    def _stream_via_subprocess(
        self,
        exe: str,
        dev: Optional[str] = None,
        raw: bool = True,
        timeout: float = 10.0,
    ) -> tuple[bytes, float]:
        """Fallback stream acquisition via subprocess invoking 'adb exec-out screencap'."""
        cmd = [exe]
        if dev:
            cmd.extend(["-s", dev])
        cmd.append("exec-out")
        cmd.append("screencap")
        if not raw:
            cmd.append("-p")

        t0 = time.perf_counter()
        res = subprocess.run(cmd, capture_output=True, timeout=timeout)
        transfer_ms = (time.perf_counter() - t0) * 1000.0

        if res.returncode == 0 and len(res.stdout) > 100:
            return res.stdout, transfer_ms
        err_msg = res.stderr.decode("utf-8", errors="ignore").strip()
        raise ScreenCaptureError(f"ADB subprocess exited with code {res.returncode}: {err_msg}")

    def capture_from_adb_with_timings(
        self,
        device_id: Optional[str] = None,
        adb_path: Optional[str] = None,
        retries: int = 3,
        retry_delay: float = 0.5,
    ) -> tuple[np.ndarray, float, float]:
        """Capture live screen directly from Android device with timings.

        Evaluates tiered capture:
        1. Raw screencap via direct ADB socket (Fastest: ~280-320ms, zero subprocess overhead).
        2. Raw screencap via subprocess (Fast: ~320-380ms).
        3. Seamless automatic fallback to PNG (screencap -p) if raw fails or format is unparseable.

        Returns:
            Tuple of (img, adb_transfer_ms, png_decode_ms).
        """
        exe = adb_path or self.config.adb_path
        dev = device_id or self.config.device_id
        use_raw = getattr(self.config, "use_raw_screencap", True)
        use_socket = getattr(self.config, "use_adb_socket", True)

        resolved_exe = shutil.which(exe) if exe else None
        if not resolved_exe:
            if exe and os.path.isfile(exe):
                resolved_exe = exe
            elif os.path.isfile(r"C:\platform-tools\adb.exe"):
                resolved_exe = r"C:\platform-tools\adb.exe"
            else:
                raise ScreenCaptureError(f"ADB executable not found at: {exe}")
        exe = resolved_exe

        last_error = ""
        fallback_reason = "NONE"

        for attempt in range(1, retries + 1):
            connect_ms = 0.0
            # Attempt 1: Raw screencap
            if use_raw:
                transport_method = "SOCKET"
                t_adb_start = time.perf_counter()
                raw_bytes: Optional[bytes] = None

                if use_socket:
                    try:
                        raw_bytes, connect_ms, _ = self._stream_via_adb_socket(device_id=dev, raw=True, timeout=8.0)
                    except Exception as sock_err:
                        fallback_reason = f"SOCKET_RAW_FAILED({sock_err})"
                        logger.debug("ADB socket streaming failed (%s); falling back to subprocess", sock_err)
                        raw_bytes = None

                if raw_bytes is None:
                    transport_method = "SUBPROCESS"
                    t_adb_start = time.perf_counter()
                    try:
                        raw_bytes, _ = self._stream_via_subprocess(exe=exe, dev=dev, raw=True, timeout=10.0)
                    except Exception as sub_err:
                        last_error = str(sub_err)
                        fallback_reason = f"SUBPROCESS_RAW_FAILED({sub_err})"
                        raw_bytes = None

                if raw_bytes is not None and len(raw_bytes) > 100:
                    t_adb_end = time.perf_counter()
                    adb_ms = (t_adb_end - t_adb_start) * 1000.0

                    t_dec_start = time.perf_counter()
                    img = parse_raw_screencap(raw_bytes)
                    t_dec_end = time.perf_counter()
                    dec_ms = (t_dec_end - t_dec_start) * 1000.0

                    if img is not None and img.shape[0] > 100 and img.shape[1] > 100:
                        self.last_capture_mode = "RAW"
                        self.last_transport_mode = transport_method
                        self.last_bytes_count = len(raw_bytes)
                        self.last_connect_ms = connect_ms
                        self.last_fallback_reason = None if fallback_reason == "NONE" else fallback_reason
                        self.last_telemetry = {
                            "attempt": attempt,
                            "mode": "RAW",
                            "transport": transport_method,
                            "bytes": len(raw_bytes),
                            "connect_ms": round(connect_ms, 1),
                            "transfer_ms": round(adb_ms, 1),
                            "decode_ms": round(dec_ms, 1),
                            "fallback": self.last_fallback_reason,
                        }
                        logger.info(
                            "[ADB_CAPTURE_DIAG] attempt=%d transport=%s mode=RAW bytes=%d connect_ms=%.1f transfer_ms=%.1f decode_ms=%.1f fallback=%s",
                            attempt, transport_method, len(raw_bytes), connect_ms, adb_ms, dec_ms, self.last_fallback_reason or "NONE"
                        )
                        return img, adb_ms, dec_ms
                    else:
                        fallback_reason = f"RAW_PARSE_RETURNED_NONE({len(raw_bytes)} bytes)"
                        logger.warning("parse_raw_screencap returned None on buffer (%d bytes); falling back to PNG", len(raw_bytes))

            # Attempt 2 (Fallback): Standard PNG screencap -p
            transport_method = "SOCKET"
            t_adb_start = time.perf_counter()
            png_bytes: Optional[bytes] = None

            if use_socket:
                try:
                    png_bytes, connect_ms, _ = self._stream_via_adb_socket(device_id=dev, raw=False, timeout=10.0)
                except Exception as sock_err:
                    fallback_reason = f"SOCKET_PNG_FAILED({sock_err})"
                    logger.debug("ADB socket PNG streaming failed (%s); falling back to subprocess", sock_err)
                    png_bytes = None

            if png_bytes is None:
                transport_method = "SUBPROCESS"
                t_adb_start = time.perf_counter()
                try:
                    png_bytes, _ = self._stream_via_subprocess(exe=exe, dev=dev, raw=False, timeout=10.0)
                except Exception as sub_err:
                    last_error = str(sub_err)
                    fallback_reason = f"SUBPROCESS_PNG_FAILED({sub_err})"
                    png_bytes = None

            if png_bytes is not None and len(png_bytes) > 100:
                t_adb_end = time.perf_counter()
                adb_ms = (t_adb_end - t_adb_start) * 1000.0

                t_dec_start = time.perf_counter()
                np_arr = np.frombuffer(png_bytes, dtype=np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                t_dec_end = time.perf_counter()
                dec_ms = (t_dec_end - t_dec_start) * 1000.0

                if img is not None and img.shape[0] > 100 and img.shape[1] > 100:
                    self.last_capture_mode = "PNG"
                    self.last_transport_mode = transport_method
                    self.last_bytes_count = len(png_bytes)
                    self.last_connect_ms = connect_ms
                    self.last_fallback_reason = fallback_reason
                    self.last_telemetry = {
                        "attempt": attempt,
                        "mode": "PNG",
                        "transport": transport_method,
                        "bytes": len(png_bytes),
                        "connect_ms": round(connect_ms, 1),
                        "transfer_ms": round(adb_ms, 1),
                        "decode_ms": round(dec_ms, 1),
                        "fallback": fallback_reason,
                    }
                    logger.info(
                        "[ADB_CAPTURE_DIAG] attempt=%d transport=%s mode=PNG bytes=%d connect_ms=%.1f transfer_ms=%.1f decode_ms=%.1f fallback=%s",
                        attempt, transport_method, len(png_bytes), connect_ms, adb_ms, dec_ms, fallback_reason
                    )
                    return img, adb_ms, dec_ms
                last_error = "cv2.imdecode failed to parse image buffer"

            if attempt < retries:
                time.sleep(retry_delay)

        raise ScreenCaptureError(
            f"ADB screencap failed after {retries} attempt(s): {last_error}"
        )

    def capture_from_adb(
        self,
        device_id: Optional[str] = None,
        adb_path: Optional[str] = None,
        retries: int = 3,
        retry_delay: float = 0.5,
    ) -> np.ndarray:
        """Capture live screen directly from Android device via ADB streaming.

        Uses 'adb exec-out screencap -p' to pipe raw PNG bytes into Python memory
        without creating temporary files on the device filesystem.

        Args:
            device_id: Target ADB serial (e.g. 'R58N62QSJZV').
            adb_path: Path to adb executable.
            retries: Number of retry attempts on transient USB/ADB dropouts.
            retry_delay: Pause in seconds between retries.

        Raises:
            ScreenCaptureError: If adb fails, device is unreachable, or stream is corrupted.
        """
        img, adb_ms, dec_ms = self.capture_from_adb_with_timings(
            device_id=device_id,
            adb_path=adb_path,
            retries=retries,
            retry_delay=retry_delay,
        )
        self.last_adb_transfer_ms = adb_ms
        self.last_png_decode_ms = dec_ms
        return img
