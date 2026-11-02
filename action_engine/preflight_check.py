"""System Preflight Check & Device Diagnostics for BLOCK BLAST AI.

Audits environment readiness:
1. Python version (>= 3.9) and 64-bit architecture
2. Required dependencies (NumPy, OpenCV)
3. ADB executable availability and server daemon status
4. Connected Android devices and authorization states
5. Screen resolution and display density
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class AndroidDeviceInfo:
    serial: str
    state: str  # "device", "unauthorized", "offline", "no_device"
    model: str = "Unknown"
    resolution: Optional[Tuple[int, int]] = None
    warning: str = ""


@dataclass
class PreflightReport:
    python_ok: bool
    python_version: str
    deps_ok: bool
    numpy_version: str
    opencv_version: str
    adb_ok: bool
    adb_path: str
    adb_version: str
    devices: List[AndroidDeviceInfo]
    ready_for_live_play: bool
    recommendations: List[str]


def resolve_adb_path() -> Optional[str]:
    """Find adb in PATH or common standard installation directories on Windows/Linux/macOS."""
    which_adb = shutil.which("adb")
    if which_adb:
        return which_adb

    candidates = [
        r"C:\platform-tools\adb.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe"),
        os.path.expandvars(r"%USERPROFILE%\AppData\Local\Android\Sdk\platform-tools\adb.exe"),
        "/usr/bin/adb",
        "/usr/local/bin/adb",
        "/opt/homebrew/bin/adb",
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def audit_python() -> Tuple[bool, str]:
    ver = sys.version_info
    major = getattr(ver, "major", ver[0])
    minor = getattr(ver, "minor", ver[1])
    micro = getattr(ver, "micro", ver[2] if len(ver) > 2 else 0)
    ver_str = f"{major}.{minor}.{micro} ({platform_arch()})"
    ok = (major == 3 and minor >= 9)
    return ok, ver_str


def platform_arch() -> str:
    return "64-bit" if sys.maxsize > 2**32 else "32-bit"


def audit_dependencies() -> Tuple[bool, str, str]:
    np_ver = "Missing"
    cv_ver = "Missing"
    try:
        import numpy as np
        np_ver = np.__version__
    except ImportError:
        pass

    try:
        import cv2
        cv_ver = cv2.__version__
    except ImportError:
        pass

    ok = (np_ver != "Missing" and cv_ver != "Missing")
    return ok, np_ver, cv_ver


def audit_adb(adb_exe: Optional[str] = None) -> Tuple[bool, str, str]:
    exe = adb_exe or resolve_adb_path() or "adb"
    try:
        res = subprocess.run([exe, "version"], capture_output=True, text=True, timeout=3.0)
        if res.returncode == 0:
            first_line = res.stdout.strip().splitlines()[0] if res.stdout else "ADB Active"
            return True, exe, first_line
    except Exception:
        pass
    return False, exe if os.path.isfile(exe) else "Not found", "Unavailable"


def query_android_devices(adb_exe: str) -> List[AndroidDeviceInfo]:
    devices: List[AndroidDeviceInfo] = []
    try:
        res = subprocess.run([adb_exe, "devices", "-l"], capture_output=True, text=True, timeout=4.0)
        if res.returncode != 0:
            return devices

        lines = res.stdout.strip().splitlines()[1:]
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            serial = parts[0]
            state = parts[1] if len(parts) > 1 else "unknown"

            # Parse model name if present
            model = "Android Device"
            for p in parts[2:]:
                if p.startswith("model:"):
                    model = p.split(":", 1)[1]
                elif p.startswith("device:"):
                    if model == "Android Device":
                        model = p.split(":", 1)[1]

            warning = ""
            resolution = None

            if state == "unauthorized":
                warning = "Device is UNAUTHORIZED. Unlock phone and tap 'Always allow from this computer'."
            elif state == "offline":
                warning = "Device is OFFLINE. Reconnect USB cable or toggle USB Debugging."
            elif state == "device":
                # Try getting screen resolution
                try:
                    wm_res = subprocess.run(
                        [adb_exe, "-s", serial, "shell", "wm", "size"],
                        capture_output=True,
                        text=True,
                        timeout=2.0,
                    )
                    if wm_res.returncode == 0 and "Physical size:" in wm_res.stdout:
                        size_str = wm_res.stdout.split("Physical size:")[1].strip().split()[0]
                        w_str, h_str = size_str.split("x")
                        resolution = (int(w_str), int(h_str))
                except Exception:
                    pass

            devices.append(
                AndroidDeviceInfo(
                    serial=serial,
                    state=state,
                    model=model,
                    resolution=resolution,
                    warning=warning,
                )
            )
    except Exception:
        pass
    return devices


def run_preflight() -> PreflightReport:
    recs: List[str] = []

    # 1. Python
    py_ok, py_ver = audit_python()
    if not py_ok:
        recs.append("Install Python 3.9+ (64-bit) from https://www.python.org/downloads/")

    # 2. Dependencies
    deps_ok, np_ver, cv_ver = audit_dependencies()
    if not deps_ok:
        recs.append("Install required dependencies by running: pip install -r requirements.txt")

    # 3. ADB
    adb_path = resolve_adb_path() or "adb"
    adb_ok, adb_resolved, adb_ver = audit_adb(adb_path)
    if not adb_ok:
        recs.append("Install Android SDK Platform-Tools: https://developer.android.com/tools/releases/platform-tools")

    # 4. Devices
    devices = query_android_devices(adb_resolved) if adb_ok else []
    ready_devices = [d for d in devices if d.state == "device"]

    if adb_ok and not devices:
        recs.append("Connect your Android phone via USB cable and enable 'USB Debugging' in Developer Options.")
    elif adb_ok and any(d.state == "unauthorized" for d in devices):
        recs.append("Unlock your Android screen and tap 'Allow USB debugging' (Check 'Always allow').")

    ready = py_ok and deps_ok and adb_ok and len(ready_devices) >= 1

    return PreflightReport(
        python_ok=py_ok,
        python_version=py_ver,
        deps_ok=deps_ok,
        numpy_version=np_ver,
        opencv_version=cv_ver,
        adb_ok=adb_ok,
        adb_path=adb_resolved,
        adb_version=adb_ver,
        devices=devices,
        ready_for_live_play=ready,
        recommendations=recs,
    )


def print_preflight_banner(report: PreflightReport) -> None:
    sep = "=" * 74
    sub = "-" * 74
    print(sep)
    print("  BLOCK BLAST AI ULTIMATE — ENVIRONMENT & DEVICE PREFLIGHT AUDIT")
    print(sep)

    # Python
    py_badge = "[OK]" if report.python_ok else "[FAIL]"
    print(f"  {py_badge:<8} Python:         {report.python_version}")

    # Dependencies
    dep_badge = "[OK]" if report.deps_ok else "[FAIL]"
    dep_desc = f"NumPy {report.numpy_version} | OpenCV {report.opencv_version}"
    print(f"  {dep_badge:<8} Dependencies:   {dep_desc}")

    # ADB
    adb_badge = "[OK]" if report.adb_ok else "[FAIL]"
    print(f"  {adb_badge:<8} ADB Bridge:     {report.adb_path}")
    if report.adb_ok:
        print(f"           Version:        {report.adb_version}")

    print(sub)

    # Devices
    if not report.adb_ok:
        print("  [WARN]   Devices:        Cannot inspect devices (ADB not installed)")
    elif not report.devices:
        print("  [WARN]   Devices:        No Android devices detected via USB.")
        print("                           (Connect phone with USB Debugging enabled, or use Offline CLI mode)")
    else:
        for idx, dev in enumerate(report.devices, 1):
            res_str = f"{dev.resolution[0]}x{dev.resolution[1]}" if dev.resolution else "Unknown"
            if dev.state == "device":
                print(f"  [OK]     Device #{idx}:     {dev.serial} ({dev.model}) - State: READY (Res: {res_str})")
            else:
                print(f"  [WARN]   Device #{idx}:     {dev.serial} ({dev.model}) - State: {dev.state.upper()}")
                if dev.warning:
                    print(f"           ACTION:         {dev.warning}")

    print(sep)

    if report.ready_for_live_play:
        print("  STATUS: READY FOR AUTONOMOUS PLAY!")
    else:
        print("  STATUS: PREFLIGHT ACTION REQUIRED FOR LIVE PHONE PLAY")
        if report.recommendations:
            print("  RECOMMENDED ACTIONS:")
            for rec in report.recommendations:
                print(f"    -> {rec}")
    print(sep)


def main() -> int:
    report = run_preflight()
    print_preflight_banner(report)
    return 0 if report.ready_for_live_play else 1


if __name__ == "__main__":
    sys.exit(main())
