"""Unit tests for action_engine/preflight_check.py.

Verifies:
1. Python version audit logic
2. Dependency detection logic
3. ADB path resolution and daemon status queries
4. ADB device table parsing (device, unauthorized, offline, no devices)
5. Preflight report assembly and banner printing
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

from action_engine.preflight_check import (
    AndroidDeviceInfo,
    PreflightReport,
    audit_adb,
    audit_dependencies,
    audit_python,
    print_preflight_banner,
    query_android_devices,
    resolve_adb_path,
    run_preflight,
)


class TestPreflightCheck(unittest.TestCase):
    """Test suite for system preflight and device diagnostic checks."""

    def test_audit_python_current_environment(self) -> None:
        """Current runtime environment must satisfy Python >= 3.9 requirement."""
        ok, ver_str = audit_python()
        self.assertTrue(ok)
        self.assertIn("64-bit", ver_str)
        self.assertIn(str(sys.version_info.major), ver_str)

    def test_audit_python_rejection_on_old_version(self) -> None:
        """Old Python versions (e.g. 3.8) must be flagged as not ok."""
        with patch.object(sys, "version_info", (3, 8, 5)):
            ok, ver_str = audit_python()
            self.assertFalse(ok)

    def test_audit_dependencies_installed(self) -> None:
        """NumPy and OpenCV must be verified successfully."""
        ok, np_ver, cv_ver = audit_dependencies()
        self.assertTrue(ok)
        self.assertNotEqual(np_ver, "Missing")
        self.assertNotEqual(cv_ver, "Missing")

    def test_audit_adb_success(self) -> None:
        """Successful subprocess execution reports ADB as available."""
        mock_res = subprocess.CompletedProcess(
            args=["adb", "version"],
            returncode=0,
            stdout="Android Debug Bridge version 1.0.41\nVersion 34.0.0",
            stderr="",
        )
        with patch("subprocess.run", return_value=mock_res):
            ok, path, ver = audit_adb("adb")
            self.assertTrue(ok)
            self.assertEqual(path, "adb")
            self.assertIn("Android Debug Bridge", ver)

    def test_audit_adb_failure_or_missing(self) -> None:
        """When adb command fails or is missing, reports unavailable."""
        with patch("subprocess.run", side_effect=FileNotFoundError("adb not found")):
            ok, path, ver = audit_adb("nonexistent_adb_path")
            self.assertFalse(ok)
            self.assertEqual(ver, "Unavailable")

    def test_query_android_devices_parsing(self) -> None:
        """Validates correct parsing of attached device statuses."""
        mock_output = (
            "List of devices attached\n"
            "M21_DEV_01            device product:m21 model:SM_M215F device:m21 transport_id:1\n"
            "UNAUTH_DEV_02         unauthorized transport_id:2\n"
            "OFFLINE_DEV_03        offline transport_id:3\n"
        )
        mock_res = subprocess.CompletedProcess(args=["adb", "devices", "-l"], returncode=0, stdout=mock_output)

        with patch("subprocess.run", return_value=mock_res):
            devices = query_android_devices("adb")

        self.assertEqual(len(devices), 3)

        d1 = devices[0]
        self.assertEqual(d1.serial, "M21_DEV_01")
        self.assertEqual(d1.state, "device")
        self.assertEqual(d1.model, "SM_M215F")

        d2 = devices[1]
        self.assertEqual(d2.serial, "UNAUTH_DEV_02")
        self.assertEqual(d2.state, "unauthorized")
        self.assertIn("UNAUTHORIZED", d2.warning)

        d3 = devices[2]
        self.assertEqual(d3.serial, "OFFLINE_DEV_03")
        self.assertEqual(d3.state, "offline")
        self.assertIn("OFFLINE", d3.warning)

    def test_query_android_devices_empty(self) -> None:
        """No devices attached returns empty list."""
        mock_output = "List of devices attached\n\n"
        mock_res = subprocess.CompletedProcess(args=["adb", "devices", "-l"], returncode=0, stdout=mock_output)

        with patch("subprocess.run", return_value=mock_res):
            devices = query_android_devices("adb")

        self.assertEqual(devices, [])

    def test_run_preflight_and_banner_render(self) -> None:
        """End-to-end report generation and formatted console banner printing."""
        report = run_preflight()
        self.assertIsInstance(report, PreflightReport)
        self.assertTrue(report.python_ok)
        self.assertTrue(report.deps_ok)

        # Banner should print cleanly without throwing exceptions
        print_preflight_banner(report)


if __name__ == "__main__":
    unittest.main()
