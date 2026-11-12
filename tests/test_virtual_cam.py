"""Unit and regression tests for Virtual Camera output manager (modules/virtual_cam.py)."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import modules.globals
from modules.virtual_cam import (
    VirtualCameraManager,
    get_virtual_camera_manager,
    HAS_PYVIRTUALCAM,
)


class VirtualCameraManagerTests(unittest.TestCase):
    def setUp(self):
        self.mgr = VirtualCameraManager()

    def tearDown(self):
        self.mgr.stop()

    def test_singleton_accessor(self):
        """Verifies get_virtual_camera_manager returns consistent singleton instance."""
        inst1 = get_virtual_camera_manager()
        inst2 = get_virtual_camera_manager()
        self.assertIs(inst1, inst2)

    def test_initial_inactive_state(self):
        """Verifies manager starts in clean inactive state."""
        self.assertFalse(self.mgr.is_active)
        self.assertEqual(self.mgr.device_name, "None")

    @unittest.skipUnless(HAS_PYVIRTUALCAM, "pyvirtualcam required for live test")
    def test_live_camera_lifecycle(self):
        """Verifies camera start, frame dispatch, and stop on host system."""
        ok = self.mgr.start(640, 480, 30.0)
        if not ok:
            # Driver might not be installed in some environments
            self.skipTest("Virtual camera driver not available on host")

        self.assertTrue(self.mgr.is_active)
        self.assertNotEqual(self.mgr.device_name, "None")

        # Send matching frame
        test_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        sent = self.mgr.send_frame(test_frame)
        self.assertTrue(sent)

        # Send mismatched frame (should auto-resize without crash)
        mismatched_frame = np.full((720, 1280, 3), 200, dtype=np.uint8)
        sent_mismatched = self.mgr.send_frame(mismatched_frame)
        self.assertTrue(sent_mismatched)

        # Stop
        self.mgr.stop()
        self.assertFalse(self.mgr.is_active)

    def test_mocked_camera_lifecycle(self):
        """Verifies start, frame send, and stop lifecycle using mocks."""
        mock_cam = MagicMock()
        mock_cam.device = "Mocked VCam"

        with patch("modules.virtual_cam.pyvirtualcam.Camera", return_value=mock_cam):
            with patch("modules.virtual_cam.HAS_PYVIRTUALCAM", True):
                started = self.mgr.start(640, 480, 30.0)
                self.assertTrue(started)
                self.assertTrue(self.mgr.is_active)
                self.assertEqual(self.mgr.device_name, "Mocked VCam")

                # Test frame dispatch
                dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                sent = self.mgr.send_frame(dummy)
                self.assertTrue(sent)
                mock_cam.send.assert_called_once()

                # Test stop
                self.mgr.stop()
                self.assertFalse(self.mgr.is_active)
                mock_cam.close.assert_called_once()

    def test_send_frame_when_inactive_returns_false(self):
        """Verifies send_frame returns False safely when camera is not started."""
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertFalse(self.mgr.send_frame(dummy))
        self.assertFalse(self.mgr.send_frame(None))

    def test_pyvirtualcam_missing_graceful_failure(self):
        """Verifies graceful handling when pyvirtualcam is not installed."""
        with patch("modules.virtual_cam.HAS_PYVIRTUALCAM", False):
            started = self.mgr.start(640, 480, 30.0)
            self.assertFalse(started)
            self.assertFalse(self.mgr.is_active)


if __name__ == "__main__":
    unittest.main()
