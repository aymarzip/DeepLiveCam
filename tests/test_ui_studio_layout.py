"""Unit tests for DeepLiveCam Studio modern PySide6 Fluent UI layout.

Verifies:
- MainWindow layout and control hierarchy.
- Presence and wiring of all cards (Media, Camera, Protection, Refinement).
- Dynamic numeric slider readout updates.
- Backward compatibility stubs.
- Hardware provider badge determination.
"""

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

import modules.globals
import modules.metadata
import modules.ui as ui

_APP = None


def setUpModule():
    global _APP
    _APP = QApplication.instance() or QApplication([])


class UIStudioLayoutTest(unittest.TestCase):
    def setUp(self):
        self.start_called = False
        self.destroy_called = False

        def fake_start():
            self.start_called = True

        def fake_destroy():
            self.destroy_called = True

        with patch.object(ui, "get_available_cameras", lambda: ([0], ["Mock Camera"])):
            self.window = ui.MainWindow(fake_start, fake_destroy)
        self.addCleanup(self.window.deleteLater)

    def test_window_title_and_dimensions(self):
        title = self.window.windowTitle()
        self.assertIn("Studio", title)
        self.assertIn(modules.metadata.version, title)
        self.assertEqual(ui.ROOT_WIDTH, 960)
        self.assertEqual(ui.ROOT_HEIGHT, 760)
        self.assertGreaterEqual(self.window.minimumWidth(), 900)
        self.assertGreaterEqual(self.window.minimumHeight(), 700)

    def test_media_card_widgets_present_and_wired(self):
        self.assertTrue(hasattr(self.window, "source_label"))
        self.assertTrue(hasattr(self.window, "target_label"))
        self.assertTrue(hasattr(self.window, "btn_select_source"))
        self.assertTrue(hasattr(self.window, "btn_random_face"))
        self.assertTrue(hasattr(self.window, "btn_swap"))
        self.assertTrue(hasattr(self.window, "btn_select_target"))

        self.assertEqual(self.window.source_label.objectName(), "imageDrop")
        self.assertEqual(self.window.target_label.objectName(), "imageDrop")
        self.assertEqual(self.window.source_label.width(), ui.SOURCE_TARGET_PREVIEW_SIZE)
        self.assertEqual(self.window.source_label.height(), ui.SOURCE_TARGET_PREVIEW_SIZE)

    def test_camera_and_virtual_cam_card_widgets(self):
        self.assertTrue(hasattr(self.window, "cb_camera"))
        self.assertTrue(hasattr(self.window, "btn_live"))
        self.assertTrue(hasattr(self.window, "cb_resolution"))
        self.assertTrue(hasattr(self.window, "cb_det_size"))
        self.assertTrue(hasattr(self.window, "sw_vcam"))
        self.assertTrue(hasattr(self.window, "sw_show_fps"))
        self.assertTrue(hasattr(self.window, "sw_color_fix"))
        self.assertTrue(hasattr(self.window, "sw_poisson"))

        self.assertEqual(self.window.btn_live.objectName(), "live")
        self.assertIn("Live", self.window.btn_live.text())

    def test_feature_protection_card_widgets(self):
        self.assertTrue(hasattr(self.window, "sw_eyes"))
        self.assertTrue(hasattr(self.window, "sw_glasses"))
        self.assertTrue(hasattr(self.window, "sw_bangs"))
        self.assertTrue(hasattr(self.window, "sw_map_faces"))
        self.assertTrue(hasattr(self.window, "s_mouth"))
        self.assertTrue(hasattr(self.window, "lbl_mouth_val"))

        self.assertEqual(self.window.lbl_mouth_val.objectName(), "valueBadge")

    def test_tuning_card_and_numeric_readouts(self):
        self.assertTrue(hasattr(self.window, "cb_enhancer"))
        self.assertTrue(hasattr(self.window, "s_transparency"))
        self.assertTrue(hasattr(self.window, "lbl_trans_val"))
        self.assertTrue(hasattr(self.window, "s_sharpness"))
        self.assertTrue(hasattr(self.window, "lbl_sharp_val"))
        self.assertTrue(hasattr(self.window, "sw_keep_fps"))
        self.assertTrue(hasattr(self.window, "sw_keep_audio"))
        self.assertTrue(hasattr(self.window, "sw_keep_frames"))
        self.assertTrue(hasattr(self.window, "sw_many_faces"))

        # Test transparency slider readout
        self.window.s_transparency.setValue(75)
        self.assertEqual(self.window.lbl_trans_val.text(), "75%")
        self.assertEqual(modules.globals.opacity, 0.75)

        # Test sharpness slider readout
        self.window.s_sharpness.setValue(25)
        self.assertEqual(self.window.lbl_sharp_val.text(), "2.5")
        self.assertEqual(modules.globals.sharpness, 2.5)

        # Test mouth mask slider readout
        self.window.s_mouth.setValue(40)
        self.assertEqual(self.window.lbl_mouth_val.text(), "40%")
        self.assertEqual(modules.globals.mouth_mask_size, 40.0)

    def test_action_buttons_and_callbacks(self):
        self.assertTrue(hasattr(self.window, "btn_start"))
        self.assertTrue(hasattr(self.window, "btn_preview"))
        self.assertTrue(hasattr(self.window, "btn_destroy"))

        self.assertEqual(self.window.btn_destroy.objectName(), "danger")
        self.window.btn_destroy.click()
        self.assertTrue(self.destroy_called)

    def test_backward_compatibility_stubs(self):
        opt = self.window._build_options_card()
        self.assertIsNotNone(opt)
        sliders = self.window._build_sliders_card()
        self.assertIsNotNone(sliders)
        img_row = self.window._build_image_row()
        self.assertIsNotNone(img_row)

    def test_hardware_provider_badge(self):
        with patch.object(modules.globals, "execution_providers", ["DmlExecutionProvider"]):
            badge = ui._get_provider_badge_text()
            self.assertIn("DirectML", badge)

        with patch.object(modules.globals, "execution_providers", ["CUDAExecutionProvider"]):
            badge = ui._get_provider_badge_text()
            self.assertIn("CUDA", badge)

        with patch.object(modules.globals, "execution_providers", ["CPUExecutionProvider"]):
            badge = ui._get_provider_badge_text()
            self.assertIn("CPU", badge)


if __name__ == "__main__":
    unittest.main()
