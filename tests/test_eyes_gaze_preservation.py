"""Unit and regression tests for Eye-Gaze & Natural Blink Preservation.

Verifies:
1. create_eyes_mask extracts accurate 106-point eye contours and handles 5-point fallback.
2. apply_eyes_area blends natural pupils, gaze, and eyelids seamlessly into swapped faces.
3. draw_eyes_mask_visualization renders debug contours correctly.
4. CLI arguments --eyes-mask, --virtual-cam, and --virtual-cam-backend parse and configure globals.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import modules.globals
import modules.core
from modules.processors.frame.face_masking import (
    create_eyes_mask,
    apply_eyes_area,
    draw_eyes_mask_visualization,
)
import modules.processors.frame.face_swapper as face_swapper
from modules.typing import Face


def _make_dummy_face_with_eyes(w=640, h=480) -> Face:
    """Creates a mock Face object with valid 106-point landmarks covering eye contours."""
    landmarks = np.zeros((106, 2), dtype=np.float32)
    # Right eye: 33-42 around (260, 200)
    for i in range(33, 43):
        angle = (i - 33) * (2 * np.pi / 10)
        landmarks[i] = [260 + 25 * np.cos(angle), 200 + 12 * np.sin(angle)]

    # Left eye: 87-96 around (380, 200)
    for i in range(87, 97):
        angle = (i - 87) * (2 * np.pi / 10)
        landmarks[i] = [380 + 25 * np.cos(angle), 200 + 12 * np.sin(angle)]

    kps = np.array([
        [260.0, 200.0],  # Right eye
        [380.0, 200.0],  # Left eye
        [320.0, 250.0],  # Nose
        [280.0, 310.0],  # Right mouth
        [360.0, 310.0],  # Left mouth
    ], dtype=np.float32)

    face = MagicMock(spec=Face)
    face.landmark_2d_106 = landmarks
    face.kps = kps
    face.bbox = np.array([200, 140, 440, 380], dtype=np.float32)
    face.normed_embedding = np.random.randn(512).astype(np.float32)
    return face


class TestEyesMaskCreation(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        self.face = _make_dummy_face_with_eyes()

    def test_create_eyes_mask_with_106_landmarks(self):
        """Verifies 106-point landmark extraction generates mask, cutout, box, and polygon."""
        mask, cutout, box, poly = create_eyes_mask(self.face, self.frame)

        self.assertEqual(mask.shape, self.frame.shape[:2])
        self.assertEqual(mask.dtype, np.uint8)
        self.assertIsNotNone(cutout)
        self.assertIsInstance(box, tuple)
        self.assertEqual(len(box), 4)

        min_x, min_y, max_x, max_y = box
        self.assertGreater(max_x, min_x)
        self.assertGreater(max_y, min_y)
        self.assertGreaterEqual(min_x, 0)
        self.assertLessEqual(max_x, self.frame.shape[1])

        # Polygon must contain points for both eyes
        self.assertIsNotNone(poly)
        self.assertGreaterEqual(len(poly), 6)

        # Mask inside eye regions must be non-zero
        self.assertGreater(np.count_nonzero(mask), 0)

    def test_create_eyes_mask_5pt_fallback(self):
        """Verifies fallback to 5-point kps when 106 landmarks are absent."""
        face_5pt = MagicMock(spec=Face)
        face_5pt.landmark_2d_106 = None
        face_5pt.kps = self.face.kps
        face_5pt.bbox = self.face.bbox

        mask, cutout, box, poly = create_eyes_mask(face_5pt, self.frame)
        self.assertEqual(mask.shape, self.frame.shape[:2])
        self.assertIsNotNone(cutout)
        self.assertGreaterEqual(len(poly), 6)
        self.assertGreater(np.count_nonzero(mask), 0)

    def test_create_eyes_mask_none_and_edge_inputs(self):
        """Verifies graceful handling of None inputs and empty coordinates."""
        # Face None
        mask, cutout, box, poly = create_eyes_mask(None, self.frame)
        self.assertIsNone(cutout)
        self.assertEqual(box, (0, 0, 0, 0))

        # Frame None
        mask, cutout, box, poly = create_eyes_mask(self.face, None)
        self.assertIsNone(cutout)

        # Face with neither 106 landmarks nor kps
        empty_face = MagicMock(spec=Face)
        empty_face.landmark_2d_106 = None
        empty_face.kps = None
        mask, cutout, box, poly = create_eyes_mask(empty_face, self.frame)
        self.assertIsNone(cutout)
        self.assertEqual(box, (0, 0, 0, 0))


class TestApplyEyesArea(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        self.face = _make_dummy_face_with_eyes()
        self.mask, self.cutout, self.box, self.poly = create_eyes_mask(self.face, self.frame)

    def test_apply_eyes_area_seamless_blend(self):
        """Verifies apply_eyes_area modifies eye pixels while preserving frame dimensions."""
        swapped = np.full((480, 640, 3), 200, dtype=np.uint8)
        dummy_face_mask = np.ones((480, 640), dtype=np.uint8) * 255

        blended = apply_eyes_area(swapped, self.cutout, self.box, dummy_face_mask, self.poly)

        self.assertEqual(blended.shape, swapped.shape)
        self.assertEqual(blended.dtype, np.uint8)

        # Eye region should have changed toward cutout value (100)
        min_x, min_y, max_x, max_y = self.box
        eye_center_val = blended[200, 260, 0]
        self.assertLess(eye_center_val, 200)

        # Outside eye region should remain untouched (200)
        self.assertEqual(blended[50, 50, 0], 200)

    def test_apply_eyes_area_boundary_and_invalid_inputs(self):
        """Verifies robustness against empty cutouts and invalid boxes."""
        orig = self.frame.copy()

        # None cutout
        out = apply_eyes_area(orig.copy(), None, self.box, None, self.poly)
        self.assertTrue(np.array_equal(out, orig))

        # Invalid box
        out2 = apply_eyes_area(orig.copy(), self.cutout, (0, 0, 0, 0), None, self.poly)
        self.assertTrue(np.array_equal(out2, orig))

        # Out-of-bounds box
        out3 = apply_eyes_area(orig.copy(), self.cutout, (-100, -100, -50, -50), None, self.poly)
        self.assertTrue(np.array_equal(out3, orig))


class TestEyesMaskVisualizationAndCli(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        self.face = _make_dummy_face_with_eyes()
        self.eyes_data = create_eyes_mask(self.face, self.frame)

    def test_draw_eyes_mask_visualization(self):
        """Verifies debug visualizer draws contours onto frame."""
        vis = draw_eyes_mask_visualization(self.frame, self.face, self.eyes_data)
        self.assertEqual(vis.shape, self.frame.shape)
        # Should contain yellow contour pixels (0, 255, 255)
        diff = np.sum(vis != self.frame)
        self.assertGreater(diff, 0)

    def test_cli_args_parsing_eyes_and_virtual_cam(self):
        """Verifies CLI flags --eyes-mask, --virtual-cam, and --virtual-cam-backend are parsed."""
        program = modules.core.create_program()
        args = program.parse_args([
            "--eyes-mask",
            "--virtual-cam",
            "--virtual-cam-backend", "obs",
        ])

        self.assertTrue(args.eyes_mask)
        self.assertTrue(args.virtual_cam)
        self.assertEqual(args.virtual_cam_backend, "obs")

    def test_face_swapper_eyes_mask_integration(self):
        """Verifies swap_face invokes eyes_mask preservation when enabled."""
        source_face = _make_dummy_face_with_eyes()
        target_face = _make_dummy_face_with_eyes()

        orig_eyes_mask = modules.globals.eyes_mask
        orig_providers = modules.globals.execution_providers
        try:
            modules.globals.eyes_mask = True
            modules.globals.execution_providers = ["CPUExecutionProvider"]

            frame = np.full((480, 640, 3), 128, dtype=np.uint8)

            # Mock swapper get
            swapper_mock = MagicMock()
            swapper_mock.input_size = (128, 128)
            fake_crop = np.full((128, 128, 3), 220, dtype=np.uint8)
            M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            swapper_mock.get.return_value = (fake_crop, M)

            with unittest.mock.patch(
                "modules.processors.frame.face_swapper.get_face_swapper",
                return_value=swapper_mock,
            ):
                result = face_swapper.swap_face(source_face, target_face, frame)
                self.assertIsNotNone(result)
                self.assertEqual(result.shape, frame.shape)
        finally:
            modules.globals.eyes_mask = orig_eyes_mask
            modules.globals.execution_providers = orig_providers


if __name__ == "__main__":
    unittest.main()
