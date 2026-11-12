"""Unit and regression tests for Eyebrows & Brow Expression Preservation and Pipeline Optimization."""

import os
import sys
import unittest
from unittest.mock import MagicMock

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import modules.globals
from modules.processors.frame.face_masking import (
    create_eyebrows_mask,
    apply_eyebrows_area,
    draw_eyebrows_mask_visualization,
    create_curved_eyebrow,
)
from modules.face_analyser import _needs_landmark
from modules.core import create_program


class EyebrowsMaskCreationTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        self.face = MagicMock()
        # Synthetic 106 landmarks: left eyebrow 97-104, right eyebrow 43-50
        lmk = np.zeros((106, 2), dtype=np.float32)
        # Left eyebrow points
        lmk[97:105] = np.array([
            [200, 150], [215, 142], [230, 138], [245, 138],
            [260, 142], [250, 145], [230, 145], [210, 148]
        ], dtype=np.float32)
        # Right eyebrow points
        lmk[43:51] = np.array([
            [300, 142], [315, 138], [330, 138], [345, 142],
            [360, 150], [350, 148], [330, 145], [310, 145]
        ], dtype=np.float32)
        self.face.landmark_2d_106 = lmk

    def test_create_eyebrows_mask_with_106_landmarks(self):
        """Verifies eyebrows mask generates valid non-empty mask, cutout, box, and polygon."""
        mask, cutout, box, polygon = create_eyebrows_mask(self.face, self.frame)
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, self.frame.shape[:2])
        self.assertIsNotNone(cutout)
        self.assertEqual(cutout.shape[2], 3)
        self.assertIsNotNone(polygon)
        self.assertGreaterEqual(len(polygon), 6)

        min_x, min_y, max_x, max_y = box
        self.assertGreater(max_x, min_x)
        self.assertGreater(max_y, min_y)

    def test_create_eyebrows_mask_none_and_edge_inputs(self):
        """Verifies eyebrows mask handles missing landmarks or None face gracefully."""
        dummy_face = MagicMock()
        dummy_face.landmark_2d_106 = None
        mask, cutout, box, polygon = create_eyebrows_mask(dummy_face, self.frame)
        self.assertIsNotNone(mask)
        self.assertIsNone(cutout)
        self.assertEqual(box, (0, 0, 0, 0))

    def test_create_curved_eyebrow_algorithm(self):
        """Verifies curved eyebrow interpolation creates smooth closed contour."""
        pts = np.array([[10, 20], [20, 15], [30, 14], [40, 16], [50, 22]], dtype=np.float32)
        curved = create_curved_eyebrow(pts)
        self.assertGreater(len(curved), len(pts))
        self.assertEqual(curved.shape[1], 2)


class ApplyEyebrowsAreaTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((300, 300, 3), 100, dtype=np.uint8)
        self.cutout = np.full((50, 120, 3), 220, dtype=np.uint8)
        self.box = (40, 30, 160, 80)
        self.polygon = np.array([
            [50, 40], [100, 35], [150, 42], [140, 60], [100, 58], [60, 62]
        ], dtype=np.float32)
        self.face_mask = np.full((300, 300), 255, dtype=np.uint8)

    def test_apply_eyebrows_area_blending(self):
        """Verifies apply_eyebrows_area blends cutout into target frame without boundary artifacts."""
        out = apply_eyebrows_area(self.frame.copy(), self.cutout, self.box, self.face_mask, self.polygon)
        self.assertEqual(out.shape, self.frame.shape)
        # Inside the polygon, pixel values should reflect blended cutout (>100)
        self.assertGreater(float(np.mean(out[45:55, 90:110])), 110.0)

    def test_apply_eyebrows_area_boundary_and_invalid_inputs(self):
        """Verifies apply_eyebrows_area survives empty/degenerate inputs gracefully."""
        orig = self.frame.copy()
        # None inputs
        self.assertTrue(np.array_equal(apply_eyebrows_area(orig, None, self.box, self.face_mask, self.polygon), orig))
        # Empty polygon
        self.assertTrue(np.array_equal(apply_eyebrows_area(orig, self.cutout, self.box, self.face_mask, np.empty((0, 2))), orig))
        # Inverted box
        self.assertTrue(np.array_equal(apply_eyebrows_area(orig, self.cutout, (100, 100, 50, 50), self.face_mask, self.polygon), orig))


class EyebrowsVisualizationAndAnalysisTests(unittest.TestCase):
    def test_draw_eyebrows_mask_visualization(self):
        """Verifies visualization function draws contour overlays without crashing."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        face = MagicMock()
        mask_data = (
            np.zeros((200, 200), dtype=np.uint8),
            np.zeros((40, 80, 3), dtype=np.uint8),
            (20, 20, 100, 60),
            np.array([[25, 25], [50, 20], [75, 25], [70, 40], [50, 40], [30, 40]], dtype=np.float32),
        )
        vis = draw_eyebrows_mask_visualization(frame, face, mask_data)
        self.assertIsNotNone(vis)
        self.assertEqual(vis.shape, frame.shape)
        # Canvas should now contain drawn polylines (>0)
        self.assertGreater(float(np.sum(vis)), 0)

    def test_needs_landmark_recognizes_eyebrows_mask(self):
        """Verifies _needs_landmark returns True when eyebrows_mask is active."""
        orig_eyebrows = getattr(modules.globals, "eyebrows_mask", False)
        orig_eyes = getattr(modules.globals, "eyes_mask", False)
        orig_glasses = getattr(modules.globals, "glasses_mask", False)
        orig_mouth = getattr(modules.globals, "mouth_mask", False)
        orig_bangs = getattr(modules.globals, "bangs_mask", False)
        orig_proc = list(getattr(modules.globals, "frame_processors", []))

        try:
            modules.globals.mouth_mask = False
            modules.globals.mouth_mask_size = 0.0
            modules.globals.glasses_mask = False
            modules.globals.glasses_mask_size = 0.0
            modules.globals.bangs_mask = False
            modules.globals.forehead_cutoff = 0.0
            modules.globals.eyes_mask = False
            modules.globals.eyes_mask_size = 0.0
            modules.globals.frame_processors = []

            modules.globals.eyebrows_mask = False
            self.assertFalse(_needs_landmark())

            modules.globals.eyebrows_mask = True
            self.assertTrue(_needs_landmark())
        finally:
            modules.globals.eyebrows_mask = orig_eyebrows
            modules.globals.eyes_mask = orig_eyes
            modules.globals.glasses_mask = orig_glasses
            modules.globals.mouth_mask = orig_mouth
            modules.globals.bangs_mask = orig_bangs
            modules.globals.frame_processors = orig_proc

    def test_core_cli_arguments_registration(self):
        """Verifies CLI parser registers --eyebrows-mask and --benchmark flags."""
        program = create_program()
        args = program.parse_args(["--eyebrows-mask", "--benchmark"])
        self.assertTrue(args.eyebrows_mask)
        self.assertTrue(args.benchmark)


if __name__ == "__main__":
    unittest.main()
