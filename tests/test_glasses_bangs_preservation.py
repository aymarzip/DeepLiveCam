"""
Unit and regression tests for:
1. Glasses preservation / occlusion-aware masking (create_glasses_mask, apply_glasses_area, draw_glasses_mask_visualization).
2. Bangs / fringe forehead hair preservation (create_bangs_mask, apply_bangs_area, _get_soft_alpha cutoff).
3. 20-30 FPS live preview pipeline cadence and asynchronous display decoupling.
4. Full hardware resource utilization (all CPU cores/threads without artificial throttling).
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import modules.globals
import modules.core
from modules.processors.frame.face_masking import (
    create_glasses_mask,
    apply_glasses_area,
    draw_glasses_mask_visualization,
    create_bangs_mask,
    apply_bangs_area,
    draw_bangs_mask_visualization,
)
import modules.processors.frame.face_swapper as face_swapper
import modules.face_analyser as face_analyser
import modules.processors.frame.face_enhancer as face_enhancer


def _make_dummy_106_landmarks():
    """Generate realistic 106 facial landmark layout on a 400x400 face."""
    lm = np.zeros((106, 2), dtype=np.float32)
    # Chin / jawline outline (0 to 32): 0=right temple, 16=bottom of chin, 32=left temple
    for i in range(33):
        angle = -np.pi / 6.0 + (7.0 * np.pi / 6.0) * (i / 32.0)
        lm[i] = [200 + 100 * np.cos(angle), 200 + 120 * np.sin(angle)]
    # Right eye (33 to 42): centered around (140, 170)
    for i in range(33, 43):
        a = 2 * np.pi * (i - 33) / 10.0
        lm[i] = [140 + 20 * np.cos(a), 170 + 10 * np.sin(a)]
    # Right eyebrow (43 to 51): above right eye, y ~ 140
    for i in range(43, 52):
        lm[i] = [115 + (i - 43) * 6.0, 142 - 4 * np.sin(np.pi * (i - 43) / 8.0)]
    # Left eye (87 to 96): centered around (260, 170)
    for i in range(87, 97):
        a = 2 * np.pi * (i - 87) / 10.0
        lm[i] = [260 + 20 * np.cos(a), 170 + 10 * np.sin(a)]
    # Left eyebrow (97 to 105): above left eye, y ~ 140
    for i in range(97, 106):
        lm[i] = [235 + (i - 97) * 6.0, 142 - 4 * np.sin(np.pi * (i - 97) / 8.0)]
    # Nose bridge (72 to 75): vertical between eyes (200, 150-180)
    for i in range(72, 76):
        lm[i] = [200, 150 + (i - 72) * 10.0]
    # Nose tip (76 to 86)
    for i in range(76, 87):
        lm[i] = [200 + (i - 81) * 3.0, 200]
    # Mouth (52 to 71)
    for i in range(52, 72):
        a = 2 * np.pi * (i - 52) / 20.0
        lm[i] = [200 + 30 * np.cos(a), 260 + 15 * np.sin(a)]
    return lm


class TestGlassesMaskPreservation(unittest.TestCase):
    """Verifies glasses occlusion-aware masking, extraction, and blending."""

    def setUp(self):
        self.frame = np.full((400, 400, 3), 180, dtype=np.uint8)
        # Draw simulated dark glasses frames on original frame
        cv2.circle(self.frame, (140, 170), 25, (20, 20, 20), 4)
        cv2.circle(self.frame, (260, 170), 25, (20, 20, 20), 4)
        cv2.line(self.frame, (165, 168), (235, 168), (20, 20, 20), 4)

        self.face = MagicMock()
        self.face.landmark_2d_106 = _make_dummy_106_landmarks()
        self.face.kps = np.array([
            [140.0, 170.0],
            [260.0, 170.0],
            [200.0, 200.0],
            [170.0, 260.0],
            [230.0, 260.0],
        ], dtype=np.float32)
        self.face.bbox = np.array([80, 100, 320, 340], dtype=np.float32)

    def test_create_glasses_mask_with_106_landmarks(self):
        mask, cutout, box, polygon = create_glasses_mask(self.face, self.frame)
        self.assertEqual(mask.shape, self.frame.shape[:2])
        self.assertEqual(mask.dtype, np.uint8)
        self.assertIsNotNone(cutout)
        self.assertIsNotNone(polygon)
        self.assertGreaterEqual(len(polygon), 3)

        min_x, min_y, max_x, max_y = box
        self.assertGreater(max_x, min_x)
        self.assertGreater(max_y, min_y)
        self.assertGreaterEqual(min_x, 0)
        self.assertGreaterEqual(min_y, 0)
        self.assertLessEqual(max_x, self.frame.shape[1])
        self.assertLessEqual(max_y, self.frame.shape[0])
        self.assertEqual(cutout.shape, (max_y - min_y, max_x - min_x, 3))
        # Mask inside the glasses box should have positive alpha
        self.assertGreater(np.max(mask[min_y:max_y, min_x:max_x]), 0)

    def test_create_glasses_mask_5pt_fallback(self):
        # Remove 106 landmarks, verify 5-point fallback
        self.face.landmark_2d_106 = None
        mask, cutout, box, polygon = create_glasses_mask(self.face, self.frame)
        self.assertEqual(mask.shape, self.frame.shape[:2])
        self.assertIsNotNone(cutout)
        self.assertIsNotNone(polygon)
        self.assertGreaterEqual(len(polygon), 3)

    def test_create_glasses_mask_none_and_edge_inputs(self):
        # Empty inputs must not crash
        mask, cutout, box, polygon = create_glasses_mask(None, self.frame)
        self.assertEqual(box, (0, 0, 0, 0))
        self.assertIsNone(cutout)

        mask2, cutout2, box2, polygon2 = create_glasses_mask(self.face, None)
        self.assertIsNone(cutout2)

        empty_face = MagicMock()
        empty_face.landmark_2d_106 = None
        empty_face.kps = None
        mask3, cutout3, box3, polygon3 = create_glasses_mask(empty_face, self.frame)
        self.assertEqual(box3, (0, 0, 0, 0))

    def test_apply_glasses_area_seamless_blend(self):
        swapped_frame = np.full((400, 400, 3), 255, dtype=np.uint8)
        mask, cutout, box, polygon = create_glasses_mask(self.face, self.frame)
        face_mask = np.full((400, 400), 255, dtype=np.uint8)

        blended = apply_glasses_area(swapped_frame, cutout, box, face_mask, polygon)
        self.assertEqual(blended.shape, swapped_frame.shape)
        self.assertEqual(blended.dtype, np.uint8)

        # The glasses center region should have pixels from the original frame (not all 255 swapped)
        min_x, min_y, max_x, max_y = box
        center_x, center_y = (min_x + max_x) // 2, (min_y + max_y) // 2
        # Near glasses bridge/lenses, original dark pixel values should be preserved
        self.assertLess(np.mean(blended[center_y - 10:center_y + 10, center_x - 10:center_x + 10]), 250)

    def test_apply_glasses_area_boundary_and_invalid_inputs(self):
        # Should return frame without raising on bad/empty inputs
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        res = apply_glasses_area(frame, None, (0, 0, 0, 0), None, None)
        self.assertEqual(res.shape, frame.shape)

        res2 = apply_glasses_area(frame, np.zeros((10, 10, 3), dtype=np.uint8), (-50, -50, -10, -10), None, np.array([[0,0],[1,1],[2,2]]))
        self.assertEqual(res2.shape, frame.shape)

    def test_draw_glasses_mask_visualization(self):
        mask, cutout, box, polygon = create_glasses_mask(self.face, self.frame)
        vis = draw_glasses_mask_visualization(self.frame, self.face, (mask, cutout, box, polygon))
        self.assertEqual(vis.shape, self.frame.shape)
        self.assertEqual(vis.dtype, np.uint8)


class TestBangsFringePreservation(unittest.TestCase):
    """Verifies bangs / fringe hair preservation on forehead."""

    def setUp(self):
        self.frame = np.full((400, 400, 3), 200, dtype=np.uint8)
        # Draw simulated dark bangs/fringe hair across upper forehead
        self.frame[80:140, 100:300] = [30, 20, 15]

        self.face = MagicMock()
        self.face.landmark_2d_106 = _make_dummy_106_landmarks()
        self.face.kps = np.array([
            [140.0, 170.0],
            [260.0, 170.0],
            [200.0, 200.0],
            [170.0, 260.0],
            [230.0, 260.0],
        ], dtype=np.float32)
        self.face.bbox = np.array([80, 100, 320, 340], dtype=np.float32)

    def test_create_bangs_mask_with_106_landmarks(self):
        modules.globals.forehead_cutoff = 40.0
        mask, cutout, box, polygon = create_bangs_mask(self.face, self.frame)
        self.assertEqual(mask.shape, self.frame.shape[:2])
        self.assertIsNotNone(cutout)
        self.assertIsNotNone(polygon)
        self.assertGreaterEqual(len(polygon), 3)

        min_x, min_y, max_x, max_y = box
        self.assertGreater(max_x, min_x)
        self.assertGreater(max_y, min_y)
        # Bangs mask must be in the upper half of the face (above eyebrows)
        eyebrow_y = np.mean(self.face.landmark_2d_106[43:52, 1])
        self.assertLessEqual(min_y, eyebrow_y)

    def test_create_bangs_mask_5pt_fallback(self):
        self.face.landmark_2d_106 = None
        mask, cutout, box, polygon = create_bangs_mask(self.face, self.frame)
        self.assertEqual(mask.shape, self.frame.shape[:2])
        self.assertIsNotNone(cutout)
        self.assertIsNotNone(polygon)

    def test_apply_bangs_area_blend(self):
        swapped_frame = np.full((400, 400, 3), 255, dtype=np.uint8)
        mask, cutout, box, polygon = create_bangs_mask(self.face, self.frame)
        face_mask = np.full((400, 400), 255, dtype=np.uint8)

        blended = apply_bangs_area(swapped_frame, cutout, box, face_mask, polygon)
        self.assertEqual(blended.shape, swapped_frame.shape)

        # Upper forehead should have preserved dark bangs hair
        min_x, min_y, max_x, max_y = box
        forehead_sample = blended[min_y + 10:min_y + 30, min_x + 20:max_x - 20]
        if forehead_sample.size > 0:
            self.assertLess(np.mean(forehead_sample), 240)

    def test_soft_alpha_forehead_cutoff(self):
        """Verifies _get_soft_alpha zeroes out the top forehead when cutoff > 0."""
        # Uncut alpha
        alpha_uncut = face_swapper._get_soft_alpha(128, forehead_cutoff=0.0)
        # Cutoff alpha (e.g. 50%)
        alpha_cut = face_swapper._get_soft_alpha(128, forehead_cutoff=50.0)

        # Top 15 rows of alpha_cut should be zero (hairline completely suppressed)
        top_cut_mean = np.mean(alpha_cut[:15, :])
        top_uncut_mean = np.mean(alpha_uncut[:15, :])

        self.assertLess(top_cut_mean, top_uncut_mean)
        self.assertEqual(float(np.min(alpha_cut[:10, :])), 0.0)

    def test_draw_bangs_mask_visualization(self):
        mask, cutout, box, polygon = create_bangs_mask(self.face, self.frame)
        vis = draw_bangs_mask_visualization(self.frame, self.face, (mask, cutout, box, polygon))
        self.assertEqual(vis.shape, self.frame.shape)


class TestResourceUtilizationAndPipeline(unittest.TestCase):
    """Verifies full resource utilization and live preview settings."""

    def test_suggest_execution_threads_utilizes_full_machine(self):
        """Must return cpu_count without capping at 16 or subtracting 2."""
        threads = modules.core.suggest_execution_threads()
        cpu_count = os.cpu_count() or 4
        self.assertEqual(threads, max(4, cpu_count))
        self.assertGreaterEqual(threads, 4)

    def test_globals_default_swap_interval_is_accelerated(self):
        """DEFAULT_SWAP_INTERVAL must be >= 2 for 20-30 FPS tracking cadence."""
        self.assertGreaterEqual(modules.globals.DEFAULT_SWAP_INTERVAL, 2)

    def test_ensure_landmarks_attaches_landmarks_for_masks(self):
        """When face lacks landmark_2d_106, ensure_landmarks runs and attaches it."""
        mock_face = MagicMock()
        mock_face.landmark_2d_106 = None
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        mock_lmk_model = MagicMock()
        def attach_lmk(fr, fc):
            fc.landmark_2d_106 = np.zeros((106, 2))
        mock_lmk_model.get.side_effect = attach_lmk

        mock_fa = MagicMock()
        mock_fa.models = {"landmark_2d_106": mock_lmk_model}

        with patch.object(face_analyser, "get_face_analyser", return_value=mock_fa):
            face_analyser.ensure_landmarks(frame, [mock_face])

        self.assertIsNotNone(mock_face.landmark_2d_106)
        self.assertEqual(mock_lmk_model.get.call_count, 1)

    def test_face_enhancer_thread_semaphore_uncapped(self):
        """THREAD_SEMAPHORE in face_enhancer must not be capped to 1 (mutex)."""
        cpu_count = os.cpu_count() or 1
        self.assertGreaterEqual(face_enhancer.THREAD_SEMAPHORE._value, max(1, cpu_count))

    def test_core_cli_args_parsing_glasses_and_bangs(self):
        """CLI arguments for glasses and bangs masking must parse into globals."""
        program = modules.core.create_program()
        args = program.parse_args(["--glasses-mask", "--bangs-mask", "--forehead-cutoff", "45.0"])
        self.assertTrue(args.glasses_mask)
        self.assertTrue(args.bangs_mask)
        self.assertEqual(args.forehead_cutoff, 45.0)

    def test_create_glasses_mask_default_size_covers_full_spectacles(self):
        """When glasses_mask is enabled with default size (0.0), coverage must enclose rims and bridge."""
        modules.globals.glasses_mask_size = 0.0
        face = MagicMock()
        face.landmark_2d_106 = _make_dummy_106_landmarks()
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        mask, cutout, box, poly = create_glasses_mask(face, frame)
        min_x, min_y, max_x, max_y = box
        # Box width should comfortably span both eyes plus frame rims
        self.assertGreater(max_x - min_x, 150)
        # Box height should cover eyebrows down below eyes
        self.assertGreater(max_y - min_y, 45)

    def test_poisson_blend_elliptical_mask_forehead_cutoff(self):
        """Poisson blend elliptical mask must respect forehead cutoff when bangs mask is active."""
        uncut = face_swapper._create_elliptical_mask((128, 128), forehead_cutoff=0.0)
        cut = face_swapper._create_elliptical_mask((128, 128), forehead_cutoff=60.0)
        self.assertLess(float(np.mean(cut[:16, :])), float(np.mean(uncut[:16, :])))

    def test_swap_face_ensures_landmarks_before_mouth_mask(self):
        """If target face has no 106 landmarks and mouth_mask is True, ensure_landmarks runs first."""
        target_face = MagicMock()
        target_face.landmark_2d_106 = None
        target_face.bbox = np.array([80, 100, 320, 340], dtype=np.float32)
        target_face.kps = np.array([[140, 170], [260, 170], [200, 200], [170, 260], [230, 260]], dtype=np.float32)

        source_face = MagicMock()
        source_face.normed_embedding = np.ones((512,), dtype=np.float32)

        frame = np.full((400, 400, 3), 128, dtype=np.uint8)
        modules.globals.mouth_mask = True
        modules.globals.mouth_mask_size = 10.0
        modules.globals.glasses_mask = False
        modules.globals.bangs_mask = False
        modules.globals.poisson_blend = False

        dummy_106 = _make_dummy_106_landmarks()
        def mock_ensure_landmarks(fr, faces):
            for fc in faces:
                fc.landmark_2d_106 = dummy_106

        with patch("modules.face_analyser.ensure_landmarks", side_effect=mock_ensure_landmarks) as mock_el, \
             patch.object(face_swapper, "get_face_swapper") as mock_get_swapper:
            mock_swapper = MagicMock()
            mock_swapper.input_size = (128, 128)
            mock_swapper.get.return_value = (np.zeros((128, 128, 3), dtype=np.uint8), np.eye(2, 3, dtype=np.float32))
            mock_get_swapper.return_value = mock_swapper
            res = face_swapper.swap_face(source_face, target_face, frame)
            self.assertEqual(mock_el.call_count, 1)
            self.assertIsNotNone(target_face.landmark_2d_106)


if __name__ == "__main__":
    unittest.main()
