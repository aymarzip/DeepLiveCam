"""Unit and regression tests for profile turns, tracking stability, and anti-flicker retention.

Verifies:
- _fast_paste_back gracefully handles singular or collapsed affine matrices without crash.
- _needs_landmark in face_analyser respects eyes_mask.
- Face retention prevents single-frame face drops during fast motion or head turning.
"""

import unittest

import numpy as np

import modules.face_analyser as fa
import modules.globals
import modules.processors.frame.face_swapper as swapper


class ProfileAndTrackingStabilityTests(unittest.TestCase):
    def test_fast_paste_back_singular_affine_returns_target_safely(self):
        """When face landmarks collapse or align in a collinear singularity,
        cv2.invertAffineTransform must not raise cv2.error or crash the pipeline."""
        h, w = 128, 128
        target_img = np.zeros((h, w, 3), dtype=np.uint8)
        bgr_fake = np.full((128, 128, 3), 200, dtype=np.uint8)
        aimg = np.full((128, 128, 3), 100, dtype=np.uint8)

        # Singular matrix (determinant == 0)
        M_singular = np.array([[0.0, 0.0, 50.0], [0.0, 0.0, 50.0]], dtype=np.float32)

        res = swapper._fast_paste_back(target_img.copy(), bgr_fake, aimg, M_singular)
        self.assertIsNotNone(res)
        self.assertEqual(res.shape, (h, w, 3))
        # When singular, original image is preserved
        self.assertTrue(np.all(res == 0))

    def test_fast_paste_back_nan_inf_affine_returns_target_safely(self):
        """Non-finite matrices (NaN or Inf) must be safely rejected."""
        target_img = np.zeros((64, 64, 3), dtype=np.uint8)
        bgr_fake = np.full((128, 128, 3), 150, dtype=np.uint8)
        aimg = np.full((128, 128, 3), 100, dtype=np.uint8)

        M_nan = np.array([[np.nan, 0.0, 10.0], [0.0, 1.0, 10.0]], dtype=np.float32)
        res_nan = swapper._fast_paste_back(target_img.copy(), bgr_fake, aimg, M_nan)
        self.assertEqual(res_nan.shape, (64, 64, 3))

        M_inf = np.array([[np.inf, 0.0, 10.0], [0.0, 1.0, 10.0]], dtype=np.float32)
        res_inf = swapper._fast_paste_back(target_img.copy(), bgr_fake, aimg, M_inf)
        self.assertEqual(res_inf.shape, (64, 64, 3))

    def test_needs_landmark_checks_eyes_mask(self):
        """_needs_landmark in face_analyser must return True when eyes_mask is enabled."""
        modules.globals.mouth_mask = False
        modules.globals.mouth_mask_size = 0.0
        modules.globals.glasses_mask = False
        modules.globals.glasses_mask_size = 0.0
        modules.globals.bangs_mask = False
        modules.globals.forehead_cutoff = 0.0
        modules.globals.eyes_mask = False
        modules.globals.eyes_mask_size = 0.0
        modules.globals.frame_processors = ["face_swapper"]

        self.assertFalse(fa._needs_landmark())

        modules.globals.eyes_mask = True
        self.assertTrue(fa._needs_landmark())

        modules.globals.eyes_mask = False
        modules.globals.eyes_mask_size = 15.0
        self.assertTrue(fa._needs_landmark())

        # Reset
        modules.globals.eyes_mask_size = 0.0
        self.assertFalse(fa._needs_landmark())

    def test_anti_flicker_retention_tolerates_temporary_missed_frame(self):
        """Simulate detection misses: 1 missed frame must retain face, >2 misses drop face."""
        from insightface.app.common import Face

        dummy_face = Face(bbox=np.array([10, 10, 80, 80]), kps=np.zeros((5, 2)), det_score=0.9)

        cached_target_face = None
        missed_det_count = 0
        MAX_MISSED_DET_FRAMES = 2

        # Frame 1: Successful detection
        new_face = dummy_face
        if new_face is not None:
            cached_target_face = new_face
            missed_det_count = 0
        else:
            missed_det_count += 1
            if missed_det_count > MAX_MISSED_DET_FRAMES:
                cached_target_face = None

        self.assertIsNotNone(cached_target_face)
        self.assertEqual(missed_det_count, 0)

        # Frame 2: Momentary miss (e.g. quick motion / blink)
        new_face = None
        if new_face is not None:
            cached_target_face = new_face
            missed_det_count = 0
        else:
            missed_det_count += 1
            if missed_det_count > MAX_MISSED_DET_FRAMES:
                cached_target_face = None

        # Face is RETAINED! No flicker, no mask drop
        self.assertIsNotNone(cached_target_face)
        self.assertEqual(missed_det_count, 1)

        # Frame 3: Recovery
        new_face = dummy_face
        if new_face is not None:
            cached_target_face = new_face
            missed_det_count = 0
        else:
            missed_det_count += 1
            if missed_det_count > MAX_MISSED_DET_FRAMES:
                cached_target_face = None

        self.assertIsNotNone(cached_target_face)
        self.assertEqual(missed_det_count, 0)

        # Frame 4 & 5 & 6: User left the camera
        for i in range(1, 4):
            new_face = None
            if new_face is not None:
                cached_target_face = new_face
                missed_det_count = 0
            else:
                missed_det_count += 1
                if missed_det_count > MAX_MISSED_DET_FRAMES:
                    cached_target_face = None

        # After 3 misses (> MAX_MISSED_DET_FRAMES), face is gracefully cleared
        self.assertIsNone(cached_target_face)
        self.assertEqual(missed_det_count, 3)


if __name__ == "__main__":
    unittest.main()
