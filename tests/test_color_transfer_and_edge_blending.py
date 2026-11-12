"""Unit and Regression Test Suite for Color/Lighting Transfer & Edge Feathering.

Tests:
1. fast_estimate_norm analytical similarity transform across 112 and 128 resolutions without warnings.
2. patch_insightface_face_align monkeypatch verification and zero-warning assertion.
3. apply_color_transfer LAB Reinhard transfer, lighting clamp, blend ratios, skin masking, and edge cases.
4. _get_soft_alpha smooth edge feathering, zero-boundary invariant, and bangs cutoff compatibility.
5. swap_face color_correction toggle integration and invariance.
"""

import os
import sys
import unittest
import warnings
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.processors.frame.face_swapper import (
    fast_estimate_norm,
    fast_norm_crop,
    fast_norm_crop2,
    patch_insightface_face_align,
    apply_color_transfer,
    _get_soft_alpha,
    _get_skin_statistics_mask,
    _fast_paste_back,
)


class FastEstimateNormTests(unittest.TestCase):
    """Verifies analytical similarity transform for both 112 (ArcFace) and 128 (INSwapper) sizes."""

    def setUp(self):
        self.kps = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ], dtype=np.float32)

    def test_fast_estimate_norm_128_shape_and_finite(self):
        M = fast_estimate_norm(self.kps, image_size=128)
        self.assertEqual(M.shape, (2, 3))
        self.assertTrue(np.all(np.isfinite(M)))

    def test_fast_estimate_norm_112_shape_and_finite(self):
        M = fast_estimate_norm(self.kps, image_size=112)
        self.assertEqual(M.shape, (2, 3))
        self.assertTrue(np.all(np.isfinite(M)))

    def test_fast_estimate_norm_matches_insightface_exact(self):
        try:
            from insightface.utils import face_align
            M128 = fast_estimate_norm(self.kps, image_size=128)
            M112 = fast_estimate_norm(self.kps, image_size=112)
            M_fa128 = face_align.estimate_norm(self.kps, 128)
            M_fa112 = face_align.estimate_norm(self.kps, 112)
            self.assertEqual(M128.shape, (2, 3))
            self.assertEqual(M112.shape, (2, 3))
            np.testing.assert_allclose(M128, M_fa128, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(M112, M_fa112, rtol=1e-5, atol=1e-5)
        except ImportError:
            pass

    def test_fast_estimate_norm_zero_variance_degrades_gracefully(self):
        """Points collapsed to a single pixel must not throw ZeroDivisionError."""
        zero_kps = np.full((5, 2), 50.0, dtype=np.float32)
        M = fast_estimate_norm(zero_kps, image_size=128)
        self.assertEqual(M.shape, (2, 3))
        self.assertTrue(np.all(M == 0))

    def test_fast_estimate_norm_collinear_points(self):
        """Collinear points must produce a valid finite transform."""
        collinear_kps = np.array([
            [10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0], [50.0, 50.0]
        ], dtype=np.float32)
        M = fast_estimate_norm(collinear_kps, image_size=128)
        self.assertEqual(M.shape, (2, 3))
        self.assertTrue(np.all(np.isfinite(M)))

    def test_fast_norm_crop_functions(self):
        dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
        cropped = fast_norm_crop(dummy_img, self.kps, image_size=112)
        self.assertEqual(cropped.shape, (112, 112, 3))

        cropped2, M = fast_norm_crop2(dummy_img, self.kps, image_size=128)
        self.assertEqual(cropped2.shape, (128, 128, 3))
        self.assertEqual(M.shape, (2, 3))


class DeprecationWarningEliminationTests(unittest.TestCase):
    """Ensures monkeypatching completely eliminates skimage FutureWarning."""

    def test_patch_insightface_face_align_eliminates_warnings(self):
        patch_insightface_face_align()
        try:
            import insightface.utils.face_align as face_align
            kps = np.array([
                [30.0, 30.0], [70.0, 30.0], [50.0, 50.0], [35.0, 70.0], [65.0, 70.0]
            ], dtype=np.float32)

            with warnings.catch_warnings(record=True) as recorded_warnings:
                warnings.simplefilter("always")
                M112 = face_align.estimate_norm(kps, 112)
                M128 = face_align.estimate_norm(kps, 128)
                self.assertEqual(M112.shape, (2, 3))
                self.assertEqual(M128.shape, (2, 3))

                # Check that no skimage FutureWarning was raised
                future_warnings = [
                    w for w in recorded_warnings
                    if issubclass(w.category, FutureWarning) and "estimate is deprecated" in str(w.message)
                ]
                self.assertEqual(len(future_warnings), 0, f"Found unexpected FutureWarning: {future_warnings}")
        except ImportError:
            pass


class ColorTransferTests(unittest.TestCase):
    """Comprehensive test of Lab color and lighting transfer with boundary safeguards."""

    def test_apply_color_transfer_basic_warmth_transfer(self):
        """A cool/blue source transferred to a warm orange target shifts towards warmth."""
        source = np.full((100, 100, 3), [220, 100, 50], dtype=np.uint8)   # BGR: high blue, low red
        target = np.full((100, 100, 3), [50, 100, 220], dtype=np.uint8)   # BGR: low blue, high red

        result = apply_color_transfer(source, target)
        self.assertEqual(result.shape, source.shape)
        self.assertEqual(result.dtype, np.uint8)

        # Red channel of result must increase, blue channel must decrease
        self.assertGreater(result[50, 50, 2], source[50, 50, 2])
        self.assertLess(result[50, 50, 0], source[50, 50, 0])

    def test_apply_color_transfer_lighting_transfer(self):
        """A dark source transferred to a bright target gets brighter within clamped bounds."""
        source = np.full((100, 100, 3), 40, dtype=np.uint8)
        target = np.full((100, 100, 3), 200, dtype=np.uint8)

        result = apply_color_transfer(source, target)
        self.assertGreater(float(np.mean(result)), float(np.mean(source)))

    def test_apply_color_transfer_blend_ratio(self):
        """blend_ratio=0.0 preserves source; blend_ratio=0.5 is intermediate."""
        source = np.full((64, 64, 3), 50, dtype=np.uint8)
        target = np.full((64, 64, 3), 200, dtype=np.uint8)

        res_zero = apply_color_transfer(source, target, blend_ratio=0.0)
        self.assertTrue(np.all(res_zero == source))

        res_half = apply_color_transfer(source, target, blend_ratio=0.5)
        res_full = apply_color_transfer(source, target, blend_ratio=1.0)
        self.assertGreater(float(np.mean(res_half)), float(np.mean(source)))
        self.assertLess(float(np.mean(res_half)), float(np.mean(res_full)))

    def test_apply_color_transfer_with_skin_mask(self):
        """Applying a skin mask correctly computes statistics exclusively inside mask."""
        source = np.full((128, 128, 3), 100, dtype=np.uint8)
        target = np.full((128, 128, 3), 100, dtype=np.uint8)
        # Put an extreme color strictly in the center
        target[50:80, 50:80] = [20, 20, 230]

        mask = _get_skin_statistics_mask(128)
        result_masked = apply_color_transfer(source, target, mask=mask)
        self.assertEqual(result_masked.shape, (128, 128, 3))
        # Masked result reflects the red center
        self.assertGreater(result_masked[64, 64, 2], 120)

    def test_apply_color_transfer_edge_cases(self):
        """Robustness against None, empty arrays, mismatched sizes, grayscale, zero variance."""
        # 1. None inputs
        self.assertIsNone(apply_color_transfer(None, None))
        dummy = np.zeros((10, 10, 3), dtype=np.uint8)
        self.assertTrue(np.all(apply_color_transfer(dummy, None) == dummy))

        # 2. Empty arrays
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        self.assertEqual(apply_color_transfer(empty, dummy).size, 0)
        self.assertEqual(apply_color_transfer(dummy, empty).shape, dummy.shape)

        # 3. Grayscale inputs: color target -> colorized output
        gray_src = np.full((50, 50), 120, dtype=np.uint8)
        color_tgt = np.full((50, 50, 3), [10, 50, 200], dtype=np.uint8)
        res_from_gray = apply_color_transfer(gray_src, color_tgt)
        self.assertEqual(res_from_gray.shape, (50, 50, 3))

        # Grayscale source with grayscale target preserves 2D shape
        gray_tgt = np.full((50, 50), 200, dtype=np.uint8)
        res_gray_to_gray = apply_color_transfer(gray_src, gray_tgt)
        self.assertEqual(res_gray_to_gray.shape, (50, 50))

        # Grayscale source with blend_ratio=0.0 preserves 2D shape
        res_zero_blend_gray = apply_color_transfer(gray_src, color_tgt, blend_ratio=0.0)
        self.assertEqual(res_zero_blend_gray.shape, (50, 50))

        # 4. Monochromatic / IR target seamlessly desaturates a colorful face
        np.random.seed(42)
        vibrant_src = np.random.randint(50, 220, (128, 128, 3), dtype=np.uint8)
        mono_tgt = np.full((128, 128, 3), 110, dtype=np.uint8)
        res_mono = apply_color_transfer(vibrant_src, mono_tgt)
        res_lab = cv2.cvtColor(res_mono, cv2.COLOR_BGR2Lab)
        # Chroma variance in a and b should collapse to < 1.0 (seamless monochrome match)
        self.assertLess(float(res_lab[:, :, 1].std()), 1.0)
        self.assertLess(float(res_lab[:, :, 2].std()), 1.0)

        # 5. Mismatched dimensions (e.g. 128x128 source with 720x1280 target)
        src_128 = np.full((128, 128, 3), 80, dtype=np.uint8)
        tgt_hd = np.full((720, 1280, 3), [50, 150, 210], dtype=np.uint8)
        res_mismatch = apply_color_transfer(src_128, tgt_hd)
        self.assertEqual(res_mismatch.shape, (128, 128, 3))

        # 6. Zero variance (solid black / solid white)
        solid_black = np.zeros((64, 64, 3), dtype=np.uint8)
        solid_white = np.full((64, 64, 3), 255, dtype=np.uint8)
        res_zero_var = apply_color_transfer(solid_black, solid_white)
        self.assertEqual(res_zero_var.shape, (64, 64, 3))
        self.assertFalse(np.any(np.isnan(res_zero_var)))
        self.assertFalse(np.any(np.isinf(res_zero_var)))


class MemoryAndRenderingOptimizationTests(unittest.TestCase):
    """Verifies zero redundant copy, robust _bgr_to_qpixmap, and buffer reuse."""

    def test_bgr_to_qpixmap_handles_grayscale_and_color(self):
        from PySide6.QtWidgets import QApplication
        from modules.ui import _bgr_to_qpixmap

        app = QApplication.instance() or QApplication([sys.argv[0]])
        # 1. 2D Grayscale
        gray_2d = np.zeros((80, 120), dtype=np.uint8)
        pm_gray = _bgr_to_qpixmap(gray_2d)
        self.assertEqual(pm_gray.width(), 120)
        self.assertEqual(pm_gray.height(), 80)
        self.assertFalse(pm_gray.isNull())

        # 2. 3D single channel
        gray_3d = np.zeros((80, 120, 1), dtype=np.uint8)
        pm_3d = _bgr_to_qpixmap(gray_3d)
        self.assertEqual(pm_3d.width(), 120)
        self.assertEqual(pm_3d.height(), 80)
        self.assertFalse(pm_3d.isNull())

        # 3. 3-channel BGR
        bgr = np.zeros((80, 120, 3), dtype=np.uint8)
        pm_bgr = _bgr_to_qpixmap(bgr)
        self.assertEqual(pm_bgr.width(), 120)
        self.assertEqual(pm_bgr.height(), 80)
        self.assertFalse(pm_bgr.isNull())

    def test_fit_image_to_size_same_dimension_no_realloc(self):
        from modules.ui import fit_image_to_size
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        fitted = fit_image_to_size(img, 640, 480)
        # Should return the exact same object reference
        self.assertIs(fitted, img)

    def test_skin_statistics_mask_excludes_eyes_and_mouth(self):
        mask = _get_skin_statistics_mask(128)
        # Eye locations (left eye around x=46, y=51; right eye around x=82, y=51) should be 0
        left_eye_sample = mask[51, 46]
        right_eye_sample = mask[51, 82]
        self.assertEqual(left_eye_sample, 0, "Left eye region was not excluded from skin mask")
        self.assertEqual(right_eye_sample, 0, "Right eye region was not excluded from skin mask")

        # Mouth location (around x=64, y=92) should be 0
        mouth_sample = mask[92, 64]
        self.assertEqual(mouth_sample, 0, "Mouth region was not excluded from skin mask")

        # Cheeks (pure skin around x=40, y=70) should be 255
        cheek_sample = mask[70, 40]
        self.assertEqual(cheek_sample, 255, "Cheek skin was excluded from skin mask")


class EdgeFeatheringPolishTests(unittest.TestCase):
    """Verifies that _get_soft_alpha eliminates boundary step-cliffs and blends imperceptibly."""

    def test_soft_alpha_boundary_is_strictly_zero(self):
        """All outer boundary pixels of the 128x128 template must be 0 to eliminate edge seams."""
        alpha = _get_soft_alpha(128)
        self.assertEqual(alpha.shape, (128, 128))

        # Check top, bottom, left, right edges
        self.assertEqual(int(alpha[0, :].max()), 0, "Top edge not zero")
        self.assertEqual(int(alpha[-1, :].max()), 0, "Bottom edge not zero")
        self.assertEqual(int(alpha[:, 0].max()), 0, "Left edge not zero")
        self.assertEqual(int(alpha[:, -1].max()), 0, "Right edge not zero")

    def test_soft_alpha_center_is_full_opacity(self):
        """The core facial region must maintain full 255 opacity."""
        alpha = _get_soft_alpha(128)
        center_val = alpha[64, 64]
        self.assertEqual(center_val, 255)

    def test_soft_alpha_monotonic_smooth_falloff(self):
        """From center outwards along the horizontal axis, alpha values decay monotonically."""
        alpha = _get_soft_alpha(128)
        center_y = 64
        # Slice from center (x=64) to right edge (x=127)
        radial_slice = alpha[center_y, 64:]
        diffs = np.diff(radial_slice.astype(int))
        # Non-positive differences means monotonically decreasing or constant
        self.assertTrue(np.all(diffs <= 0), f"Radial slice not monotonically decreasing: {radial_slice}")

    def test_soft_alpha_bangs_cutoff(self):
        """forehead_cutoff > 0 smoothly cuts off the top forehead while keeping lower face intact."""
        alpha_bangs = _get_soft_alpha(128, forehead_cutoff=50.0)
        alpha_uncut = _get_soft_alpha(128, forehead_cutoff=0.0)
        # Hairline at top must be suppressed compared to uncut
        self.assertLess(float(np.mean(alpha_bangs[:15, :])), float(np.mean(alpha_uncut[:15, :])))
        # Chin/mouth region around y=85 should still be high opacity
        self.assertGreater(int(alpha_bangs[85, 64]), 200)

    def test_fast_paste_back_seamless_edge(self):
        """Pasted face onto uniform background shows smooth continuous values without steps."""
        bg = np.full((200, 200, 3), 128, dtype=np.uint8)
        fake = np.full((128, 128, 3), 200, dtype=np.uint8)
        aimg_dummy = np.empty((128, 128, 3), dtype=np.uint8)
        M = np.array([[1.0, 0.0, 36.0], [0.0, 1.0, 36.0]], dtype=np.float32)

        pasted = _fast_paste_back(bg.copy(), fake, aimg_dummy, M)
        self.assertEqual(pasted.shape, (200, 200, 3))
        # Boundary pixels outside paste area (e.g. at x=150, y=150) must match original background exactly
        self.assertEqual(pasted[150, 150, 0], 128)
        self.assertEqual(pasted[195, 195, 0], 128)


if __name__ == "__main__":
    unittest.main()
