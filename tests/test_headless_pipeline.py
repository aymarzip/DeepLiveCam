"""
Integration test for DeepLiveCam headless pipeline execution.
Verifies end-to-end face swap processing on a real test frame without Qt GUI,
ensuring face detection, ONNX swapper inference on DirectML, and feathered
paste-back blending operate correctly.
"""

import os
import sys
import unittest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import modules.globals
from modules.core import decode_execution_providers
from modules.face_analyser import (
    get_one_face,
    detect_one_face_fast,
    reset_face_analyser,
)
from modules.processors.frame.face_swapper import (
    get_face_swapper,
    swap_face,
    _fast_paste_back,
)
from modules import imread_unicode


class HeadlessPipelineIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Configure globals for headless DirectML test execution
        modules.globals.headless = True
        modules.globals.frame_processors = ["face_swapper"]
        modules.globals.execution_providers = decode_execution_providers(["dml"])
        modules.globals.det_size = 320
        modules.globals.mouth_mask = False
        modules.globals.poisson_blend = False
        modules.globals.opacity = 1.0

        reset_face_analyser()

        cls.source_img_path = os.path.join(
            PROJECT_ROOT, "tests/fixtures/source_face.jpg"
        )
        cls.target_img_path = os.path.join(
            PROJECT_ROOT, "tests/fixtures/target_face.jpg"
        )

        assert os.path.exists(cls.source_img_path), f"Missing {cls.source_img_path}"
        assert os.path.exists(cls.target_img_path), f"Missing {cls.target_img_path}"

        cls.source_img = imread_unicode(cls.source_img_path)
        assert cls.source_img is not None, "Failed to load source image"

        cls.target_frame = imread_unicode(cls.target_img_path)
        assert cls.target_frame is not None, "Failed to load target image"

    def test_source_face_analysis_produces_embedding(self):
        """Verifies get_one_face detects source face and extracts 512-D identity embedding."""
        source_face = get_one_face(self.source_img)
        self.assertIsNotNone(source_face, "Source face should be detected")
        self.assertTrue(hasattr(source_face, "bbox"), "Face should have bbox")
        self.assertTrue(hasattr(source_face, "kps"), "Face should have kps")
        self.assertTrue(
            hasattr(source_face, "normed_embedding"),
            "Face should have normed_embedding",
        )
        self.assertIsNotNone(source_face.normed_embedding)
        self.assertEqual(
            source_face.normed_embedding.shape,
            (512,),
            "ArcFace embedding must be 512-dimensional",
        )
        # Normalized embedding should have unit L2 norm (~1.0)
        norm = np.linalg.norm(source_face.normed_embedding)
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_target_face_fast_detection(self):
        """Verifies detect_one_face_fast detects target face without recognition overhead."""
        target_face = detect_one_face_fast(self.target_frame)
        self.assertIsNotNone(target_face, "Target face should be detected")
        self.assertTrue(hasattr(target_face, "bbox"), "Target face should have bbox")
        self.assertEqual(len(target_face.bbox), 4)
        self.assertTrue(hasattr(target_face, "kps"), "Target face should have kps")
        self.assertEqual(target_face.kps.shape, (5, 2))
        self.assertGreater(target_face.det_score, 0.5)

    def test_end_to_end_face_swap_modifies_face_region(self):
        """Verifies full end-to-end swap_face produces valid image with swapped face."""
        source_face = get_one_face(self.source_img)
        target_face = detect_one_face_fast(self.target_frame)

        orig_copy = self.target_frame.copy()
        swapped = swap_face(source_face, target_face, self.target_frame.copy())

        self.assertIsInstance(swapped, np.ndarray)
        self.assertEqual(swapped.shape, orig_copy.shape)
        self.assertEqual(swapped.dtype, np.uint8)

        # Swapping must modify pixels in the target image
        diff_count = int(np.sum(swapped != orig_copy))
        self.assertGreater(
            diff_count,
            1000,
            f"Face swap should modify face pixels, but only {diff_count} differed",
        )

        # Output pixels must stay valid uint8 [0, 255]
        self.assertGreaterEqual(swapped.min(), 0)
        self.assertLessEqual(swapped.max(), 255)

    def test_swap_with_none_face_returns_original(self):
        """Verifies edge case safety: missing faces return unmodified frame."""
        source_face = get_one_face(self.source_img)
        orig = self.target_frame.copy()

        # Target face None
        out_no_target = swap_face(source_face, None, orig.copy())
        self.assertTrue(np.array_equal(out_no_target, orig))

        # Source face None
        target_face = detect_one_face_fast(self.target_frame)
        out_no_source = swap_face(None, target_face, orig.copy())
        self.assertTrue(np.array_equal(out_no_source, orig))

    def test_fast_paste_back_blending(self):
        """Verifies low-level _fast_paste_back blends face crop correctly into frame."""
        swapper = get_face_swapper()
        self.assertIsNotNone(swapper, "Swapper should initialize")

        source_face = get_one_face(self.source_img)
        target_face = detect_one_face_fast(self.target_frame)

        with modules.globals.dml_lock:
            bgr_fake, M = swapper.get(
                self.target_frame, target_face, source_face, paste_back=False
            )

        self.assertIsNotNone(bgr_fake)
        self.assertIsNotNone(M)
        self.assertEqual(bgr_fake.shape, (128, 128, 3))
        self.assertEqual(M.shape, (2, 3))

        _face_size = swapper.input_size[0]
        _aimg_dummy = np.empty((_face_size, _face_size, 3), dtype=np.uint8)

        blended = _fast_paste_back(self.target_frame.copy(), bgr_fake, _aimg_dummy, M)
        self.assertEqual(blended.shape, self.target_frame.shape)
        self.assertEqual(blended.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
