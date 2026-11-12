"""
Integration test for ONNX Runtime DirectML Execution Provider on Intel Arc GPU.
Verifies that DirectML is available, correctly decoded by DeepLiveCam, and capable
of initializing sessions and executing tensor inference on the host D3D12 hardware.
"""

import os
import sys
import unittest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import onnxruntime as ort
import modules.globals
import modules.platform_info as platform_info
from modules.core import decode_execution_providers


class DirectMLExecutionProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.available_providers = ort.get_available_providers()
        cls.has_dml = "DmlExecutionProvider" in cls.available_providers

    def test_directml_provider_registered(self):
        """Verifies onnxruntime has registered DmlExecutionProvider."""
        self.assertTrue(
            self.has_dml,
            f"DmlExecutionProvider not found in available providers: {self.available_providers}",
        )

    def test_platform_info_detects_dml(self):
        """Verifies modules.platform_info properly identifies DirectML capability."""
        self.assertTrue(
            platform_info.HAS_DML_PROVIDER,
            "platform_info.HAS_DML_PROVIDER should be True on this host",
        )

    def test_decode_execution_providers_dml(self):
        """Verifies decode_execution_providers properly maps 'dml' to DmlExecutionProvider."""
        decoded = decode_execution_providers(["dml"])
        self.assertIn("DmlExecutionProvider", decoded)

    def test_decode_execution_providers_cpu(self):
        """Verifies decode_execution_providers fallback for 'cpu'."""
        decoded = decode_execution_providers(["cpu"])
        self.assertIn("CPUExecutionProvider", decoded)
        self.assertNotIn("DmlExecutionProvider", decoded)

    def test_directml_session_execution(self):
        """Initializes a real DirectML session and verifies tensor inference on GPU."""
        if not self.has_dml:
            self.skipTest("DirectML not available on this platform")

        # Test inference using ArcFace feature extractor model
        model_path = os.path.join(PROJECT_ROOT, "models/buffalo_l/w600k_r50.onnx")
        if not os.path.exists(model_path):
            model_path = os.path.join(PROJECT_ROOT, "models/models/buffalo_l/w600k_r50.onnx")

        self.assertTrue(os.path.exists(model_path), f"Model not found: {model_path}")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        )

        active_providers = session.get_providers()
        self.assertIn("DmlExecutionProvider", active_providers)

        input_name = session.get_inputs()[0].name
        # w600k_r50 expects (1, 3, 112, 112)
        dummy_input = np.zeros((1, 3, 112, 112), dtype=np.float32)

        with modules.globals.dml_lock:
            outputs = session.run(None, {input_name: dummy_input})

        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), 1)
        self.assertIsInstance(outputs[0], np.ndarray)
        self.assertEqual(outputs[0].shape, (1, 512))

    def test_dml_lock_is_functional(self):
        """Verifies dml_lock synchronization primitive is available and operational."""
        import threading
        self.assertIsInstance(modules.globals.dml_lock, type(threading.Lock()))
        acquired = modules.globals.dml_lock.acquire(timeout=1.0)
        self.assertTrue(acquired, "Failed to acquire dml_lock")
        modules.globals.dml_lock.release()


if __name__ == "__main__":
    unittest.main()
