"""Unit and regression tests for PersonaManager and multi-source face embedding fusion."""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.persona_manager import Persona, PersonaManager


class PersonaDataClassTests(unittest.TestCase):
    def test_persona_serialization(self):
        """Verifies Persona model serializes to and deserializes from dictionary."""
        dummy_emb = [float(x) for x in np.random.randn(512)]
        persona = Persona(
            id="test-id-1",
            name="Alice",
            hotkey="F1",
            image_paths=["/path/to/img1.jpg", "/path/to/img2.jpg"],
            thumbnail_path="/path/to/thumb.jpg",
            fused_embedding=dummy_emb,
        )

        d = persona.to_dict()
        self.assertEqual(d["id"], "test-id-1")
        self.assertEqual(d["name"], "Alice")
        self.assertEqual(d["hotkey"], "F1")
        self.assertEqual(len(d["image_paths"]), 2)
        self.assertEqual(d["thumbnail_path"], "/path/to/thumb.jpg")

        restored = Persona.from_dict(d)
        self.assertEqual(restored.id, persona.id)
        self.assertEqual(restored.name, persona.name)
        self.assertEqual(restored.hotkey, persona.hotkey)
        self.assertEqual(restored.image_paths, persona.image_paths)
        self.assertIsNotNone(restored.get_embedding_array())
        self.assertEqual(len(restored.get_embedding_array()), 512)

    def test_persona_empty_embedding(self):
        """Verifies get_embedding_array returns None when no embedding is stored."""
        persona = Persona(
            id="test-empty",
            name="Empty",
            hotkey="F2",
            image_paths=[],
        )
        self.assertIsNone(persona.get_embedding_array())


class PersonaManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mgr = PersonaManager(base_dir=self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_initial_empty_state(self):
        """Verifies newly created manager has no personas."""
        self.assertEqual(len(self.mgr.get_personas()), 0)
        self.assertIsNone(self.mgr.get_active_persona())

    @patch("modules.persona_manager.PersonaManager.fuse_embeddings_from_images")
    @patch("modules.persona_manager.PersonaManager.create_thumbnail")
    def test_add_and_retrieve_persona(self, mock_thumb, mock_fuse):
        """Verifies adding a persona persists and updates list."""
        mock_fuse.return_value = (np.ones(512, dtype=np.float32) / np.sqrt(512), "dummy1.jpg")
        mock_thumb.return_value = "dummy_thumb.jpg"

        persona = self.mgr.add_persona(
            name="Bob",
            image_paths=["dummy1.jpg", "dummy2.jpg"],
            hotkey="F1",
            custom_id="bob-001",
        )

        self.assertIsNotNone(persona)
        self.assertEqual(persona.id, "bob-001")
        self.assertEqual(persona.name, "Bob")
        self.assertEqual(persona.hotkey, "F1")
        self.assertEqual(len(self.mgr.get_personas()), 1)

        # Retrieval by ID and hotkey
        by_id = self.mgr.get_persona_by_id("bob-001")
        self.assertIsNotNone(by_id)
        self.assertEqual(by_id.name, "Bob")

        by_hotkey = self.mgr.get_persona_by_hotkey("f1")  # Case insensitive
        self.assertIsNotNone(by_hotkey)
        self.assertEqual(by_hotkey.id, "bob-001")

    @patch("modules.persona_manager.PersonaManager.fuse_embeddings_from_images")
    @patch("modules.persona_manager.PersonaManager.create_thumbnail")
    def test_hotkey_conflict_resolution(self, mock_thumb, mock_fuse):
        """Verifies assigning an existing hotkey releases it from previous persona."""
        mock_fuse.return_value = (np.ones(512, dtype=np.float32) / np.sqrt(512), "dummy.jpg")
        mock_thumb.return_value = "dummy_thumb.jpg"

        p1 = self.mgr.add_persona(name="P1", image_paths=["dummy.jpg"], hotkey="F1", custom_id="p1")
        p2 = self.mgr.add_persona(name="P2", image_paths=["dummy.jpg"], hotkey="F1", custom_id="p2")

        self.assertEqual(p2.hotkey, "F1")
        refreshed_p1 = self.mgr.get_persona_by_id("p1")
        self.assertEqual(refreshed_p1.hotkey, "")
        self.assertEqual(self.mgr.get_persona_by_hotkey("F1").id, "p2")

    @patch("modules.persona_manager.PersonaManager.fuse_embeddings_from_images")
    @patch("modules.persona_manager.PersonaManager.create_thumbnail")
    def test_delete_persona(self, mock_thumb, mock_fuse):
        """Verifies deleting persona removes it from list and clears active if selected."""
        mock_fuse.return_value = (np.ones(512, dtype=np.float32) / np.sqrt(512), "dummy.jpg")
        mock_thumb.return_value = "dummy_thumb.jpg"

        p = self.mgr.add_persona(name="ToDelete", image_paths=["dummy.jpg"], hotkey="F3", custom_id="del-1")
        self.mgr.set_active_persona(p.id)
        self.assertEqual(self.mgr.get_active_persona().id, "del-1")

        deleted = self.mgr.delete_persona("del-1")
        self.assertTrue(deleted)
        self.assertEqual(len(self.mgr.get_personas()), 0)
        self.assertIsNone(self.mgr.get_active_persona())

    def test_build_synthetic_face(self):
        """Verifies build_synthetic_face returns a valid insightface Face object."""
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        persona = Persona(
            id="synth-1",
            name="Synthetic",
            hotkey="F4",
            image_paths=[],
            fused_embedding=emb.tolist(),
        )

        face = self.mgr.build_synthetic_face(persona)
        self.assertIsNotNone(face)
        self.assertTrue(hasattr(face, "normed_embedding"))
        self.assertTrue(hasattr(face, "embedding"))
        self.assertTrue(hasattr(face, "bbox"))
        self.assertTrue(hasattr(face, "det_score"))
        np.testing.assert_allclose(face.normed_embedding, emb, rtol=1e-5)


class MultiSourceEmbeddingFusionTests(unittest.TestCase):
    @patch("modules.persona_manager.imread_unicode")
    @patch("os.path.exists", return_value=True)
    def test_fuse_embeddings_mathematical_norm(self, mock_exists, mock_imread):
        """Verifies multi-source fusion normalizes the average of embeddings to unit L2 norm."""
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        # Create 3 synthetic face embeddings with random angles
        emb1 = np.random.randn(512).astype(np.float32)
        emb1 /= np.linalg.norm(emb1)
        emb2 = np.random.randn(512).astype(np.float32)
        emb2 /= np.linalg.norm(emb2)
        emb3 = np.random.randn(512).astype(np.float32)
        emb3 /= np.linalg.norm(emb3)

        class DummyFace:
            def __init__(self, e):
                self.normed_embedding = e

        faces = [DummyFace(emb1), DummyFace(emb2), DummyFace(emb3)]

        import modules.face_analyser as fa

        with patch.object(fa, "get_one_face", side_effect=faces):
            fused_emb, first_path = PersonaManager.fuse_embeddings_from_images(["img1.jpg", "img2.jpg", "img3.jpg"])

        self.assertIsNotNone(fused_emb)
        self.assertEqual(first_path, "img1.jpg")
        self.assertEqual(fused_emb.shape, (512,))

        # Verify unit norm ||E||_2 == 1.0
        norm = np.linalg.norm(fused_emb)
        self.assertAlmostEqual(norm, 1.0, places=5)

        # Verify mathematical direction matches average
        expected_avg = (emb1 + emb2 + emb3) / 3.0
        expected_normed = expected_avg / np.linalg.norm(expected_avg)
        np.testing.assert_allclose(fused_emb, expected_normed, rtol=1e-5)

    @patch("modules.face_analyser.imread_unicode")
    @patch("os.path.exists", return_value=True)
    def test_create_fused_face_helper(self, mock_exists, mock_imread):
        """Verifies create_fused_face helper in modules.face_analyser."""
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        emb1 = np.random.randn(512).astype(np.float32)
        emb1 /= np.linalg.norm(emb1)
        emb2 = np.random.randn(512).astype(np.float32)
        emb2 /= np.linalg.norm(emb2)

        class DummyFace:
            def __init__(self, e):
                self.normed_embedding = e
                self.bbox = np.array([10, 10, 80, 80], dtype=np.float32)
                self.kps = np.zeros((5, 2), dtype=np.float32)
                self.det_score = 0.98

        faces = [DummyFace(emb1), DummyFace(emb2)]

        import modules.face_analyser as fa

        with patch.object(fa, "get_one_face", side_effect=faces):
            fused_face = fa.create_fused_face(["pic1.jpg", "pic2.jpg"])

        self.assertIsNotNone(fused_face)
        self.assertEqual(fused_face.det_score, 0.98)
        self.assertAlmostEqual(np.linalg.norm(fused_face.normed_embedding), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
