"""Pytest test session configuration.

Ensures fast vectorized landmark alignment monkeypatch is active from the very start
of the test run, eliminating skimage SimilarityTransform deprecation warnings.
"""

try:
    from modules.processors.frame.face_swapper import patch_insightface_face_align
    patch_insightface_face_align()
except Exception:
    pass
