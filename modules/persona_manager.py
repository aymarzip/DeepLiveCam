"""Persona Manager for DeepLiveCam Studio.

Provides:
- Persona data model with multi-source image paths and fused embeddings.
- Mathematical multi-angle embedding fusion:
    E_avg = (1/K) * sum(E_i), E_fused = E_avg / ||E_avg||_2
- Local persistence in `personas/personas.json` and thumbnail generation in `personas/thumbs/`.
- Hotkey binding and retrieval (e.g. F1-F8, 1-8).
"""

from __future__ import annotations

import json
import os
import uuid
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from modules import imread_unicode


@dataclass
class Persona:
    id: str
    name: str
    hotkey: str
    image_paths: List[str]
    thumbnail_path: str = ""
    fused_embedding: Optional[List[float]] = None
    created_at: float = field(default_factory=time.time)

    def get_embedding_array(self) -> Optional[np.ndarray]:
        if self.fused_embedding is None:
            return None
        return np.asarray(self.fused_embedding, dtype=np.float32)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "hotkey": self.hotkey,
            "image_paths": self.image_paths,
            "thumbnail_path": self.thumbnail_path,
            "fused_embedding": self.fused_embedding,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Persona:
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", "Unnamed"),
            hotkey=data.get("hotkey", ""),
            image_paths=data.get("image_paths", []),
            thumbnail_path=data.get("thumbnail_path", ""),
            fused_embedding=data.get("fused_embedding"),
            created_at=data.get("created_at", time.time()),
        )


class PersonaManager:
    """Manages persona profiles, multi-source embeddings, and hotkeys."""

    _instance: Optional[PersonaManager] = None

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "personas")
        self.base_dir = base_dir
        self.thumbs_dir = os.path.join(self.base_dir, "thumbs")
        self.config_path = os.path.join(self.base_dir, "personas.json")
        self._personas: List[Persona] = []
        self._active_persona_id: Optional[str] = None

        os.makedirs(self.thumbs_dir, exist_ok=True)
        self.load()

    @classmethod
    def get_instance(cls) -> PersonaManager:
        if cls._instance is None:
            cls._instance = PersonaManager()
        return cls._instance

    def load(self) -> List[Persona]:
        if not os.path.exists(self.config_path):
            self._personas = []
            return self._personas
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._personas = [Persona.from_dict(item) for item in data.get("personas", [])]
            self._active_persona_id = data.get("active_persona_id")
        except (json.JSONDecodeError, OSError):
            self._personas = []
        return self._personas

    def save(self) -> None:
        try:
            os.makedirs(self.base_dir, exist_ok=True)
            data = {
                "active_persona_id": self._active_persona_id,
                "personas": [p.to_dict() for p in self._personas],
            }
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    def get_personas(self) -> List[Persona]:
        return list(self._personas)

    def get_persona_by_id(self, persona_id: str) -> Optional[Persona]:
        for p in self._personas:
            if p.id == persona_id:
                return p
        return None

    def get_persona_by_hotkey(self, hotkey: str) -> Optional[Persona]:
        normalized = hotkey.strip().upper()
        for p in self._personas:
            if p.hotkey.strip().upper() == normalized:
                return p
        return None

    def get_active_persona(self) -> Optional[Persona]:
        if self._active_persona_id is None:
            return None
        return self.get_persona_by_id(self._active_persona_id)

    def set_active_persona(self, persona_id: Optional[str]) -> None:
        self._active_persona_id = persona_id
        self.save()

    @staticmethod
    def fuse_embeddings_from_images(image_paths: List[str]) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Extracts and averages face embeddings across multiple images.

        Returns (fused_embedding_array, first_valid_image_path).
        """
        from modules.face_analyser import get_one_face

        embeddings = []
        first_valid_path = None

        for path in image_paths:
            if not os.path.exists(path):
                continue
            frame = imread_unicode(path)
            if frame is None:
                continue
            face = get_one_face(frame)
            if face is not None and getattr(face, "normed_embedding", None) is not None:
                emb = np.asarray(face.normed_embedding, dtype=np.float32)
                embeddings.append(emb)
                if first_valid_path is None:
                    first_valid_path = path

        if not embeddings:
            return None, None

        # Mean vector
        stacked = np.stack(embeddings, axis=0)
        avg = np.mean(stacked, axis=0)
        norm = np.linalg.norm(avg)
        if norm > 1e-9:
            fused = avg / norm
        else:
            fused = avg

        return fused.astype(np.float32), first_valid_path

    def create_thumbnail(self, image_path: str, persona_id: str, size: int = 120) -> str:
        """Crops a square thumbnail and saves to the thumbs directory."""
        thumb_name = f"{persona_id}.jpg"
        thumb_path = os.path.join(self.thumbs_dir, thumb_name)
        img = imread_unicode(image_path)
        if img is None:
            return ""

        h, w = img.shape[:2]
        crop_size = min(h, w)
        y0 = (h - crop_size) // 2
        x0 = (w - crop_size) // 2
        cropped = img[y0:y0 + crop_size, x0:x0 + crop_size]
        resized = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)

        cv2.imwrite(thumb_path, resized)
        return thumb_path

    def add_persona(
        self,
        name: str,
        image_paths: List[str],
        hotkey: str = "",
        custom_id: Optional[str] = None,
    ) -> Optional[Persona]:
        """Creates and persists a new persona with fused embeddings."""
        if not image_paths:
            return None

        persona_id = custom_id or str(uuid.uuid4())[:8]
        fused_emb, valid_path = self.fuse_embeddings_from_images(image_paths)
        if valid_path is None:
            return None

        thumb_path = self.create_thumbnail(valid_path, persona_id)
        emb_list = fused_emb.tolist() if fused_emb is not None else None

        persona = Persona(
            id=persona_id,
            name=name or f"Persona {len(self._personas) + 1}",
            hotkey=hotkey,
            image_paths=list(image_paths),
            thumbnail_path=thumb_path,
            fused_embedding=emb_list,
        )

        # If hotkey is already used, release it from other persona
        if hotkey:
            for p in self._personas:
                if p.hotkey.upper() == hotkey.upper():
                    p.hotkey = ""

        self._personas.append(persona)
        self.save()
        return persona

    def delete_persona(self, persona_id: str) -> bool:
        initial_len = len(self._personas)
        self._personas = [p for p in self._personas if p.id != persona_id]
        if self._active_persona_id == persona_id:
            self._active_persona_id = None
        if len(self._personas) != initial_len:
            thumb_path = os.path.join(self.thumbs_dir, f"{persona_id}.jpg")
            if os.path.exists(thumb_path):
                try:
                    os.remove(thumb_path)
                except OSError:
                    pass
            self.save()
            return True
        return False

    def build_synthetic_face(self, persona: Persona) -> Optional[object]:
        """Builds a Face-compatible object for face_swapper from a Persona."""
        from insightface.app.common import Face

        emb = persona.get_embedding_array()
        if emb is None:
            return None

        face = Face(
            bbox=np.array([0.0, 0.0, 128.0, 128.0], dtype=np.float32),
            kps=np.zeros((5, 2), dtype=np.float32),
            det_score=1.0,
        )
        face.embedding = emb
        return face
