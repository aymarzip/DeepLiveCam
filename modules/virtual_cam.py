"""Virtual Camera output manager for DeepLiveCam.

Streams processed video frames directly to system virtual cameras (OBS Virtual Camera,
UnityCapture, etc.) using pyvirtualcam, making DeepLiveCam immediately selectable as a
video input device in Zoom, Discord, Telegram, and browser meetings.
"""

import logging
import threading
from typing import Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import pyvirtualcam
    from pyvirtualcam import PixelFormat
    HAS_PYVIRTUALCAM = True
except ImportError:
    pyvirtualcam = None
    PixelFormat = None
    HAS_PYVIRTUALCAM = False


class VirtualCameraManager:
    """Thread-safe virtual camera lifecycle and streaming manager."""

    def __init__(self) -> None:
        self._cam: Optional["pyvirtualcam.Camera"] = None
        self._lock = threading.Lock()
        self._active = False
        self._dimensions: Optional[Tuple[int, int]] = None
        self._fps: float = 30.0
        self._device_name: str = "None"
        self._backend: Optional[str] = None

    @property
    def is_active(self) -> bool:
        """Returns True if the virtual camera is currently streaming."""
        return self._active and self._cam is not None

    @property
    def device_name(self) -> str:
        """Returns the name of the active virtual camera device."""
        return self._device_name

    def start(
        self,
        width: int,
        height: int,
        fps: float = 30.0,
        backend: Optional[str] = None,
    ) -> bool:
        """Initializes and opens the virtual camera device.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
            fps: Desired frame rate (default 30.0).
            backend: Optional backend identifier (e.g., 'obs', 'unitycapture').

        Returns:
            True if successfully opened, False otherwise.
        """
        if not HAS_PYVIRTUALCAM:
            logger.warning(
                "pyvirtualcam is not installed. Virtual camera output is unavailable."
            )
            return False

        with self._lock:
            # If already active with the same configuration, keep running
            if self._active and self._cam is not None and self._dimensions == (width, height):
                return True

            # Close existing session if reconfiguring dimensions
            if self._cam is not None:
                self._stop_unlocked()

            try:
                # pyvirtualcam takes width, height, fps, fmt, and optional backend
                kwargs = {
                    "width": int(width),
                    "height": int(height),
                    "fps": float(fps),
                    "fmt": PixelFormat.BGR,
                }
                if backend and backend != "auto":
                    kwargs["backend"] = backend

                self._cam = pyvirtualcam.Camera(**kwargs)
                self._active = True
                self._dimensions = (int(width), int(height))
                self._fps = float(fps)
                self._backend = backend
                self._device_name = getattr(self._cam, "device", "Virtual Camera")
                logger.info(
                    "Virtual camera opened successfully: %s (%dx%d @ %.1f FPS)",
                    self._device_name,
                    width,
                    height,
                    fps,
                )
                return True
            except Exception as exc:
                logger.error("Failed to start virtual camera: %s", exc)
                self._active = False
                self._cam = None
                self._device_name = "None"
                return False

    def send_frame(self, frame: np.ndarray) -> bool:
        """Sends a processed frame to the virtual camera device.

        Automatically resizes the frame if its resolution does not match the
        configured virtual camera dimensions.

        Args:
            frame: uint8 BGR image numpy array.

        Returns:
            True if frame was sent, False otherwise.
        """
        if not self._active or self._cam is None or frame is None:
            return False

        with self._lock:
            if not self._active or self._cam is None:
                return False

            try:
                target_w, target_h = self._dimensions
                fh, fw = frame.shape[:2]

                # Ensure dimensions match
                if (fw, fh) != (target_w, target_h):
                    send_img = cv2.resize(
                        frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR
                    )
                else:
                    send_img = frame

                # Ensure memory is contiguous C-order uint8
                if not send_img.flags["C_CONTIGUOUS"] or send_img.dtype != np.uint8:
                    send_img = np.ascontiguousarray(send_img, dtype=np.uint8)

                self._cam.send(send_img)
                return True
            except Exception as exc:
                logger.warning("Error sending frame to virtual camera: %s", exc)
                return False

    def _stop_unlocked(self) -> None:
        """Internal helper to close the camera while holding the lock."""
        if self._cam is not None:
            try:
                self._cam.close()
            except Exception as exc:
                logger.debug("Exception closing virtual camera: %s", exc)
            finally:
                self._cam = None
        self._active = False
        self._dimensions = None
        self._device_name = "None"

    def stop(self) -> None:
        """Safely stops and releases the virtual camera device."""
        with self._lock:
            self._stop_unlocked()
            logger.info("Virtual camera stopped.")


# Module singleton instance
_VIRTUAL_CAM_MANAGER: Optional[VirtualCameraManager] = None
_INIT_LOCK = threading.Lock()


def get_virtual_camera_manager() -> VirtualCameraManager:
    """Returns the process-wide VirtualCameraManager instance."""
    global _VIRTUAL_CAM_MANAGER
    if _VIRTUAL_CAM_MANAGER is None:
        with _INIT_LOCK:
            if _VIRTUAL_CAM_MANAGER is None:
                _VIRTUAL_CAM_MANAGER = VirtualCameraManager()
    return _VIRTUAL_CAM_MANAGER
