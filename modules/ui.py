"""PySide6 UI for Deep-Live-Cam.

Public API kept stable for the rest of the codebase:
    init(start, destroy, lang) -> _Window
        Returned object has .mainloop() that core.py calls.
    update_status(text)
        Thread-safe; routed through Qt signal when called off-UI.
    check_and_ignore_nsfw(target, destroy=None) -> bool
"""

from __future__ import annotations

import os
import platform
import queue
import sys
import tempfile
import threading
import time
import webbrowser
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps
from PySide6.QtCore import (
    QObject,
    QThread,
    QTimer,
    Qt,
    Signal,
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

import modules.globals
import modules.metadata
from modules.capturer import get_video_frame, get_video_frame_total
from modules.face_analyser import (
    add_blank_map,
    detect_many_faces_fast,
    detect_one_face_fast,
    ensure_landmarks,
    get_one_face,
    get_unique_faces_from_target_image,
    get_unique_faces_from_target_video,
    has_valid_map,
    reset_face_analyser,
    simplify_maps,
)
from modules.gettext import LanguageManager
from modules.gpu_processing import gpu_cvt_color, gpu_flip, gpu_resize
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import (
    has_image_extension,
    is_image,
    is_video,
)
from modules import imread_unicode
from modules.video_capture import VideoCapturer

if platform.system() == "Windows":
    from pygrabber.dshow_graph import FilterGraph

import json


# ─── constants ────────────────────────────────────────────────────────────

ROOT_HEIGHT = 760
ROOT_WIDTH = 960

PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200
PREVIEW_DEFAULT_WIDTH = 640
PREVIEW_DEFAULT_HEIGHT = 360

POPUP_WIDTH = 750
POPUP_HEIGHT = 810
POPUP_SCROLL_WIDTH = 720
POPUP_SCROLL_HEIGHT = 700

POPUP_LIVE_WIDTH = 900
POPUP_LIVE_HEIGHT = 820
POPUP_LIVE_SCROLL_WIDTH = 870
POPUP_LIVE_SCROLL_HEIGHT = 700

MAPPER_PREVIEW_SIZE = 100
SOURCE_TARGET_PREVIEW_SIZE = 170


# ─── modern Fluent dark stylesheet ────────────────────────────────────────

QSS = """
QMainWindow, QDialog {
    background-color: #0f141c;
    color: #e6edf3;
}
QWidget {
    color: #e6edf3;
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, "Roboto", "Helvetica Neue", Arial, sans-serif;
    font-size: 10pt;
}

QGroupBox {
    background-color: #161f2c;
    border: 1px solid #232d3d;
    border-radius: 10px;
    margin-top: 14px;
    padding-top: 14px;
    padding-bottom: 8px;
    padding-left: 10px;
    padding-right: 10px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: #58a6ff;
}

QPushButton {
    background-color: #0078d4;
    color: #ffffff;
    border: 1px solid #1084d8;
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: 600;
    font-size: 9.5pt;
}
QPushButton:hover {
    background-color: #1084d8;
    border-color: #2997ea;
}
QPushButton:pressed {
    background-color: #0067b8;
    border-color: #005a9e;
}
QPushButton:disabled {
    background-color: #21262d;
    color: #484f58;
    border: 1px solid #30363d;
}
QPushButton#secondary {
    background-color: #212936;
    color: #c9d1d9;
    border: 1px solid #303b4b;
}
QPushButton#secondary:hover {
    background-color: #2c3748;
    color: #ffffff;
    border-color: #3f4c60;
}
QPushButton#secondary:pressed {
    background-color: #1a222f;
}
QPushButton#danger {
    background-color: #b91c1c;
    color: #ffffff;
    border: 1px solid #dc2626;
}
QPushButton#danger:hover {
    background-color: #dc2626;
}
QPushButton#danger:pressed {
    background-color: #991b1b;
}
QPushButton#live {
    background-color: #107c41;
    color: #ffffff;
    border: 1px solid #16a34a;
}
QPushButton#live:hover {
    background-color: #15803d;
    border-color: #22c55e;
}
QPushButton#live:pressed {
    background-color: #0d5e32;
}

QComboBox {
    background-color: #1a2332;
    color: #e6edf3;
    border: 1px solid #2d3b4e;
    border-radius: 6px;
    padding: 5px 8px;
    min-height: 20px;
}
QComboBox:hover {
    border-color: #0078d4;
}
QComboBox QAbstractItemView {
    background-color: #161f2c;
    color: #e6edf3;
    selection-background-color: #0078d4;
    border: 1px solid #2d3b4e;
}

QCheckBox {
    spacing: 8px;
    padding: 3px 0;
    color: #c9d1d9;
    font-size: 9.5pt;
}
QCheckBox::indicator {
    width: 34px;
    height: 18px;
    border-radius: 9px;
    background-color: #242f3d;
    border: 1px solid #38465a;
}
QCheckBox::indicator:hover {
    background-color: #2c3849;
    border-color: #4a5d78;
}
QCheckBox::indicator:checked {
    background-color: #0078d4;
    border: 1px solid #1084d8;
}
QCheckBox::indicator:checked:hover {
    background-color: #1084d8;
    border-color: #2997ea;
}

QSlider::groove:horizontal {
    height: 6px;
    background: #253142;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #ffffff;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
    border: 2px solid #0078d4;
}
QSlider::handle:horizontal:hover {
    background: #f0f6fc;
    border: 2px solid #58a6ff;
}
QSlider::sub-page:horizontal {
    background: #0078d4;
    border-radius: 3px;
}

QLabel#imageDrop {
    background-color: #121822;
    border: 2px dashed #263345;
    border-radius: 8px;
    color: #8b949e;
    font-size: 9.5pt;
}
QLabel#imageDrop:hover {
    border-color: #0078d4;
    color: #58a6ff;
}
QLabel#statusLabel {
    color: #8b949e;
    font-size: 9pt;
}
QLabel#linkLabel {
    color: #58a6ff;
    font-size: 9pt;
}
QLabel#linkLabel:hover {
    text-decoration: underline;
    color: #79c0ff;
}
QLabel#valueBadge {
    background-color: #1b2636;
    color: #38bdf8;
    border: 1px solid #2d3d54;
    border-radius: 4px;
    padding: 2px 6px;
    font-family: "Consolas", monospace;
    font-size: 9pt;
    font-weight: 600;
    min-width: 36px;
}
QLabel#headerTitle {
    font-size: 14pt;
    font-weight: 700;
    color: #ffffff;
}
QLabel#headerSubtitle {
    font-size: 8.5pt;
    color: #8b949e;
}
QLabel#hardwareBadge {
    background-color: #17263c;
    color: #38bdf8;
    border: 1px solid #234268;
    border-radius: 11px;
    padding: 3px 10px;
    font-weight: 600;
    font-size: 8.5pt;
}

QScrollArea {
    border: none;
    background: transparent;
}

QFrame#card {
    background-color: #161f2c;
    border: 1px solid #232d3d;
    border-radius: 10px;
}
"""


# ─── module-level state ───────────────────────────────────────────────────

_APP: Optional[QApplication] = None
_MAIN: Optional["MainWindow"] = None
_PREVIEW: Optional["PreviewWindow"] = None
_WEBCAM_PREVIEW: Optional["WebcamPreviewWindow"] = None
# How long to wait quietly for a live-preview worker to notice the stop flag
# before saying so. Shutdown then keeps waiting: it has to finish.
WORKER_SHUTDOWN_GRACE_MS = 30000
_MAPPER: Optional["MapperDialog"] = None
_LIVE_MAPPER: Optional["LiveMapperDialog"] = None
_LANG: Optional[LanguageManager] = None
_BRIDGE: Optional["_UIBridge"] = None


def _(text: str) -> str:
    """Translate via LanguageManager; falls back to identity."""
    if _LANG is None:
        return text
    return _LANG._(text)


# Preserve original cwd state for file dialogs.
_RECENT_SOURCE_DIR: Optional[str] = None
_RECENT_TARGET_DIR: Optional[str] = None
_RECENT_OUTPUT_DIR: Optional[str] = None

# QFileDialog filter strings, built from the canonical extension sets in
# globals so every dialog stays in sync (no hand-copied lists to drift).
_IMAGE_FILE_FILTER = "Images (" + " ".join(
    f"*{ext}" for ext in modules.globals.IMAGE_EXTENSIONS
) + ")"
_MEDIA_FILE_FILTER = "Media (" + " ".join(
    f"*{ext}" for ext in (*modules.globals.IMAGE_EXTENSIONS, *modules.globals.VIDEO_EXTENSIONS)
) + ")"
_VIDEO_FILE_FILTER = "Videos (" + " ".join(
    f"*{ext}" for ext in modules.globals.VIDEO_EXTENSIONS
) + ")"


# ─── image utilities ─────────────────────────────────────────────────────


def fit_image_to_size(image, width: int, height: int):
    """BGR ndarray → BGR ndarray scaled to fit within (width, height)."""
    if width is None and height is None or width <= 0 or height <= 0:
        return image
    h, w = image.shape[:2]
    ratio_w = width / w
    ratio_h = height / h
    ratio = min(ratio_w, ratio_h)
    new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
    if new_size == (w, h):
        return image
    return gpu_resize(image, dsize=new_size)


def _bgr_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    """Zero-copy / fast ndarray → QPixmap supporting BGR, Grayscale, and RGBA."""
    h, w = bgr.shape[:2]
    if not bgr.flags['C_CONTIGUOUS']:
        bgr = np.ascontiguousarray(bgr)
    if bgr.ndim == 2:
        qimg = QImage(bgr.data, w, h, bgr.strides[0], QImage.Format.Format_Grayscale8)
    elif bgr.ndim == 3 and bgr.shape[2] == 1:
        sq = bgr[:, :, 0]
        if not sq.flags['C_CONTIGUOUS']:
            sq = np.ascontiguousarray(sq)
        qimg = QImage(sq.data, w, h, sq.strides[0], QImage.Format.Format_Grayscale8)
    elif bgr.ndim == 3 and bgr.shape[2] == 4:
        qimg = QImage(bgr.data, w, h, bgr.strides[0], QImage.Format.Format_ARGB32)
    else:
        bgr_fmt = getattr(QImage.Format, "Format_BGR888", None)
        if bgr_fmt is not None:
            qimg = QImage(bgr.data, w, h, bgr.strides[0], bgr_fmt)
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _pil_to_qpixmap(image: Image.Image) -> QPixmap:
    """PIL.Image → QPixmap."""
    image = image.convert("RGBA")
    data = image.tobytes("raw", "RGBA")
    qimg = QImage(data, image.width, image.height, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


def render_image_preview(image_path: str, size: Tuple[int, int]) -> QPixmap:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return _pil_to_qpixmap(image)


def render_video_preview(
    video_path: str, size: Tuple[int, int], frame_number: int = 0
) -> Optional[QPixmap]:
    capture = cv2.VideoCapture(video_path)
    try:
        if frame_number:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        has_frame, frame = capture.read()
        if not has_frame:
            return None
        image = Image.fromarray(gpu_cvt_color(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return _pil_to_qpixmap(image)
    finally:
        capture.release()


# ─── persistence ─────────────────────────────────────────────────────────


def save_switch_states():
    state = {
        "keep_fps": modules.globals.keep_fps,
        "keep_audio": modules.globals.keep_audio,
        "keep_frames": modules.globals.keep_frames,
        "many_faces": modules.globals.many_faces,
        "map_faces": modules.globals.map_faces,
        "poisson_blend": modules.globals.poisson_blend,
        "color_correction": modules.globals.color_correction,
        "nsfw_filter": modules.globals.nsfw_filter,
        "live_mirror": modules.globals.live_mirror,
        "live_resizable": modules.globals.live_resizable,
        "fp_ui": modules.globals.fp_ui,
        "show_fps": modules.globals.show_fps,
        "mouth_mask": modules.globals.mouth_mask,
        "show_mouth_mask_box": modules.globals.show_mouth_mask_box,
        "mouth_mask_size": modules.globals.mouth_mask_size,
        "glasses_mask": getattr(modules.globals, "glasses_mask", False),
        "glasses_mask_size": getattr(modules.globals, "glasses_mask_size", 0.0),
        "show_glasses_mask_box": False,
        "bangs_mask": getattr(modules.globals, "bangs_mask", False),
        "forehead_cutoff": getattr(modules.globals, "forehead_cutoff", 0.0),
        "show_bangs_mask_box": False,
        "eyes_mask": getattr(modules.globals, "eyes_mask", False),
        "eyes_mask_size": getattr(modules.globals, "eyes_mask_size", 0.0),
        "show_eyes_mask_box": False,
        "virtual_cam": getattr(modules.globals, "virtual_cam", False),
        "capture_resolution": list(modules.globals.capture_resolution),
        "det_size": modules.globals.det_size,
        "enable_interpolation": getattr(modules.globals, "enable_interpolation", False),
        "swap_interval": getattr(modules.globals, "swap_interval", modules.globals.DEFAULT_SWAP_INTERVAL),
    }
    try:
        with open("switch_states.json", "w") as f:
            json.dump(state, f)
    except OSError:
        pass


def load_switch_states():
    try:
        with open("switch_states.json", "r") as f:
            state = json.load(f)
        modules.globals.keep_fps = state.get("keep_fps", True)
        modules.globals.keep_audio = state.get("keep_audio", True)
        modules.globals.keep_frames = state.get("keep_frames", False)
        modules.globals.many_faces = state.get("many_faces", False)
        modules.globals.map_faces = state.get("map_faces", False)
        modules.globals.poisson_blend = state.get("poisson_blend", False)
        modules.globals.color_correction = state.get("color_correction", False)
        modules.globals.nsfw_filter = state.get("nsfw_filter", False)
        modules.globals.live_mirror = state.get("live_mirror", False)
        modules.globals.live_resizable = state.get("live_resizable", False)
        modules.globals.fp_ui = state.get("fp_ui", {"face_enhancer": False})
        modules.globals.show_fps = state.get("show_fps", False)
        modules.globals.enable_interpolation = state.get("enable_interpolation", False)
        modules.globals.swap_interval = state.get("swap_interval", modules.globals.DEFAULT_SWAP_INTERVAL)
        # Masks start disabled or initialized safely on launch
        modules.globals.mouth_mask_size = 0.0
        modules.globals.mouth_mask = False
        modules.globals.show_mouth_mask_box = False
        modules.globals.glasses_mask = state.get("glasses_mask", False)
        modules.globals.glasses_mask_size = float(state.get("glasses_mask_size", 0.0))
        modules.globals.show_glasses_mask_box = False
        modules.globals.bangs_mask = state.get("bangs_mask", False)
        modules.globals.forehead_cutoff = float(state.get("forehead_cutoff", 0.0))
        modules.globals.show_bangs_mask_box = False
        modules.globals.eyes_mask = state.get("eyes_mask", False)
        modules.globals.eyes_mask_size = float(state.get("eyes_mask_size", 0.0))
        modules.globals.show_eyes_mask_box = False
        modules.globals.virtual_cam = state.get("virtual_cam", False)
        # Validate persisted camera settings before trusting them — a hand-
        # edited/corrupt file shouldn't push a bad size into cv2/insightface.
        res = state.get("capture_resolution")
        if isinstance(res, (list, tuple)) and len(res) == 2:
            try:
                w, h = int(res[0]), int(res[1])
                if w > 0 and h > 0:
                    modules.globals.capture_resolution = (w, h)
            except (TypeError, ValueError):
                pass
        if state.get("det_size") in (160, 320, 640):
            modules.globals.det_size = int(state["det_size"])
    except FileNotFoundError:
        pass
    except (OSError, json.JSONDecodeError):
        pass


# ─── thread-safe status bridge ───────────────────────────────────────────


class _UIBridge(QObject):
    """Single QObject that owns cross-thread signals."""

    statusChanged = Signal(str)


def _emit_status(text: str) -> None:
    if _BRIDGE is None:
        print(text)
        return
    _BRIDGE.statusChanged.emit(text)


# ─── public API ──────────────────────────────────────────────────────────


def update_status(text: str) -> None:
    """Thread-safe status update — uses signal if called off-UI thread."""
    _emit_status(_(text))
    if _APP is not None and QThread.currentThread() is _APP.thread():
        # On UI thread — flush events so the user sees the update during
        # long synchronous start() runs.
        _APP.processEvents()


def check_and_ignore_nsfw(target, destroy: Optional[Callable] = None) -> bool:
    from numpy import ndarray
    from modules.predicter import predict_frame, predict_image, predict_video

    check_nsfw = None
    if isinstance(target, str):
        check_nsfw = predict_image if has_image_extension(target) else predict_video
    elif isinstance(target, ndarray):
        check_nsfw = predict_frame

    if check_nsfw and check_nsfw(target):
        if destroy:
            destroy(to_quit=False)
        update_status("Processing ignored!")
        return True
    return False


# ─── camera enumeration (unchanged from tk version) ──────────────────────


def get_available_cameras() -> Tuple[List[int], List[str]]:
    if platform.system() == "Windows":
        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()
            if devices:
                return list(range(len(devices))), devices
            return [], ["No cameras found"]
        except Exception as exc:
            print(f"Error detecting cameras: {exc}")
            return [], ["No cameras found"]

    if platform.system() == "Darwin":
        return [0, 1], ["Camera 0", "Camera 1"]

    # Linux probe
    indices: List[int] = []
    names: List[str] = []
    for i in range(10):
        cap = cv2.VideoCapture(f"/dev/video{i}")
        if cap.isOpened():
            indices.append(i)
            names.append(f"Camera {i}")
            cap.release()
    return (indices, names) if names else ([], ["No cameras found"])


# ─── main window ─────────────────────────────────────────────────────────


def _make_image_drop(text: str, size: Tuple[int, int]) -> QLabel:
    label = QLabel(text)
    label.setObjectName("imageDrop")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setFixedSize(size[0], size[1])
    label.setText(text)
    return label


class _Switch(QWidget):
    """Compact toggle switch with label + optional tooltip."""

    toggled = Signal(bool)

    def __init__(self, text: str, initial: bool, tooltip: str = ""):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._checkbox = QCheckBox(text)
        self._checkbox.setChecked(initial)
        self._checkbox.toggled.connect(self.toggled.emit)
        if tooltip:
            self._checkbox.setToolTip(tooltip)
        layout.addWidget(self._checkbox)
        layout.addStretch(1)

    def isChecked(self) -> bool:
        return self._checkbox.isChecked()

    def setChecked(self, value: bool) -> None:
        self._checkbox.setChecked(value)


def _get_provider_badge_text() -> str:
    providers = getattr(modules.globals, "execution_providers", [])
    for p in providers:
        p_lower = p.lower()
        if "dml" in p_lower:
            return "⚡ DirectML GPU"
        if "cuda" in p_lower:
            return "⚡ CUDA GPU"
        if "rocm" in p_lower:
            return "⚡ ROCm GPU"
        if "coreml" in p_lower:
            return "⚡ CoreML Apple"
    return "⚙ CPU Mode"


class MainWindow(QMainWindow):
    def __init__(self, start_cb: Callable, destroy_cb: Callable):
        super().__init__()
        load_switch_states()
        self._start_cb = start_cb
        self._destroy_cb = destroy_cb

        self.setWindowTitle(
            f"{modules.metadata.name} Studio {modules.metadata.version} {modules.metadata.edition}"
        )
        self.setMinimumSize(920, 720)
        self.resize(ROOT_WIDTH, ROOT_HEIGHT)

        # ── Initialize switches first ─────────────────────────────────────
        def make_switch(field, label, tip):
            sw = _Switch(_(label), getattr(modules.globals, field, False), _(tip))
            sw.toggled.connect(
                lambda v, f=field: (
                    setattr(modules.globals, f, v),
                    save_switch_states(),
                )
            )
            return sw

        self.sw_keep_fps = make_switch("keep_fps", "Keep fps",
                                       "Output video keeps the original frame rate")
        self.sw_keep_audio = make_switch("keep_audio", "Keep audio",
                                         "Copy audio track from the source video to output")
        self.sw_keep_frames = make_switch("keep_frames", "Keep frames",
                                          "Keep extracted frames on disk after processing")
        self.sw_many_faces = make_switch("many_faces", "Many faces",
                                         "Swap every detected face, not just the primary one")
        self.sw_poisson = make_switch("poisson_blend", "Poisson Blend",
                                      "Blend face edges smoothly using Poisson blending")
        self.sw_color_fix = make_switch("color_correction", "Fix Blueish Cam",
                                        "Fix blue/green color cast from some webcams")
        self.sw_show_fps = make_switch("show_fps", "Show FPS",
                                       "Display frames-per-second counter on the live preview")
        self.sw_glasses = make_switch("glasses_mask", "Glasses Protect",
                                      "Preserve original glasses frames and eye contours")
        self.sw_bangs = make_switch("bangs_mask", "Bangs Protect",
                                    "Preserve forehead hair and bangs from being overwritten")
        self.sw_eyes = make_switch("eyes_mask", "Eyes & Gaze Protect",
                                   "Preserve natural eye gaze, pupil movements, and blinking")
        self.sw_vcam = make_switch("virtual_cam", "Virtual Camera",
                                   "Stream live output directly to virtual webcam (Zoom, Discord, OBS)")
        self.sw_map_faces = _Switch(_("Map faces"), modules.globals.map_faces,
                                    _("Manually assign which source face maps to which target face"))
        self.sw_map_faces.toggled.connect(self._on_map_faces_toggled)

        # ── Root layout ───────────────────────────────────────────────────
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        # Header
        layout.addLayout(self._build_header_row())

        # 2-column studio layout
        studio_layout = QHBoxLayout()
        studio_layout.setSpacing(14)

        # Left Column: Media & Identity (370px fixed width)
        left_col = QVBoxLayout()
        left_col.setSpacing(10)
        left_col.addWidget(self._build_media_card())
        left_col.addLayout(self._build_action_row())
        left_col.addStretch(1)

        # Right Column: Controls & Virtual Broadcast (flexible width)
        right_col = QVBoxLayout()
        right_col.setSpacing(10)
        right_col.addWidget(self._build_camera_card())
        right_col.addWidget(self._build_protection_card())
        right_col.addWidget(self._build_tuning_card())
        right_col.addStretch(1)

        studio_layout.addLayout(left_col, 0)
        studio_layout.addLayout(right_col, 1)
        layout.addLayout(studio_layout, 1)

        # Bottom status bar
        status_bar = QHBoxLayout()
        status_bar.setContentsMargins(4, 0, 4, 0)
        self._status_label = QLabel(_("Ready"))
        self._status_label.setObjectName("statusLabel")
        status_bar.addWidget(self._status_label)
        status_bar.addStretch(1)

        footer = QLabel("DeepLiveCam Studio • v" + str(modules.metadata.version))
        footer.setObjectName("linkLabel")
        footer.setCursor(Qt.CursorShape.PointingHandCursor)
        footer.mousePressEvent = lambda _e: webbrowser.open("https://deeplivecam.net")
        status_bar.addWidget(footer)

        layout.addLayout(status_bar)

    # ── header row ───────────────────────────────────────────────────────

    def _build_header_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(4, 2, 4, 2)

        brand_col = QVBoxLayout()
        brand_col.setSpacing(2)
        title = QLabel("DeepLiveCam Studio")
        title.setObjectName("headerTitle")
        subtitle = QLabel("AI Real-Time Face Swap & Virtual Broadcast Suite")
        subtitle.setObjectName("headerSubtitle")
        brand_col.addWidget(title)
        brand_col.addWidget(subtitle)
        row.addLayout(brand_col)

        row.addStretch(1)

        badge = QLabel(_get_provider_badge_text())
        badge.setObjectName("hardwareBadge")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(badge)

        return row

    # ── media card (left column) ─────────────────────────────────────────

    def _build_media_card(self) -> QGroupBox:
        card = QGroupBox(_("Media & Identity"))
        layout = QVBoxLayout(card)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 16, 12, 12)

        # Source face
        self.source_label = _make_image_drop(
            _("Source face"), (SOURCE_TARGET_PREVIEW_SIZE, SOURCE_TARGET_PREVIEW_SIZE)
        )
        layout.addWidget(self.source_label, alignment=Qt.AlignmentFlag.AlignCenter)

        src_row = QHBoxLayout()
        src_row.setSpacing(6)
        self.btn_select_source = QPushButton(_("Select a face"))
        self.btn_select_source.setToolTip(
            _("Choose the source face image to swap onto the target")
        )
        self.btn_select_source.clicked.connect(self._on_select_source)

        self.btn_random_face = QPushButton("🎲")
        self.btn_random_face.setObjectName("secondary")
        self.btn_random_face.setFixedWidth(42)
        self.btn_random_face.setToolTip(
            _("Get a random face from thispersondoesnotexist.com")
        )
        self.btn_random_face.clicked.connect(self._on_random_face)
        src_row.addWidget(self.btn_select_source)
        src_row.addWidget(self.btn_random_face)
        layout.addLayout(src_row)

        # Swap button
        swap_row = QHBoxLayout()
        self.btn_swap = QPushButton("⇅ " + _("Swap Direction"))
        self.btn_swap.setObjectName("secondary")
        self.btn_swap.setToolTip(_("Swap source and target images"))
        self.btn_swap.clicked.connect(self._on_swap_paths)
        swap_row.addStretch(1)
        swap_row.addWidget(self.btn_swap)
        swap_row.addStretch(1)
        layout.addLayout(swap_row)

        # Target media
        self.target_label = _make_image_drop(
            _("Target"), (SOURCE_TARGET_PREVIEW_SIZE, SOURCE_TARGET_PREVIEW_SIZE)
        )
        layout.addWidget(self.target_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.btn_select_target = QPushButton(_("Select a target"))
        self.btn_select_target.setToolTip(
            _("Choose the target image or video to apply face swap to")
        )
        self.btn_select_target.clicked.connect(self._on_select_target)
        layout.addWidget(self.btn_select_target)

        return card

    # ── action row (left column bottom) ──────────────────────────────────

    def _build_action_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        self.btn_start = QPushButton("▶ " + _("Start"))
        self.btn_start.setToolTip(_("Begin processing the target image/video with selected face"))
        self.btn_start.clicked.connect(self._on_start)

        self.btn_preview = QPushButton("👁 " + _("Preview"))
        self.btn_preview.setObjectName("secondary")
        self.btn_preview.setToolTip(_("Show/hide a preview of the processed output"))
        self.btn_preview.clicked.connect(self._on_toggle_preview)

        self.btn_destroy = QPushButton("⏹ " + _("Stop"))
        self.btn_destroy.setObjectName("danger")
        self.btn_destroy.setToolTip(_("Stop processing and close the application"))
        self.btn_destroy.clicked.connect(lambda: self._destroy_cb())

        row.addWidget(self.btn_start, 2)
        row.addWidget(self.btn_preview, 1)
        row.addWidget(self.btn_destroy, 1)
        return row

    # ── camera & virtual cam card (right column) ─────────────────────────

    def _build_camera_card(self) -> QGroupBox:
        card = QGroupBox(_("Camera & Virtual Broadcast"))
        grid = QGridLayout(card)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.setContentsMargins(12, 16, 12, 12)

        # Row 0: Camera selector + Live button
        grid.addWidget(QLabel(_("Camera:")), 0, 0)
        self._camera_indices, self._camera_names = get_available_cameras()

        self.cb_camera = QComboBox()
        if not self._camera_names or self._camera_names[0] == "No cameras found":
            self.cb_camera.addItem("No cameras found")
            self.cb_camera.setEnabled(False)
            cam_ok = False
        else:
            self.cb_camera.addItems(self._camera_names)
            cam_ok = True
        self.cb_camera.setToolTip(_("Select which camera to use for live mode"))
        grid.addWidget(self.cb_camera, 0, 1)

        self.btn_live = QPushButton("🎥 " + _("Live"))
        self.btn_live.setObjectName("live")
        self.btn_live.setEnabled(cam_ok)
        self.btn_live.setToolTip(_("Start real-time face swap using webcam"))
        self.btn_live.clicked.connect(self._on_live)
        grid.addWidget(self.btn_live, 0, 2)

        # Row 1: Resolution & Det Size
        grid.addWidget(QLabel(_("Resolution:")), 1, 0)
        self.cb_resolution = QComboBox()
        self._resolution_options = [
            ("640 x 480", (640, 480)),
            ("960 x 540 (qHD)", (960, 540)),
            ("1280 x 720 (HD)", (1280, 720)),
            ("1920 x 1080 (FHD)", (1920, 1080)),
        ]
        for label, _wh in self._resolution_options:
            self.cb_resolution.addItem(label)
        cur = tuple(modules.globals.capture_resolution)
        idx = next((i for i, (_, wh) in enumerate(self._resolution_options) if wh == cur), 0)
        self.cb_resolution.setCurrentIndex(idx)
        self.cb_resolution.currentIndexChanged.connect(self._on_resolution_change)
        self.cb_resolution.setToolTip(_(
            "Requested webcam resolution. 640x480 is native on virtually every webcam."
        ))
        grid.addWidget(self.cb_resolution, 1, 1)

        det_row = QHBoxLayout()
        det_row.setSpacing(6)
        det_row.addWidget(QLabel(_("Det size:")))
        self.cb_det_size = QComboBox()
        self._det_size_options = [160, 320, 640]
        for v in self._det_size_options:
            self.cb_det_size.addItem(f"{v}x{v}")
        cur_det = int(getattr(modules.globals, 'det_size', modules.globals.DEFAULT_DET_SIZE))
        if cur_det not in self._det_size_options:
            cur_det = (modules.globals.DEFAULT_DET_SIZE
                       if modules.globals.DEFAULT_DET_SIZE in self._det_size_options
                       else self._det_size_options[-1])
        idx = self._det_size_options.index(cur_det)
        self.cb_det_size.setCurrentIndex(idx)
        self.cb_det_size.currentIndexChanged.connect(self._on_det_size_change)
        self.cb_det_size.setToolTip(_(
            "Face detection resolution. Lower = faster, higher = better accuracy at distance."
        ))
        det_row.addWidget(self.cb_det_size)
        grid.addLayout(det_row, 1, 2)

        # Row 2: Live broadcast toggles
        toggles_grid = QGridLayout()
        toggles_grid.setHorizontalSpacing(14)
        toggles_grid.setVerticalSpacing(4)
        toggles_grid.addWidget(self.sw_vcam, 0, 0)
        toggles_grid.addWidget(self.sw_show_fps, 0, 1)
        toggles_grid.addWidget(self.sw_color_fix, 1, 0)
        toggles_grid.addWidget(self.sw_poisson, 1, 1)
        grid.addLayout(toggles_grid, 2, 0, 1, 3)

        grid.setColumnStretch(1, 1)
        return card

    # ── protection card (right column) ───────────────────────────────────

    def _build_protection_card(self) -> QGroupBox:
        card = QGroupBox(_("Face Feature Preservation & Masks"))
        layout = QVBoxLayout(card)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 16, 12, 12)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(4)
        grid.addWidget(self.sw_eyes, 0, 0)
        grid.addWidget(self.sw_glasses, 0, 1)
        grid.addWidget(self.sw_bangs, 1, 0)
        grid.addWidget(self.sw_map_faces, 1, 1)
        layout.addLayout(grid)

        # Mouth mask slider row
        mouth_row = QHBoxLayout()
        mouth_row.setSpacing(8)
        lbl_mouth = QLabel(_("Mouth Mask:"))
        lbl_mouth.setMinimumWidth(85)
        mouth_row.addWidget(lbl_mouth)

        self.s_mouth = QSlider(Qt.Orientation.Horizontal)
        self.s_mouth.setRange(0, 100)
        cur_mouth = int(getattr(modules.globals, "mouth_mask_size", 0.0))
        self.s_mouth.setValue(cur_mouth)
        self.s_mouth.valueChanged.connect(self._on_mouth_mask_change)
        self.s_mouth.sliderPressed.connect(self._on_mouth_mask_pressed)
        self.s_mouth.sliderReleased.connect(self._on_mouth_mask_released)
        self.s_mouth.setToolTip(
            _("0% = use swapped mouth, 100% = expose original mouth to chin area")
        )
        mouth_row.addWidget(self.s_mouth, 1)

        self.lbl_mouth_val = QLabel(f"{cur_mouth}%")
        self.lbl_mouth_val.setObjectName("valueBadge")
        self.lbl_mouth_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mouth_row.addWidget(self.lbl_mouth_val)

        layout.addLayout(mouth_row)
        return card

    # ── tuning & quality card (right column) ─────────────────────────────

    def _build_tuning_card(self) -> QGroupBox:
        card = QGroupBox(_("Refinement & Quality"))
        layout = QVBoxLayout(card)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 16, 12, 12)

        # Enhancer selector row
        enh_row = QHBoxLayout()
        enh_row.setSpacing(8)
        lbl_enh = QLabel(_("Face Enhancer:"))
        lbl_enh.setMinimumWidth(85)
        enh_row.addWidget(lbl_enh)

        self.cb_enhancer = QComboBox()
        self.cb_enhancer.addItems(["None", "GFPGAN", "GPEN-512", "GPEN-256"])
        initial = "None"
        if modules.globals.fp_ui.get("face_enhancer", False):
            initial = "GFPGAN"
        elif modules.globals.fp_ui.get("face_enhancer_gpen512", False):
            initial = "GPEN-512"
        elif modules.globals.fp_ui.get("face_enhancer_gpen256", False):
            initial = "GPEN-256"
        self.cb_enhancer.setCurrentText(initial)
        self.cb_enhancer.currentTextChanged.connect(self._on_enhancer_change)
        self.cb_enhancer.setToolTip(_("Select a face enhancement model (None = no enhancement)"))
        enh_row.addWidget(self.cb_enhancer, 1)
        layout.addLayout(enh_row)

        # Transparency slider row
        trans_row = QHBoxLayout()
        trans_row.setSpacing(8)
        lbl_trans = QLabel(_("Transparency:"))
        lbl_trans.setMinimumWidth(85)
        trans_row.addWidget(lbl_trans)

        self.s_transparency = QSlider(Qt.Orientation.Horizontal)
        self.s_transparency.setRange(0, 100)
        cur_trans = int(getattr(modules.globals, "opacity", 1.0) * 100)
        self.s_transparency.setValue(cur_trans)
        self.s_transparency.valueChanged.connect(lambda v: self._on_transparency_change(v / 100.0))
        self.s_transparency.setToolTip(
            _("Blend between original and swapped face (0% = original, 100% = fully swapped)")
        )
        trans_row.addWidget(self.s_transparency, 1)

        self.lbl_trans_val = QLabel(f"{cur_trans}%")
        self.lbl_trans_val.setObjectName("valueBadge")
        self.lbl_trans_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        trans_row.addWidget(self.lbl_trans_val)
        layout.addLayout(trans_row)

        # Sharpness slider row
        sharp_row = QHBoxLayout()
        sharp_row.setSpacing(8)
        lbl_sharp = QLabel(_("Sharpness:"))
        lbl_sharp.setMinimumWidth(85)
        sharp_row.addWidget(lbl_sharp)

        self.s_sharpness = QSlider(Qt.Orientation.Horizontal)
        self.s_sharpness.setRange(0, 50)
        cur_sharp = float(getattr(modules.globals, "sharpness", 0.0))
        self.s_sharpness.setValue(int(cur_sharp * 10))
        self.s_sharpness.valueChanged.connect(lambda v: self._on_sharpness_change(v / 10.0))
        self.s_sharpness.setToolTip(_("Sharpen the enhanced face output"))
        sharp_row.addWidget(self.s_sharpness, 1)

        self.lbl_sharp_val = QLabel(f"{cur_sharp:.1f}")
        self.lbl_sharp_val.setObjectName("valueBadge")
        self.lbl_sharp_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sharp_row.addWidget(self.lbl_sharp_val)
        layout.addLayout(sharp_row)

        # Video / general toggles grid
        toggles_grid = QGridLayout()
        toggles_grid.setHorizontalSpacing(14)
        toggles_grid.setVerticalSpacing(4)
        toggles_grid.addWidget(self.sw_keep_fps, 0, 0)
        toggles_grid.addWidget(self.sw_keep_audio, 0, 1)
        toggles_grid.addWidget(self.sw_keep_frames, 1, 0)
        toggles_grid.addWidget(self.sw_many_faces, 1, 1)
        layout.addLayout(toggles_grid)

        return card

    # ── Backward-compatibility stubs ─────────────────────────────────────

    def _build_options_card(self) -> QGroupBox:
        return self._build_protection_card()

    def _build_sliders_card(self) -> QGroupBox:
        return self._build_tuning_card()

    def _build_image_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(self.source_label)
        row.addWidget(self.btn_swap)
        row.addWidget(self.target_label)
        return row

    def _on_resolution_change(self, idx: int) -> None:
        if 0 <= idx < len(self._resolution_options):
            _label, wh = self._resolution_options[idx]
            modules.globals.capture_resolution = wh
            save_switch_states()
            # Pre-translate the static template; append the dynamic size after
            # so the message still participates in localization (update_status
            # only translates whole-string keys).
            update_status(
                _("Capture resolution set (applies on next Live start):")
                + f" {wh[0]}x{wh[1]}"
            )

    def _on_det_size_change(self, idx: int) -> None:
        if 0 <= idx < len(self._det_size_options):
            v = self._det_size_options[idx]
            modules.globals.det_size = v
            reset_face_analyser()
            save_switch_states()
            update_status(
                _("Face detection size set (analyser re-inits on next call):")
                + f" {v}x{v}"
            )

    # ── slot handlers ────────────────────────────────────────────────────

    def set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def _on_select_source(self) -> None:
        global _RECENT_SOURCE_DIR
        if _PREVIEW is not None:
            _PREVIEW.hide()
        path, _filter = QFileDialog.getOpenFileName(
            self, _("select an source image"),
            _RECENT_SOURCE_DIR or "",
            _IMAGE_FILE_FILTER,
        )
        if path and is_image(path):
            modules.globals.source_path = path
            _RECENT_SOURCE_DIR = os.path.dirname(path)
            self.source_label.setPixmap(
                render_image_preview(path, (SOURCE_TARGET_PREVIEW_SIZE, SOURCE_TARGET_PREVIEW_SIZE))
            )
            self.source_label.setText("")
        elif not path:
            return
        else:
            modules.globals.source_path = None
            self.source_label.clear()
            self.source_label.setText(_("Source face"))

    def _on_select_target(self) -> None:
        global _RECENT_TARGET_DIR
        if _PREVIEW is not None:
            _PREVIEW.hide()
        path, _filter = QFileDialog.getOpenFileName(
            self, _("select an target image or video"),
            _RECENT_TARGET_DIR or "",
            _MEDIA_FILE_FILTER,
        )
        if not path:
            return
        if is_image(path):
            modules.globals.target_path = path
            _RECENT_TARGET_DIR = os.path.dirname(path)
            self.target_label.setPixmap(
                render_image_preview(path, (SOURCE_TARGET_PREVIEW_SIZE, SOURCE_TARGET_PREVIEW_SIZE))
            )
            self.target_label.setText("")
        elif is_video(path):
            modules.globals.target_path = path
            _RECENT_TARGET_DIR = os.path.dirname(path)
            pm = render_video_preview(path, (SOURCE_TARGET_PREVIEW_SIZE, SOURCE_TARGET_PREVIEW_SIZE))
            if pm:
                self.target_label.setPixmap(pm)
                self.target_label.setText("")
        else:
            modules.globals.target_path = None
            self.target_label.clear()
            self.target_label.setText(_("Target"))

    def _on_random_face(self) -> None:
        if _PREVIEW is not None:
            _PREVIEW.hide()
        try:
            response = requests.get(
                "https://thispersondoesnotexist.com/random-person.jpeg",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                raise ValueError(f"expected an image, got {content_type!r}")
            temp_path = os.path.join(tempfile.gettempdir(), "deep_live_cam_random_face.jpg")
            staging_path = f"{temp_path}.part"
            # Download into a staging file and validate there. Writing straight
            # to temp_path would destroy the image currently in use whenever a
            # later fetch comes back bad, and content-type alone does not prove
            # the bytes decode.
            try:
                with open(staging_path, "wb") as f:
                    f.write(response.content)
                pixmap = render_image_preview(
                    staging_path, (SOURCE_TARGET_PREVIEW_SIZE, SOURCE_TARGET_PREVIEW_SIZE)
                )
                if imread_unicode(staging_path) is None:
                    raise ValueError("downloaded image could not be decoded")
                os.replace(staging_path, temp_path)
            except Exception:
                try:
                    os.remove(staging_path)
                except OSError:
                    pass
                raise
            modules.globals.source_path = temp_path
            self.source_label.setPixmap(pixmap)
            self.source_label.setText("")
        except Exception as exc:
            print(f"Failed to fetch random face: {exc}")

    def _on_swap_paths(self) -> None:
        global _RECENT_SOURCE_DIR, _RECENT_TARGET_DIR
        sp = modules.globals.source_path
        tp = modules.globals.target_path
        if not (sp and tp and is_image(sp) and is_image(tp)):
            return
        modules.globals.source_path, modules.globals.target_path = tp, sp
        _RECENT_SOURCE_DIR = os.path.dirname(tp)
        _RECENT_TARGET_DIR = os.path.dirname(sp)
        if _PREVIEW is not None:
            _PREVIEW.hide()
        self.source_label.setPixmap(
            render_image_preview(tp, (SOURCE_TARGET_PREVIEW_SIZE, SOURCE_TARGET_PREVIEW_SIZE))
        )
        self.target_label.setPixmap(
            render_image_preview(sp, (SOURCE_TARGET_PREVIEW_SIZE, SOURCE_TARGET_PREVIEW_SIZE))
        )
        self.source_label.setText("")
        self.target_label.setText("")

    def _on_map_faces_toggled(self, value: bool) -> None:
        modules.globals.map_faces = value
        save_switch_states()
        if not value:
            close_mapper_window()

    def _on_enhancer_change(self, choice: str) -> None:
        key_map = {
            "None": None,
            "GFPGAN": "face_enhancer",
            "GPEN-512": "face_enhancer_gpen512",
            "GPEN-256": "face_enhancer_gpen256",
        }
        for key in ("face_enhancer", "face_enhancer_gpen256", "face_enhancer_gpen512"):
            _update_tumbler(key, False)
        selected = key_map.get(choice)
        if selected:
            _update_tumbler(selected, True)
        save_switch_states()

    def _on_transparency_change(self, value: float) -> None:
        modules.globals.opacity = value
        pct = int(value * 100)
        if hasattr(self, "lbl_trans_val"):
            self.lbl_trans_val.setText(f"{pct}%")
        if pct == 0:
            modules.globals.fp_ui["face_enhancer"] = False
            update_status("Transparency set to 0% - Face swapping disabled.")
        elif pct == 100:
            modules.globals.face_swapper_enabled = True
            update_status("Transparency set to 100%.")
        else:
            modules.globals.face_swapper_enabled = True
            update_status(f"Transparency set to {pct}%")

    def _on_sharpness_change(self, value: float) -> None:
        modules.globals.sharpness = value
        if hasattr(self, "lbl_sharp_val"):
            self.lbl_sharp_val.setText(f"{value:.1f}")
        update_status(f"Sharpness set to {value:.1f}")

    def _on_mouth_mask_change(self, value: float) -> None:
        modules.globals.mouth_mask_size = value
        modules.globals.mouth_mask = value > 0
        if hasattr(self, "lbl_mouth_val"):
            self.lbl_mouth_val.setText(f"{int(value)}%")
        if value <= 0:
            modules.globals.show_mouth_mask_box = False

    def _on_mouth_mask_pressed(self) -> None:
        if modules.globals.mouth_mask_size > 0:
            modules.globals.show_mouth_mask_box = True

    def _on_mouth_mask_released(self) -> None:
        modules.globals.show_mouth_mask_box = False

    def _on_start(self) -> None:
        if _MAPPER is not None and _MAPPER.isVisible():
            update_status("Please complete pop-up or close it.")
            return
        if modules.globals.map_faces:
            modules.globals.source_target_map = []
            if is_image(modules.globals.target_path):
                update_status("Getting unique faces")
                get_unique_faces_from_target_image()
            elif is_video(modules.globals.target_path):
                update_status("Getting unique faces")
                get_unique_faces_from_target_video()
            if modules.globals.source_target_map:
                _open_mapper_dialog(self._start_cb, modules.globals.source_target_map)
            else:
                update_status("No faces found in target")
        else:
            self._select_output_and_start()

    def _select_output_and_start(self) -> None:
        global _RECENT_OUTPUT_DIR
        if is_image(modules.globals.target_path):
            path, _f = QFileDialog.getSaveFileName(
                self, _("save image output file"),
                os.path.join(_RECENT_OUTPUT_DIR or "", "output.png"),
                _IMAGE_FILE_FILTER,
            )
        elif is_video(modules.globals.target_path):
            path, _f = QFileDialog.getSaveFileName(
                self, _("save video output file"),
                os.path.join(_RECENT_OUTPUT_DIR or "", "output.mp4"),
                _VIDEO_FILE_FILTER,
            )
        else:
            return
        if path:
            modules.globals.output_path = path
            _RECENT_OUTPUT_DIR = os.path.dirname(path)
            self._start_cb()

    def _on_toggle_preview(self) -> None:
        if _PREVIEW is None:
            return
        if _PREVIEW.isVisible():
            _PREVIEW.hide()
        elif modules.globals.source_path and modules.globals.target_path:
            _PREVIEW.init_for_target()
            _PREVIEW.refresh_frame(0)
            _PREVIEW.show()

    def _on_live(self) -> None:
        idx = self.cb_camera.currentIndex()
        if idx < 0 or idx >= len(self._camera_indices):
            update_status("No camera available")
            return
        camera_index = self._camera_indices[idx]
        if _LIVE_MAPPER is not None and _LIVE_MAPPER.isVisible():
            update_status("Source x Target Mapper is already open.")
            _LIVE_MAPPER.raise_()
            return
        if not modules.globals.map_faces:
            if modules.globals.source_path is None:
                update_status("Please select a source image first")
                return
            from modules.face_analyser import get_face_analyser
            from modules.processors.frame.face_swapper import get_face_swapper
            get_face_analyser()
            get_face_swapper()
            _open_webcam_preview(camera_index)
        else:
            modules.globals.source_target_map = []
            _open_live_mapper_dialog(camera_index, modules.globals.source_target_map)

    def closeEvent(self, event):
        # Treat OS-level close as Destroy click
        self._destroy_cb()
        event.accept()


def _update_tumbler(var: str, value: bool) -> None:
    modules.globals.fp_ui[var] = value
    save_switch_states()
    # If we're currently in a live preview, refresh frame processors so
    # toggling enhancers takes effect immediately.
    if _WEBCAM_PREVIEW is not None and _WEBCAM_PREVIEW.isVisible():
        get_frame_processors_modules(modules.globals.frame_processors)


# ─── preview window (still-image / video scrub) ──────────────────────────


class PreviewWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(_("Preview"))
        self.resize(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._image_label, 1)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.valueChanged.connect(self.refresh_frame)
        layout.addWidget(self._slider)

    def init_for_target(self) -> None:
        if is_image(modules.globals.target_path):
            self._slider.hide()
        elif is_video(modules.globals.target_path):
            total = get_video_frame_total(modules.globals.target_path)
            self._slider.setRange(0, max(0, total - 1))
            self._slider.setValue(0)
            self._slider.show()

    def refresh_frame(self, frame_number: int = 0) -> None:
        if not (modules.globals.source_path and modules.globals.target_path):
            return
        update_status("Processing...")
        temp_frame = get_video_frame(modules.globals.target_path, frame_number)
        if modules.globals.nsfw_filter and check_and_ignore_nsfw(temp_frame):
            return
        source_frame = imread_unicode(modules.globals.source_path)
        if source_frame is None:
            update_status(
                f"Could not read source image: {modules.globals.source_path}"
            )
            return
        from modules.processors.frame.core import get_frame_processors_modules as _gfpm
        for fp in _gfpm(modules.globals.frame_processors):
            temp_frame = fp.process_frame(get_one_face(source_frame), temp_frame)
        # Fit to current widget size while preserving aspect ratio.
        h, w = temp_frame.shape[:2]
        bound_w = min(PREVIEW_MAX_WIDTH, max(self.width(), PREVIEW_DEFAULT_WIDTH))
        bound_h = min(PREVIEW_MAX_HEIGHT, max(self.height(), PREVIEW_DEFAULT_HEIGHT))
        ratio = min(bound_w / w, bound_h / h)
        new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
        temp_frame = cv2.resize(temp_frame, new_size, interpolation=cv2.INTER_LANCZOS4)
        self._image_label.setPixmap(_bgr_to_qpixmap(temp_frame))
        update_status("Processing succeed!")


# ─── webcam preview window ───────────────────────────────────────────────


class _CaptureWorker(QThread):
    """Reads frames from the camera into a bounded queue. Drops on overflow."""

    def __init__(self, cap, capture_queue: queue.Queue, stop_event: threading.Event):
        super().__init__()
        self._cap = cap
        self._queue = capture_queue
        self._stop = stop_event
        self._latest_frame = None
        self._frame_lock = threading.Lock()

    def get_latest_frame(self):
        with self._frame_lock:
            return self._latest_frame

    def run(self) -> None:
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret:
                self._stop.set()
                break
            with self._frame_lock:
                self._latest_frame = frame
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(frame)
                except queue.Full:
                    pass


class _ProcessingWorker(QThread):
    """Pulls raw frames, runs detect/swap/enhance, pushes processed frames."""

    def __init__(self, capture_queue, processed_queue, stop_event, camera_fps: float):
        super().__init__()
        self._cq = capture_queue
        self._pq = processed_queue
        self._stop = stop_event
        self._fps = camera_fps

    def run(self) -> None:
        frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
        source_image = None
        last_source_path = None
        reported_source_error = None
        prev_time = time.time()
        fps_update_interval = 0.5
        frame_count = 0
        fps = 0.0
        det_interval = max(2, round(self._fps * 0.1)) if self._fps and self._fps > 0 else 3
        frames_since_det = det_interval  # Force detection on first frame
        cached_target_face = None
        cached_many_faces = None
        prev_raw_gray = None

        from modules.virtual_cam import get_virtual_camera_manager
        vcam_mgr = get_virtual_camera_manager()

        while not self._stop.is_set():
            try:
                frame = self._cq.get(timeout=0.05)
            except queue.Empty:
                continue

            temp_frame = frame
            if modules.globals.live_mirror:
                temp_frame = gpu_flip(temp_frame, 1)

            if not modules.globals.map_faces:
                if (
                    modules.globals.source_path
                    and modules.globals.source_path != last_source_path
                ):
                    source_frame = imread_unicode(modules.globals.source_path)
                    if source_frame is None:
                        # Clear the cached path along with the image. It keeps
                        # a source that is only transiently unreadable (still
                        # being written, on a volume that just went away) from
                        # being given up on, and it keeps a bad pick from
                        # sticking: with the path cached, A -> unreadable B -> A
                        # never reloaded A, so swapping stayed off. Retrying
                        # every frame is cheap; the message is reported once per
                        # path so it cannot flood the status line.
                        source_image = None
                        last_source_path = None
                        if reported_source_error != modules.globals.source_path:
                            reported_source_error = modules.globals.source_path
                            update_status(
                                f"Could not read source image: {modules.globals.source_path}"
                            )
                    else:
                        last_source_path = modules.globals.source_path
                        reported_source_error = None
                        source_image = get_one_face(source_frame)

                should_detect = False
                if modules.globals.many_faces:
                    if not cached_many_faces or frames_since_det >= (det_interval - 1):
                        should_detect = True
                else:
                    if cached_target_face is None:
                        should_detect = True
                    elif hasattr(cached_target_face, "det_score") and (
                        cached_target_face.det_score is None or cached_target_face.det_score < 0.5
                    ):
                        should_detect = True
                    elif frames_since_det >= (det_interval - 1):
                        should_detect = True

                curr_raw_gray = None
                if not should_detect:
                    frames_since_det += 1
                    # Immediate re-detection if head motion is detected between raw frames
                    if prev_raw_gray is not None and cached_target_face is not None and getattr(cached_target_face, 'bbox', None) is not None:
                        try:
                            curr_raw_gray = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2GRAY) if temp_frame.ndim == 3 else temp_frame
                            bbox = cached_target_face.bbox.astype(int)
                            h, w = curr_raw_gray.shape[:2]
                            x1, y1 = max(0, bbox[0]), max(0, bbox[1])
                            x2, y2 = min(w, bbox[2]), min(h, bbox[3])
                            if (x2 - x1) > 20 and (y2 - y1) > 20:
                                diff = float(np.mean(cv2.absdiff(curr_raw_gray[y1:y2, x1:x2], prev_raw_gray[y1:y2, x1:x2])))
                                if diff > 18.0:
                                    should_detect = True
                        except Exception:
                            pass

                # Store pre-swap raw frame for clean motion detection on subsequent frames (reuse curr_raw_gray)
                try:
                    if curr_raw_gray is not None:
                        prev_raw_gray = curr_raw_gray
                    else:
                        prev_raw_gray = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2GRAY) if temp_frame.ndim == 3 else temp_frame
                except Exception:
                    prev_raw_gray = None

                if should_detect:
                    frames_since_det = 0
                    if modules.globals.many_faces:
                        cached_target_face = None
                        cached_many_faces = detect_many_faces_fast(temp_frame)
                    else:
                        cached_target_face = detect_one_face_fast(temp_frame)
                        cached_many_faces = None

                cached_faces = None
                if cached_many_faces:
                    cached_faces = cached_many_faces
                elif cached_target_face is not None:
                    cached_faces = [cached_target_face]

                # Fast detection skips the 2d106 landmark model, but mouth, glasses,
                # and bangs masks need it. Attach landmarks on demand (computed once per
                # detection cycle — the helper no-ops if already present).
                mask_needs_landmarks = (
                    modules.globals.mouth_mask
                    or getattr(modules.globals, "mouth_mask_size", 0.0) > 0
                    or getattr(modules.globals, "glasses_mask", False)
                    or getattr(modules.globals, "glasses_mask_size", 0.0) > 0
                    or getattr(modules.globals, "bangs_mask", False)
                    or getattr(modules.globals, "forehead_cutoff", 0.0) > 0
                    or getattr(modules.globals, "eyes_mask", False)
                    or getattr(modules.globals, "eyes_mask_size", 0.0) > 0
                )
                if mask_needs_landmarks and cached_faces:
                    ensure_landmarks(temp_frame, cached_faces)

                for fp in frame_processors:
                    if fp.NAME == "DLC.FACE-ENHANCER":
                        if modules.globals.fp_ui["face_enhancer"]:
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                    elif fp.NAME == "DLC.FACE-ENHANCER-GPEN256":
                        if modules.globals.fp_ui.get("face_enhancer_gpen256", False):
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                    elif fp.NAME == "DLC.FACE-ENHANCER-GPEN512":
                        if modules.globals.fp_ui.get("face_enhancer_gpen512", False):
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                    elif fp.NAME == "DLC.FACE-SWAPPER":
                        swapped_bboxes = []
                        if modules.globals.many_faces and cached_many_faces:
                            for t_face in cached_many_faces:
                                temp_frame = fp.swap_face(source_image, t_face, temp_frame)
                                if hasattr(t_face, "bbox") and t_face.bbox is not None:
                                    swapped_bboxes.append(t_face.bbox.astype(int))
                        elif cached_target_face is not None:
                            temp_frame = fp.swap_face(
                                source_image, cached_target_face, temp_frame
                            )
                            if (
                                hasattr(cached_target_face, "bbox")
                                and cached_target_face.bbox is not None
                            ):
                                swapped_bboxes.append(cached_target_face.bbox.astype(int))
                        temp_frame = fp.apply_post_processing(temp_frame, swapped_bboxes)
                    else:
                        temp_frame = fp.process_frame(source_image, temp_frame)
            else:
                modules.globals.target_path = None
                for fp in frame_processors:
                    if fp.NAME == "DLC.FACE-ENHANCER":
                        if modules.globals.fp_ui["face_enhancer"]:
                            temp_frame = fp.process_frame_v2(temp_frame)
                    elif fp.NAME in ("DLC.FACE-ENHANCER-GPEN256", "DLC.FACE-ENHANCER-GPEN512"):
                        fp_key = fp.NAME.split(".")[-1].lower().replace("-", "_")
                        if modules.globals.fp_ui.get(fp_key, False):
                            temp_frame = fp.process_frame_v2(temp_frame)
                    else:
                        temp_frame = fp.process_frame_v2(temp_frame)

            current_time = time.time()
            frame_count += 1
            if current_time - prev_time >= fps_update_interval:
                fps = frame_count / (current_time - prev_time)
                frame_count = 0
                prev_time = current_time

            if modules.globals.show_fps:
                cv2.putText(
                    temp_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                )

            try:
                self._pq.put_nowait(temp_frame)
            except queue.Full:
                try:
                    self._pq.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._pq.put_nowait(temp_frame)
                except queue.Full:
                    pass

            if getattr(modules.globals, "virtual_cam", False):
                if not vcam_mgr.is_active:
                    fh, fw = temp_frame.shape[:2]
                    vcam_mgr.start(
                        fw,
                        fh,
                        self._fps or 30.0,
                        getattr(modules.globals, "virtual_cam_backend", "auto"),
                    )
                vcam_mgr.send_frame(temp_frame)
            elif vcam_mgr.is_active:
                vcam_mgr.stop()

        if vcam_mgr is not None and vcam_mgr.is_active:
            vcam_mgr.stop()


class WebcamPreviewWindow(QWidget):
    def __init__(self, camera_index: int):
        super().__init__()
        self.setWindowTitle("Live Preview")
        self.resize(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._image_label, 1)

        self._cap = VideoCapturer(camera_index)
        req_w, req_h = modules.globals.capture_resolution
        if not self._cap.start(req_w, req_h, 60):
            update_status("Failed to start camera")
            QTimer.singleShot(0, self.close)
            return

        camera_fps = self._cap.actual_fps
        print(
            f"[webcam] Camera running at {self._cap.actual_width}x"
            f"{self._cap.actual_height}@{camera_fps:.0f}fps"
        )

        # Enforce accelerated real-time tracking cadence for live preview (20-30 FPS)
        if getattr(modules.globals, "swap_interval", 1) <= 1:
            modules.globals.swap_interval = 3
        if not getattr(modules.globals, "enable_interpolation", False):
            modules.globals.enable_interpolation = True

        self._capture_queue: queue.Queue = queue.Queue(maxsize=2)
        self._processed_queue: queue.Queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()

        self._capture_worker = _CaptureWorker(
            self._cap, self._capture_queue, self._stop_event
        )
        self._processing_worker = _ProcessingWorker(
            self._capture_queue, self._processed_queue, self._stop_event, camera_fps
        )
        self._capture_worker.start()
        self._processing_worker.start()

        # Poll at ~2x camera fps so we never block but also don't burn CPU.
        poll_ms = max(1, min(16, int(500 / max(camera_fps, 1))))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(poll_ms)

    def _tick(self) -> None:
        if self._stop_event.is_set():
            self.close()
            return
        bgr_frame = None
        try:
            bgr_frame = self._processed_queue.get_nowait()
            self._last_rendered_frame = bgr_frame
        except queue.Empty:
            pass

        if bgr_frame is None:
            # While background inference is warming up or between initial frames,
            # or when no source image is active, display the live camera capture
            # immediately so UI preview never freezes.
            if (not hasattr(self, "_last_rendered_frame") or self._last_rendered_frame is None
                    or not modules.globals.source_path):
                bgr_frame = self._capture_worker.get_latest_frame()
                if bgr_frame is not None and modules.globals.live_mirror:
                    bgr_frame = gpu_flip(bgr_frame, 1)
            else:
                return

        if bgr_frame is not None:
            bgr_frame = fit_image_to_size(bgr_frame, self.width(), self.height())
            self._image_label.setPixmap(_bgr_to_qpixmap(bgr_frame))

    def closeEvent(self, event) -> None:
        # __init__ can bail out before these exist (e.g. the camera fails to open),
        # and closeEvent still runs.
        stop_event = getattr(self, "_stop_event", None)
        if stop_event is not None:
            stop_event.set()
        timer = getattr(self, "_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
        # Shutdown has to actually complete. Qt aborts the process when a
        # running QThread is destroyed, and holding a straggler in a module
        # global only moves that abort to interpreter exit. Both loops re-check
        # the stop flag between bounded operations (a camera read, a 50ms queue
        # get, one inference), so waiting terminates; a first-time model load
        # is simply slow, which is what the grace period absorbs quietly.
        for name in ("_capture_worker", "_processing_worker"):
            worker = getattr(self, name, None)
            if worker is None:
                continue
            if not worker.wait(WORKER_SHUTDOWN_GRACE_MS):
                print(
                    f"[webcam] still waiting for {name[1:]} to stop...",
                    flush=True,
                )
                worker.wait()
        # Only now: releasing the capture while the capture thread could still
        # be inside cap.read() is not safe.
        cap = getattr(self, "_cap", None)
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        try:
            from modules.virtual_cam import get_virtual_camera_manager
            get_virtual_camera_manager().stop()
        except Exception:
            pass
        global _WEBCAM_PREVIEW
        if _WEBCAM_PREVIEW is self:
            _WEBCAM_PREVIEW = None
        event.accept()


def _open_webcam_preview(camera_index: int) -> None:
    global _WEBCAM_PREVIEW
    if _WEBCAM_PREVIEW is not None:
        _WEBCAM_PREVIEW.close()
    _WEBCAM_PREVIEW = WebcamPreviewWindow(camera_index)
    _WEBCAM_PREVIEW.show()


# ─── mapper dialogs (image/video + live) ────────────────────────────────


def _make_thumb(cv2_img: np.ndarray) -> QPixmap:
    rgb = gpu_cvt_color(cv2_img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).resize(
        (MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE), Image.LANCZOS
    )
    return _pil_to_qpixmap(image)


class MapperDialog(QDialog):
    """Source × Target mapper for image / video processing."""

    def __init__(self, start_cb: Callable, mapping: list):
        super().__init__(_MAIN)
        self._start_cb = start_cb
        self._map = mapping
        self.setWindowTitle(_("Source x Target Mapper"))
        self.resize(POPUP_WIDTH, POPUP_HEIGHT)
        layout = QVBoxLayout(self)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll, 1)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        btn_submit = QPushButton(_("Submit"))
        btn_submit.clicked.connect(self._on_submit)
        layout.addWidget(btn_submit, alignment=Qt.AlignmentFlag.AlignCenter)

        self._rebuild()

    def set_status(self, text: str) -> None:
        self._status.setText(_(text))

    def _rebuild(self) -> None:
        body = QWidget()
        grid = QGridLayout(body)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        for item in self._map:
            row = item["id"]
            btn = QPushButton(_("Select source image"))
            btn.setFixedWidth(200)
            btn.clicked.connect(lambda _c, n=row: self._select_source(n))
            grid.addWidget(btn, row, 0)

            src_label = QLabel(f"S-{row}")
            src_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            src_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            src_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(src_label, row, 1)
            if "source" in item:
                src_label.setPixmap(_make_thumb(item["source"]["cv2"]))
                src_label.setText("")

            x_label = QLabel("×")
            x_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(x_label, row, 2)

            tgt_label = QLabel(f"T-{row}")
            tgt_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            tgt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tgt_label.setStyleSheet("border: 1px solid #555;")
            grid.addWidget(tgt_label, row, 3)
            if "target" in item:
                tgt_label.setPixmap(_make_thumb(item["target"]["cv2"]))
                tgt_label.setText("")

        grid.setRowStretch(grid.rowCount(), 1)
        self._scroll.setWidget(body)

    def _select_source(self, row: int) -> None:
        path, _f = QFileDialog.getOpenFileName(
            self, _("select an source image"),
            _RECENT_SOURCE_DIR or "",
            _IMAGE_FILE_FILTER,
        )
        if not path:
            return
        cv2_img = imread_unicode(path)
        face = get_one_face(cv2_img)
        if face is None:
            self.set_status("Face could not be detected in last upload!")
            return
        x_min, y_min, x_max, y_max = face["bbox"]
        self._map[row]["source"] = {
            "cv2": cv2_img[int(y_min):int(y_max), int(x_min):int(x_max)],
            "face": face,
        }
        self._rebuild()

    def _on_submit(self) -> None:
        if has_valid_map():
            self.accept()
            _MAIN._select_output_and_start()
        else:
            self.set_status("Atleast 1 source with target is required!")


class LiveMapperDialog(QDialog):
    """Source × Target mapper for live webcam mode."""

    def __init__(self, camera_index: int, mapping: list):
        super().__init__(_MAIN)
        self._camera_index = camera_index
        self._map = mapping
        self.setWindowTitle(_("Source x Target Mapper"))
        self.resize(POPUP_LIVE_WIDTH, POPUP_LIVE_HEIGHT)
        layout = QVBoxLayout(self)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll, 1)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        btn_row = QHBoxLayout()
        for text, slot in (
            (_("Add"), self._on_add),
            (_("Clear"), self._on_clear),
            (_("Submit"), self._on_submit),
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        self._rebuild()

    def set_status(self, text: str) -> None:
        self._status.setText(_(text))

    def _rebuild(self) -> None:
        body = QWidget()
        grid = QGridLayout(body)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        for item in self._map:
            row = item["id"]
            btn_s = QPushButton(_("Select source image"))
            btn_s.setFixedWidth(200)
            btn_s.clicked.connect(lambda _c, n=row: self._select_face(n, "source"))
            grid.addWidget(btn_s, row, 0)

            src_label = QLabel(f"S-{row}")
            src_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            src_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            src_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(src_label, row, 1)
            if "source" in item:
                src_label.setPixmap(_make_thumb(item["source"]["cv2"]))
                src_label.setText("")

            x_label = QLabel("×")
            x_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(x_label, row, 2)

            btn_t = QPushButton(_("Select target image"))
            btn_t.setFixedWidth(200)
            btn_t.clicked.connect(lambda _c, n=row: self._select_face(n, "target"))
            grid.addWidget(btn_t, row, 3)

            tgt_label = QLabel(f"T-{row}")
            tgt_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            tgt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tgt_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(tgt_label, row, 4)
            if "target" in item:
                tgt_label.setPixmap(_make_thumb(item["target"]["cv2"]))
                tgt_label.setText("")

        grid.setRowStretch(grid.rowCount(), 1)
        self._scroll.setWidget(body)

    def _select_face(self, row: int, kind: str) -> None:
        path, _f = QFileDialog.getOpenFileName(
            self, _("select an source image"),
            _RECENT_SOURCE_DIR or "",
            _IMAGE_FILE_FILTER,
        )
        if not path:
            return
        cv2_img = imread_unicode(path)
        face = get_one_face(cv2_img)
        if face is None:
            self.set_status("Face could not be detected in last upload!")
            return
        x_min, y_min, x_max, y_max = face["bbox"]
        self._map[row][kind] = {
            "cv2": cv2_img[int(y_min):int(y_max), int(x_min):int(x_max)],
            "face": face,
        }
        self._rebuild()

    def _on_add(self) -> None:
        add_blank_map()
        self._rebuild()
        self.set_status("Please provide mapping!")

    def _on_clear(self) -> None:
        for item in self._map:
            item.pop("source", None)
            item.pop("target", None)
        self._rebuild()
        self.set_status("All mappings cleared!")

    def _on_submit(self) -> None:
        if has_valid_map():
            simplify_maps()
            self.set_status("Mappings successfully submitted!")
            self.accept()
            _open_webcam_preview(self._camera_index)
        else:
            self.set_status("At least 1 source with target is required!")


def _open_mapper_dialog(start_cb: Callable, mapping: list) -> None:
    global _MAPPER
    close_mapper_window()
    _MAPPER = MapperDialog(start_cb, mapping)
    _MAPPER.show()


def _open_live_mapper_dialog(camera_index: int, mapping: list) -> None:
    global _LIVE_MAPPER
    close_mapper_window()
    _LIVE_MAPPER = LiveMapperDialog(camera_index, mapping)
    _LIVE_MAPPER.show()


def close_mapper_window() -> None:
    global _MAPPER, _LIVE_MAPPER
    if _MAPPER is not None:
        _MAPPER.close()
        _MAPPER = None
    if _LIVE_MAPPER is not None:
        _LIVE_MAPPER.close()
        _LIVE_MAPPER = None


# ─── entry point ─────────────────────────────────────────────────────────


def _close_live_windows() -> None:
    """Close every window that owns worker threads, so the threads stop."""
    for win in (_WEBCAM_PREVIEW, _LIVE_MAPPER):
        if win is None:
            continue
        try:
            win.close()
        except Exception:
            pass


class _Window:
    """Thin wrapper exposing .mainloop() for core.py compatibility."""

    def __init__(self, app: QApplication, main_window: MainWindow):
        self._app = app
        self._main = main_window

    def mainloop(self) -> None:
        self._main.show()
        self._app.exec()


def init(
    start: Callable[[], None], destroy: Callable[[], None], lang: str
) -> _Window:
    global _APP, _MAIN, _PREVIEW, _LANG, _BRIDGE

    _LANG = LanguageManager(lang)
    if QApplication.instance() is None:
        _APP = QApplication(sys.argv)
    else:
        _APP = QApplication.instance()
    _APP.setStyleSheet(QSS)

    _BRIDGE = _UIBridge()
    def _destroy_with_cleanup(*args, **kwargs):
        # core.destroy() ends in a bare quit(), which raises SystemExit and tears
        # the interpreter down without unwinding Qt. Destroying a running QThread
        # is fatal in Qt, so stop the live-preview workers before that happens.
        _close_live_windows()
        return destroy(*args, **kwargs)

    _MAIN = MainWindow(start, _destroy_with_cleanup)
    _PREVIEW = PreviewWindow()

    # Route status updates onto the UI thread regardless of caller.
    _BRIDGE.statusChanged.connect(_MAIN.set_status)

    return _Window(_APP, _MAIN)
