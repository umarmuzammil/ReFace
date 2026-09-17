import os
import re
import cv2
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QLineEdit, QPushButton, QSizePolicy, QComboBox, QToolButton,
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QPixmap, QImage

from .icons import icon


class VideoCanvas(QWidget):
    """Always-visible display area. Shows branded splash when empty."""

    frame_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("VideoCanvas")
        self.setMinimumSize(480, 320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._label)

        self._current_frame = None
        self._show_splash()

    def _show_splash(self):
        self._label.setStyleSheet("""
            color: #444;
            font-size: 20px;
            font-weight: 300;
            letter-spacing: 2px;
        """)
        self._label.setText(
            '<span style="color:#555;font-size:32px;font-weight:200;">&#9670;</span>'
            '<br>'
            '<span style="color:#666;font-size:22px;font-weight:300;letter-spacing:3px;">'
            'RE:FACE</span>'
            '<br>'
            '<span style="color:#444;font-size:12px;font-weight:400;letter-spacing:1px;">'
            'V 2 . 1</span>'
        )

    def display(self, frame_bgr):
        if frame_bgr is None:
            self._show_splash()
            self._label.setPixmap(QPixmap())
            return
        self._current_frame = frame_bgr
        self._label.setText("")
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pm = QPixmap.fromImage(qimg).scaled(
            self._label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(pm)

    def clear(self):
        self._current_frame = None
        self._label.setPixmap(QPixmap())
        self._show_splash()


class VideoPlayer(QWidget):
    """Integrated video player with timeline, play/pause, mark, source selector."""

    frame_changed = Signal(int)
    mark_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cap = None
        self._frames = []
        self._total = 0
        self._fps = 0.0
        self._pos = 0
        self._current_frame = None
        self._playing = False
        self._source = "video"
        self._source_frames = []
        self._project_dir = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Canvas
        self.canvas = VideoCanvas()
        root.addWidget(self.canvas, 1)

        # Controls bar
        ctrl = QWidget()
        ctrl.setObjectName("PlayerControls")
        cl = QHBoxLayout(ctrl)
        cl.setContentsMargins(10, 4, 10, 4)
        cl.setSpacing(6)

        # Left: play + time
        self.play_btn = QPushButton()
        self.play_btn.setIcon(icon("play", size=16, color="#ccc"))
        self.play_btn.setFixedSize(28, 28)
        self.play_btn.setToolTip("Play / Pause")
        self.play_btn.clicked.connect(self.toggle_play)
        cl.addWidget(self.play_btn)

        self.time_input = QLineEdit("00:00")
        self.time_input.setFixedWidth(62)
        self.time_input.returnPressed.connect(self._jump_time)
        cl.addWidget(self.time_input)

        self.frame_input = QLineEdit("0")
        self.frame_input.setFixedWidth(55)
        self.frame_input.returnPressed.connect(self._jump_frame)
        cl.addWidget(self.frame_input)

        # Mark button (promote frame)
        self.mark_btn = QPushButton()
        self.mark_btn.setObjectName("MarkButton")
        self.mark_btn.setIcon(icon("bookmark", size=16, color="#4ade80"))
        self.mark_btn.setFixedSize(28, 28)
        self.mark_btn.setToolTip("Mark Frame (promote face)")
        self.mark_btn.clicked.connect(lambda: self.mark_clicked.emit())
        cl.addWidget(self.mark_btn)

        # Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self._seek)
        self.slider.sliderReleased.connect(lambda: self._seek(self.slider.value()))
        cl.addWidget(self.slider, 1)

        # Right: frame info + source selector
        self._lbl_info = QLabel("0 / 0")
        self._lbl_info.setFixedWidth(90)
        cl.addWidget(self._lbl_info)

        self.source_combo = QComboBox()
        self.source_combo.setFixedWidth(160)
        self._source_options = [
            ("Video", None),
            ("Marked Faces", "marked_faces"),
            ("Matched Frames", "matched_frames"),
            ("People (YOLO)", "detected_people"),
            ("Sharp Frames", "sharp_frames"),
        ]
        for label, _ in self._source_options:
            self.source_combo.addItem(label)
        self.source_combo.currentTextChanged.connect(self._on_source_change)
        cl.addWidget(self.source_combo)

        root.addWidget(ctrl)

        # Play timer
        self._play_timer = QTimer()
        self._play_timer.timeout.connect(self._advance_frame)

    # Public API

    def load_video(self, path):
        self.release()
        self._source = "video"
        if not os.path.isfile(path):
            return False
        self._video_path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            self._cap = None
            return False
        self._total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 24.0
        self.slider.setRange(0, max(0, self._total - 1))
        self._pos = 0
        self._seek(0)
        return True

    def load_frames(self, folder, source_name="Frames"):
        self.release()
        self._source = "frames"
        if not os.path.isdir(folder):
            return False
        pat = re.compile(r"_(\d+)\.", re.I)
        files = sorted(
            [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))],
            key=lambda f: int(m.group(1)) if (m := pat.match(f)) else 0,
        )
        for f in files:
            img = cv2.imread(os.path.join(folder, f))
            if img is not None:
                self._frames.append(img)
        if not self._frames:
            return False
        self._total = len(self._frames)
        self._fps = 25.0
        self.slider.setRange(0, self._total - 1)
        self._pos = 0
        self._seek(0)
        return True

    def set_project_dir(self, project_dir):
        self._project_dir = project_dir
        self._update_source_availability()

    def _update_source_availability(self):
        """Enable/disable source items based on whether folders have files."""
        for i, (label, folder_name) in enumerate(self._source_options):
            if folder_name is None:
                self.source_combo.setItemData(i, True)
                continue
            if not hasattr(self, "_project_dir") or not self._project_dir:
                self.source_combo.setItemData(i, False)
                continue
            folder = os.path.join(self._project_dir, folder_name)
            has_files = False
            if os.path.isdir(folder):
                for entry in os.listdir(folder):
                    full = os.path.join(folder, entry)
                    if os.path.isfile(full) and entry.lower().endswith((".jpg", ".jpeg", ".png")):
                        has_files = True
                        break
                    if os.path.isdir(full):
                        for f in os.listdir(full):
                            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                                has_files = True
                                break
                    if has_files:
                        break
            self.source_combo.setItemData(i, has_files)

    def _on_source_change(self, name):
        # Find folder for this option
        folder_name = None
        for label, fn in self._source_options:
            if label == name:
                folder_name = fn
                break

        if folder_name is None:
            # "Video" — reload video capture
            if self._video_path and os.path.isfile(self._video_path):
                self.stop_play()
                if self._cap:
                    self._cap.release()
                self._source = "video"
                self._cap = cv2.VideoCapture(self._video_path)
                if self._cap.isOpened():
                    self._total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 24.0
                    self.slider.setRange(0, max(0, self._total - 1))
                    self._pos = 0
                    self._seek(0)
            return

        # Image source — load from project folder
        if not hasattr(self, "_project_dir") or not self._project_dir:
            return
        folder = os.path.join(self._project_dir, folder_name)
        if not os.path.isdir(folder):
            return
        self.stop_play()
        if self._cap:
            self._cap.release()
            self._cap = None
        self._source = name
        self._frames.clear()
        self._collect_images(folder)
        if self._frames:
            self._total = len(self._frames)
            self.slider.setRange(0, self._total - 1)
            self._pos = 0
            self._seek(0)

    def release(self):
        self.stop_play()
        if self._cap:
            self._cap.release()
            self._cap = None
        self._frames.clear()
        self._total = 0
        self._fps = 0.0
        self._pos = 0
        self._current_frame = None
        self.canvas.clear()
        self.slider.setRange(0, 0)
        self._lbl_info.setText("0 / 0")
        self.source_combo.blockSignals(True)
        self.source_combo.setCurrentIndex(0)
        self.source_combo.blockSignals(False)

    def toggle_play(self):
        if self._playing:
            self.stop_play()
        else:
            self.start_play()

    def start_play(self):
        if self._total == 0:
            return
        self._playing = True
        self.play_btn.setIcon(icon("x", size=16, color="#fff"))
        interval = int(1000 / self._fps) if self._fps > 0 else 33
        self._play_timer.start(interval)

    def stop_play(self):
        self._playing = False
        self._play_timer.stop()
        self.play_btn.setIcon(icon("play", size=16, color="#ccc"))

    def get_frame_num(self):
        return self._pos

    def get_current_frame(self):
        return self._current_frame

    @property
    def fps(self):
        return self._fps

    @property
    def total_frames(self):
        return self._total

    # Internal

    def _advance_frame(self):
        if self._pos + 1 >= self._total:
            self.stop_play()
            return
        self._seek(self._pos + 1)

    def _collect_images(self, folder):
        """Recursively collect images from folder and subdirectories."""
        pat = re.compile(r"_(\d+)\.", re.I)
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        files = []
        for entry in os.listdir(folder):
            full = os.path.join(folder, entry)
            if os.path.isfile(full) and entry.lower().endswith(exts):
                files.append(full)
            elif os.path.isdir(full):
                for f in os.listdir(full):
                    if f.lower().endswith(exts):
                        files.append(os.path.join(full, f))
        files.sort(key=lambda f: int(m.group(1)) if (m := pat.match(os.path.basename(f))) else 0)
        for fp in files:
            img = cv2.imread(fp)
            if img is not None:
                self._frames.append(img)

    def _seek(self, n):
        if self._total == 0 or n < 0 or n >= self._total:
            return
        frame = None
        if self._source == "video" and self._cap and self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, n)
            ok, frame = self._cap.read()
            if not ok:
                frame = None
        elif self._frames:
            frame = self._frames[n]
        if frame is None:
            return
        self._current_frame = frame
        self._pos = n
        self.canvas.display(frame)
        self._lbl_info.setText(f"{self._pos} / {self._total}")
        self.slider.blockSignals(True)
        self.slider.setValue(n)
        self.slider.blockSignals(False)
        self.time_input.setText(self._fmt(n))
        self.frame_input.setText(str(n))
        self.frame_changed.emit(n)

    def _jump_time(self):
        parts = self.time_input.text().strip().split(":")
        try:
            if len(parts) == 2:
                t = int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                t = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            else:
                return
            self._seek(int(t * self._fps))
        except ValueError:
            pass

    def _jump_frame(self):
        try:
            self._seek(int(self.frame_input.text()))
        except ValueError:
            pass

    def _fmt(self, n):
        if self._fps <= 0:
            return "00:00"
        s = n / self._fps
        return f"{int(s // 60):02d}:{int(s % 60):02d}"
