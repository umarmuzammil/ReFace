import sys
import os
import json
import concurrent.futures
import threading
import time

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QComboBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QProgressBar, QStatusBar,
    QMessageBox, QApplication, QToolButton, QScrollArea,
    QGridLayout, QFrame,
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QAction, QKeySequence, QPixmap, QImage, QIcon

APP_VERSION = "2.1.0"
APP_NAME = "Re:Face"

from .styles import DARK_THEME
from .icons import icon
from .activity_bar import ActivityBar
from .video_player import VideoPlayer


def _sidebar_label(text):
    lbl = QLabel(text)
    lbl.setObjectName("SidebarTitle")
    return lbl


def _sidebar_section(text):
    lbl = QLabel(text)
    lbl.setObjectName("SidebarSection")
    return lbl


class SidebarPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self.setFixedWidth(260)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

    def add_title(self, text):
        self._layout.addWidget(_sidebar_label(text))

    def add_section(self, text):
        self._layout.addWidget(_sidebar_section(text))

    def add_button(self, text, slot, tooltip=""):
        btn = QPushButton(text)
        btn.setObjectName("SidebarAction")
        if tooltip:
            btn.setToolTip(tooltip)
        btn.clicked.connect(slot)
        self._layout.addWidget(btn)

    def add_row(self, widgets):
        row = QHBoxLayout()
        row.setContentsMargins(14, 0, 14, 0)
        row.setSpacing(6)
        for w in widgets:
            row.addWidget(w)
        container = QWidget()
        container.setLayout(row)
        self._layout.addWidget(container)

    def add_widget(self, w):
        self._layout.addWidget(w)

    def add_stretch(self):
        self._layout.addStretch()

    def add_separator(self):
        line = QFrame()
        line.setObjectName("SidebarSeparator")
        line.setFrameShape(QFrame.Shape.HLine)
        self._layout.addWidget(line)


class FaceThumbnail(QWidget):
    deleted = Signal(object)

    def __init__(self, size=70, parent=None):
        super().__init__(parent)
        self.setObjectName("FaceThumb")
        self.setFixedSize(size, size)
        self._cv_img = None
        self._pixmap = None

        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("background: transparent;")
        self._label.setGeometry(2, 2, size - 4, size - 4)

        self._del_btn = QPushButton("\u2715")
        self._del_btn.setObjectName("DeleteBtn")
        self._del_btn.setParent(self)
        self._del_btn.move(size - 18, 2)
        self._del_btn.clicked.connect(lambda: self.deleted.emit(self))
        self._del_btn.hide()

    def enterEvent(self, event):
        self._del_btn.show()

    def leaveEvent(self, event):
        self._del_btn.hide()

    def set_face_image(self, cv_img):
        if cv_img is None:
            return
        self._cv_img = cv_img
        self._label.setText("")
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data.tobytes(), w, h, ch * w, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self._apply_pixmap()

    def _apply_pixmap(self):
        if self._pixmap is None:
            return
        s = self._label.width()
        if s < 10:
            s = 66
        scaled = self._pixmap.scaled(s, s, Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)
        self._label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_pixmap()

    def get_cv_image(self):
        return self._cv_img


class FaceGrid(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(14, 4, 14, 4)
        self._layout.setSpacing(6)
        self._thumbs = []
        self._cols = 3

    def add_face(self, cv_img, face_path=None):
        idx = len(self._thumbs)
        thumb = FaceThumbnail(size=70)
        if cv_img is not None:
            thumb.set_face_image(cv_img)
        thumb.deleted.connect(self._remove_thumb)
        thumb._face_path = face_path
        row, col = divmod(idx, self._cols)
        self._layout.addWidget(thumb, row, col)
        self._thumbs.append(thumb)

    def _remove_thumb(self, thumb):
        if thumb in self._thumbs:
            self._thumbs.remove(thumb)
        path = getattr(thumb, "_face_path", None)
        thumb.setParent(None)
        thumb.deleteLater()
        self._relayout()
        if path and os.path.isfile(path):
            os.remove(path)
            base = os.path.splitext(path)[0]
            for ext in ["_embedding.npz", "_faces.png"]:
                p = base + ext
                if os.path.isfile(p):
                    os.remove(p)

    def _relayout(self):
        for i, t in enumerate(self._thumbs):
            row, col = divmod(i, self._cols)
            self._layout.addWidget(t, row, col)

    def count(self):
        return len(self._thumbs)

    def clear_faces(self):
        for t in self._thumbs[:]:
            t.setParent(None)
            t.deleteLater()
        self._thumbs.clear()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(1100, 700)
        self.resize(1400, 850)
        self.setStyleSheet(DARK_THEME)

        # Set window icon from SVG
        ico = icon("scan", size=64, color="#0a64ff")
        self.setWindowIcon(ico)

        self.utils_instance = None
        self.model_loader = None
        self.face_processor = None

        # Background worker state
        self._worker_running = False
        self._worker_stop = False
        self._worker_progress = 0
        self._worker_status = ""
        self._worker_result = None
        self._worker_callback = None
        self._worker_timer = QTimer()
        self._worker_timer.timeout.connect(self._poll_worker)

        # Video path for reloading
        self._video_path = None

        # Central
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Menu
        self._build_menu()

        # Toolbar
        root.addWidget(self._build_toolbar())

        # Body: activity bar | sidebar | canvas
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self.activity_bar = ActivityBar()
        self.activity_bar.view_changed.connect(self._on_view_change)
        body.addWidget(self.activity_bar)

        self.sidebar = QWidget()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(260)
        self._sidebar_stack = {}
        self._sidebar_layout = QVBoxLayout(self.sidebar)
        self._sidebar_layout.setContentsMargins(0, 0, 0, 0)
        self._sidebar_layout.setSpacing(0)
        body.addWidget(self.sidebar)

        self.video_player = VideoPlayer()
        self.video_player.mark_clicked.connect(self._on_mark_frame)
        body.addWidget(self.video_player, 1)

        root.addLayout(body, 1)

        # Permanent progress strip (full-width, below video, above status bar)
        self.progress_strip = QWidget()
        self.progress_strip.setObjectName("ProgressStrip")
        self.progress_strip.setFixedHeight(5)
        self.progress_strip.setStyleSheet("background: #333;")
        self._progress_bar = QWidget(self.progress_strip)
        self._progress_bar.setStyleSheet("background: #3b82f6;")
        self._progress_bar.setGeometry(0, 0, 0, 5)
        root.addWidget(self.progress_strip)

        # Status
        self._build_status_bar()

        # Sidebar panels
        self._build_sidebar_panels()

        # Default
        self.activity_bar.select("Extract")
        self._on_view_change("Extract")

    # ── Menu ──

    def _build_menu(self):
        mb = self.menuBar()

        file_menu = mb.addMenu("&File")
        act_open = QAction("&Open Video...", self)
        act_open.setShortcut(QKeySequence("Ctrl+O"))
        act_open.triggered.connect(self._open_video)
        file_menu.addAction(act_open)
        act_project = QAction("&Load Project...", self)
        act_project.setShortcut(QKeySequence("Ctrl+L"))
        act_project.triggered.connect(self._load_project)
        file_menu.addAction(act_project)
        file_menu.addSeparator()
        act_quit = QAction("&Quit", self)
        act_quit.setShortcut(QKeySequence("Ctrl+Q"))
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        view_menu = mb.addMenu("&View")
        for name in ["Extract", "Faces", "People", "Sharp"]:
            act = QAction(name, self)
            act.triggered.connect(lambda checked, n=name: self.activity_bar.select(n) or self._on_view_change(n))
            view_menu.addAction(act)

        help_menu = mb.addMenu("&Help")
        act_guide = QAction("&Guide", self)
        act_guide.setShortcut(QKeySequence("F1"))
        act_guide.triggered.connect(self._show_guide)
        help_menu.addAction(act_guide)
        help_menu.addSeparator()
        act_about = QAction("&About", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    # ── Toolbar ──

    def _build_toolbar(self):
        bar = QWidget()
        bar.setObjectName("Toolbar")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(6, 2, 6, 2)
        bl.setSpacing(2)

        # Left: main actions
        main_btns = [
            ("folder", "Open Video", self._open_video),
            ("bookmark", "Mark Frame", self._on_mark_frame),
            ("scissors", "Extract Frames", self._run_extract),
            ("scan", "Process Marked Faces", self._run_face_match),
            ("sliders", "Sharpen", self._run_sharpen),
            ("download", "Export", self._run_sort),
        ]
        for icon_name, tooltip, slot in main_btns:
            btn = QToolButton()
            btn.setIcon(icon(icon_name, size=18))
            btn.setToolTip(tooltip)
            btn.setAutoRaise(True)
            btn.setFixedSize(30, 30)
            btn.clicked.connect(slot)
            bl.addWidget(btn)

        self.stop_btn = QToolButton()
        self.stop_btn.setIcon(icon("x", size=18, color="#ff4444"))
        self.stop_btn.setToolTip("Stop Running Task")
        self.stop_btn.setAutoRaise(True)
        self.stop_btn.setFixedSize(30, 30)
        self.stop_btn.clicked.connect(self._stop_worker)
        self.stop_btn.setVisible(False)
        bl.addWidget(self.stop_btn)

        bl.addStretch()

        # Right: utility icons
        util_btns = [
            ("folder-open", "Open Project Folder", self._open_project_dir),
            ("folder-open", "Open Output Folder", self._open_output_dir),
        ]
        for icon_name, tooltip, slot in util_btns:
            btn = QToolButton()
            btn.setIcon(icon(icon_name, size=16, color="#777"))
            btn.setToolTip(tooltip)
            btn.setAutoRaise(True)
            btn.setFixedSize(26, 26)
            btn.clicked.connect(slot)
            bl.addWidget(btn)

        return bar

    # ── Status Bar ──

    def _build_status_bar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.status_label = QLabel("Ready")
        sb.addWidget(self.status_label)

    def _set_progress(self, pct):
        w = self.width()
        self._progress_bar.setGeometry(0, 0, int(w * pct / 100), 5)
        self._progress_bar.setStyleSheet("background: #eab308;")
        self._progress_bar.show()

    def _hide_progress(self):
        self._progress_bar.setGeometry(0, 0, 0, 5)
        self._progress_bar.setStyleSheet("background: #3b82f6;")
        self._progress_bar.hide()

    def _start_worker(self, target, callback=None):
        if self._worker_running:
            QMessageBox.warning(self, "Busy", "A process is already running. Please wait.")
            return False
        self._worker_running = True
        self._worker_stop = False
        self._worker_progress = 0
        self._worker_status = ""
        self._worker_result = None
        self._worker_callback = callback
        self._set_progress(0)
        self.stop_btn.setVisible(True)

        def wrapped():
            try:
                target()
            except Exception as e:
                self._worker_status = f"Error: {e}"
                print(f"Worker error: {e}")
            finally:
                self._worker_running = False

        t = threading.Thread(target=wrapped, daemon=True)
        t.start()
        self._worker_timer.start(150)
        return True

    def _poll_worker(self):
        if not self._worker_running:
            self._worker_timer.stop()
            if self._worker_status:
                self.status_label.setText(self._worker_status)
            self._hide_progress()
            self.stop_btn.setVisible(False)
            if self._worker_callback:
                self._worker_callback(self._worker_result)
            return
        self._set_progress(self._worker_progress)
        if self._worker_status:
            self.status_label.setText(self._worker_status)

    def _stop_worker(self):
        if self._worker_running:
            self._worker_stop = True
            self._worker_status = "Stopping..."
            self.status_label.setText("Stopping...")

    def _show_guide(self):
        from PySide6.QtWidgets import QDialog, QTextBrowser
        dlg = QDialog(self)
        dlg.setWindowTitle("Re:Face Guide")
        dlg.setMinimumSize(600, 500)
        dlg.setStyleSheet(DARK_THEME)
        lay = QVBoxLayout(dlg)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setStyleSheet("background: #1e1e1e; color: #ccc; border: none; font-size: 13px; padding: 16px;")
        browser.setHtml(f"""
        <h2 style="color:#0a64ff;">{APP_NAME} v{APP_VERSION} — User Guide</h2>
        <h3 style="color:#4ade80;">Quick Start</h3>
        <ol>
        <li><b>Open Video</b> — File &gt; Open Video (Ctrl+O) or toolbar folder icon</li>
        <li><b>Mark Faces</b> — scrub to a clear face frame, click the <span style="color:#4ade80;">&#9733;</span> Mark button</li>
        <li><b>Extract Frames</b> — switch to Extract view, click "Extract Frames" (parallel FFmpeg)</li>
        <li><b>Process Marked Faces</b> — switch to Faces view, click "Process Marked Faces" to search all frames</li>
        <li><b>Export</b> — save matched frames or detected people from the Faces panel</li>
        </ol>

        <h3 style="color:#4ade80;">Views (Activity Bar)</h3>
        <ul>
        <li><b>Extract</b> — configure scale/quality and extract frames from video</li>
        <li><b>Faces</b> — mark faces, run face matching, export results</li>
        <li><b>People</b> — YOLO person detection with padding control</li>
        <li><b>Sharp</b> — apply sharpness filters and save output</li>
        </ul>

        <h3 style="color:#4ade80;">Face Matching Workflow</h3>
        <ol>
        <li>Mark 1+ faces from different frames as references</li>
        <li>Click "Process Marked Faces" — scans all extracted frames</li>
        <li>Results saved to <code>face_data.json</code></li>
        <li>Click "Save Frames with Marked Faces" to extract matching frames</li>
        </ol>

        <h3 style="color:#4ade80;">Sharpness Filters</h3>
        <p>Scene-aware extraction: detects shot changes via SSIM, picks sharpest frames per shot, skips motion-blurred frames.</p>
        <p>Methods: Unsharp Mask, High-Pass, Laplacian, Sobel, BRISQUE, FFT</p>

        <h3 style="color:#4ade80;">Requirements</h3>
        <ul>
        <li>Python 3.10+</li>
        <li>FFmpeg (must be in PATH)</li>
        <li>PySide6, opencv-python, numpy, insightface, onnxruntime, scikit-learn</li>
        <li>NVIDIA GPU optional (auto-detects CUDA)</li>
        </ul>

        <h3 style="color:#4ade80;">Project Folder Structure</h3>
        <pre style="color:#999;">video_project/
├── extracted_frames/    FFmpeg frames
├── marked_faces/        Face crops + embeddings
├── matched_frames/      Face match results
├── detected_people/     YOLO person crops
├── sharp_frames/        Sharpness output
└── face_data.json       Match data</pre>

        <p style="color:#666; margin-top:20px;">Author: <b>Umar Muzammil</b></p>
        """)
        lay.addWidget(browser)
        btn = QPushButton("Close")
        btn.clicked.connect(dlg.close)
        lay.addWidget(btn)
        dlg.exec()

    def _show_about(self):
        QMessageBox.about(self, f"About {APP_NAME}",
            f"<h2>{APP_NAME}</h2>"
            f"<p>Version {APP_VERSION}</p>"
            "<p>Video face detection, matching, and sharp frame extraction.</p>"
            "<hr>"
            "<p><b>Author:</b> Umar Muzammil</p>"
            "<p><b>Stack:</b> PySide6 / OpenCV / InsightFace / YOLO</p>"
            "<p>FFmpeg for parallel frame extraction.</p>"
        )

    # ── Helpers ──

    def _has_project(self):
        return self.utils_instance is not None

    def _has_frames(self):
        if not self._has_project():
            return False
        d = self.utils_instance.get_frames_dir()
        if not os.path.isdir(d):
            return False
        for entry in os.listdir(d):
            subdir = os.path.join(d, entry)
            if os.path.isdir(subdir) and any(f.endswith((".jpg", ".png")) for f in os.listdir(subdir)):
                return True
        return any(f.endswith((".jpg", ".png")) for f in os.listdir(d))

    def _has_images_recursive(self, path):
        for entry in os.listdir(path):
            full = os.path.join(path, entry)
            if os.path.isfile(full) and entry.lower().endswith((".jpg", ".jpeg", ".png")):
                return True
            if os.path.isdir(full) and any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in os.listdir(full)):
                return True
        return False

    def _has_faces(self):
        if not self._has_project():
            return False
        d = self.utils_instance.get_detected_faces_dir()
        return os.path.isdir(d) and any(f.endswith(".npz") for f in os.listdir(d))

    def _ask_extract_frames(self):
        reply = QMessageBox.question(
            self, "Frames Not Found",
            "No extracted frames found.\nExtract frames first?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._run_extract()
        return reply == QMessageBox.StandardButton.Yes

    # ── Mark Frame ──

    def _on_mark_frame(self):
        if not self._has_project():
            QMessageBox.warning(self, "No Project", "Open a video first (File > Open Video).")
            return
        frame = self.video_player.get_current_frame()
        if frame is None:
            QMessageBox.warning(self, "No Frame", "No frame to mark. Play or scrub the video first.")
            return

        from facekit.pipeline.retinaface import ModelLoader, FaceProcessor

        out_dir = self.utils_instance.get_detected_faces_dir()
        os.makedirs(out_dir, exist_ok=True)
        frame_num = self.video_player.get_frame_num()

        if not self.face_processor:
            self.status_label.setText("Loading face model...")
            QApplication.processEvents()
            self.model_loader = ModelLoader()
            app = self.model_loader.load_model()
            if app is None:
                QMessageBox.critical(self, "Model Error",
                    "Failed to load face detection model.\nCheck that onnxruntime is installed correctly.")
                self.status_label.setText("Ready")
                return
            self.face_processor = FaceProcessor(app, self.utils_instance)

        faces = self.face_processor.app.get(frame)
        if not faces:
            QMessageBox.information(self, "No Face Detected",
                f"No face found in frame {frame_num}.\nTry a frame where a face is clearly visible.")
            return

        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face.bbox.astype(int)
            h, w = frame.shape[:2]
            bw, bh = x2 - x1, y2 - y1
            pad = int(max(bw, bh) * 0.3)
            x1c, y1c = max(0, x1 - pad), max(0, y1 - pad)
            x2c, y2c = min(w, x2 + pad), min(h, y2 + pad)
            face_crop = frame[y1c:y2c, x1c:x2c]

            suffix = f"_face{i}" if len(faces) > 1 else ""
            face_path = os.path.join(out_dir, f"frame_{frame_num:06d}{suffix}.png")
            cv2.imwrite(face_path, face_crop)

            emb_path = os.path.join(out_dir, f"frame_{frame_num:06d}{suffix}_embedding.npz")
            np.savez(emb_path, embeddings=np.array([face.embedding]))

            self.face_grid.add_face(face_crop, face_path=face_path)

        self.status_label.setText(f"Marked frame {frame_num} \u2014 {len(faces)} face(s) saved")
        self._refresh_source_dropdown()
        self.activity_bar.select("Faces")
        self._on_view_change("Faces")

    # ── Sidebar Panels ──

    def _build_sidebar_panels(self):
        # ── EXTRACT ──
        ext = SidebarPanel()
        ext.add_title("Frame Extraction")

        ext.add_section("Scale")
        self.rescale_combo = QComboBox()
        self.rescale_combo.addItems(["100%", "50%", "25%", "12.5%", "6.25%"])
        self.rescale_combo.setCurrentIndex(2)
        ext.add_widget(self.rescale_combo)

        ext.add_section("JPEG Quality")
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setValue(80)
        self.quality_spin.setSuffix("%")
        self.quality_spin.setToolTip("100% = best quality, 1% = lowest")
        ext.add_widget(self.quality_spin)

        ext.add_section("Actions")
        ext.add_button("Extract Frames", self._run_extract)
        ext.add_stretch()
        self._register_panel("Extract", ext)

        # ── FACES ──
        fac = SidebarPanel()
        fac.add_title("Face Detection")

        # Marked faces on TOP
        fac.add_section("Marked Faces")
        self.face_grid = FaceGrid()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.face_grid)
        scroll.setFixedHeight(160)
        scroll.setStyleSheet("background: transparent; border: none;")
        fac.add_widget(scroll)

        fac.add_separator()

        # Process
        fac.add_section("Process")
        fac.add_button("Process Marked Faces", self._run_face_match,
                        "Search all extracted frames for faces matching your marked faces")

        fac.add_separator()

        fac.add_button("Save Frames with Marked Faces", self._run_save_matched,
                        "Extract video frames where matching faces were found")

        self.cropping_cb = QCheckBox("Crop faces on save")
        fac.add_widget(self.cropping_cb)
        fac.add_stretch()
        self._register_panel("Faces", fac)

        # ── FILTERS ──
        flt = SidebarPanel()
        flt.add_title("Sharpness Filters")

        flt.add_section("Method")
        self.sharpness_combo = QComboBox()
        self.sharpness_combo.addItems([
            "Unsharp Mask", "High-Pass Filter", "Laplacian",
            "Sobel", "Custom Kernel", "BRISQUE", "FFT"
        ])
        flt.add_widget(self.sharpness_combo)

        flt.add_section("Source")
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Raw", "Matched Frames", "Matched People"])
        flt.add_widget(self.source_combo)

        flt.add_section("Parameters")
        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(1, 24)
        self.sample_spin.setValue(1)
        self.sample_spin.setToolTip("Keep top N sharpest frames per second")
        flt.add_row([QLabel("Keep per Second"), self.sample_spin])

        flt.add_button("Save Sharp Frames", self._run_sharpen)

        flt.add_separator()
        flt.add_section("Sort & Rank")
        self.topn_spin = QSpinBox()
        self.topn_spin.setRange(1, 1000)
        self.topn_spin.setValue(100)
        flt.add_row([QLabel("Top N"), self.topn_spin])
        flt.add_button("Sort Sharp Frames", self._run_sort)

        flt.add_stretch()
        self._register_panel("Sharp", flt)

        # ── PEOPLE ──
        ppl = SidebarPanel()
        ppl.add_title("People Detection")
        ppl.add_section("Options")
        ppl.add_button("Detect People (YOLO)", self._run_figures,
                        "Find and crop person regions from video")
        self.padding_spin = QDoubleSpinBox()
        self.padding_spin.setRange(0.0, 0.5)
        self.padding_spin.setSingleStep(0.01)
        self.padding_spin.setValue(0.12)
        pad_row = QHBoxLayout()
        pad_row.setContentsMargins(14, 0, 14, 0)
        pad_row.setSpacing(6)
        pad_row.addWidget(QLabel("Padding"))
        pad_row.addWidget(self.padding_spin)
        pad_container = QWidget()
        pad_container.setLayout(pad_row)
        ppl.add_widget(pad_container)
        ppl.add_separator()
        ppl.add_stretch()
        self._register_panel("People", ppl)

    def _register_panel(self, name, widget):
        self._sidebar_stack[name] = widget
        self._sidebar_layout.addWidget(widget)
        widget.hide()

    def _on_view_change(self, name):
        for n, w in self._sidebar_stack.items():
            w.setVisible(n == name)

    # ── File Actions ──

    def _open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Videos (*.mp4 *.mov *.avi *.mkv *.webm)"
        )
        if not path:
            return
        self._init_project(path)
        if self.video_player.load_video(path):
            self.status_label.setText(f"Loaded: {os.path.basename(path)}")
            self._refresh_source_dropdown()
        else:
            QMessageBox.warning(self, "Error", "Failed to load video.")

    def _load_project(self):
        folder = QFileDialog.getExistingDirectory(self, "Load Project")
        if not folder:
            return
        # Find the video file: check folder, then parent, then sibling (strip _project suffix)
        video_exts = (".mp4", ".mov", ".avi", ".mkv", ".webm")
        video_path = None
        for search_dir in [folder, os.path.dirname(folder)]:
            for f in os.listdir(search_dir):
                if f.lower().endswith(video_exts):
                    video_path = os.path.join(search_dir, f)
                    break
            if video_path:
                break
        if not video_path:
            # Try stripping _project suffix from folder name
            base = folder.rstrip("\\/").rstrip("/").rstrip("\\")
            if base.endswith("_project"):
                candidate = base[:-8]
                if os.path.isdir(candidate):
                    for f in os.listdir(candidate):
                        if f.lower().endswith(video_exts):
                            video_path = os.path.join(candidate, f)
                            break
        if video_path:
            self._init_project(video_path)
            if self.video_player.load_video(video_path):
                self.status_label.setText(f"Project: {self.utils_instance.get_base_name()}")
                self._refresh_source_dropdown()
            else:
                QMessageBox.warning(self, "Error", "Failed to load video from project.")
        else:
            QMessageBox.information(self, "No Video", "No video file found in this project folder.")

    def _init_project(self, path):
        from facekit.pipeline.utils import ProjectUtils
        self.utils_instance = ProjectUtils(path)
        self.utils_instance.create_project_dir()
        self.status_label.setText(f"Project: {self.utils_instance.get_base_name()}")

    def _open_project_dir(self):
        if self.utils_instance:
            d = self.utils_instance.get_project_dir()
            if d and os.path.isdir(d):
                os.startfile(d)

    def _open_output_dir(self):
        if self.utils_instance:
            d = self.utils_instance.get_output_dir()
            if d and os.path.isdir(d):
                os.startfile(d)

    def _refresh_source_dropdown(self):
        if self.utils_instance:
            self.video_player.set_project_dir(self.utils_instance.get_project_dir())

    # ── Pipeline Actions ──

    def _run_extract(self):
        if not self.utils_instance:
            QMessageBox.warning(self, "No project", "Open a video first.")
            return
        from facekit.pipeline.frames_extract import (
            detect_hardware_acceleration, get_video_duration, get_video_fps,
            extract_frames_segment,
        )

        vp = self.utils_instance.get_file_path()
        out = self.utils_instance.get_frames_dir()
        os.makedirs(out, exist_ok=True)
        hw = detect_hardware_acceleration()
        dur = get_video_duration(vp)
        fps = get_video_fps(vp)
        rescale = [1, 2, 4, 8, 16][self.rescale_combo.currentIndex()]
        qual = max(1, 31 - int((self.quality_spin.value() - 1) / 100 * 30))
        sr = 1
        nd = 8
        sd = dur / nd

        def run():
            total = 0
            with concurrent.futures.ProcessPoolExecutor(max_workers=nd) as pool:
                futs = [
                    pool.submit(extract_frames_segment, vp, out, i * sd, sd, i, fps, hw, rescale, qual, sr)
                    for i in range(nd)
                ]
                for i, f in enumerate(concurrent.futures.as_completed(futs)):
                    try:
                        _, c = f.result()
                        total += c
                    except Exception as e:
                        print(f"Error: {e}")
                    self._worker_progress = int((i + 1) / nd * 100)
                    self._worker_status = f"Extracting frames... {i+1}/{nd}"
            self._worker_status = f"Extracted {total} frames"
            self._worker_result = total

        def done(result):
            self._refresh_source_dropdown()

        self._start_worker(run, callback=done)

    def _run_face_match(self):
        if not self._has_project():
            QMessageBox.warning(self, "No Project", "Open a video first.")
            return
        if not self._has_frames():
            self._ask_extract_frames()
            return
        from facekit.pipeline.retinaface import ModelLoader, FaceProcessor

        if not self.face_processor:
            self.model_loader = ModelLoader()
            app = self.model_loader.load_model()
            if app is None:
                QMessageBox.critical(self, "Model Error",
                    "Failed to load face detection model.\nCheck that onnxruntime is installed correctly.")
                return
            self.face_processor = FaceProcessor(app, self.utils_instance)

        ed = self.utils_instance.get_detected_faces_dir()
        emb_files = [f for f in os.listdir(ed) if f.endswith(".npz")]
        if not emb_files:
            QMessageBox.information(self, "No Faces", "No marked faces found.\nMark some faces first.")
            return
        total_emb = len(emb_files)

        # Count total frames for granular progress
        frames_dir = self.utils_instance.get_frames_dir()
        total_frames = 0
        if os.path.isdir(frames_dir):
            for entry in os.listdir(frames_dir):
                seg = os.path.join(frames_dir, entry)
                if os.path.isdir(seg):
                    total_frames += len([f for f in os.listdir(seg) if f.endswith(".jpg")])

        def run():
            matched_total = 0
            for i, f in enumerate(emb_files):
                if self._worker_stop:
                    self._worker_status = "Face search stopped"
                    break
                emb = self.face_processor.load_embedding(os.path.join(ed, f))
                if emb is None:
                    continue
                result = self.face_processor.compare_face_embedding(
                    emb,
                    progress_callback=lambda p, s="": self._update_match_progress(i, total_emb, p, s),
                    stop_check=lambda: self._worker_stop,
                )
                matched_total += len(result)
            self._worker_status = f"Face search complete \u2014 {matched_total} match(es) found"
            self._worker_progress = 100

        def done(result):
            self._refresh_source_dropdown()

        self._start_worker(run, callback=done)

    def _update_match_progress(self, emb_idx, total_emb, frame_pct, status_text):
        base = int(100 * (emb_idx) / total_emb)
        self._worker_progress = min(100, base + int(100 * frame_pct / total_emb))
        if status_text:
            self._worker_status = status_text

    def _run_figures(self):
        if not self._has_project():
            QMessageBox.warning(self, "No Project", "Open a video first.")
            return
        if not self._has_frames():
            self._ask_extract_frames()
            return
        from facekit.pipeline.yolov import crop_and_save

        def run():
            self._worker_status = "Detecting people (YOLO)..."
            self._worker_progress = 20
            crop_and_save(self.utils_instance.get_file_path(), padding_factor=self.padding_spin.value())
            self._worker_status = "Person detection complete"
            self._worker_progress = 100

        def done(result):
            self._refresh_source_dropdown()

        self._start_worker(run, callback=done)

    def _run_save_matched(self):
        if not self._has_project():
            QMessageBox.warning(self, "No Project", "Open a video first.")
            return
        if not self._has_frames():
            self._ask_extract_frames()
            return
        from facekit.pipeline.frames_extract import (
            detect_hardware_acceleration, get_video_duration, get_video_fps,
            extract_frames_target,
        )
        jp = os.path.join(self.utils_instance.get_project_dir(), "face_data.json")
        if not os.path.isfile(jp):
            QMessageBox.warning(self, "No Face Data",
                "Run 'Process Marked Faces' first to identify frames with matching faces.")
            return
        with open(jp) as f:
            fd = json.load(f)
        if not fd:
            QMessageBox.information(self, "No Matches", "No matching frames found. Try marking more faces.")
            return

        vp = self.utils_instance.get_file_path()
        out = self.utils_instance.get_saved_frames_dir()
        os.makedirs(out, exist_ok=True)
        hw = detect_hardware_acceleration()
        dur = get_video_duration(vp)
        fps = get_video_fps(vp)
        rescale = [1, 2, 4, 8, 16][self.rescale_combo.currentIndex()]
        qual = max(1, 31 - int((self.quality_spin.value() - 1) / 100 * 30))
        nd = 8
        sd = dur / nd

        def run():
            total = 0
            with concurrent.futures.ProcessPoolExecutor(max_workers=nd) as pool:
                futs = [
                    pool.submit(extract_frames_target, vp, out, fd, i * sd, sd, i, fps, hw, 1, rescale, qual, self.cropping_cb.isChecked())
                    for i in range(nd)
                ]
                for i, f in enumerate(concurrent.futures.as_completed(futs)):
                    try:
                        _, c = f.result()
                        total += c
                    except Exception as e:
                        print(f"Error: {e}")
                    self._worker_progress = int(100 * (i + 1) / nd)
                    self._worker_status = f"Saving matched frames... {i+1}/{nd}"
            self._worker_status = f"Saved {total} matched frames"
            self._worker_result = total

        def done(result):
            self._refresh_source_dropdown()

        self._start_worker(run, callback=done)

    def _run_sharpen(self):
        if not self._has_project():
            QMessageBox.warning(self, "No Project", "Open a video first.")
            return
        if not self._has_frames():
            self._ask_extract_frames()
            return
        from facekit.pipeline.sharpness import extract_sharp_frames

        m = self.sharpness_combo.currentText()
        s = self.source_combo.currentText()
        sr = self.sample_spin.value()

        def run():
            self._worker_status = f"Applying {m}..."
            self._worker_progress = 20
            extract_sharp_frames(self.utils_instance, m, s, sr)
            self._worker_status = "Sharp frames extracted"
            self._worker_progress = 100

        def done(result):
            self._refresh_source_dropdown()

        self._start_worker(run, callback=done)

    def _run_sort(self):
        if not self._has_project():
            QMessageBox.warning(self, "No Project", "Open a video first.")
            return
        from facekit.pipeline.sorter import rank_by_sharpness

        m = self.sharpness_combo.currentText()
        s = self.source_combo.currentText()
        d = os.path.join(self.utils_instance.get_output_dir(), f"{s}_{m}")
        if not os.path.isdir(d):
            QMessageBox.warning(self, "No Output", f"No sharp frames found.\nRun 'Extract Sharp Frames' first.\nExpected: {d}")
            return

        def run():
            self._worker_status = f"Sorting by {m}..."
            self._worker_progress = 30
            out = rank_by_sharpness(d, method=m, top_n=self.topn_spin.value())
            self._worker_status = f"Sorted {m} frames saved to: {out}"
            self._worker_progress = 100

        self._start_worker(run)
