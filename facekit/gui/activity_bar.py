from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PySide6.QtCore import Signal, Qt

from .icons import icon


class ActivityBar(QWidget):
    """Vertical icon bar (VSCode-style) for switching pipeline views."""

    view_changed = Signal(str)

    _BUTTONS = [
        ("play", "Video", "Video preview"),
        ("scissors", "Extract", "Frame extraction"),
        ("scan", "Faces", "Face detection & matching"),
        ("users", "People", "Person detection"),
        ("zap", "Sharp", "Sharpness filters & output"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ActivityBar")
        self.setFixedWidth(42)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 6)
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._buttons: dict[str, QPushButton] = {}

        for icon_name, name, tooltip in self._BUTTONS:
            btn = QPushButton()
            btn.setIcon(icon(icon_name, size=20))
            btn.setCheckable(True)
            btn.setToolTip(tooltip)
            btn.setFixedSize(32, 32)
            btn.clicked.connect(lambda checked, n=name: self._on_click(n))
            layout.addWidget(btn, 0, Qt.AlignmentFlag.AlignHCenter)
            self._buttons[name] = btn

        layout.addStretch()

    def _on_click(self, name):
        for n, btn in self._buttons.items():
            btn.setChecked(n == name)
        self.view_changed.emit(name)

    def select(self, name):
        for n, btn in self._buttons.items():
            btn.setChecked(n == name)
