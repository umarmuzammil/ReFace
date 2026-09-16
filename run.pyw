import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from facekit.gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Re:Face")
    app.setApplicationVersion("2.1.0")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
