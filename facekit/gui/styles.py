DARK_THEME = """
/* ═══════════════════════════════════════════════════
   macOS Dark Mode Inspired Theme
   ═══════════════════════════════════════════════════ */

/* ── Global ── */
QMainWindow, QDialog {
    background-color: #1e1e1e;
    color: #cccccc;
    font-family: -apple-system, "SF Pro Text", "Helvetica Neue", "Segoe UI", sans-serif;
    font-size: 13px;
}
QWidget {
    background-color: transparent;
    color: #cccccc;
}

/* ── Menu Bar ── */
QMenuBar {
    background-color: #2d2d2d;
    color: #bbbbbb;
    border-bottom: 1px solid #222;
    padding: 2px 0;
}
QMenuBar::item {
    padding: 4px 12px;
    border-radius: 4px;
}
QMenuBar::item:selected {
    background-color: #3a3a3a;
    color: #fff;
}
QMenu {
    background-color: #2d2d2d;
    color: #cccccc;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    padding: 4px 0;
}
QMenu::item {
    padding: 5px 24px 5px 12px;
}
QMenu::item:selected {
    background-color: #0a64ff;
    color: #ffffff;
}
QMenu::separator {
    height: 1px;
    background: #3a3a3a;
    margin: 4px 8px;
}

/* ── Toolbar (icon strip below menu) ── */
#Toolbar {
    background-color: #2d2d2d;
    border-bottom: 1px solid #222;
    padding: 3px 6px;
}
#Toolbar QPushButton {
    background: transparent;
    color: #999;
    border: none;
    border-radius: 5px;
    padding: 5px 6px;
    min-width: 28px;
    min-height: 28px;
}
#Toolbar QPushButton:hover {
    background-color: #3a3a3a;
    color: #fff;
}
#Toolbar QPushButton:pressed {
    background-color: #444;
}
#Toolbar QPushButton:disabled {
    color: #555;
}
#ToolbarSeparator {
    background-color: #3a3a3a;
    max-width: 1px;
    margin: 4px 4px;
}

/* ── Activity Bar (left icon strip) ── */
#ActivityBar {
    background-color: #2d2d2d;
    border-right: 1px solid #222;
    max-width: 42px;
}
#ActivityBar QPushButton {
    background: transparent;
    border: none;
    border-radius: 5px;
    padding: 9px;
    color: #777;
    min-width: 32px;
    min-height: 32px;
}
#ActivityBar QPushButton:hover {
    background-color: #3a3a3a;
    color: #ccc;
}
#ActivityBar QPushButton:checked {
    background-color: #3a3a3a;
    color: #0a64ff;
}

/* ── Sidebar Panel ── */
#Sidebar {
    background-color: #252525;
    border-right: 1px solid #222;
    min-width: 220px;
    max-width: 280px;
}
#SidebarTitle {
    color: #999;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    padding: 8px 14px 4px;
}
#SidebarSection {
    color: #666;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 8px 14px 4px;
}
#SidebarSeparator {
    max-height: 1px;
    background-color: #333;
    margin: 4px 14px;
}

/* ── Sidebar Controls ── */
#Sidebar QPushButton {
    background-color: #333;
    color: #ccc;
    border: none;
    border-radius: 4px;
    padding: 7px 10px;
    font-size: 12px;
    text-align: center;
}
#Sidebar QPushButton:hover {
    background-color: #404040;
    color: #fff;
}
#Sidebar QPushButton#SidebarAction {
    background-color: #3a3a3a;
    color: #ddd;
    font-weight: 500;
    padding: 10px 14px;
    margin: 3px 14px;
    border-radius: 0;
    min-height: 20px;
}
#Sidebar QPushButton#SidebarAction:hover {
    background-color: #0a64ff;
    color: #fff;
}

/* ── Mark Button (player controls) ── */
#MarkButton {
    background-color: #1a5c2a;
    color: #4ade80;
    border: 1px solid #22c55e;
    border-radius: 4px;
    font-weight: 600;
}
#MarkButton:hover {
    background-color: #22c55e;
    color: #fff;
}

/* ── Face Thumbnail ── */
#FaceThumb {
    background-color: #333;
    border-radius: 4px;
    border: 1px solid #444;
}
#FaceThumb:hover {
    border-color: #ef4444;
}
#FaceThumb #DeleteBtn {
    background-color: #ef4444;
    color: #fff;
    border: none;
    border-radius: 8px;
    font-size: 10px;
    font-weight: bold;
    max-width: 16px;
    max-height: 16px;
    min-width: 16px;
    min-height: 16px;
    padding: 0;
}
#FaceThumb #DeleteBtn:hover {
    background-color: #dc2626;
}
#Sidebar QComboBox, #Sidebar QSpinBox, #Sidebar QDoubleSpinBox {
    background-color: #333;
    color: #ccc;
    border: 1px solid #444;
    border-radius: 5px;
    padding: 4px 8px;
    font-size: 12px;
}
#Sidebar QComboBox:hover, #Sidebar QSpinBox:hover, #Sidebar QDoubleSpinBox:hover {
    border-color: #0a64ff;
}
#Sidebar QComboBox::drop-down {
    border: none;
    width: 20px;
}
#Sidebar QComboBox QAbstractItemView {
    background-color: #2d2d2d;
    color: #ccc;
    border: 1px solid #444;
    selection-background-color: #0a64ff;
    border-radius: 4px;
}

/* ── Video Canvas ── */
#VideoCanvas {
    background-color: #111;
}
#VideoCanvas QLabel {
    background-color: #111;
    color: #444;
    font-size: 16px;
}

/* ── Player Controls ── */
#PlayerControls {
    background-color: #2d2d2d;
    border-top: 1px solid #222;
    padding: 4px 8px;
}
#PlayerControls QSlider::groove:horizontal {
    border: none;
    height: 3px;
    background: #444;
    border-radius: 2px;
}
#PlayerControls QSlider::handle:horizontal {
    background: #fff;
    border: none;
    width: 12px;
    height: 12px;
    margin: -5px 0;
    border-radius: 6px;
}
#PlayerControls QSlider::sub-page:horizontal {
    background: #0a64ff;
    border-radius: 2px;
}
#PlayerControls QLineEdit {
    background-color: #333;
    color: #ccc;
    border: 1px solid #444;
    border-radius: 4px;
    padding: 3px 6px;
    font-size: 12px;
    font-family: "SF Mono", "Menlo", "Consolas", monospace;
}
#PlayerControls QLineEdit:focus {
    border-color: #0a64ff;
}
#PlayerControls QLabel {
    color: #777;
    font-size: 11px;
}

/* ── Status Bar ── */
QStatusBar {
    background-color: #0a64ff;
    color: #fff;
    font-size: 11px;
    padding: 2px 8px;
}
QStatusBar QLabel {
    color: rgba(255,255,255,0.8);
    font-size: 11px;
    padding: 0 4px;
}

/* ── Progress Bar ── */
QProgressBar {
    border: none;
    border-radius: 3px;
    background-color: #333;
    text-align: center;
    color: #ccc;
    font-size: 11px;
    max-height: 14px;
}
QProgressBar::chunk {
    background-color: #0a64ff;
    border-radius: 3px;
}

/* ── Scrollbar (macOS thin style) ── */
QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background: #555;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #888;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar:horizontal {
    background: transparent;
    height: 8px;
    margin: 2px;
}
QScrollBar::handle:horizontal {
    background: #555;
    border-radius: 4px;
    min-width: 30px;
}
QScrollBar::handle:horizontal:hover {
    background: #888;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* ── Checkbox ── */
QCheckBox {
    color: #ccc;
    font-size: 12px;
    spacing: 6px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid #555;
    background-color: #333;
}
QCheckBox::indicator:checked {
    background-color: #0a64ff;
    border-color: #0a64ff;
}

/* ── Tooltips ── */
QToolTip {
    background-color: #3a3a3a;
    color: #fff;
    border: none;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 11px;
}

/* ── Bookmarks ── */
#BookmarkItem {
    background-color: #333;
    border-radius: 5px;
    padding: 4px;
}
#BookmarkItem QPushButton {
    background-color: #444;
    color: #ccc;
    border: none;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 11px;
}
#BookmarkItem QPushButton:hover {
    background-color: #555;
}
#BookmarkItem #RemoveBtn {
    background-color: #5a2020;
    color: #ff6b6b;
    max-width: 20px;
}
#BookmarkItem #RemoveBtn:hover {
    background-color: #ff4444;
    color: #fff;
}
"""
