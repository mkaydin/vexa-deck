"""Themed window controls with compositor-managed movement and resizing."""

import pixel_theme as theme
from pixel_theme import font
from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class CaptionButton(QPushButton):
    def __init__(self, action, callback, parent):
        super().__init__(parent)
        self.action = action
        self.restored = False
        self.setObjectName("captionClose" if action == "Close" else "captionButton")
        self.setFixedSize(40, 28)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setProperty("keyboardFocus", False)
        self.setAccessibleName(action)
        self.setToolTip(action)
        self.clicked.connect(callback)

    def set_keyboard_focus(self, enabled):
        self.setProperty("keyboardFocus", enabled)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def focusInEvent(self, event):
        self.set_keyboard_focus(
            event.reason()
            in (
                Qt.FocusReason.TabFocusReason,
                Qt.FocusReason.BacktabFocusReason,
                Qt.FocusReason.ShortcutFocusReason,
            )
        )
        super().focusInEvent(event)

    def focusOutEvent(self, event):
        self.set_keyboard_focus(False)
        super().focusOutEvent(event)

    def mousePressEvent(self, event):
        self.set_keyboard_focus(False)
        super().mousePressEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(QPen(QColor(theme.PINK if self.action == "Close" else theme.CYAN), 1.5))
        x, y = (self.width() - 12) // 2, (self.height() - 12) // 2
        if self.action == "Close":
            painter.drawLine(x + 1, y + 1, x + 11, y + 11)
            painter.drawLine(x + 11, y + 1, x + 1, y + 11)
        elif self.action == "Minimize":
            painter.drawLine(x, y + 10, x + 12, y + 10)
        elif self.restored:
            painter.drawRect(x + 3, y, 9, 9)
            painter.drawRect(x, y + 3, 9, 9)
        else:
            painter.drawRect(x, y, 12, 12)


class TitleBar(QFrame):
    def __init__(self, window):
        super().__init__(window)
        self.host = window
        self.setObjectName("titleBar")
        self.setFixedHeight(36)
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 4, 8, 4)
        row.setSpacing(4)
        icon = QLabel()
        icon.setPixmap(window.windowIcon().pixmap(24, 24))
        icon.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        row.addWidget(icon)
        self.title = QLabel(window.windowTitle())
        self.title.setFont(font(10, 700))
        self.title.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        row.addWidget(self.title)
        row.addStretch()
        self.minimize_button = CaptionButton("Minimize", window.showMinimized, self)
        self.maximize_button = CaptionButton("Maximize", self.toggle_maximized, self)
        self.close_button = CaptionButton("Close", window.close, self)
        for button in (self.minimize_button, self.maximize_button, self.close_button):
            row.addWidget(button)
        window.windowTitleChanged.connect(self.title.setText)

    def sync_state(self):
        maximized = self.host.isMaximized()
        self.maximize_button.restored = maximized
        action = "Restore" if maximized else "Maximize"
        self.maximize_button.setAccessibleName(action)
        self.maximize_button.setToolTip(action)
        self.maximize_button.update()

    def toggle_maximized(self):
        if self.host.isMaximized():
            self.host.showNormal()
        else:
            self.host.showMaximized()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            handle = self.host.windowHandle()
            if handle is not None and handle.startSystemMove():
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle_maximized()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class ResizeController(QObject):
    """Delegate edge drags to Qt's native Wayland/X11 resize operation."""

    BORDER = 6

    def __init__(self, window):
        super().__init__(window)
        self.host = window
        self._cursor_widget = None
        self._previous_cursor = None
        self.host.installEventFilter(self)
        QTimer.singleShot(0, self.track_widgets)

    def track_widgets(self):
        self.host.setMouseTracking(True)
        for child in self.host.findChildren(QWidget):
            # Scope native window handling to our widgets. A global Python
            # event filter also wraps Chromium's private QObjects on startup.
            if child.window() is not self.host:
                continue
            child.setMouseTracking(True)
            child.installEventFilter(self)

    def edges_at(self, position):
        edges = Qt.Edge(0)
        if self.host.isMaximized() or self.host.isFullScreen():
            return edges
        if position.x() < self.BORDER:
            edges |= Qt.Edge.LeftEdge
        elif position.x() >= self.host.width() - self.BORDER:
            edges |= Qt.Edge.RightEdge
        if position.y() < self.BORDER:
            edges |= Qt.Edge.TopEdge
        elif position.y() >= self.host.height() - self.BORDER:
            edges |= Qt.Edge.BottomEdge
        return edges

    def clear_cursor(self):
        if self._cursor_widget is not None:
            self._cursor_widget.setCursor(self._previous_cursor)
            self._cursor_widget = None

    def eventFilter(self, watched, event):
        kind = event.type()
        if watched is self.host and kind == QEvent.Type.WindowStateChange:
            self.host.title_bar.sync_state()
            self.clear_cursor()
        if kind not in (QEvent.Type.MouseMove, QEvent.Type.MouseButtonPress, QEvent.Type.Leave):
            return False
        if not isinstance(watched, QWidget) or watched.window() is not self.host:
            return False
        if kind == QEvent.Type.Leave:
            self.clear_cursor()
            return False
        edges = self.edges_at(watched.mapTo(self.host, event.position().toPoint()))
        if kind == QEvent.Type.MouseButtonPress:
            handle = self.host.windowHandle()
            if edges and event.button() == Qt.MouseButton.LeftButton and handle is not None:
                return handle.startSystemResize(edges)
            return False
        self.clear_cursor()
        if edges:
            horizontal = bool(edges & (Qt.Edge.LeftEdge | Qt.Edge.RightEdge))
            vertical = bool(edges & (Qt.Edge.TopEdge | Qt.Edge.BottomEdge))
            if horizontal and vertical:
                cursor = (
                    Qt.CursorShape.SizeFDiagCursor
                    if edges
                    in (Qt.Edge.TopEdge | Qt.Edge.LeftEdge, Qt.Edge.BottomEdge | Qt.Edge.RightEdge)
                    else Qt.CursorShape.SizeBDiagCursor
                )
            else:
                cursor = (
                    Qt.CursorShape.SizeHorCursor if horizontal else Qt.CursorShape.SizeVerCursor
                )
            self._cursor_widget, self._previous_cursor = watched, watched.cursor()
            watched.setCursor(cursor)
        return False


def window_layout(window, root):
    """Install shared chrome and return the padded content layout."""
    window.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
    shell = QVBoxLayout(root)
    shell.setContentsMargins(0, 0, 0, 0)
    shell.setSpacing(0)
    window.title_bar = TitleBar(window)
    shell.addWidget(window.title_bar)
    body = QWidget()
    shell.addWidget(body, 1)
    content = QVBoxLayout(body)
    content.setContentsMargins(12, 12, 12, 12)
    content.setSpacing(8)
    window.resize_controller = ResizeController(window)
    return content
