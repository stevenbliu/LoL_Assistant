import json
import mss
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QRect, QTimer, pyqtSignal, QPoint
from PyQt5.QtGui import QPainter, QColor, QPen
from win32_helper import make_non_clickable, make_clickable


class OverlayWindow(QWidget):
    ocrRequested = pyqtSignal(list)
    toggleDrawingSignal = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)
        self.setMouseTracking(True)

        self.sct = mss.mss()
        self.monitor = self.sct.monitors[2]
        self.resize(self.monitor["width"], self.monitor["height"])
        self.move(self.monitor["left"], self.monitor["top"])

        self.rects = []
        self.start = None
        self.end = None
        self.ocr_timer = None

        self.drawing_enabled = False
        self.box_file = "boxes.json"

        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.StrongFocus)

        self.toggleDrawingSignal.connect(self.toggle_drawing)
        self.show()
        self.hwnd = self.winId().__int__()

    def toggle_drawing(self):
        self.drawing_enabled = not self.drawing_enabled
        self.setAttribute(Qt.WA_TransparentForMouseEvents, not self.drawing_enabled)
        print(f"Drawing mode: {'ON' if self.drawing_enabled else 'OFF'}")
        self.update()

    def mousePressEvent(self, event):
        if self.drawing_enabled and event.button() == Qt.LeftButton:
            self.start = event.pos()
            self.end = self.start
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing_enabled and self.start:
            self.end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drawing_enabled and self.start and self.end:
            rect = QRect(self.start, self.end).normalized()
            self.rects.append(rect)
            self.start = None
            self.end = None
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.drawing_enabled:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 255, 0, 80))
            painter.drawRect(0, 0, self.width(), self.height())

            painter.setPen(QPen(QColor(0, 255, 0, 160), 3))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(0, 0, self.width() - 1, self.height() - 1)

            painter.setPen(QPen(QColor(255, 0, 0, 180), 3))
            for rect in self.rects:
                painter.drawRect(rect)

            if self.start and self.end:
                painter.drawRect(QRect(self.start, self.end).normalized())

            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 255, 0, 180))
            painter.drawRect(10, 10, 120, 30)
            painter.setPen(QColor(0, 0, 0))
            painter.drawText(20, 30, "DRAWING ON")

    def keyPressEvent(self, event):
        key = event.key()

        if key == Qt.Key_Escape:
            if self.ocr_timer and self.ocr_timer.isActive():
                self.ocr_timer.stop()
            self.close()
        elif key == Qt.Key_C:
            self.rects.clear()
            self.update()
        elif key in (Qt.Key_Return, Qt.Key_Enter):
            if not self.ocr_timer:
                self.ocr_timer = QTimer()
                self.ocr_timer.timeout.connect(self.emit_ocr_request)
            if self.ocr_timer.isActive():
                self.ocr_timer.stop()
                print("OCR stopped.")
            else:
                self.ocr_timer.start(2000)
                print("OCR started every 2s.")
        elif key == Qt.Key_S:
            self.save_boxes()
        elif key == Qt.Key_L:
            self.load_boxes()

    def get_mss_bboxes(self):
        return [
            {
                "top": rect.top() + self.monitor["top"],
                "left": rect.left() + self.monitor["left"],
                "width": rect.width(),
                "height": rect.height(),
            }
            for rect in self.rects
        ]

    def emit_ocr_request(self):
        bboxes = self.get_mss_bboxes()
        if bboxes:
            self.ocrRequested.emit(bboxes)

    def save_boxes(self):
        with open(self.box_file, "w") as f:
            json.dump(self.get_mss_bboxes(), f, indent=2)
            print(f"Saved {len(self.rects)} boxes to {self.box_file}")

    def load_boxes(self):
        try:
            with open(self.box_file, "r") as f:
                bboxes = json.load(f)
            self.rects = [
                QRect(
                    QPoint(
                        b["left"] - self.monitor["left"], b["top"] - self.monitor["top"]
                    ),
                    QPoint(
                        b["left"] - self.monitor["left"] + b["width"],
                        b["top"] - self.monitor["top"] + b["height"],
                    ),
                )
                for b in bboxes
            ]
            self.update()
            print(f"Loaded {len(self.rects)} boxes from {self.box_file}")
        except Exception as e:
            print(f"Failed to load boxes: {e}")
