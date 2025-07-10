import pytest
from PyQt5.QtCore import Qt, QRect, QPoint
from overlay.window import OverlayWindow


@pytest.fixture
def overlay(qtbot):
    w = OverlayWindow()
    qtbot.addWidget(w)
    return w


def test_initial_state(overlay):
    assert overlay.drawing_enabled is False
    assert overlay.rects == []


def test_toggle_drawing(overlay):
    overlay.toggle_drawing()
    assert overlay.drawing_enabled is True
    overlay.toggle_drawing()
    assert overlay.drawing_enabled is False


def test_mouse_events_add_rect(overlay, qtbot):
    overlay.toggle_drawing()
    start_point = QPoint(10, 10)
    end_point = QPoint(100, 100)

    # Simulate mouse press
    qtbot.mousePress(overlay, Qt.LeftButton, pos=start_point)
    assert overlay.start == start_point

    # Simulate mouse move
    qtbot.mouseMove(overlay, pos=end_point)
    assert overlay.end == end_point

    # Simulate mouse release
    qtbot.mouseRelease(overlay, Qt.LeftButton, pos=end_point)
    assert len(overlay.rects) == 1
    rect = overlay.rects[0]
    assert rect.contains(start_point)
    assert rect.contains(end_point)


def test_clear_rects(overlay):
    overlay.rects.append(QRect(0, 0, 10, 10))
    overlay.rects.append(QRect(5, 5, 10, 10))
    overlay.rects.clear()
    assert overlay.rects == []


def test_get_mss_bboxes(overlay):
    # Add rects and test conversion to MSS bbox dicts
    overlay.monitor = {"top": 0, "left": 0, "width": 100, "height": 100}
    overlay.rects.append(QRect(1, 2, 3, 4))
    bboxes = overlay.get_mss_bboxes()
    assert bboxes == [{"top": 2, "left": 1, "width": 3, "height": 4}]
