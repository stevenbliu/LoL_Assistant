import pytest
import numpy as np
from overlay.ocr import OCRProcessor


class DummySct:
    def grab(self, bbox):
        # Return a dummy image array (all white)
        return np.ones((bbox["height"], bbox["width"], 4), dtype=np.uint8) * 255


@pytest.fixture
def ocr():
    ocr = OCRProcessor()
    ocr.sct = DummySct()  # Inject dummy screen grabber
    return ocr


def test_process_no_boxes(ocr, capsys):
    ocr.process([])
    captured = capsys.readouterr()
    assert "Running OCR on 0 box(es)..." in captured.out


def test_process_single_box(ocr, capsys):
    bbox = {"top": 0, "left": 0, "width": 10, "height": 10}
    ocr.process([bbox])
    captured = capsys.readouterr()
    assert "Running OCR on 1 box(es)..." in captured.out
    assert "Box 1: OCR result:" in captured.out
