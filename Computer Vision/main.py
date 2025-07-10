import sys
from PyQt5.QtWidgets import QApplication
from threading import Thread

from overlay.overlay_window import OverlayWindow
from ocr.processor import OCRProcessor
from utils.hotkeys import listen_hotkey


def main():
    app = QApplication(sys.argv)

    overlay = OverlayWindow()
    ocr = OCRProcessor()
    overlay.ocrRequested.connect(ocr.process)

    hotkey_thread = Thread(target=listen_hotkey, args=(overlay,), daemon=True)
    hotkey_thread.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
