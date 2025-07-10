import mss
import numpy as np
import cv2
import pytesseract


class OCRProcessor:
    def __init__(self):
        self.sct = mss.mss()

    def process(self, bboxes):
        print(f"Running OCR on {len(bboxes)} box(es)...")
        for i, bbox in enumerate(bboxes):
            img = np.array(self.sct.grab(bbox))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            config = r"--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789"
            text = pytesseract.image_to_string(thresh, config=config).strip()
            text = text if text.isdigit() else "N/A"
            print(f"Box {i+1}: OCR result: {text}")
