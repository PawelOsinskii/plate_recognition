import re

import cv2
from pytesseract import pytesseract
from collections import Counter

pytesseract.tesseract_cmd = 'Tesseract-OCR/tesseract.exe'


def recognize(input_path):
    image_to_recognition = cv2.imread(input_path)
    return detect_text(image_to_recognition)


def detect_text(plate):
    string_list = []
    for i in range(3, 14):
        config = r'--tessdata-dir "Tesseract-OCR\tessdata"'
        config += r' --psm {}'.format(i)
        text = pytesseract.image_to_string(plate, config=config)
        text = re.sub(r'\W+', '', text)
        if len(text) > 0:
            string_list.append(text)
        print("License plate number (PSM {}):".format(i), text)
    most_common_word = [word for word, word_count in Counter(string_list).most_common(1)]
    return most_common_word[0]
