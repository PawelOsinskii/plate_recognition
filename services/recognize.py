import re

import cv2
import imutils
import numpy as np
from pytesseract import pytesseract
from collections import Counter

pytesseract.tesseract_cmd = 'Tesseract-OCR/tesseract.exe'


def recognize(input_path, original_path):
    image_to_recognition = cv2.imread(input_path)
    text = detect_text(image_to_recognition, original_path)
    if len(text) < 5:
        text = second_try(original_path)
    return text


def detect_text(plate, original_path):
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


def second_try(path_input):
    def detect(edges):
        contours = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        location = None
        detected = 0

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            if len(approx) == 4:
                location = approx
                break
        if location is None:
            raise NoCountoursDetected
        else:
            detected = 1

        return detected, location

    def mask_plate(gray, original, location):
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1, )
        new_image = cv2.bitwise_and(original, original, mask=mask)

        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        plate = new_image[topx:bottomx + 1, topy:bottomy + 1]

        return plate

    original = cv2.imread(path_input)
    original = cv2.resize(original, (600, 400))
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 13, 15, 15)
    edges = cv2.Canny(gray, 30, 200)
    try:
        detected, location = detect(edges)
    except NoCountoursDetected:
        return "is not detected"
    plate = mask_plate(gray, original, location)
    return detect_text(plate, path_input)


class NoCountoursDetected(Exception):
    pass
