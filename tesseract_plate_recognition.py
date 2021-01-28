import re


import cv2
import imutils
import numpy as np
import pytesseract


def load_and_edit(filename):
    original = cv2.imread(filename)
    original = cv2.resize(original, (600, 400))
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 13, 15, 15)
    edges = cv2.Canny(gray, 30, 200)
    return original, gray, edges


def detect(edges):
    contours = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    location = None

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            location = approx
            break

    if location is None:
        print("No contour detected")
        sys.exit()
    else:
        detected = 1

    return detected, location


def display_img(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def mask_plate(gray, original, location):
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1, )
    new_image = cv2.bitwise_and(original, original, mask=mask)

    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    plate = new_image[topx:bottomx + 1, topy:bottomy + 1]

    return plate


def detect_text(plate):
    string_list = []
    for i in range(3, 14):
        config = r'--tessdata-dir "C:\Program Files\Tesseract-OCR\tessdata"'
        config += r' --psm {}'.format(i)
        text = pytesseract.image_to_string(plate, config=config)
        text = re.sub(r'\W+', '', text)
        if len(text) > 0:
            string_list.append(text)
        print("License plate number (PSM {}):".format(i), text)
    most_common_word = [word for word, word_count in Counter(string_list).most_common(1)]
    print("\nFinal result: " + most_common_word[0])


# 1-23
# 2, 8
filename = r"E:\DCIM\100NCD90\canvas.jpg"
# 1-10
# 1, 9
# filename = 'bad_plates/9.jpg'

original, gray, edges = load_and_edit(filename)
display_img('Original', original)
display_img('Gray', gray)
display_img('Edges', edges)
detected, location = detect(edges)
copy = original.copy()
if detected == 1:
    cv2.drawContours(copy, [location], -1, (0, 0, 255), 3)
display_img('Marked plate', copy)
plate = mask_plate(gray, original, location)

display_img('License plate', plate)

detect_text(plate)