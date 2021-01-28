# imports
import re
import imutils
from collections import Counter
import cv2
import numpy as np
import pytesseract
import tensorflow as tf
from flask import Flask, flash, request, redirect
from flask import render_template
from matplotlib import pyplot as plt

app = Flask(__name__)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

import os

plate = 'brak tablicy'

UPLOAD_FOLDER = '/images/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


# load model
model = tf.keras.models.load_model("my_model.h5")

IMAGE_SIZE = 224
app.config["IMAGE_UPLOADS"] = "static/images_to_algorithm"


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            print(image)
            path_original = 'obraz.jpg'
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], path_original))
            plate_string = do_algorithm('static/images_to_algorithm/obraz.jpg')
            global plate
            plate = plate_string
            return redirect(request.url)

    return render_template("upload_image.html")


def do_algorithm(path_original):
    global plate
    plate = "brak tablicy"
    X1 = []
    margin = 10
    img = cv2.imread(path_original)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    X1.append(np.array(img))
    images_matrix_rgb_2_recognize = np.array(X1)
    images_matrix_normalized_2_recognize = images_matrix_rgb_2_recognize / 255
    y_cnn_result = model.predict(images_matrix_normalized_2_recognize)
    our_index = 0
    y_cnn_result = y_cnn_result * 255  # na początku znormalizowałem dzieląc przez 255, powinienem przez IMAGE_SIZE
    plt.imshow(img)
    top_x = int(y_cnn_result[our_index][0])
    top_y = int(y_cnn_result[our_index][1])
    bottom_y = int(y_cnn_result[our_index][3])
    bottom_x = int(y_cnn_result[our_index][2])
    cropped = img[bottom_y - margin:top_y + margin, bottom_x - margin:top_x + margin]
    path = "static/images_to_algorithm/cropped.png"
    path2 = "static/images_to_algorithm/cropped2.png"
    cv2.imwrite(path, cropped)
    cropped_load = cv2.imread(path, 0)
    th3 = cv2.adaptiveThreshold(cropped_load, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(path2, th3)

    # zwiększanie obrazka o 3
    scale = 3
    #margin = margin * scale
    img = cv2.resize(img, (IMAGE_SIZE * scale, IMAGE_SIZE * scale))
    top_x = top_x * scale
    top_y = top_y * scale
    bottom_y = bottom_y * scale
    bottom_x = bottom_x * scale
    img = np.array(img)
    cropped = img[bottom_y - margin:top_y + margin, bottom_x - margin:top_x + margin]
    path3 = "static/images_to_algorithm/cropped3.png"
    cv2.imwrite(path3, cropped)
    cropped_load = cv2.imread(path, 0)
    th3 = cv2.adaptiveThreshold(cropped_load, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(path3, th3)
    image_to_recognition = cv2.imread(path3)


    # rozpoznany obraz
    recognition_string = detect_text(image_to_recognition)

    #jesli wynik jest podejrzanie za krotki to jeszcze raz
    recognition_string = recognition_string.replace(' ', '')
    print(len(recognition_string))
    if len(recognition_string) < 5:
        recognition_string = second_try(path_original)
    return recognition_string


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
    return most_common_word[0]


@app.route('/get_plate', methods=['GET', 'POST'])
def get_plate():
    return plate




#druga próba
def second_try(path2):
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
            print("No contour detected")
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

    original = cv2.imread(path2)
    original = cv2.resize(original, (600, 400))
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 13, 15, 15)
    edges = cv2.Canny(gray, 30, 200)
    detected, location = detect(edges)
    if location is None:
        global plate
        return plate
    plate = mask_plate(gray, original, location)
    return detect_text(plate)



if __name__ == '__main__':
    app.run(host='0.0.0.0')
