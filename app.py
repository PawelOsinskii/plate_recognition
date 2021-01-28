# imports
import re
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

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# load model
model = tf.keras.models.load_model("my_model.h5")

IMAGE_SIZE = 224
app.config["IMAGE_UPLOADS"] = r"C:\Users\posinski\PycharmProjects\plate_recognition"


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            print(image)
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], 'images_to_algorithm/obraz.jpg'))
            plate_string = do_algorithm('images_to_algorithm/obraz.jpg')
            global plate
            plate = plate_string
            return redirect(request.url)

    return render_template("upload_image.html")


def do_algorithm(path):
    X1 = []
    margin = 10
    img = cv2.imread(path)
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
    path = "images_to_algorithm/cropped.png"
    path2 = "images_to_algorithm/cropped2.png"
    cv2.imwrite(path, cropped)
    cropped_load = cv2.imread(path, 0)
    th3 = cv2.adaptiveThreshold(cropped_load, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(path2, th3)
    plate = cv2.imread(path2)
    return detect_text(plate)


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
    return "\nFinal result: " + most_common_word[0]


@app.route('/get_plate', methods=['GET', 'POST'])
def get_plate():
    return plate


if __name__ == '__main__':
    app.run(host='0.0.0.0')
