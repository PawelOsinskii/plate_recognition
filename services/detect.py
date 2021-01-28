import cv2
import numpy as np
import tensorflow as tf


def detect(input_image_path, output_image_path, image_size):
    model = tf.keras.models.load_model("my_model.h5")
    X1 = []
    margin = 10
    img = cv2.imread(input_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    X1.append(np.array(img))
    images_matrix_rgb_2_recognize = np.array(X1)
    images_matrix_normalized_2_recognize = images_matrix_rgb_2_recognize / 255  # normalize
    y_cnn_result = model.predict(images_matrix_normalized_2_recognize)
    our_index = 0
    y_cnn_result = y_cnn_result * 255  # when I have build model I divided x and y by 255
    top_x = int(y_cnn_result[our_index][0])
    top_y = int(y_cnn_result[our_index][1])
    bottom_y = int(y_cnn_result[our_index][3])
    bottom_x = int(y_cnn_result[our_index][2])
    cropped = img[bottom_y - margin:top_y + margin, bottom_x - margin:top_x + margin]
    cv2.imwrite(output_image_path, cropped)


def threshold(input_image_path, output_image_path):
    cropped_load = cv2.imread(input_image_path, 0)
    th3 = cv2.adaptiveThreshold(cropped_load, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(output_image_path, th3)