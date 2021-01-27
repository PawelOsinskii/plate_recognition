from flask import Flask
#imports
from tensorflow.keras.models import Sequential
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
#import cv2
import os
import glob
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf

app = Flask(__name__)


#load model
model = tf.keras.models.load_model("my_model.h5")

IMAGE_SIZE = 224


@app.route('/')
def hello_world():
    return 'Hello World!'


if __name__ == '__main__':

    app.run(host='0.0.0.0')
