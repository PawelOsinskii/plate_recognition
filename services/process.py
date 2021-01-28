import os
import uuid

from services import detect

IMAGES_PATH = "static/images"
IMAGE_SIZE = 224


def process(image):
    image_id = str(uuid.uuid4())
    original_image_path = path_build(1, image_id)
    image.save(original_image_path)

    car_plate_detected_path = path_build(2, image_id)
    top_x, top_y, bottom_x, bottom_y = detect.detect(original_image_path, car_plate_detected_path, IMAGE_SIZE)

    image_with_threshold = path_build(3, image_id)
    detect.threshold(car_plate_detected_path, image_with_threshold)
    resized_image_path = path_build(4, image_id)
    detect.resize(original_image_path, resized_image_path, IMAGE_SIZE, top_x, top_y, bottom_x, bottom_y)



    return {
        'id': image_id,
        'text': 'SCZ89771',
    }


def path_build(step, image_id):
    return str(os.path.join(IMAGES_PATH, f"{image_id}-step-{step}.jpg"))
