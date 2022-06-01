from __future__ import print_function

import os

# window
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(f'{BASE_DIR}')

import numpy as np

from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model

from utils import RotNetDataGenerator, angle_error


def process_images(input_path,
                   batch_size=64, crop=True):
    model = load_model("D:\\work\\ERP\\workspace\\RotNet\\rotnet_models\\rotnet_street_view_resnet50_keras2.hdf5", custom_objects={'angle_error': angle_error}, compile=False)

    extensions = ['.jpg', '.jpeg', '.bmp', '.png']

    if os.path.isfile(input_path) or input_path[:4].lower()=="http":
        image_paths = [input_path]

    else:
        image_paths = [os.path.join(input_path, f)
                       for f in os.listdir(input_path)
                       if os.path.splitext(f)[1].lower() in extensions]


    predictions = model.predict_generator(
        RotNetDataGenerator(
            image_paths,
            input_shape=(224, 224, 3),
            batch_size=batch_size,
            one_hot=True,
            preprocess_func=preprocess_input,
            rotate=False,
            crop_largest_rect=True,
            crop_center=True
        )
    )

    predicted_angles = np.argmax(predictions, axis=1)
    print(predicted_angles)
    return predicted_angles



if __name__ == '__main__':
    # print('Processsing input image(s)...')
    process_images("D:\\work\\ERP\\workspace\\baidu_login\\some_demo\\img.jpg")
