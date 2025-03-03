from __future__ import print_function

import os

# window
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(f'{BASE_DIR}')

import cv2
import numpy as np
import argparse

from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model

from utils import RotNetDataGenerator, crop_largest_rectangle, angle_error, rotate


def process_images(model, input_path, output_path,
                   batch_size=64, crop=True):
    extensions = ['.jpg', '.jpeg', '.bmp', '.png']

    output_is_image = False
    if os.path.isfile(input_path):
        image_paths = [input_path]
        if os.path.splitext(output_path)[1].lower() in extensions:
            output_is_image = True
            output_filename = output_path
            output_path = os.path.dirname(output_filename)
    else:
        image_paths = [os.path.join(input_path, f)
                       for f in os.listdir(input_path)
                       if os.path.splitext(f)[1].lower() in extensions]
        if os.path.splitext(output_path)[1].lower() in extensions:
            print('Output must be a directory!')

    predictions = model.predict_generator(
        RotNetDataGenerator(
            image_paths,
            input_shape=(224, 224, 3),
            batch_size=64,
            one_hot=True,
            preprocess_func=preprocess_input,
            rotate=False,
            crop_largest_rect=True,
            crop_center=True
        )
    )

    predicted_angles = np.argmax(predictions, axis=1)

    if output_path == '':
        output_path = '.'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for path, predicted_angle in zip(image_paths, predicted_angles):
        image = cv2.imread(path)
        rotated_image = rotate(image, -predicted_angle)
        print(path,predicted_angle)
        if crop:
            size = (image.shape[0], image.shape[1])
            rotated_image = crop_largest_rectangle(rotated_image, -predicted_angle, *size)
        if not output_is_image:
            output_filename = os.path.join(output_path, os.path.basename(path))
        cv2.imwrite(output_filename, rotated_image)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('model', help='Path to model')
    # parser.add_argument('input_path', help='Path to image or directory')
    # parser.add_argument('-o', '--output_path', help='Output directory')
    # parser.add_argument('-b', '--batch_size', help='Batch size for running the network')
    # parser.add_argument('-c', '--crop', dest='crop', default=False, action='store_true',
    #                     help='Crop out black borders after rotating')
    # args = parser.parse_args()

    print('Loading model...')
    model_location = load_model("D:\\work\\ERP\\workspace\\RotNet\\rotnet_models\\rotnet_street_view_resnet50_keras2.hdf5", custom_objects={'angle_error': angle_error}, compile=False)
    # output_path = args.output_path if args.output_path else args.input_path
    # output_path="I:\\pythonProject\\RotNet\\data\\test_examples"
    print('Processsing input image(s)...')
    # process_images(model_location, args.input_path, output_path,
    #                args.batch_size, args.crop)
    process_images(model_location, "D:\\work\\ERP\\workspace\\baidu_login\\some_demo\\img.jpg",
                   "D:\\work\\ERP\\workspace\\RotNet\\data\\test_login")