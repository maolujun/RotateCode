import cv2
import os

# window
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(f'{BASE_DIR}')


from utils import generate_rotated_image
import random
root_path="D:\\work\ERP\\workspace\\baidu_login\\some_demo"
create_path="D:\\work\ERP\\workspace\\baidu_login\\angle_demo"
filenames=os.listdir(root_path)
for filename in filenames:
    image = cv2.imread(os.path.join(root_path,filename), 1)
    h, w = image.shape[:2]
    if h != 224 or w != 224:
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)

    rotation_angle = random.randint(0,360)
    rotated_image = generate_rotated_image(
                    image,
                    rotation_angle
                )
    cv2.imwrite(f"{create_path}\\{rotation_angle}_{filename}", rotated_image)