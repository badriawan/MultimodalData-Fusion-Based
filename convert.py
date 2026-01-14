import os
import cv2
import numpy as np

input_dir = "/Users/yusuf/Documents/Work_PhD/S3/UTP/Dummy_Test/MS2_dataset/thermal_left/"
output_dir = "/Users/yusuf/Documents/Work_PhD/S3/UTP/Dummy_Test/MS2_dataset/thermal_left_uint8/"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    img16 = cv2.imread(os.path.join(input_dir, fname), cv2.IMREAD_UNCHANGED)
    img8 = cv2.normalize(img16, None, 0, 255, cv2.NORM_MINMAX)
    img8 = img8.astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, fname), img8)
