import cv2
import os
import glob
import itertools
import numpy as np
import matplotlib.pyplot as plt

#1-1. 이미지 파일 리스트 생성
folder_path = "./data/label_v2"

for m in os.listdir(folder_path):
    # print(f"상위폴더: {m}")

    for n in os.listdir(folder_path + '/' + m):
        # print(f"하위폴더: {n}")

        for f in os.listdir(folder_path + '/' + m + '/' + n):
            img_path = "./data/label_v2/" +m+'/'+n+'/'+f
            print(f"img_path:\n{img_path}")

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_input = img.copy()

            # convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            img_rgb = img.copy()

            # normalize input
            img_rgb = (img_rgb / 255.).astype(np.float32)

            # convert RGB to LAB
            img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
            # only L channel to be used
            img_l = img_lab[:, :, 0]

            input_img = cv2.resize(img_l, (224, 224))
            input_img -= 50 # subtract 50 for mean-centering

            # plot images
            fig = plt.figure(figsize=(10, 5))
            fig.add_subplot(1, 2, 1)
            plt.imshow(img_rgb)
            fig.add_subplot(1, 2, 2)
            plt.axis('off')
            plt.imshow(input_img, cmap='gray')

# 6,600장 out of memory
