'''
from matplotlib import pyplot
from matplotlib.image import imread

# define location of dataset
folder = 'D:/project/data/flow_from/fighting'

# plot first few images
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# define filename
	filename = folder + 'dog.' + str(i) + '.jpg'
	# load image pixels
	image = imread(filename)
	# plot raw pixel data
	pyplot.imshow(image)
# show the figure
pyplot.show()



import os, random
import shutil

random.choice([x for x in os.listdir("D:/project/data/flow_from/fighting") \
               if os.path.isfile(os.path.join("D:/project/data/flow_from/fighting", x))])

import os
import shutil
import numpy as np
from keras.preprocessing import image

# The path to the directory where the original dataset was uncompressed
base_dir = 'D:/project/data'
img_dir = 'D:/project/data/binary'

print(f"num of img: {len(os.listdir(img_dir))}")
# num of img: 2000


# a picture of one cat as an example
# img_name = 'cat.10.jpg'
img_path = "D:/project/data/binary"
# img_path = os.path.join(cats30_dir, img_name)

# Preprocess the image into a 4D tensor using keras.preprocessing
img = image.load_img(img_path, target_size=(250, 250))
img_tensor = image.img_to_array(img)

# expand a dimension (3D -> 4D)
img_tensor = np.expand_dims(img_tensor, axis=0)
print(f"img.shape: {img_tensor.shape}")

# scaling into [0, 1]
img_tensor /= 255.
'''


from PIL import Image
import os, glob, sys, numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

img_dir = 'D:/project/data/flow_from'
categories = ['fighting', 'normal']
np_classes = len(categories)

image_w, image_h = 224, 224
pixel = image_h * image_w * 3

x = []
y = []

for idx, fighting in enumerate(categories):
    img_dir_detail = img_dir + "/" + fighting
    files = glob.glob(img_dir_detail+"/*.jpg")

    for i, file in enumerate(files):
        try:
            img = Image.open(file)
            img = img.convert("RGB")
            img = img.resize((image_w, image_h))
            data = np.asarray(img)
            #Y는 0 아니면 1이니까 idx값으로 넣는다.
            x.append(data)
            y.append(idx)
            if i % 20 == 0:
                print(fighting, " : ", file)
        except:
            print(fighting, str(i)+" 번째에서 에러 ")
            
x = np.array(x)
y = np.array(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=99, \
                                                    validation_split=0.2, test_size=0.2)

binary_data = (x_train, x_test, y_train, y_test)
np.save("D:/project/data/flow_from/binary_data.npy", binary_data)