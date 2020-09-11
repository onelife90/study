import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from natsort import natsorted
import glob
from PIL import Image
import numpy as np
from keras.utils import np_utils

#1. show binary origin image 
# define location of dataset
main_folder = 'D:/project/data/flow_from'
directory = os.listdir(main_folder)

# plot for each folder
for each in directory:
    fig = plt.figure() 
    cur_folder = main_folder + "/" + each
    print(f"cur_folder:{cur_folder}")
    # read 3 images for each folder 
    for i, file in enumerate(os.listdir(cur_folder)[0:3]):
        fullpath = main_folder + "/" + each + "/" + file
        print(fullpath)
        img = mpimg.imread(fullpath)
        fig.add_subplot(2, 3, i+1)
        plt.imshow(img)
    #plt.show()

#2. image down size
# sorted fight images
fight = []    
resized_fight = []  

for filename in natsorted(glob.glob('D:/project/data/flow_from/fighting/*.jpg')):
    fight_open = Image.open(filename) 
    fight.append(fight_open)    

# append resized images to list
for fight_img in fight:       
    fight_img = fight_img.resize((224, 224))
    resized_fight.append(fight_img)  

# save resized images to new folder
for i, new in enumerate(resized_fight):
    new.save(f"{'D:/project/data/binary/fight'},{i},{'.jpg'}") 

# sorted normal images
normal = []
resized_normal = [] 

for filename in natsorted(glob.glob('D:/project/data/flow_from/normal/*.jpg')):
    normal_open = Image.open(filename) 
    normal.append(normal_open)    

# append resized images to list
for normal_img in normal:       
    normal_img = normal_img.resize((224, 224))
    resized_normal.append(normal_img)  

# save resized images to new folder
for i, new in enumerate(resized_normal):
    new.save(f"{'D:/project/data/binary/normal'},{i},{'.jpg'}") 


categories = ["fight", "normal"]  

x = []
y = []

for index, categorie in enumerate(categories):
    label = [0 for i in range(2)]
    # label[index] = 1

    # image file path
    image_dir = main_folder + "/" + categorie + '/'
    files = glob.glob(image_dir +  "*.jpg")

    for img, filename in enumerate(files):
        img = Image.open(filename)
        img = img.convert("RGB")
        data = np.asarray(img)
        x.append(data)
        y.append(label)

x = np.array(x)
y = np.array(y)

print(y)

print("x.shape :", x.shape)   # (200, 64, 64, 3)
print("y.shape :", y.shape)   # (200, 2)

# numpy로 최종 저장
np.save('./data/x_data.npy', x)
np.save('./data/y_data.npy', y)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import glob
import glob

x = np.load('./data/x_data.npy')
y = np.load('./data/y_data.npy')

print("x.shape :", x.shape)
print("y.shape :", y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state = 77, shuffle = True
)

print("x_train.shape :", x_train.shape)  # (160, 64, 64, 3)
print("x_test.shape :", x_test.shape)    # (40, 64, 64, 3)
print("y_train.shape :", y_train.shape)  # (160, 2)
print("y_test.shape :", y_test.shape)    # (40, 2)





# Scaler 사용하기 위해 Reshape
x_train = x_train.reshape(x_train.shape[0], 64*64*3)
x_test = x_test.reshape(x_test.shape[0], 64*64*3)

# MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# CNN 모델에 맞게 Reshape
x_train = x_train.reshape(x_train.shape[0], 64, 64, 3)
x_test = x_test.reshape(x_test.shape[0], 64, 64, 3)



# 2. 모델 구성

model = Sequential()

model.add(Conv2D(32, (2, 2), activation = 'relu', input_shape = (64, 64, 3))) 
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.25))

model.add(Conv2D(64, (2, 2)))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.25))

model.add(Conv2D(128, (2, 2)))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Dropout(0.25))
model.add(Dense(32))
model.add(Dropout(0.1))
model.add(Dense(16))
model.add(Dropout(0.1))
model.add(Dense(8))
model.add(Dense(2, activation = 'sigmoid'))

model.summary()


# 3. 컴파일, 훈련
modelpath = './check/check--{epoch:02d}--{val_loss:.4f}.hdf5'

cp = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', 
                     save_best_only = True, mode = 'auto')

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', 
                                            metrics = ['acc'])

hist = model.fit(x_train, y_train, epochs = 50, batch_size = 32, 
          validation_split = 0.3, verbose = 1, callbacks = [cp])










# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 32)
print("LOSS :", loss)
print("ACC :", acc)


# 시각화
plt.figure(figsize = (12, 10))

plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker = 'o', 
                   c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = 'o', 
                 c = 'blue', label = 'val_loss')
plt.grid()
plt.title('Training and Val loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Training loss', 'Val loss'], 
                     loc = 'upper right')

plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'], marker = 'o', 
              c = 'purple', label = 'acc' )
plt.plot(hist.history['val_acc'], marker = 'o', 
               c = 'green', label = 'val_loss')
plt.grid()
plt.title('Training and Val accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['Training acc', 'Val acc'], 
                    loc = 'upper left')

plt.show()
'''