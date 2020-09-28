import os
import glob
import cv2
import tensorflow as tf
import numpy as np
from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Nadam
from keras.layers import LeakyReLU
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
import time, datetime


#1-1. define images to figure (openCV resize, MinmaxScaler)
def load_train(train_path, image_size, classes):
    images = []
    labels = []

    print('Going to read training images')
    # classes = ["fighting", "normal"]
    for fields in classes:
        index = classes.index(fields)
        print(f'Now going to read {fields} files index: {index}')
        path = os.path.join(train_path, fields, '*jpg')

        # make file list
        files = glob.glob(path)
        for file in files:
            # image down-size
            image = cv2.imread(file,0)
            image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_AREA)

            # cv2.imshow('image', image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows() 

            # MinmaxScaler apply to x data
            image = image.astype(np.float32) / 255.0
            image1 = image.reshape(224,224,1)
            images.append(image1)

            # y data
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

#1-2. images to figure params
train_path = "D:/project/data/flow_from"
image_size = 224
classes = ["fighting", "normal"]

# result = return = images(x data), labels(y data)
result = load_train(train_path, image_size, classes)

print(f"x:\n{result[0]}")
# x: 
# [[[[0.46274513]
#    [0.46274513]
#    [0.45882356]
#    ...
#    [0.37647063]
#    [0.37647063]
#    [0.37647063]]

print(f"y:\n{result[1]}")
# y: 
# [[1. 0.]
#  [1. 0.]
#  [1. 0.]
#  ...
#  [0. 1.]
#  [0. 1.]
#  [0. 1.]]


# images(x data), labels(y data)
x = result[0]
y = result[1]

#1-3. train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

print(f"x_train.shape: {x_train.shape}, x_test.shape: {x_test.shape}") 
# x_train.shape: (1600, 224, 224, 1), x_test.shape: (400, 224, 224, 1)
print(f"y_train.shape: {y_train.shape}, y_test.shape: {y_test.shape}")
# y_train.shape: (1600, 2), y_test.shape: (400, 2)


#2-1. define cnn network model
def build_model(drop=0.1, optimizer='adam', learning_rate=0.1, activation='relu', kernel_size=(2,2)):
    inputs = tf.keras.Input(shape=(224,224,1), name='inputs')
    x = tf.keras.layers.Conv2D(32, kernel_size=kernel_size, padding="same", activation=activation)(inputs)
    x = tf.keras.layers.Dropout(drop)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=kernel_size, padding="same", activation=activation)(x)
    x = tf.keras.layers.Dropout(drop)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=kernel_size, padding="same", activation=activation)(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=kernel_size, padding="same", activation=activation)(x)
    x = tf.keras.layers.Dropout(drop)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(16)(x)
    outputs = tf.keras.layers.Dense(2, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    opt = optimizer(learning_rate=learning_rate)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    return model

#2-2. wrap keras model (for RandomizedSearchCV)
model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=build_model, verbose=1)

#2-3. set params to RandomizedSearchCV
def create_hyperparameters():
    
    batches = [8, 16, 24, 32, 40]
    optimizers = [RMSprop, Adam, Adadelta, SGD, Adagrad, Nadam]
    learning_rate = [1e-5, 1e-4, 1e-3, 1e-2]
    dropout = [0.1, 0.2, 0.3]
    activation = ['tanh', 'relu', 'elu', "selu", "softmax", "sigmoid", LeakyReLU()]
    kernel_size = [2, 3, 4]
    epochs = [100, 300, 500]
    pool_size = [(2,2),(3,3)]

    # return: {key, value}
    return{"batch_size": batches, "optimizer":optimizers,
           "learning_rate": learning_rate, "drop": dropout, 
           "activation": activation, "kernel_size": kernel_size, "epochs": epochs}

hyperparams = create_hyperparameters()

#2-4. set RandomizedSearchCV
search = RandomizedSearchCV(model, hyperparams, cv=3)

# time check
start = time.time()

#3. fit
search.fit(x_train, y_train)

#4. best_params_
print(f"best_params of cnn network:\n {search.best_params_}")
# best_params of cnn network:
# {'optimizer': <class 'keras.optimizers.Nadam'>, 'learning_rate': 1e-05, 'kernel_size': 3, 
#  'epochs': 300, 'drop': 0.3, 'batch_size': 32, 'activation': 'relu'}

#5. acc
acc = search.score(x_test, y_test)
print(f"acc: {acc}")
# acc: 0.8262500166893005


# time check
sec = time.time() - start
times = str(datetime.timedelta(seconds=sec)).split(".")
times = times[0]

print(f"times: {times}")
times_list.append(times)

