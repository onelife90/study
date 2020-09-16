import os
import glob
import cv2
import tensorflow as tf
import numpy as np
from natsort import natsorted
from PIL import Image
from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Nadam
from keras.layers import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import RandomizedSearchCV


#images = [cv2.imread(file) for file in glob.glob("D:/project/data/binary/*.jpg")]
#print(images)



def load_train(train_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    cls = []

    print('Going to read training images')
    for fields in classes:
        index = classes.index(fields)
        print('Now going to read {} files (Index: {})'.format(fields, index))
        path = os.path.join(train_path, fields, '*g')
        files = glob.glob(path)
        for file in files:
            image = cv2.imread(file,0)
            image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
            #cv2.imwrite('image', image)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows() 
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)
            image1 = image.reshape(224,224,1)
            images.append(image1)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(file)
            img_names.append(flbase)
            cls.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    return images, labels, img_names, cls

train_path = "D:/project/data/flow_from"
image_size = 224
classes = ["fighting", "normal"]

result = load_train(train_path, image_size, classes)

print(result)

x = np.array(result[0])
y = np.array(result[1])

#1-2. save to numpy
np.save('D:/project/data/x_data.npy', x)
np.save('D:/project/data/y_data.npy', y)

x = np.load('D:/project/data/x_data.npy')
y = np.load('D:/project/data/y_data.npy')

#1-3. train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

print(f"x_train.shape: {x_train.shape}, x_test.shape: {x_test.shape}") 
# x_train.shape: (1600, 224, 224, 1), x_test.shape: (400, 224, 224, 1)
print(f"y_train.shape: {y_train.shape}, y_test.shape: {y_test.shape}")
# y_train.shape: (1600, 2), y_test.shape: (400, 2)

#2. cnn network modeling
def build_model(drop=0.1, optimizer='adam', learning_rate=0.1, activation='relu'):
    inputs = tf.keras.Input(shape=(224,224,1), name='inputs')
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3,3), padding='same')(x)
    x = tf.keras.layers.Dropout(drop)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=(3,3), padding='same')(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=(3,3),padding='same')(x)
    x = tf.keras.layers.Dropout(drop)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(16, padding='same')(x)
    outputs = tf.keras.layers.Dense(2, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    opt = optimizer(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])
    return model
model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=build_model, verbose=1)
                    
#2-1. set pipeline
pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])

#2-2. set params
def create_hyperparameters():
    
    batches = [16,32,64,128,256]
    optimizers = [RMSprop, Adam, Adadelta, SGD, Adagrad, Nadam]
    learning_rate = np.linspace(0.1,1.0,10).tolist()
    dropout = np.linspace(0.1,0.5,5).tolist()
    activation = ['tanh', 'relu', 'elu', "selu", "softmax", "sigmoid", LeakyReLU()]

    # return: {key, value}
    return{"batch_size": batches, "optimizer":optimizers,
           "learning_rate": learning_rate, "drop": dropout, 
           "activation": activation}

hyperparameters = create_hyperparameters()

#2-3. set RandomizedSearchCV
search = RandomizedSearchCV(pipe, hyperparameters, cv=3, n_jobs=-1)
search.fit(x_train, y_train)
# Check the list of available parameters with `estimator.get_params().keys()`.
# error solving..

print(f"best_params of cnn network:\n {search.best_params_}")
# {'model__optimizer': 'rmsprop', 'model__drop': 0.2, 'model__batch_size': 128}
acc = search.score(x_test, y_test)
print(f"acc: {acc}")
# acc:  0.699999988079071






#'''