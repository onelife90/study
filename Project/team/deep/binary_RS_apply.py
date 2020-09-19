import os
import glob
import cv2
import tensorflow as tf
import numpy as np
from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Nadam
from keras.layers import LeakyReLU
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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
inputs = tf.keras.Input(shape=(224,224,1), name='inputs')
x = tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation='relu')(inputs)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)
x = tf.keras.layers.Conv2D(64, kernel_size=2, padding="same", activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)
x = tf.keras.layers.Conv2D(128, kernel_size=3, padding="same", activation='relu')(x)
x = tf.keras.layers.Conv2D(256, kernel_size=3, padding="same", activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(16)(x)
outputs = tf.keras.layers.Dense(2, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

#dot_img_file = '/tmp/model_1.png'
#tf.keras.utils.plot_model(model, to_file="D:/Study/Project/team/deep/model.png", show_shapes=True)

opt = tf.keras.optimizers.Nadam(learning_rate=1e-05)

#3. compile & fit
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])
hist = model.fit(x_train, y_train, batch_size=32, epochs=300, validation_split=0.2)

#4. evaluate
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print(f"loss: {loss}")
print(f"acc: {acc}")


#4. visualization
loss = hist.history['loss']
acc = hist.history['acc']
val_loss = hist.history['val_loss']
val_acc = hist.history['val_acc']

plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(loss, marker='.', c='red', label='loss')
plt.plot(val_loss, marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()

plt.subplot(2,1,2)
plt.plot(acc, marker='.', c='red', label='acc')
plt.plot(val_acc, marker='.', c='blue', label='val_acc')
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend()
plt.show()

# best_params of cnn network:
# {'optimizer': <class 'keras.optimizers.Nadam'>, 'learning_rate': 1e-05, 'kernel_size': 3, 
#  'epochs': 300, 'drop': 0.3, 'batch_size': 32, 'activation': 'relu'}