from keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential, load_model
from keras.utils.np_utils import to_categorical

model = Sequential()
model.add(Conv2D(32, (2,2), strides=(1,1), padding='same', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
model.add(Conv2D(32, (2,2), strides=(1,1), padding='same'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('sigmoid'))
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()
