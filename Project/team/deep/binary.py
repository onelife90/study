import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# modeling
inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu')(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(2, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs = outputs)

model.summary()


# compile & fit
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# parameters
classes = ['normal', 'fighting']
input_size = (224, 224, 3)
n_classes = 2
batch_size = 32
epochs = 10

steps_per_epoch_train = int(2000 / batch_size)

datagen = ImageDataGenerator(
    horizontal_flip=False,
    validation_split=0.2
)

dataset_path = 'D:project/data/flow_from'

train_gen = datagen.flow_from_directory(
    directory=dataset_path,
    batch_size=batch_size,
    target_size=input_size[:-1],
    shuffle=True,
    subset='training'
)

val_gen = datagen.flow_from_directory(
    directory=dataset_path,
    batch_size=batch_size,
    target_size=input_size[:-1],
    shuffle=True,
    subset='validation'
)

model.fit_generator(generator=train_gen, steps_per_epoch=steps_per_epoch_train,
                    epochs=epochs, validation_data=val_gen)

    