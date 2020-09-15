import os
import tensorflow as tf
import numpy as np
import glob
from natsort import natsorted
from PIL import Image
from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Nadam
from keras.layers import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import RandomizedSearchCV

#1. image down size
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


#1-1. image dataset
groups_folder_path = "D:/project/data/binary"
categories = ["fight", "normal"]  

x = []
y = []

for index, categorie in enumerate(categories):
    label = [0 for i in range(2)]
    # label[index] = 1

    # image file path
    image_dir = groups_folder_path + categorie + '/'
    files = glob.glob(image_dir +  "*.jpg")

    for img, filename in enumerate(files):
        img = Image.open(filename)
        img = img.convert("RGB")
        data = np.array(img)
        # data = np.asarray(img)
        x.append(data)
        y.append(label)

# x = np.array(x)
# y = np.array(y)

#1-2. save to numpy
np.save('D:/project/data/x_data.npy', x)
np.save('D:/project/data/y_data.npy', y)

x = np.load('D:/project/data/x_data.npy')
y = np.load('D:/project/data/x_data.npy')

#1-3. train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size = 0.2, 
                                                    random_state = 42)#,
                                                    # validation_split=0.2)

print(f"x_train.shape: {x_train.shape}, x_test.shape: {x_test.shape}")  
print(f"y_train.shape: {y_train.shape}, y_test.shape: {y_test.shape}")

#2. cnn network modeling
def build_model(drop=0.1, optimizer='adam', learning_rate=0.1, activation='relu'):
    inputs = tf.keras.Input(shape=(224,224,3), name='inputs')
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3,3), padding='same')(x)
    x = tf.keras.layers.Dropout(drop)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=(3,3), padding='same')(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=(3,3),padding='same')(x)
    x = tf.keras.layers.Dropout(0.1)
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
search = RandomizedSearchCV(pipe, hyperparameters, cv=3)
search.fit(x_train, y_train)

print(f"best_params of cnn network:\n {search.best_params_}")
# {'model__optimizer': 'rmsprop', 'model__drop': 0.2, 'model__batch_size': 128}
acc = search.score(x_test, y_test)
print(f"acc: {acc}")
# acc:  0.699999988079071








