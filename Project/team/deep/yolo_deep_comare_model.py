from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import ResNet50V2, ResNet152V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input 
from efficientnet.tfkeras import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5
from efficientnet.tfkeras import EfficientNetB6, EfficientNetB7, EfficientNetL2, preprocess_input as preprocess_eff
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten, Dropout, Dense, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os, shutil
import time
import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.backend import clear_session
from tensorflow.keras.utils import multi_gpu_model
import tensorflow as tf
import cv2
tf.compat.v1.disable_eager_execution()

## callback
def callbacks(model_path, patience):
    callbacks = [ReduceLROnPlateau(patience=patience),  # TODO Change to cyclic LR
                 ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True)]
                 # EarlyStopping(patience=patience)  # EarlyStopping needs to be placed last, due to a bug fixed in tf2.2
    return callbacks

# resnet부분이 얼려져있는 모델 리턴
def frozen_models(input_size, n_classes):
    models = [ResNet50V2, ResNet152V2, EfficientNetB0, EfficientNetB1, EfficientNetB2,\
              EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7]
    models_name = ['ResNet50V2', 'ResNet152V2','EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', \
                   'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7']

    for model_name, model in zip(models_name, models):
        model_ = model(include_top=False, input_shape=input_size)
        for layer in model_.layers:
            layer.trainable = False
        x = Flatten()(model_.output)
        x = Dropout(0.5)(x)
        x = Dense(n_classes, activation='softmax')(x)
        frozen_model = Model(model_.input, x)
        yield (model_name, frozen_model)

classes = ['normal', 'fighting']
times_list, acc_list = [], []
# input_size = (3840,2160,3)
input_size = (224,224,3)
n_classes = 2
batch_size = 32
epochs_finetune = 2
epochs_fulltune = 2


## 모델링 ##
for model_name, model in frozen_models(input_size, n_classes):

    model.compile(loss='binary_crossentropy', optimizer=Adam(),metrics=['acc'])

    dataset_path = 'D:/python_module/darknet-master/build/darknet/x64/project/flow_from'

    total_image_num = len(os.listdir(dataset_path + '/' + classes[0])) + len(os.listdir(dataset_path + '/' + classes[1])) ## 3905
    steps_per_epoch_train = int(total_image_num / batch_size)
    model_path_finetune = 'model_finetuned.h5'

    def preprocessing(image):
        if model_name[0] == 'R':
            image = preprocess_input(image)
        else:
            image = preprocess_eff(image)
        return image

    ## 이미지 제네레이터
    ## preprocess_input : resnet에서 제공해주는 전처리함수
    datagen = ImageDataGenerator(preprocessing_function=preprocessing,
                                 zoom_range=0.1,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 validation_split= 0.2)

    train_gen = datagen.flow_from_directory(directory=dataset_path,
                                            shuffle=True,
                                            batch_size=batch_size,
                                            target_size=input_size[:-1],
                                            classes=classes,
                                            subset='training')

    val_gen = datagen.flow_from_directory(directory=dataset_path,
                                          batch_size=batch_size,
                                          target_size=input_size[:-1],
                                          classes=classes,
                                          shuffle=True,
                                          subset='validation')

    start = time.time()

    ## 얼려져있는 모델 fit
    model.fit_generator(generator=train_gen,
                        steps_per_epoch=steps_per_epoch_train,
                        epochs=epochs_finetune,
                        callbacks=callbacks(model_path=model_path_finetune, patience=5),
                        validation_data=val_gen)


    ## 저장된 최적의 모델 가져옴
    model.load_weights(model_path_finetune)

    ## 얼려져있던 레이어 해동
    for layer in model.layers:
        layer.trainable = True

    model.compile(loss='binary_crossentropy', optimizer=Adam(),metrics=['acc'])
    
    ## 해동한 모델 다시 fit
    model_path_full = 'model_full.h5'
    hist = model.fit_generator(generator=train_gen,
                               steps_per_epoch=steps_per_epoch_train,
                               epochs=epochs_fulltune,
                               callbacks=callbacks(model_path=model_path_full, patience=20),
                               validation_data=val_gen)

    model.load_weights(model_path_full)


    # 소요 시간
    sec = time.time() - start
    times = str(datetime.timedelta(seconds=sec)).split(".")
    times = times[0]

    print(times)
    times_list.append(times)


    # 시각화
    plt.figure(figsize = (15, 13))


    plt.subplot(2, 1, 1)
    plt.plot(hist.history['loss'], marker = '.', c = 'blue', label = 'loss')         
    plt.plot(hist.history['val_loss'], marker = '.', c = 'red', label = 'val_loss')   
    plt.grid() 
    plt.title('loss')      
    plt.ylabel('loss')      
    plt.xlabel('epoch')          
    plt.legend(loc = 'upper right')

    plt.subplot(2, 1, 2) 
    plt.plot(hist.history['acc'], marker = '.', c = 'blue', label = 'acc')
    plt.plot(hist.history['val_acc'], marker = '.', c = 'red', label = 'val_acc')
    plt.grid() 
    plt.title('acc')      
    plt.ylabel('acc')      
    plt.xlabel('epoch')          
    plt.legend(['acc', 'val_acc'])

    plt.savefig('fig_' + model_name + '.png')
    clear_session()
    # plt.show()  


    # test_path = 'D:/python_module/darknet-master/build/darknet/x64/project/testing'
    # testdata_gen = ImageDataGenerator(
    #     preprocessing_function=preprocessing
    # )

    # test_gen = testdata_gen.flow_from_directory(
    #     directory=test_path,
    #     batch_size=1,
    #     target_size=input_size[:-1],
    #     classes=classes)
    # acc = model.evaluate_generator(test_gen)
    # print(acc)
    # acc_list.append(acc)

print(times_list)
print(acc_list)

# # 1-10
# # Epoch 10/10
# # 122/122 [==============================] - 633s 5s/step - loss: 2.6466 - acc: 0.6279 - val_loss: 0.6671 - val_acc: 0.6504
# # 2:03:10

# # 10-100
# # Epoch 19/100
# # 122/122 [==============================] - 635s 5s/step - loss: 2.0300 - acc: 0.6191 - val_loss: 0.7091 - val_acc: 0.6280
# # Epoch 00019: val_loss did not improve from 0.55209
# # 5:14:35

# # 100-1000

# # Epoch 00048: val_loss did not improve from 0.45523
# # Epoch 49/1000
# # 122/122 [==============================] - 627s 5s/step - loss: 2.6062 - acc: 0.6435 - val_loss: 0.6719 - val_acc: 0.6447
# # 11:10:03

# # Found 3124 images belonging to 2 classes.
# # Found 781 images belonging to 2 classes.