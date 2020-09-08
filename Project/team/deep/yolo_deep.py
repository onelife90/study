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

 #1-1. callback 함수 정의
def callbacks(model_path, patience):
    callbacks = [ReduceLROnPlateau(patience=3), 
                 ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True),
                 EarlyStopping(patience=patience)]
    return callbacks

                # Plateau==안정 수준에 달하다
                # 경사하강법에 의해 학습 하는 경우
                # 이차함수 그래프를 생각해보면 볼록한 공간이 2개 이상 나오는데 처음 만난 볼록한 구간에서 가중치를 계속 계산하게 되는 현상 발생
                # Local Minima에 빠져버린 경우, 쉽게 빠져나오지 못하고 갇혀버리게 됨
                # 이 때 learning rate를 늘리거나 줄여주는 방법으로 빠져나오는 효과 기대 가능


#1-2. finetuning 모델 정의
def frozen_resnet(input_size, n_classes):
    model_ = ResNet50V2(include_top=False, input_tensor=Input(shape=input_size))
    # include_top = False : flatten() layer 전 층까지만 가져다 쓴다
    # include_top = True : flatten() layer 까지 다 가져 와서 쓴다

    for layer in model_.layers:
        layer.trainable = False # frozen 상태로 가중치만 가져온다
    # x = Flatten(input_shape=model_.output_shape[1:])(model_.layers[-1].output)
    x = Flatten()(model_.layers[-1].output)
    x = Dense(n_classes, activation='softmax')(x)
    # output layer, n_classes : 클래스 개수(라벨의 개수)
    frozen_model = Model(model_.input, x)
    # 함수형 모델이므로 마지막에 input과 output을 Model()에 넣어준다

    return frozen_model

# 1-3. 변수 정의
classes = ['normal', 'fighting']
# input_size = (3840,2160,3)
input_size = (416,416,3) # 훈련에 사용할 이미지 크기 설정
n_classes = 2
batch_size = 32
epochs_finetune = 1 # fine tuning 시 사용할 epoch
epochs_fulltune = 10 # full tuning 시 사용할 epoch

#2. finetuning 모델 구성
model = frozen_resnet(input_size, n_classes)

#3. finetuning 컴파일
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['acc'])
# categorical_crossentropy를 사용하는 이유는?
# flow_from_directory를 사욯하면 클래스가 원핫인코딩 되어 나옴

#4-1. ImageDataGenerator 사용
datagen = ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=True,
                             zoom_range=0.3, width_shift_range=0.3, height_shift_range=0.3, validation_split= 0.2)

    # preprocessing_function: 각 인풋(416,416,3)에 적용되는 함수. resnet50_v2의 default input
    # 이미지가 크기 재조절되고 증강된 후에 함수가 작동합니다. 
    # horizontal_flip: 불리언. 인풋을 50% 확률로 무작위 수평 뒤집기. 원본 이미지에 수평 비대칭성이 없을 때 효과적. 즉, 뒤집어도 자연스러울 때 사용
    # zoom_range: 부동소수점 혹은 [하한, 상산]. 무작위 줌의 범위
    # width_shift_range, height_shift_range = 부동소수점: < 1인 경우 전체 가로넓이에서의 비율, >= 1인 경우 픽셀의 개수
    # 전체 가로 넓이=1라 가정하면, -0.3~0.3의 구간으로 너비가 이동해서 전처리됨

#4-2. ImageDataGenerator에 사용할 변수 정의
dataset_path = 'D:/python_module/darknet-master/build/darknet/x64/project/flow_from'
steps_per_epoch_train = int((1620) / batch_size)
# model_path_finetune = 'model_finetuned.h5'
model_path_finetune = './model_finetuned.h5'

#4-3. flow_from_directory(폴더명을 label)로 tarin/val
train_gen = datagen.flow_from_directory(directory=dataset_path, batch_size=batch_size, target_size=input_size[:-1],
                                        class_mode = 'binary', shuffle=True, subset='training')

val_gen = datagen.flow_from_directory(directory=dataset_path, batch_size=batch_size, target_size=input_size[:-1],
                                      class_mode = 'binary', shuffle=True, subset='validation')

#5. finetunuing flow_from_directory를 사용한 증강된 data 생성
model.fit_generator(generator=train_gen, steps_per_epoch=steps_per_epoch_train,epochs=epochs_finetune,
                    callbacks=callbacks(model_path=model_path_finetune, patience=5), validation_data=val_gen)

#6. finetuning 가중치 load
model.load_weights(model_path_finetune)

#7. fulltraining 모델 구성
for layer in model.layers:
    layer.trainable = True

#8. fulltraining 모델 컴파일
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['acc'])

#9. fulltraining 모델 저장
model_path_full = 'model_full.h5'

#10. fulltraining flow_from_directory를 사용한 증강된 data 생성
model.fit_generator(generator=train_gen, steps_per_epoch=steps_per_epoch_train, epochs=epochs_fulltune,
                    callbacks=callbacks(model_path=model_path_full, patience=5), validation_data=val_gen)

#11. fulltraining 가중치 load
model.load_weights(model_path_full)