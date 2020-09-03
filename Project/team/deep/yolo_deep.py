from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Flatten, Dropout, Dense, Input
from keras.preprocessing.image import ImageDataGenerator

 # callback 정의
def callbacks(model_path, patience):
    callbacks = [
        ReduceLROnPlateau(patience=3),  
        # Plateau==안정 수준에 달하다
        # 경사하강법에 의해 학습 하는 경우
        # 이차함수 그래프를 생각해보면 볼록한 공간이 2개 이상 나오는데 처음 만난 볼록한 구간에서 가중치를 계속 계산하게 되는 현상 발생
        # Local Minima에 빠져버린 경우, 쉽게 빠져나오지 못하고 갇혀버리게 됨
        # 이 때 learning rate를 늘리거나 줄여주는 방법으로 빠져나오는 효과 기대 가능

        ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True),
        EarlyStopping(patience=patience)
    ]
    return callbacks


# finetuning case 1 (Conv는 Frozen, Fc layer는 train)
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


classes = ['normal', 'fighting']
# input_size = (3840,2160,3)
input_size = (416,416,3) # 훈련에 사용할 이미지 크기 설정
n_classes = 2
batch_size = 32
epochs_finetune = 1 # fine tuning 시 사용할 epoch
epochs_fulltune = 10 # full tuning 시 사용할 epoch

## 모델링 ##
model = frozen_resnet(input_size, n_classes)
# model = frozen_resnet((416, 416, 3), 2)
# 416, 416, 3 : input 
# n_classes : output (normal, fight 2개)


model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(),
    metrics=['acc']
)

steps_per_epoch_train = int((1620) / batch_size)
# model_path_finetune = 'model_finetuned.h5'
model_path_finetune = './model_finetuned.h5'

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    validation_split= 0.2)

    # preprocessing_function: 각 인풋(416,416,3)에 적용되는 함수. resnet50_v2의 default input
    # 이미지가 크기 재조절되고 증강된 후에 함수가 작동합니다. 
    # horizontal_flip: 불리언. 인풋을 50% 확률로 무작위 수평 뒤집기. 원본 이미지에 수평 비대칭성이 없을 때 효과적. 즉, 뒤집어도 자연스러울 때 사용
    # zoom_range: 부동소수점 혹은 [하한, 상산]. 무작위 줌의 범위
    # width_shift_range, height_shift_range = 부동소수점: < 1인 경우 전체 가로넓이에서의 비율, >= 1인 경우 픽셀의 개수
    # 전체 가로 넓이=1라 가정하면, -0.3~0.3의 구간으로 너비가 이동해서 전처리됨



# dataset_path = 'D:/python_module/darknet-master/build/darknet/x64/project/flow_from'
dataset_path = 'D:/deepstudy/project/cnn/data/train'

train_gen = datagen.flow_from_directory(
    directory=dataset_path,
    batch_size=batch_size,
    target_size=input_size[:-1],
    class_mode = 'binary',
    shuffle=True,
    subset='training')

val_gen = datagen.flow_from_directory(
    directory=dataset_path,
    batch_size=batch_size,
    target_size=input_size[:-1],
    class_mode = 'binary',
    shuffle=True,
    subset='validation')

model.fit_generator(generator=train_gen,
                    steps_per_epoch=steps_per_epoch_train,
                    epochs=epochs_finetune,
                    callbacks=callbacks(
                        model_path=model_path_finetune,
                        patience=5),
                    validation_data=val_gen
                    )

model.load_weights(model_path_finetune)

for layer in model.layers:
    layer.trainable = True

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(),
    metrics=['acc']
)

model_path_full = 'model_full.h5'
model.fit_generator(generator=train_gen,
                    steps_per_epoch=steps_per_epoch_train,
                    epochs=epochs_fulltune,
                    callbacks=callbacks(
                        model_path=model_path_full,
                        patience=10),
                    validation_data=val_gen
                    )

model.load_weights(model_path_full)




# fine tuning #
# resnet 모델 끌고 온 건 frozen, 묶어두고
# 아래에 직접 추가해준 레이어만 훈련시킨다(FC layer만)
# trainable = False 하면 해당 레이어는 훈련을 하지 않음
# frozen, 묶어둔다는 의미는 이미 사전훈련된 가중치를 더 건드리지 않는 것

# resnet : 이미 imagenet이라는 데이터셋으로 사전훈련된 가중치를 제공
# resnet의 가중치를 frozen 상태에서 fit을 한다 == 이 가중치들은 움직이지 않고
# 아래 추가해준 layer ex.flatten, dense 등만 바뀐다
# trainable false한 레이어들은 그 가중치에서 고정,
# true 이면 fit 할 때 epoch 마다 그 레이어들 계속 변동

# false == 기존의 가중치가 변하지 않음을 의미

# fine tuning -> full tuning
# 훈련을 두 번 해 준다

# 사전 훈련된 가중치는 frozen 되어 있는 레이어에 이미 있고
# 마지막에 추가한 layer(FC layer들)는 훈련 때마다 계속 가중치들이 변동 된다

# 사전 훈련된 가중치는 얼려서 업데이트를 하지 않고
# 추가해준 레이어들은 업데이트를 하는 방식

# fine tuning : 사전 훈련된 가중치를 얼려서 훈련
# full tuning : 사전 훈련 가중치 얼리지 않고 통째로 다 훈련시킴

# 위의 코드는 fine tuning fit 한 후, full tuning fit 한 번 더 해 주는 코드
# fit을 2번 해 준 것이다