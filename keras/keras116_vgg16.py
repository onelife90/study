# vgg16 : 레이어가 16개
# keras.applications은 사전 훈련된 여러 네트워크를 제공

from keras.applications import VGG16, VGG19, Xception, ResNet101, ResNet101V2, ResNet152, ResNet152V2, ResNet50, ResNet50V2
from keras.applications import InceptionV3, InceptionResNetV2, MobileNet, MobileNetV2, DenseNet121, DenseNet169, DenseNet169, DenseNet201
from keras.applications import NASNetLarge, NASNetMobile
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Activation
from keras.optimizers import Adam

#2. 모델 구성
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3)) #(None,224,224,3)
# vgg16.summary()
'''
applications = [VGG19, Xception, ResNet101, ResNet101V2, ResNet152,ResNet152V2, ResNet50, 
                ResNet50V2, InceptionV3, InceptionResNetV2,MobileNet, MobileNetV2, 
                DenseNet121, DenseNet169, DenseNet201, NASNetLarge, NASNetMobile]

for app in applications:
    take_model = app()
'''
# 모델 구성에 파라미터를 찾아서 엮으시오
model = Sequential()

model.add(vgg16)
model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10, activation='softmax'))

model.summary()
