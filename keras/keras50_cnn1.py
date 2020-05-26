# CNN convolutional 복합적인 : 이미지 분석에 최적화
# 이미지를 잘라서 이어 붙이면, 연속적인 데이터
# 이미지를 자르는 게 CNN 데이터의 기초!

from keras.models import Sequential
from keras.layers import Conv2D
# model.summary를 보면 3차원으로 레이어가 구성되는데 왜 Conv2D일까?
# 가로, 세로만 쓰겠다는 의미

#2. 모델 구성
model = Sequential()
model.add(Conv2D(10, (2,2), padding='valid', input_shape=(10,10,1)))
# 10=Conv_filter (2,2)=Conv_kernel_size or kernel_size=2
# 공식 용어 input_shape=(height, width, channel)
# input_shape=3차원 그림(가로,세로,명암) 명암=1(흑백),3(컬러)
# (10000,10,10,1)==가로,세로 10짜리 흑백 이미지 10000장. 행무시로 input_shape=(10,10,1)
# 머신은 한번 자르는 것가지고는 이해를 잘 못함. 또 잘라줘야 함
model.add(Conv2D(7, (2,2)))
model.add(Conv2D(5, (2,2)))

model.summary()
# 단순히 레이어 마다 자른 것만 아니고 증폭까지 되어있다. params?
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 9, 9, 10)          50
_________________________________filter=10으로 온다_______________
conv2d_2 (Conv2D)            (None, 8, 8, 7)           287
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 7, 7, 5)           145
=================================================================
Total params: 482
Trainable params: 482
Non-trainable params: 0
_________________________________________________________________

[output_shape 계산]
1. input_shape_height - kernel_h + 1 = next_layer_height
2. input_shape_width - kernel_w + 1 = next_layer_wdith
3. Conv_filter = next_layer_channel
'''

# padding을 입는 이유?
# kernel_size로 자르면 특성 추출에서 상대적으로 데이터 손실의 우려가 있다
# ex) 왼쪽 귀만 나온 사진 -> 훈련을 1번밖에 못한 꼴
# padding='same'을 입력하면 input_shape의 (height, width)가 동일해져서 다음 레이어로 전달
# 그 padding을 오른쪽 상단에 입힐지, 왼쪽 하단에 입힐지는 머신이 판단
# padding_default='valid'

# maxpooling 쓰레기 값을 버리고 이미지 특성에 잘 맞는 부분만 추출

# 과제3 Conv2D의 parameter 계산법
# number_parameters = out_channels(=pre_filter) * (in_channels * kernel_h * kernel_w + 1)  # 1 for bias
