# 머신러닝 기법을 적용하여 케라스 모델 완성
# wine.csv 마지막 컬럼 y. 슬라이싱
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Model, Input
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score

#1-1. 데이터 읽어오기
wine = pd.read_csv('./data/csv/winequality-white.csv', index_col=None, header=0, sep=';', encoding='CP949')
# print(wine)   # [4898 rows x 12 columns]
# wine.info()
 #   Column                Non-Null Count  Dtype
# ---  ------                --------------  -----
#  0   fixed acidity         4898 non-null   float64
#  1   volatile acidity      4898 non-null   float64
#  2   citric acid           4898 non-null   float64
#  3   residual sugar        4898 non-null   float64
#  4   chlorides             4898 non-null   float64
#  5   free sulfur dioxide   4898 non-null   float64
#  6   total sulfur dioxide  4898 non-null   float64
#  7   density               4898 non-null   float64
#  8   pH                    4898 non-null   float64
#  9   sulphates             4898 non-null   float64
#  10  alcohol               4898 non-null   float64
#  11  quality               4898 non-null   int64
# dtypes: float64(11), int64(1)

#1-2. numpy 저장
wine = wine.values
# print(type(wine))   # <class 'numpy.ndarray'>
np.save('./data/winequality-white.npy', arr=wine)

#2-1. 데이터 불러오기
wine = np.load('./data/winequality-white.npy', allow_pickle=True)
# print(wine.shape)   # (4898, 12)

#2-2. 데이터 x,y 자르기
x = wine[:, 0:11]
y = wine[:, 11]
# print(x)
# print(x.shape)  # (4898, 11)
# print(y)
# print(y.shape)  # (4898, )

#2-3. 데이터 전처리
pca = PCA(n_components=10)
x = pca.fit_transform(x)
# print(x.shape)  # (4898, 10)

scaler = MinMaxScaler()
x = scaler.fit_transform(x)
# print(x)

y = np_utils.to_categorical(y)
# print(y.shape)  # (4898, 10)

#2-4. x,y reshape
x = x.reshape(-1,1,5,2)
# print(x.shape)  # (4898, 1, 2, 5)

#2-5. train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=73, train_size=0.8)

#2-5. 모델 구성
input1 = Input(shape=(1,5,2))
dense1 = Conv2D(10, (2,2), padding='same')(input1)
dense1 = Conv2D(20, (2,2), padding='same')(dense1)
dense1 = Conv2D(40, (2,2), padding='same')(dense1)
dense1 = Conv2D(60, (2,2), padding='same')(dense1)
dense1 = Conv2D(80, (2,2), padding='same')(dense1)
dense1 = Dropout(0.1)(dense1)
dense1 = MaxPooling2D(pool_size=3, padding='same')(dense1)
dense1 = Conv2D(70, (2,1), padding='same')(dense1)
dense1 = Conv2D(50, (2,1), padding='same')(dense1)
dense1 = Conv2D(30, (2,1), padding='same')(dense1)
dense1 = MaxPooling2D(pool_size=2, padding='same')(dense1)
dense1 = Flatten()(dense1)
output1 = Dense(10, activation='softmax')(dense1)

model = Model(inputs=input1, outputs=output1)

#2-5. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
modelpath = './model/wine/{epoch:02d}-{val_loss:.4f}.hdf5'
check_p = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True)
model.fit(x_train, y_train, epochs=100, batch_size=10, validation_split=0.2, callbacks=[check_p])

#2-6. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=10)
print("loss: ", loss)
print("acc: ", acc)
y_pred = model.predict(x_test)

# loss:  1.1247760133475673
# acc:  0.5035714507102966
