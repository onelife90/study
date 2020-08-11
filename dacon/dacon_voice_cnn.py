import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import tensorflow as tf
from tqdm import tqdm
from glob import glob
from scipy.io import wavfile
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping

# wav파일로부터 데이터를 불러오는 함수, 파일 경로를 리스트 형태로 입력
def data_loader(files):
    out=list()
    # tqdm : 작업 진행표시바
    for file in tqdm(files):
        fs, data = wavfile.read(file)
        out.append(data)
    out = np.array(out)
    return out

# wav 파일로부터 feature를 생성
x_data = glob('./train-002/*.wav')
x_data = data_loader(x_data)
# print(f'x_data: {x_data.shape}')    # x_data: (100000, 16000)

# 매 8번째 데이터만 사용
x_data = x_data[:, ::8]
# 최대값 30000으로 나누어 정규화
x_data = x_data/30000
# cnn 모델 3차원 데이터 reshape
x_data = x_data.reshape(-1,x_data.shape[1],1)

# 정답값 불러오기
y_data = pd.read_csv('./train_answer.csv', index_col=0)
y_data = y_data.values

# feature, label shape 확인
print(f'x_data.shape: {x_data.shape}, y_data.shape: {y_data.shape}')
# x_data.shape: (100000, 2000, 1), y_data.shape: (100000, 30)

# 모델을 만듭니다.
model = Sequential()
model.add(Conv1D(16, 32, activation='relu', input_shape=(x_data.shape[1], x_data.shape[2])))
model.add(MaxPooling1D())
model.add(Conv1D(16, 32, activation='relu'))
model.add(MaxPooling1D())
model.add(Conv1D(16, 32, activation='relu'))
model.add(MaxPooling1D())
model.add(Conv1D(16, 32, activation='relu'))
model.add(MaxPooling1D())
model.add(Conv1D(16, 32, activation='relu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(30, activation='softmax'))

# 컴파일
model.compile(loss=tf.keras.losses.KLDivergence(), optimizer='adam')

# 모델 폴더를 생성합니다.
model_path = 'D:/Study/model/voice'
if not os.path.exists(model_path):
  os.mkdir(model_path)

# Validation 점수가 가장 좋은 모델만 저장합니다.
model_file_path = model_path + 'Epoch_{epoch:03d}_Val_{val_loss:.3f}.hdf5'
checkpoint = ModelCheckpoint(filepath=model_file_path, monitor='val_loss', verbose=1, save_best_only=True)

# 5회 간 Validation 점수가 좋아지지 않으면 중지합니다.
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# 모델을 학습시킵니다.
history = model.fit(
    x_data, y_data, 
    epochs=100, batch_size=256, validation_split=0.2, shuffle=True, callbacks=[checkpoint, early_stopping])
