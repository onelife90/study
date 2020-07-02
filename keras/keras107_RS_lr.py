# 100번 복붙해서 lr, optimizer를 넣고 튜닝하시오
# lr 넣기 : 숫자라 np.linspace를 쓰고 리스트로 바꿔줘야 하므로 .tolist()를 붙여줘야함
# optimizer 넣기 : import 받아서 랜덤서치의 파라미터에 리스트 형태로 넣어줘야함
# LSTM -> Dense로 바꿀 것

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, LSTM
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

#1. 데이터
(x_train, y_train), (x_test,y_test) = mnist.load_data()
# print(x_train.shape)    # (60000, 28, 28)
# print(x_test.shape)     # (10000, 28, 28)

#1-1. 데이터 전처리
x_train = x_train.reshape(-1,x_train.shape[1]*x_train.shape[2])/255
x_test = x_test.reshape(-1,x_test.shape[1]*x_test.shape[2])/255
# /255 == MinmaxScaler와 같은 기능

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# print(y_train.shape)    # (60000, 10)
# print(y_test.shape)     # (10000, 10)

from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Nadam

#2-1. 모델 구성(함수)
# 랜덤서치를 사용할 것인데, 사이킷런에 있는 것이기 때문에 모델을 만들어야 함
# 랜덤서치의 첫번재 매개 변수가 model이기 때문에 모델 자체를 함수로 만들겠다
def build_model(drop=0.1, optimizer='Adam', learning_rate=0.1):
    inputs = Input(shape=(28*28, ), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='output')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['acc'])
    return model
    # 모델 만든 함수에서는 compile까지만, fit은 그리드서치에서 실행(cv가 있기 때문에)
    # 사이킷런에 쓸수 있게 KerasClassifier를 wrapping 한 것(모델, 파라미터, cv을 wrap해서 사용하겠다)

# 랜덤서치에 들어갈 첫번째 모델 구성
from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=build_model, verbose=1)
# build_fn: 호출가능한 함수 혹은 클레스 인스턴스

#2-2. 파라미터 구성(함수) 
# 랜덤서치의 두번째 매개변수인 parameter도 함수로 정의
def create_hyperparameters():
    batches = [100,200,300,400,500]
    optimizers = [RMSprop, Adam, Adadelta, SGD, Adagrad, Nadam]
    learning_rate = np.linspace(0.1,1.0,10).tolist()
    dropout = np.linspace(0.1,0.5,5).tolist()
    return{"batch_size": batches, "optimizer":optimizers, "learning_rate": learning_rate, "drop": dropout}
    # 하이퍼파라미터에 대한 매개변수들이 key, values라 딕셔너리형으로 return
hyperparameters = create_hyperparameters()

#2-3. 랜덤서치 만들기
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

search = RandomizedSearchCV(model, hyperparameters, cv=3) #n_jobs=-1)

search.fit(x_train, y_train)
print(search.best_params_)
# {'optimizer': <class 'keras.optimizers.Adadelta'>, 'learning_rate': 0.9, 'drop': 0.1, 'batch_size': 200}
acc = search.score(x_test, y_test)
print("acc: ", acc)
# acc:  0.9620000123977661
