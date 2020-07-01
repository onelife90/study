# RandomizedSearchCV
# CNN 모델 구성 4차원

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

#1. 데이터
(x_train, y_train), (x_test,y_test) = mnist.load_data()
# print(x_train.shape)    # (60000, 28, 28)
# print(x_test.shape)     # (10000, 28, 28)

#1-1. 데이터 전처리
x_train = x_train.reshape(-1,28,28,1)/255
x_test = x_test.reshape(-1,28,28,1)/255
# cnn 모델 구성(4차원으로 reshape)
# /255 == MinmaxScaler와 같은 기능

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# print(y_train.shape)    # (60000, 10)
# print(y_test.shape)     # (10000, 10)

#2-1. 모델 구성(함수)
# 그리드서치를 사용할 것인데, 사이킷런에 있는 것이기 때문에 모델을 만들어야 함
# 그리드서치의 첫번재 매개 변수가 model이기 때문에 모델 자체를 함수로 만들겠다
def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(28,28,1), name='input')
    x = Conv2D(512, (2,2), activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Conv2D(256, (2,2), activation='relu', name='hidden2')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Conv2D(128, (2,2), activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax', name='output')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    return model
    # 모델 만든 함수에서는 compile까지만, fit은 그리드서치에서 실행(cv가 있기 때문에)
    # 사이킷런에 쓸수 있게 KerasClassifier를 wrapping 한 것(모델, 파라미터, cv을 wrap해서 사용하겠다)

# 그리드서치에 들어갈 첫번째 모델 구성
from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=build_model, verbose=1)
# build_fn: 호출가능한 함수 혹은 클래스 인스턴스

#2-2. 파라미터 구성(함수) 
# 그리드서치의 두번째 매개변수인 parameter도 함수로 정의
def create_hyperparameters():
    batches = [100,200,300,400,500]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1,0.5,5)
    return{"batch_size": batches, "optimizer":optimizers, "drop": dropout}
    # 하이퍼파라미터에 대한 매개변수들이 key, values라 딕셔너리형으로 return
hyperparameters = create_hyperparameters()

#2-3. 그리드서치 만들기
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model, hyperparameters, cv=3)

search.fit(x_train, y_train)
print(search.best_params_)

acc = search.score(x_test, y_test)
print("acc: ", acc)
