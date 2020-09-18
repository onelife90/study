# iris 케라스 파이프라인 구성
# 당연히 RandomizedSearchCV 구성
# 압축 : '사이킷런 배경인 iris 데이터에 케라스 모델을 쓰고 싶으니 wrapping하고 파이프라인으로 랜덤서치를 쓰겠다'
# 순서
#1) pipeline 사용할 것이니 x 데이터 전처리 생략. 대신, iris는 다중분류이므로 y만 one-hot 인코딩
#2) 사이킷런에서 사용할 def 모델 구성 / model = KerasClassifier
#3) 파이프라인 구성 / pipe=Pipeline([전처리(), 모델()])
#4) 사이킷런에서 사용할 def 파라미터 구성
#5) RandomizedSearch 구성 / search=RandomizedSearchCV(pipe(model 대신 들어가는 것), 하이퍼파라미터, cv)
#6) 훈련 search.fit
#8) search 모델 최적의 매개변수 search.best_params_
#9) search.score

import numpy as np
from sklearn.datasets import load_iris
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import RandomizedSearchCV

#1-1. 데이터 불러오기
iris = load_iris()
x = iris.data
y = iris.target

#1-2. one-hot 인코딩
y = np_utils.to_categorical(y)
# print(y.shape)  # (150, 3)

#1-2. train/test 분리
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=99, test_size=0.2)
# print(x_train.shape)    # (120, 4)
# print(x_test.shape)     # (30, 4)
# print(y_train.shape)    # (120, 3)
# print(y_test.shape)     # (30, 3)

#2-1. 모델 구성
def build_model(drop=0.1, optimizer='adam'):
    inputs =Input(shape=(4, ), name='inputs')
    x = Dense(512, activation='relu', name='hid1')(inputs)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu', name='hid2')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu', name='hid3')(x)
    outputs = Dense(3, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    return model
model = KerasClassifier(build_fn=build_model, verbose=1)

# 파이프라인 구성
pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
# pipe = make_pipeline(StandardScaler(), model())

#2-2. 파라미터 구성
def create_hyperparameters():
    batches = [128,256,512]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1,0.2,0.3,0.4,0.5]
    return{"model__batch_size":batches, "model__optimizer":optimizers, "model__drop":dropout}
hyperparameters = create_hyperparameters()

#2-3. 랜덤서치 만들기
search = RandomizedSearchCV(pipe, hyperparameters, cv=3)
search.fit(x_train, y_train)

print("현재 search 모델의 매개변수: ")
print(search.best_params_)
# {'model__optimizer': 'rmsprop', 'model__drop': 0.2, 'model__batch_size': 128}
acc = search.score(x_test, y_test)
print("acc: ", acc)
# acc:  0.699999988079071