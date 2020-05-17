#train_test_split
# 지금껏 데이터를 손으로 잘라서 사용했으나 train set과 test set을 손쉽게 분리 가능한 train_test_split를 사용
# 하지만, 과적합의 우려가 있기 때문에 validation set을 추가하여 방지
# 왜쓰나요? ex) 차를 왜 타죠? 걸어가도 목적지에 가는데. 더 빠르고 효율적이기에 사용
# 훈련이 처음부터 안되면 초(1-1) -> 초(1-2)로 단계 점프가 안된다. 따라서 전체 데이터 셋을 각 비율로 나눠서 훈련

# from sklearn..model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split( x,y, random_state=66, test_size=0.4, shuffle=False)
# shuffle 파라미터의 디폴트 값은 True. 데이터를 섞는게 효율적
# 아직 validation 데이터가 없으므로 train_test_split를 한번 더 적용
# x_val, x_test, y_val, y_test = train_tset_split(x_test, y_test, random_state=66, test_size=0.5)
# val_size는 test 데이터 셋의 50%를 배분

# 함수형 모델
# 케라스 딥러닝에서는 순차적 모델과 함수형 모델로 구성하는 방식
# 앞으로 모델이 길어지고 앙상블 등 여러가지 기법을 사용하고자 하면 함수형 모델은 필수
# 입력되는 컬럼(열)이 2개 이상인 경우
# x = np.array([range(100), range(301,401)]) 100개 짜리 2덩어리 == (100,2)
# 모델에 입력하기 위해서는 행과 열이 맞아야함. DNN 구조에서는 dimention이 가장 중요하여 행무시 열우선!
# 따라서 행과 열을 바는 transpose()라는 함수를 써서 바꿔줘야함
# x = np.transpose(x) 변수 초기화 필수! 실행하면, (100,2)
# 2열이 되었고 input_dim=2로 데이터 구조가 변경
# 함수형 모델에서 가장 많이 쓰는 구조는 input 여러개, output 1개
