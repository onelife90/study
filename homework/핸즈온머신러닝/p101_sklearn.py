# 사이킷런의 설계 철학

#1. 추정기 estimator : 데이터셋을 기반으로 일련의 모델 파라미터들을 추정하는 객체. 추정 자체는 fit() 메서드에 의해 수행
#2. 변환기 transfomer : fit_transform
#3. 예측기 predictor : score() 메서드
# 검사가능 : 모든 추정기의 학습된 모델 파라미터도 접미사로 언더바를 붙여서 공개 인스턴스 변수로 제공 all_estimateors_
# 조합성 : 기존의 구성요소를 최대한 사용. pipeline
# 기본값 : 대부분의 매개변수에 디폴트 값
