# yolo_v3_params 설명

#1. 파일
#.cfg(configure file)==모델 구조 및 train 관련 설정이 들어있는 파일
#darknet53.conv.74==가중치 파일

#2. fit관련 params
# mini_batch = batch/subdivisions
# subdivisions = 블록으로 세분화. 블록의 이미지는 GPU에서 병렬로 실행
# GPU가 사용하는 것이 mini_batch
# subdivisions이 큰 값이면 mini_batch가 낮아진다. 즉, GPU가 덜 부담스러워짐

#3. optimizator
# lr, burn_in, max_batches, policy, steps, scales
#ex) steps=4800,5400 / scales=.1,.1
# 4800 배치와 5400 배치 사이에서 lr을 적용 시킨다
# 4800steps: lr*0.1, 5400steps: lr*0.1
# 이것이 가능한 이유는 모멘텀 알고리즘을 사용하여 가중치 변화에 적용하기 때문에
# new_gradient = momentum * previous_gradient + (1- momentum ) * gradient_of_current_batch 
