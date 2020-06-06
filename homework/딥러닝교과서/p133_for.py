# 리스트의 요소를 모두 출력하고 싶을 때 자주 사용하는 것이 for문
# for 변수 in 데이터셋:
# for 문 뒤에는 콜론: 이 들어가는 것을 잊지말긔

storage = [1,2,3,4]
# for 문으로 변수 storage의 요소 출력
for i in storage:
    print(i)
# 1
# 2
# 3
# 4

# break를 이용해서 반복 처리 종료. if문과 함께 사용되는 경우 만흥ㅁ
for i in storage:
    print(i)
    if i==4:
        break;
# 1
# 2
# 3
# 4
# break도 종료 시 ; 세미클론 필수!
