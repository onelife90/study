from matplotlib import pyplot as plt

movies = ["기생충", "애마부인", "카사블랑카", "용서는 없다", "빵야빵야"]
num_oscars = [5,11,3,8,10]

plt.bar(range(len(movies)), num_oscars)     # x좌표, y좌표 설정

plt.title("최애영화")
plt.ylabel("# od Academy Awards")           # y축 레이블 추가

plt.xticks(range(len(movies)), movies)      # x축 각 막대의 중앙에 영화 제목을 레이블로 추가

plt.show()
