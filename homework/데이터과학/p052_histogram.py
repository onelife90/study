from matplotlib import pyplot as plt
from collections import Counter

grades = [83,95,91,87,70,0,85,82,100,67,73,77,0]

histogram = Counter(min(grade // 10*10, 90) for grade in grades)
# 10점 단위로 그룹화. 100점은 90점에 속함

plt.bar([x+5 for x in histogram.keys()],    # 각 막대를 오른쪽으로 5만큼 옮기고 ==x+5
        histogram.values(),                 # 각 막대의 높이를 정해 주고 values
        10,                                 # 너비는 10으로 하자
        edgecolor=(0,0,0))                  # 각 막대의 테두리는 검은색

plt.axis([-5,105,0,5])      # x축 범위==-5~105    # y축 범위==0~5

plt.xticks([10*i for i in range(11)])   # x축 레이블 0,10,20,...,100
plt.xlabel("Decile")
plt.ylabel("# of Students")
plt.title("Distribution of Exam 1 Grades")
plt.show()
