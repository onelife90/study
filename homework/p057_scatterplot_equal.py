from matplotlib import pyplot as plt

test_1_grades = [99,90,85,97,80]
test_2_grades = [100,85,60,90,70]

# 축의 범위를 균등하게 분할하는 명령어 추가
plt.axis("equal")

plt.scatter(test_1_grades, test_2_grades)
plt.title("Axes Aren't Compaarable")
plt.xlabel("test 1 grade")
plt.ylabel("test 2 grade")
plt.show()
