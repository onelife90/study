from matplotlib import pyplot as plt
from collections import Counter

mentions = [500,505]
years = [2017,2018]

plt.bar(years, mentions, 0.8)
plt.xticks(years)
plt.ylabel("# of times I heard someone say 'data science'")

# useOffset을 쓰지 않으면 x축에 0,1 레이블을 달게 됨
plt.ticklabel_format(useOffset=False)

# y축의 범위를 지정해 주지 않았으므로 500 이상의 부분만 보여줌
plt.axis([2016.5, 2018.5, 499, 506])
plt.title("Look at the 'Huge' Increase!")
plt.show()
