from matplotlib import pyplot as plt
from collections import Counter

mentions = [500,505]
years = [2017,2018]

plt.bar(years, mentions, 0.8)
plt.xticks(years)
plt.ylabel("# of times I heard someone say 'data science'")

# useOffset을 쓰지 않으면 x축에 0,1 레이블을 달게 됨
plt.ticklabel_format(useOffset=False)

plt.axis([2016.5, 2018.5, 0, 550])
plt.title("Not So Huge Anymore!")
plt.show()
