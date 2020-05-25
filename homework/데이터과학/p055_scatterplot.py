from matplotlib import pyplot as plt

friends = [70,65,72,63,71,64,60,64,67]
minutes = [175,170,205,120,220,130,105,145,190]
labels = ['a', 'b', 'c', 'd', 'f', 'g', 'h', 'i']

plt.scatter(friends, minutes)

# 각 포인트에 label, frined_count, minute_count 레이블을 달자
for label, friend_count, minute_count in zip(labels, friends, minutes):
    # annotate=타입 명시
    plt.annotate(label,         
    xy=(friend_count, minute_count),
    xytext=(5,-5),
    textcoords='offset points')

plt.title("Daily Minutes vs. Number of Friends")
plt.xlabel("# of friends")
plt.ylabel("daily minutes spend on the site")
plt.show()
