from matplotlib import pyplot as plt

variance = [1,2,4,8,16,32,64,128,256]
bias_squared = [256,128,64,32,16,8,4,2,1]
total_error = [x+y for x,y in zip(variance, bias_squared)]
xs = [i for i, _ in enumerate(variance)]

# 한 차트에 여러개 선을 그리기 위해 plt.plot을 여러번 호출 가능
plt.plot(xs, variance,      'g-', label='variance')         # 실선
plt.plot(xs, bias_squared,   'r-.', label='bias^2')         # 일점쇄선
plt.plot(xs, total_error,    'b:', label='total error')     # 점선    

plt.legend(loc=9)                       # legend=범례
plt.xlabel("model complexity")
plt.xticks([])
plt.title("The Bias-Variance Tradeoff")
plt.show()
