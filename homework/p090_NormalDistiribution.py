# 정규분포(normal distribution) 그 유명한 종형 곡선 모양의 분포
# 평균과 표준편차(시그마)의 두 파라미터로 정의
# 평균은 종의 중심이 어디인지를 나타내며 표준편차는 종의 폭이 얼마나 넓은지를 나타냄

import math
SQRT_TWO_PI = math.sqrt(2*math.pi)

def normal_pdf(x:float, mu:float=0, sigma:float=1) -> float:
    return(math.exp(-(x-mu))**2 /2/sigma**2/(SQRT_TWO_PI*sigma))

import matplotlib.pyplot as plt
xs = [x/10.0 for x in range(-50,50)]
plt.plot(xs, [normal_pdf(x,sigma=1) for x in xs], '-', label='mu=0, sigma=1')
plt.plot(xs, [normal_pdf(x,sigma=2) for x in xs], '--', label='mu=0, sigma=2')
plt.plot(xs, [normal_pdf(x,sigma=0.5) for x in xs], ':', label='mu=0, sigma=0.5')
plt.plot(xs, [normal_pdf(x,mu=-1) for x in xs], '-.', label='mu=-1, sigma=1')
plt.legend()
plt.title("Various Normal pdfs")
plt.show()

# 표준정규분포 μ=0 이고 σ=1인 정규분포를 의미
# Z를 표준정규분포의 확률변수를 나타낸다면 X=σZ+μ
