# 키별 통계량 산출. 열의 평균값 산출
import pandas as pd

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)
df.columns = ["","Alchol","Malic acid", "Ash", "Alcalinity of ash","Magnesium","Total phenols", "Flavanoids", "Nonflavanoid phenols", 
              "Proanthocyanins", "Color intensity", "Hue", "0D280/0D315 of diluted wines", "Proline"]
print(df["Magnesium"].mean())
# 99.74157303370787
