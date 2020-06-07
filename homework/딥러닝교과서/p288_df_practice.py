import  pandas as pd
df1 = pd.DataFrame([["apple","Friut",120],
                    ["orange","Fruit",60],
                    ["banana","Fruit",100],
                    ["pumpkin","Vegetable",150],
                    ["potato","Vegetable",80]],
                    columns=["Name","Type","Price"])
df2 = pd.DataFrame([["onion","Vegetable",60],
                    ["carrot","Vegetable",50],
                    ["beans","Vegetable",100],
                    ["grape","Fruit",160],
                    ["kiwi","Fruit",80]],
                    columns=["Name","Type","Price"])
df3 = pd.concat([df1, df2],axis=0)
df_fruit = df3.loc[df3["Type"]=="Fruit"]
df_fruit = df_fruit.sort_values(by="Price")
df_veg = df3.loc[df3["Type"]=="Vegetable"]
df_veg = df_veg.sort_values(by="Price")
print(sum(df_fruit[:3]["Price"]) + sum(df_veg[:3]["Price"]))         # 430

# 종합 문제
import pandas as pd
index = ["광수", "민호", "소희", "태양", "영희"]
columns = ["국어", "수학", "사회", "과학", "영어"]
data = [[30,45,12,45,87],[65,47,83,17,58],[64,63,86,57,46],[38,47,62,91,63],[65,36,85,94,36]]
df =pd.DataFrame(data, index=index, columns=columns)
# df에 "체육"이라는 새 열을 만들고 pe_column의 데이터를 추가
pe_column = pd.Series([56,43,73,82,62],index=["광수","민호","소희","태양","영희"])
df["체육"] = pe_column
print(df)
#     국어  수학  사회  과학  영어  체육
# 광수  30  45  12  45  87  56
# 민호  65  47  83  17  58  43
# 소희  64  63  86  57  46  73
# 태양  38  47  62  91  63  82
# 영희  65  36  85  94  36  62
print()

df1 = df.sort_values(by="수학", ascending=True)
print(df1)
#     국어  수학  사회  과학  영어  체육
# 영희  65  36  85  94  36  62
# 광수  30  45  12  45  87  56
# 민호  65  47  83  17  58  43
# 태양  38  47  62  91  63  82
# 소희  64  63  86  57  46  73
print()

df2 = df1+5
print(df2)
#     국어  수학  사회  과학  영어  체육
# 영희  70  41  90  99  41  67
# 광수  35  50  17  50  92  61
# 민호  70  52  88  22  63  48
# 태양  43  52  67  96  68  87
# 소희  69  68  91  62  51  78
print()

print(df2.describe().loc[["mean","max","min"]])
#         국어    수학    사회    과학    영어    체육
# mean  57.4  52.6  70.6  65.8  63.0  68.2
# max   70.0  68.0  91.0  99.0  92.0  87.0
# min   35.0  41.0  17.0  22.0  41.0  48.0
