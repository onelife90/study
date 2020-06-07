# 이름이 다른 열을 key로 결합
# 두개의 df에 같은 정보지만 이름이 다른 컬럼일 경우, 각 컬럼을 별도로 지정
# pd.merge(왼쪽df, 오른쪽df, left_on="왼쪽df의 컬럼", right_on="오른쪽df의 컬럼", how="결합방식")
import pandas as pd
order_df = pd.DataFrame([[1000,2546,103],
                         [1001,4352,102],
                         [1002,342,101]],
                         columns=["id","item_id","customer_id"])
customer_df = pd.DataFrame([[101, "광수"],
                            [102, "민호"],
                            [103, "소희"]],
                            columns=["id", "name"])
#order_df를 바탕으로 "id"를 customer_df와 결합하여 order_df에 대입
order_df = pd.merge(order_df, customer_df, left_on="customer_id", right_on="id", how="inner")
print(order_df)
#    id_x  item_id  customer_id  id_y name
# 0  1000     2546          103   103   소희
# 1  1001     4352          102   102   민호
# 2  1002      342          101   101   광수

# 인덱스를 key로 결합
# df간의 결합에 사용하는 key가 인덱스라면?
# (left_on, right_on)대신 left_index=True, right_index=True로 지정
import pandas as pd
order_df = pd.DataFrame([[1000,2546,103],
                         [1001,4352,102],
                         [1002,342,101]],
                         columns=["id","item_id","customer_id"])
customer_df = pd.DataFrame([["광수"],
                            ["민호"],
                            ["소희"]],
                            columns=["name"])
customer_df.index = [101,102,103]
# customer_df를 바탕으로 "name"을 order_df와 결합하여 order_df에 대입
order_df = pd.merge(order_df, customer_df, left_on="customer_id", right_index=True, how="inner")
print(order_df)
#      id  item_id  customer_id name
# 0  1000     2546          103   소희
# 1  1001     4352          102   민호
# 2  1002      342          101   광수

# 왼쪽 df의 인덱스를 key로 할 경우, left_index=True로 설정.
# 두개 다 바꿔줄 필요 X
