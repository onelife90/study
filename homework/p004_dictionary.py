users = [
  { "id": 0 , "name":"Hero" },
  { "id": 1 , "name":"Dunn" },
  { "id": 2 , "name":"Sue" },
  { "id": 3 , "name":"Chi" },
  { "id": 4 , "name":"Thor" },
  { "id": 5 , "name":"Clive" },
  { "id": 6 , "name":"Hicks" },
  { "id": 7 , "name":"Devin" },
  { "id": 8 , "name":"Kate" },
  { "id": 9 , "name":"Klein" },
]    

friendship_pairs = [(0,1), (0,2), (1,2), (1,3), (2,3), (3,4),
                    (4,5), (5,6), (5,7), (6,8), (7,8), (8,9)]

friendships = {user["id"]: [] for user in users}

for i, j in friendship_pairs:
friendships[i].append(j)
friendships[j].append(i)

def number_of_friends(user):
  """user의 친구는 몇 명일까?"""
  user_id = user["id"]
  friends_ids = friendships[user_id]
  return len(friends_ids)

total_connections = sum(number_of_friends(user)
                        for user in users)

# 친구가 가장 많은 사람을 알아보자. 오름차순으로 사용자를 정렬.
num_users = len(users)
avg_connections = total_connections / num_users

