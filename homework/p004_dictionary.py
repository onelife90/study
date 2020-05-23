users = [
    {"id":0, "name": "Hero"},
    {"id":1, "name": "Dunn"},
    {"id":2, "name": "Sue"},
    {"id":3, "name": "Chi"},
    {"id":4, "name": "Thor"},
    {"id":5, "name": "Clive"},
    {"id":6, "name": "Hicks"},
    {"id":7, "name": "Devin"},
    {"id":8, "name": "Kate"},
    {"id":9, "name": "Klein"}
]

friendship_pairs = [(0,1), (0,2), (1,2), (1,3), (2,3), (3,4),
                    (4,5), (5,6), (5,7), (6,8), (7,8), (8,9)]

friendships = {user["id"]: [] for user in users}
print(friendships)
for i, j in friendship_pairs:
    friendships[i].append(j)
    friendships[j].append(i)

def num_of_friends(user):
    """user의 친구는 몇 명일까?"""
    user_id=user["id"]
    friend_ids = friendships[user_id]
    return len(friend_ids)

total_connections = sum(num_of_friends(user)
                        for user in users)
num_users = len(users)

total_connections = total_connections / num_users
print(total_connections)

# 친구가 가장 많은 사람을 오름차순으로 정렬해보자
num_friends_by_id = [(user["id"], num_of_friends(user))
                    for user in users]
print(num_friends_by_id)

num_friends_by_id.sort(
    key=lambda id_and_friends: id_and_friends[1],
    reverse=True)
print(num_friends_by_id)

# def foaf_ids_bad(user):
#     return[foaf_id
#             for friend_id in friendships[user["id"]
#             for foaf_id in friendships[friend_id]]]
