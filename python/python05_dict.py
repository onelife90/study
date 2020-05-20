#3. 딕셔너리    #중복X
# {키 : 벨류}
# {key : value}     # {호출값(인덱스): 값}

a = {1: 'hi', 2 : 'hello'}       #1이라는 키에는 무조건 hi가 들어감
print(a)
print(a[1])         #hi     #딕셔너리에서는 []안이 key가 됨

b = {'hi' : 1, 'hello': 2}  
print(b['hello'])   #2

# 딕셔너리 요소 삭제
# 수정, 삭제 가능. 중복 X
del a[1]        
print(a)        #{2: 'hello}    
del a[2]
print(a)        #{}

a = {1:'a', 1:'b', 1:'c'}   
print(a[1])        #c       #처음부터 중복된 값 빼고 마지막이 출력

b = {1:'a', 2:'a', 3:'a'}       # 인간A:100점, 인간B:100점, 인간C:100점
print(b)        #{1: 'a', 2: 'a', 3: 'a'}

a = {'name':'yun', 'phone':'010', 'birth':'0511'}       #키 값에는 정수형, 문자형 다 사용가능
print(a.keys())     #['name', 'phone', 'birth']
print(a.values())   #['yun', '010', '0511']
print(type(a))
print(a.get('name'))
print(a['name'])
print(a.get('phone'))
print(a['phone'])
