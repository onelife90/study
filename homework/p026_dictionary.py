empty_dict = {}
emrty_dict2 = dict()
grades = {"Joel":80, "Tim":95}

try:
    kates_grade = grades["Kate"]    # 딕셔너리에 존재하지 않는 키를 입력하면 KeyError가 발생
except KeyError:
    print("no grade for Kate!")
