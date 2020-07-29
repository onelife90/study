import os, shutil

# 변경 전 파일이 있는 폴더
folder_path = 'D:/project/data/frame'

# 23-3_cam01_assault01_place02_night_spring
for m in os.listdir(folder_path):
    # kicking, puching, pushing
    for n in os.listdir(folder_path + '/' + m):
        # frame %d.jpg 최종 변경할 파일
        for f in os.listdir(folder_path + '/' + m + '/' + n):

            if os.path.isdir(folder_path + '/' + m + '/' + n + '/' + f):
                # shutil.rmtree==지정된 폴더와 하위 폴더, 파일을 모두 삭제
                shutil.rmtree(folder_path + '/' + m + '/' + n + '/' + f)

            old_name = folder_path + '/' + m + '/' + n + '/' + f 
            # print(f"old_name : {old_name}")
            # old_name : D:/project/data/frame/453-4_cam02_assault01_place01_day_summer/throwing/throwing_frame9.jpg
            
            new_name = folder_path + '/' + m + '/' + n + '/' + m + '_' + f 
            print(f"new_name : {new_name}")
            # D:/project/data/frame/453-4_cam02_assault01_place01_day_summer/throwing/453-4_cam02_assault01_place01_day_summer_throwing_frame27.jpg

            rename = os.rename(old_name, new_name)
print("변경 완료")
