import cv2
import os,shutil
from xml.etree.ElementTree import parse
 
# opencv의 함수인 VideoCapture 함수를 사용

# 사진 파일 위치
main_file_path = 'D:/Study/videos/captures/0cut'       
# xml 위치
main_xml_path = 'C:/Users/bitcamp/Downloads/labelImg-master/data/labelings'   
# 같은 사진 저장할 위치
main_save_path = 'C:/Users/bitcamp/Downloads/labelImg-master/data/imgs'   
 
 
if os.path.isdir(main_save_path):
    shutil.rmtree(main_save_path)
 
os.mkdir(main_save_path)
 
count = 0
for m in os.listdir(main_file_path):
    for n in os.listdir(main_file_path + '/' + m):
        file_path = main_file_path + '/' + m + '/' + n # 사진 들어가있는 폴더 주소
        file_list = os.listdir(file_path) # 동영상 파일이 들어가 있는 폴더 파일 리스트
 
        # print("file_list :",len(file_list))
        for i in file_list :
            for xmlfiles in os.listdir(main_xml_path): # 각각 파일명과 xml이름을 가져옴 
                if xmlfiles[:-4] == i[:-4]:
                    shutil.copy(src=file_path + '/' + i, dst= main_save_path + '/' + i)
