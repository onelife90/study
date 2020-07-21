import cv2
import os,shutil
# xml 파싱하기 위한 api 호출
from xml.etree.ElementTree import parse

main_file_path = 'D:/project/data/videos'
# action별로 특징추출한 프레임 저장할 path
main_save_path = 'D:/project/data/videos/newcapture'

# m은 main_file_path의 하위폴더들
for m in os.listdir(main_file_path):
    # n은 main_file_path>하위폴더>하위폴더
    for n in os.listdir(main_file_path + '/' + m):

        file_path = main_file_path + '/' + m + '/' + n
        # 동영상 파일이 들어가 있는 폴더 파일 리스트
        file_list = os.listdir(file_path)

        print(file_list)
# ['17-1_cam01_assault03_place03_night_spring.mp4', 
# '17-1_cam01_assault03_place03_night_spring.xml', 
# '17-1_cam01_assault03_place03_night_summer.mp4', 
# '17-1_cam01_assault03_place03_night_summer.xml', 
# '17-1_cam02_assault03_place03_night_spring.mp4', 
# '17-1_cam02_assault03_place03_night_spring.xml', 
# '17-1_cam02_assault03_place03_night_summer.mp4', 
# '17-1_cam02_assault03_place03_night_summer.xml']

        count = 0

        for i in file_list :
            if i[-3:] != 'mp4':
                continue
            path = file_path + '/' + i # 폴더의 각 파일에 대한 경로

            if os.path.isdir(main_save_path + '/' + i[:-4]):
                # shutil.rmtree==지정된 폴더와 하위 폴더, 파일을 모두 삭제
                shutil.rmtree(main_save_path + '/' + i[:-4])

            # main_save_path에 파일명 그대로 저장할 폴더 생성
            os.mkdir(main_save_path +'/' + i[:-4])

            # xml 파일명도 동일하기에 i[:-3] + 'xml'
            file_xml = i[:-3] + 'xml'
            # file_path 하위 폴더에 file_xml이 파싱되어 tree 생성
            tree = parse(file_path + '/' + file_xml)

            # getroot()==최상단 나무 뿌리에서 "object" 항목을 찾아 파싱하여 모든 "action"을 찾아주오
            for action in tree.getroot().find("object").findall("action"):
                # actionname.text==파싱한 xml 파일에서 actionname만 뽑아오는 .text
                action_name = action.find("actionname").text

                # 예시= main_save_path/파일명/pushing이라면
                if os.path.isdir(main_save_path + '/' + i[:-4] +'/' + action_name):
                    # shutil.rmtree==지정된 폴더와 하위 폴더, 파일을 모두 삭제
                    shutil.rmtree(main_save_path + '/' + i[:-4]+'/' + action_name)
                
                # 중복된 폴더, 파일 삭제 후 저장할 폴더 생성    
                os.mkdir(main_save_path +'/' + i[:-4]+'/' + action_name)

                # findall("frame")[0]==start frame이기 때문에
                # print(action.findall("frame")[0].find("start").text)
                
                # 파싱한 xml 파일에서 frame을 계속 돌아서 찾아라
                for frame in action.findall("frame"):
                    vidcap = cv2.VideoCapture(path)
                    
                    # 시작은 start frame의 7번째 앞부터
                    start = int(frame.find("start").text) - 7
                    # 끝은 end frame의 7번째 후까지
                    end = int(frame.find("end").text) + 7

                    # CAP_PROP_POS_FRAMES==현재 프레임 개수와 start 프레임 개수를 동일하게 설정(초기화)
                    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)

                    ret = True
                    while(ret) :
                        # return 값과 image를 읽어온다
                        ret, image = vidcap.read()
                        # 캡쳐된 전체 프레임 번호
                        now = int(vidcap.get(1))

                        # 캡쳐된 전체 프레임 번호가 end 프레임 번호보다 크면 break
                        if(now > end) :
                            break

                        # 7프레임당 1개 저장
                        if(now % 7 == 0) :
                            print('Saved frame number :' + str(int(vidcap.get(1))))
                            cv2.imwrite(main_save_path + '/' + i[:-4] + '/' + action_name +'/' + action_name + '_frame%d.jpg' % now, image) # 새롭게 .jpg 파일로 저장
                            print('Saved frame%d.jpg' % count)
                            count += 1

                    vidcap.release()

print("캡쳐 완료")
