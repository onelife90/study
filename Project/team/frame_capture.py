import cv2
import os,shutil

# opencv의 함수인 VideoCapture 함수를 사용
main_file_path = 'D:/project/data/videos'

# os.listdir==해당 디렉토리에 있는 파일들의 리스트를 구하기
# m은 main_file_path의 하위폴더들
for m in os.listdir(main_file_path):
    # n은 main_file_path>하위폴더>하위폴더
    for n in os.listdir(main_file_path + '/' + m):
        file_path = main_file_path + '/' + m + '/' + n
        file_list = os.listdir(file_path) 
        
        # 동영상 파일이 들어가 있는 폴더 파일 리스트
        print(file_list)
        
        count = 0
        for i in file_list :
            # .xml파일도 있기 때문에 무시하고 continue
            if i[-3:] != 'mp4':
                continue
            # 폴더의 각 파일에 대한 경로
            path = file_path + '/' + i 

            # i[:-4]를 하는 이유는 확장자를 제거하기 위해
            if os.path.isdir('D:/project/data/videos/capture/' + i[:-4]):
                # shutil.rmtree==지정된 폴더와 하위 폴더, 파일을 모두 삭제
                shutil.rmtree('D:/project/data/videos/capture/' + i[:-4])
            os.mkdir('D:/project/data/videos/capture/' + i[:-4])

            vidcap = cv2.VideoCapture(path)
            ret = True
            while(ret):
                # return 값과 image를 읽어온다
                ret, image = vidcap.read() 
                if(ret==False):
                    break

                # 5프레임 당 1프레임만 저장
                if(int(vidcap.get(1))%5==0):
                    print('Saved frame number :' + str(int(vidcap.get(1))))

                    # i[:-4]를 하는 이유는 확장자를 제거하기 위해
                    cv2.imwrite('D:/project/data/videos/capture/' + i[:-4] + '/' + 'frame%d.jpg' % count, image)
                    print('Saved frame%d.jpg' % count)
                    count += 1

            vidcap.release()

print("캡쳐 완료")
