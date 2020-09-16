# import glob : 파일들의 리스트를 뽑을 때 사용

import glob
from sklearn.model_selection import train_test_split as tts

def file_path_save():

    file_names = []
    # "파일 경로"에 있는 .jpg 파일을 정렬해서 보여주거라
    files = sorted(glob.glob("C:/darknet-master/darknet-master/build/darknet/x64/train/data/*.jpg"))
    
    train_files, val_files = tts(files, shuffle=True, train_size=0.8, random_state=99)
    # print(f'len(train_files): {len(train_files)}') # len(train_files): 528
    # print(f'len(val_files): {len(val_files)}')     # len(val_files): 132

    for i in range(len(train_files)):
        # 'a' 추가모드 - 파일의 마지막에 새로운 내용을 추가 시킬 때 사용
        f = open("C:/darknet-master/darknet-master/build/darknet/x64/train/train.txt", 'a')
        # files가 0번째부터 train_files의 수대로 경로가 적히는 리스트 train.txt 형성
        f.write(files[i]+"\n")

    for j in range(len(val_files)):
        f = open("C:/darknet-master/darknet-master/build/darknet/x64/train/valid.txt", 'a')
        f.write(files[j]+"\n")

if __name__ == '__main__':
    file_path_save()

print("생성 완료")
