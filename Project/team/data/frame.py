import cv2

def video2frame(invideofilename, save_path):
    vidcap = cv2.VideoCapture('2-2_cam01_assault01_place04_night_spring.mp4')
    count=0
    while True:
        suc, img = vidcap.read()
        if not suc:
            break
        print(f"Read a new frame: {suc}")
        frame = "{}.jpg".format("{0:05d}".format(count))
        cv2.imwrite('./project/data/violence/outsidedoor_05/2-2' + frame, img)
        count += 1
    print(f"count: {count}" +"images are extracted in" + "here: {save_path}")

video2frame('2-2_cam01_assault01_place04_night_spring.mp4', './project/data/violence/outsidedoor_05/2-2')