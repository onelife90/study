from flask import Flask, render_template, Response, request
# emulated camera
import cv2
import numpy as np
from threading import Thread
from darknet import darknet

# 1. make file path
config_path = "C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/train/my_yolov3.cfg"
# config_path = "D:/python_module/darknet-master/build\darknet/x64/project/myyolov3.cfg"
weigh_path = "C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/train/my_yolov3_final.weights"
# weigh_path = "D:/python_module/darknet-master/build/darknet/x64/project/backup/myyolov3_final.weights"
meta_path = "C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/train/my_data.data"
# meta_path = "D:/python_module/darknet-master/build\darknet/x64/project/my.data"
video_path = "C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/data/video/4-1.mp4"
# video_path = "D:/python_module/darknet-master/build\darknet/x64/project/22-2.mp4"
threshold = 0.5

video1 = "C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/data/video/16-3.mp4"
video2 = "C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/data/video/4-1.mp4"

# video1 = "D:/python_module/darknet-master/build\darknet/x64/project/22-2.mp4"
# video2 = "D:/python_module/darknet-master/build\darknet/x64/project/26-4.mp4"

# 2. relevant file load
# load to ascii code cfg(model) file, weight file
net = darknet.load_net(bytes(config_path, "ascii"),
                       bytes(weigh_path, "ascii"), 0)
# load to ascii code data(path about classes, train, valid, names, backup) file
meta = darknet.load_meta(bytes(meta_path, "ascii"))
# VideoCapture: video frame capture
cap = cv2.VideoCapture(video_path)


# 3. WebcamVideoStream : use thread
class WebcamVideoStream:
    def __init__(self, src=0):
        print("init")
        # initialize the video camera stream and read the first frame from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should be stopped
        self.stopped = False

    def start(self):
        print("start thread")
        # start the thread to read frames from the video stream
        # using Thread to update defnition. parameter is tuple((self.grabbed, self.frame) = self.stream.read())
        t = Thread(target=self.update, args=())
        # daemon thread is a thread that runs in the background. 
        # Shutdown immediately after main thread is closed
        t.daemon = True
        t.start()
        return self

    # update frame
    def update(self):
        print("read")
        while True:
            if self.stopped:
                return

            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


# make video list(index,video) tuple
video = [(1, video1), (2, video2)]
app = Flask(__name__)


@app.route('/')
def run():
    rows = video
    # read videl list in camera_index.html
    return render_template("camera_index.html", rows=rows)


@app.route('/index')
def index():
    # http://192.168.0.120:5000/index?id=2
    # request.args is url parameter dictionary(check the camera_index.html)
    ids = request.args.get('id')
    print(ids)
    rows = video
    return render_template('camera.html', rows=[rows[int(ids)-1]])


def gen(camera):
    # """Video streaming generator function."""
    i = 0
    while True:
        i += 1
        image = camera.read()
        image = cv2.resize(image, dsize=(640, 480),
                           interpolation=cv2.INTER_AREA)
        print(i)
        # if not ret:
        # break
        frame = darknet.nparray_to_image(image)
        r = darknet.detect_image(net, meta, frame, thresh=.5, hier_thresh=.5, nms=.45, debug=False)
        print(f"r:\n{r}")
        # [(b'normal', 0.9838562607765198, (337.76190185546875, 226.85903930664062, 41.72311782836914, 109.13109588623047)), 
        # (b'normal', 0.907978355884552, (302.71875, 253.96533203125, 41.06242752075195, 113.02967834472656)), 
        # (b'normal', 0.8925231695175171, (377.8631286621094, 233.21629333496094, 32.55954360961914, 110.92288970947266))]
        boxes = []

        for k in range(len(r)):
            width = r[k][2][2]
            height = r[k][2][3]
            center_x = r[k][2][0]
            center_y = r[k][2][1]
            bottomLeft_x = center_x - (width / 2)
            bottomLeft_y = center_y - (height / 2)
            x, y, w, h = bottomLeft_x, bottomLeft_y, width, height
            mytexts = r[k][0]
            mythresh = r[k][1]
            boxes.append((x, y, w, h, mytexts, mythresh))
        print("next")

        for k in range(len(boxes)):
            x, y, w, h, texts, threshs = boxes[k]
            top = max(0, np.floor(x + 0.5).astype(int))
            left = max(0, np.floor(y + 0.5).astype(int))
            right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
            bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))
            # cv2.rectangle(image, (top, left), (right, bottom), (0, 255, 0), 1)

            if texts.decode('utf-8') == 'normal':
                cv2.rectangle(image, (top, left),
                              (right, bottom), (255, 0, 0), 2)
                cv2.putText(image, texts.decode('utf-8') + '(' + str(threshs*100)
                            [:5] + '%)', (top, left-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

            elif texts.decode('utf-8') == 'fighting':
                cv2.rectangle(image, (top, left),
                              (right, bottom), (0, 0, 255), 2)
                # mark probablity
                cv2.putText(image, texts.decode('utf-8') + '(' + str(threshs*100)
                            [:5] + '%)', (top, left-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))


        # cv2.imshow('frame', image)
        
        ret, jpeg = cv2.imencode('.jpg', image)

        # darknet dynamic allocation
        darknet.free_image(frame)

        if jpeg is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        else:
            print("frame is none")


@app.route('/index/video_feed')
def video_feed():
    ids = request.args.get('id')
    print(ids)
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(WebcamVideoStream(src=video[int(ids)-1][1]).start()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True, threaded=True)