import time
from datetime import datetime
import subprocess as sp
import cv2
import numpy as np



from keras.models import model_from_json
detect_file1 = "./detect_model/hand1_color.json"
detect_file2 = "./detect_model/hand1_color.h5"
json = open(detect_file1).read()
model = model_from_json(json)
model.load_weights(detect_file2)


def commands(num):
    cmd = {
            1:["gvim"],
            2:["mate-system-monitor"],
            3:["mate-calc"],
            4:["mate-terminal"],
            5:["mpg123", "Cyber16-1.mp3"],
            6:["mpg123", "Cyber16-2.mp3"]
            }
    call_cmd = sp.Popen(cmd[num])
    return cmd[num]

class Timer(object):
    def __enter__(self):
        self.start = datetime.now()
    def __exit__(self, *exc):
        diff = (datetime.now() - self.start).microseconds / 1000
        print("time: {}ms".format(diff))


def detection(path):
    with Timer():

        detect_frame = path[0:250, 0:250]
        detect_frame = cv2.resize(detect_frame, (100,100))
        detect_frame = detect_frame / 255.0
        frame_data = detect_frame.reshape(1,100,100,3)
        pred = model.predict(frame_data)
        result = pred.argmax()
        return result



cap = cv2.VideoCapture(0)
gesture_mode = False
while True:

    _,frame = cap.read()
    frame = cv2.flip(frame, 1)
    res = detection(frame)

    if gesture_mode==False and res == 5:
        commands(5)
        cv2.putText(frame,
            str(res)+"Please Hand Gesture", (200,200),
            cv2.FONT_HERSHEY_PLAIN, 3, (255,255,0))
        gesture_mode = True
        print(str(res) + "  loop 1")

    else:
        cv2.putText(frame,
            str(res)+"Not Hand Detection", (200,200),
            cv2.FONT_HERSHEY_PLAIN, 3, (255,255,0))
        print(str(res)+ "  loop 2")
    if gesture_mode and (res==1 or res==2 or res==3 or res==4):
        commands(res)
        commands(6)
        gesture_mode = False
        print(str(res) + "  loop 3")

    cv2.imshow("frame", frame)
    if cv2.waitKey(1)&0xFF == ord("q"):
        break

with Timer():
    arr = []
    for i in range(10000000):
        arr.append(i)


cv2.destroyAllWindows()
cap.release()
