import subprocess as sp
import cv2
import numpy as np
from keras.models import model_from_json

detect_file1 = "./detect_model/class6.json"
detect_file2 = "./detect_model/class_6.h5"
json = open(detect_file1).read()
model = model_from_json(json)
model.load_weights(detect_file2)
gesture_mode = False

def commands(num):
    """ Run Command """
    cmd = {
            1:[""],
            2:[""],
            3:[""],
            4:[""],
            5:["mpg123", "Cyber16-1.mp3"],
            6:["mpg123", "Cyber16-2.mp3"]
            }
    call_cmd = sp.Popen(cmd[num])
    return cmd[num]


def detection(path):
    """ Hand Detection """
    detect_frame = path[0:250, 0:250]
    detect_frame = cv2.resize(detect_frame, (100,100))
    detect_frame = detect_frame / 255.0
    frame_data = detect_frame.reshape(1,100,100,3)
    pred = model.predict(frame_data)
    result = pred.argmax()
    return result


def event(frame, num):
    """ Run branch"""
    global gesture_mode
    result = (res==1 or res==2 or res==3 or res==4)
    if gesture_mode==False and res == 5:
        cv2.putText(frame,
            str(res)+"Please Hand Gesture", (0,350),
            cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0))
        commands(5)
        gesture_mode = True

    elif gesture_mode and result: 
        cv2.putText(frame,
            str(res)+"Please Hand Gesture", (0,350),
            cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0))
        gesture_mode = False
        commands(res)

    elif gesture_mode:
        cv2.putText(frame,
            str(res)+"Please Hand Gesture", (0,350),
            cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0))

    else:
        cv2.putText(frame,
            str(res)+"Not Detect Hand Gesture", (0,350),
            cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0))

cap = cv2.VideoCapture(0)

while True:

    _,frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (0,0),(260,260),(0,0,255),3)
    res = detection(frame)
    event(frame, res)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1)&0xFF == ord("q"):
        break


cv2.destroyAllWindows()
cap.release()
