from imutils.video import VideoStream
from imutils import face_utils
from max30100 import max30100
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

import RPi.GPIO as GPIO
from time import sleep

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

buzzer_pin = 19
temp = 23

GPIO.setup(temp, GPIO.IN)

GPIO.setup(17, GPIO.OUT)
GPIO.output(17, False)
GPIO.setup(buzzer_pin, GPIO.OUT)
GPIO.output(buzzer_pin, False)

mx30 = max30100.MAX30100()
mx30.enable_spo2()

def euclidean_dist(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

def eye_aspect_ratio(eye):
    A = euclidean_dist(eye[1], eye[5])
    B = euclidean_dist(eye[2], eye[4])
    C = euclidean_dist(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True, help="path to where the face cascade resides")
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=int, default=0, help="boolean used to indicate if TrafficHat should be used")
args = vars(ap.parse_args())

if args["alarm"] > 0:
    from gpiozero import TrafficHat
    th = TrafficHat()
    print("[INFO] using TrafficHat alarm...")

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0
ALARM_ON = False

print("[INFO] loading facial landmark predictor...")
detector = cv2.CascadeClassifier(args["cascade"])
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    if args["alarm"] > 0:
                        th.buzzer.blink(0.1, 0.1, 10, background=True)
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                GPIO.output(17, True)
        else:
            COUNTER = 0
            ALARM_ON = False
            GPIO.output(17, False)

        cv2.putText(frame, "EAR: {:.3f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    mx30.read_sensor()
    time.sleep(0.1)

    if len(mx30.buffer_red) >= 2:
        red, ir = mx30.buffer_red[-2:]
        print("red:")
        print(red)
        print("ir:")
        print(ir)

        if red >= 3500 and red <= 13900 and ir >= 3500 and ir <= 13900:
            heart_rate = 70 + (red - 3500) * (120 - 68) / (13900 - 3500)
            oxygen_saturation = 98 + (ir - 3300) * (91 - 98) / (13900 - 3500)

            heart_rate = max(60, min(120, heart_rate))
            oxygen_saturation = max(91, min(100, oxygen_saturation))

            if heart_rate < 69:
                print("Driver Health Condition: heart beat low alert!")
                GPIO.output(buzzer_pin, GPIO.HIGH)
            elif heart_rate > 115:
                print("Driver Health Condition: heart beat high alert!")
                GPIO.output(buzzer_pin, GPIO.HIGH)

            if oxygen_saturation < 92:
                print("Driver Health Condition: Oxygen low Alert")
                GPIO.output(buzzer_pin, GPIO.HIGH)
            elif oxygen_saturation > 100:
                print("Driver Health Condition: Oxygen high Alert")
                GPIO.output(buzzer_pin, GPIO.HIGH)
        else:
            heart_rate = 0
            oxygen_saturation = 0
            GPIO.output(buzzer_pin, GPIO.LOW)

        print("Heart Rate:", heart_rate, "bpm")
        print("Oxygen Saturation:", oxygen_saturation, "%")

        # ✅ Temperature GPIO input read
        if GPIO.input(temp) == GPIO.HIGH:
            print("Temperature is normal")
        elif GPIO.input(temp) == GPIO.LOW:
            print("Temperature is High")
            GPIO.output(buzzer_pin, GPIO.HIGH)
            time.sleep(2)
            GPIO.output(buzzer_pin, GPIO.LOW)

    if key == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
vs.stop()

