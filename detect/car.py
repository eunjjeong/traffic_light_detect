from __future__ import print_function
import cv2
import argparse

def detectAndDisplay(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 3)
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('Capture - car detection', frame)

parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--car_cascade', help='Path to cars cascade.', default='traffic_light.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()

car_cascade_name = args.car_cascade

car_cascade = cv2.CascadeClassifier()

if not car_cascade.load(cv2.samples.findFile(car_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)

camera_device = args.camera
cap = cv2.VideoCapture(camera_device)

while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)
    if cv2.waitKey(10) == 27:
        break