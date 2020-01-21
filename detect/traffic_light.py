from __future__ import print_function
import cv2
import argparse

def detectAndDisplay(frame):
    red_light = False
    green_light = False
    yellow_light = False

    v=0
    threshold = 150
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    traffic_light = traffic_light_cascade.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in traffic_light:
        cv2.rectangle(frame, (x+5, y+5), (x+w-5, y+h-5), (255, 255, 255), 2)
        v = y + h -5

        # stop
        # if w / h == 1:
        #     cv2.putText(frame, 'STOP', (x, y -10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # traffic light
        # else:
        if w / h != 1:
            roi = gray[y+10:y+h-10,x+10:x+w-10]
            mask = cv2.GaussianBlur(roi, (25,25), 0)
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(mask)

            # check if light is on
            if maxVal - minVal > threshold:
                cv2.circle(roi, maxLoc, 5, (255,0,0), 2)

                # RED light
                if 1.0/8 * (h-30) < maxLoc[1] < 4.0/8 * (h-30):
                    cv2.putText(frame, 'RED', (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    red_light = True

                # GREEN light
                elif 5.5 / 8 * (h - 30) < maxLoc[1] < h - 30:
                    cv2.putText(frame, 'GREEN', (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    green_light = True

    cv2.imshow('Capture - car detection', frame)

parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--traffic_light_cascade', help='Path to traffic light cascade.', default='traffic_light.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()

traffic_light_cascade_name = args.traffic_light_cascade
print()

traffic_light_cascade = cv2.CascadeClassifier()

if not traffic_light_cascade.load(cv2.samples.findFile(traffic_light_cascade_name)):
    print('--(!)Error loading traffic light cascade')
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