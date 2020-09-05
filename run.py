import pdb
import cv2
import matplotlib
import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time

CLASSES = ["gus", "mimi"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
stream = cv2.VideoCapture("gus.mp4")
#time.sleep(2.0)
fps = FPS().start()
# load the cat detector Haar cascade, then detect cat faces
# in the input image
detector = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    ret, frame = stream.read()
    #frame = imutils.resize(frame, width=400)
    # grab the frame dimensions and convert it to a blob
    #(h, w) = frame.shape[:2]
    #blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
    #    0.007843, (300, 300), 127.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(gray, scaleFactor=1.3,
        minNeighbors=10, minSize=(75, 75))
    # loop over the cat faces and draw a rectangle surrounding each
    for (i, (x, y, w, h)) in enumerate(rects):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
        cv2.putText(frame, "Cat #{}".format(i + 1), (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    # update the FPS counter
    fps.update()
