import pdb
import cv2
import matplotlib
import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time
import tensorflow as tf

def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

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

interpreter = tf.lite.Interpreter(
    model_path="new_mobile_model.tflite", num_threads=None)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# check the type of the input tensor
floating_model = input_details[0]['dtype'] == np.float32
# NxHxWxC, H:1, W:2
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
dim = (width, height)
labels = load_labels("class_labels.txt")

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    ret, frame = stream.read()
    # frame = Image.open(args.image).resize((width, height))
    #frame = imutils.resize(frame, width=width, height=height)
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
        frame_cropped = frame[y:y+h, x:x+w]
        frame_resized = cv2.resize(frame_cropped, dim, interpolation = cv2.INTER_AREA)
        #cv2.imshow("crop", frame_resized)

        # add N dim
        input_data = np.expand_dims(frame_resized, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - 0) / 255

        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        results = np.squeeze(output_data)

        top_k = results.argsort()[-5:][::-1]

        cv2.putText(frame, "Cat #{}".format(labels[top_k[0]]), (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # show the output frame
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    # update the FPS counter
    fps.update()
