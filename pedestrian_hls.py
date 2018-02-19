# import the necessary packages
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import subprocess as sp
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--videos", required=True, help="video to analyze")
# args = vars(ap.parse_args())
FFMPEG_BIN = "ffmpeg"

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Capture input from video
WEBURL = "http://devimages.apple.com/"
VIDEO_URL = WEBURL + "iphone/samples/bipbop/bipbopall.m3u8"

pipe = sp.Popen([ FFMPEG_BIN, "-i", VIDEO_URL,
           "-loglevel", "quiet", # no text output
           "-an",   # disable audio
           "-f", "image2pipe",
           "-pix_fmt", "bgr24",
           "-vcodec", "rawvideo", "-"],
           stdin = sp.PIPE, stdout = sp.PIPE)

# loop over video image slices
while (True):
    # Capture frame-by-frame
    # skip 20 frames
    #for i in range(20):
    #    cap.grab()

    #ret, frame = cap.read()
    raw_image = pipe.stdout.read(480*360*3)

    # transform the byte read into a numpy array
    image =  np.fromstring(raw_image, dtype='uint8')
    frame = image.reshape((360,480,3))

    # throw away the data in the pipe's buffer.
    pipe.stdout.flush()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # load the image and resize it to (1) reduce detection time
    # and (2) improve detection accuracy
    image = imutils.resize(frame, width=min(400, frame.shape[1]))
    orig = image.copy()

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
        padding=(8, 8), scale=1.05)

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    people = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    print("People found in frame: {} people".format(len(people)))

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in people:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # show the output images
    cv2.imshow("After NMS", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
