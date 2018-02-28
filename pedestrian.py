# import the necessary packages
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--videos", required=True, help="video to analyze")
args = vars(ap.parse_args())

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Capture input from video
cap = cv2.VideoCapture(args["videos"])

# Track count intervals
prev = 0
log = [0,0,0,0]
p = 0

# loop over video image slices
while (True):
    # Capture frame-by-frame
    # skip 20 frames
    for i in range(5):
        cap.grab()

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # load the image and resize it to (1) reduce detection time
    # and (2) improve detection accuracy
    image = imutils.resize(frame, width=min(400, frame.shape[1]))
    orig = image.copy()

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
        padding=(8, 8), scale=1.05)

    # draw the original bounding boxes
    # for (x, y, w, h) in rects:
    #     cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    people = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    print("People found in frame: {} people".format(len(people)))

    log[p] = len(people)
    p = (p+1) % 4
    if p == 0:
        print("Max in range: {}".format(np.max(log)))
        prev = np.max(log)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in people:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # show some information on the number of bounding boxes
    # filename = args["videos"]
    # print("[INFO] {}: {} original boxes, {} after suppression".format(
    #     filename, len(rects), len(pick)))

    # show the output images
    # cv2.imshow("Before NMS", orig)
    cv2.imshow("After NMS", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
