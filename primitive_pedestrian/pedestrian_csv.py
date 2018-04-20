# import the necessary packages
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--videos", required=True, help="video to analyze")
args = vars(ap.parse_args())

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Capture input from video
cap = cv2.VideoCapture(args["videos"])

# Number of people in previous frame
previousFrame = 0
# The number of people in the most recent frame that has been verified to not be a blip
lastStableFrame = 0

output="Event Type, Time\n"

# loop over video image slices
while (True):
    # Capture frame-by-frame
    # skip 20 frames
    for i in range(20):
        cap.grab()

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # load the image and resize it to (1) reduce detection time
    # and (2) improve detection accuracy
    image = imutils.resize(frame, width=min(1000, frame.shape[1]))
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

    # check if the number of people in frame is the same for at least 2 consecutive frames
    if len(people) == previousFrame and previousFrame != lastStableFrame:
        if previousFrame > lastStableFrame:
            eventType = "entrance"
        else:
            eventType = "exit"
        for x in range(abs(previousFrame - lastStableFrame)):
            print("Recorded an " + eventType)
            output += eventType.capitalize() + "," + str(time.time()) + "\n"
        lastStableFrame = previousFrame
    previousFrame = len(people)

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

csv = open("output.csv", "w")
csv.write(output)
csv.close()
