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

left_count = 0
right_count = 0

# loop over video image slices
while (True):
    # Capture frame-by-frame
    # skip 20 frames
    for i in range(4):
        cap.grab()

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # load the image and resize it to (1) reduce detection time
    # and (2) improve detection accuracy
    image = imutils.resize(frame, width=min(1000, frame.shape[1]))
    orig = image.copy()
    height, width, channels = image.shape
    mid = int(width/2)

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

    left_last = left_count
    right_last = right_count
    left_count = 0
    right_count = 0

    # draw divider
    cv2.line(image, (mid, 0), (mid, height), (255, 255, 0), 2)

    # draw the final bounding boxes
    for xA, yA, xB, yB in people:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        
        if ((xA+xB)/2 <= mid):
            left_count += 1
        else:
            right_count += 1

    font = cv2.FONT_HERSHEY_SIMPLEX
    line_aa = cv2.LINE_AA  
    cv2.putText(image,str(left_count),(0,40),font,1,(0,0,255),2,line_aa)
    cv2.putText(image,str(right_count),(mid,40),font,1,(0,0,255),2,line_aa)

    print("left frame: {}".format(left_count))
    print("change in left: {}".format(left_count - left_last))

    print("right frame: {}".format(right_count))
    print("change in right: {}".format(right_count - right_last))

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
