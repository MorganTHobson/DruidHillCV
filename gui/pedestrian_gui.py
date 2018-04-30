# import the necessary packages
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

import sys

import time
import csv

# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--videos", required=True, help="video to analyze")
#args = vars(ap.parse_args())

cap = None

def analyze_vid(video_name):
    output = ["Time", "Type", "Direction", "Total"]
    SKIP_FRAMES = 10

    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Capture input from video
    cap = cv2.VideoCapture(video_name)
    #csv_output = open("output.csv", "w")
    entrance = True 

    # loop over video image slices
    with open("output.csv", "w") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(output)

        while (True):
            # Capture frame-by-frame
            # skip frames to catch up with real time
            for i in range(SKIP_FRAMES):
                cap.grab()

            ret, frame = cap.read()

            # prevent crashing - only continues if image is read
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # load the image and resize it to (1) reduce detection time
            # and (2) improve detection accuracy
            image = imutils.resize(frame, width=min(750, frame.shape[1]))
            orig = image.copy()

            # detect people in the image
            (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                padding=(8, 8), scale=1.05)

            # apply non-maxima suppression to the bounding boxes using a
            # fairly large overlap threshold to try to maintain overlapping
            # boxes that are still people
            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
            people = non_max_suppression(rects, probs=None, overlapThresh=0.65)

            #print("People found in frame: {} people".format(len(people)))
            #writer = csv.writer(f, delimiter=',')
            #print("here!", time.localtime())

            local_time = time.localtime()
            timeString  = time.strftime("%Y-%m-%d %H:%M:%S", local_time)

            newrow = [timeString, "Pedestrian", entrance, len(people)]
            writer.writerow(newrow)

            # draw the final bounding boxes
            for (xA, yA, xB, yB) in people:
                cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

        # show some information on the number of bounding boxes
        # filename = args["videos"]
        # print("[INFO] {}: {} original boxes, {} after suppression".format(
        #     filename, len(rects), len(pick)))

        # show the output images
        # cv2.imshow("Before NMS", orig)
            cv2.imshow("Object Detection via IMUTILS", image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    cap.release()
    cv2.destroyAllWindows()
#def destroy():
#    cap.release()
#    cv2.destroyAllWindows()
#    sys.exit()

# write to csv
#csv.close()
