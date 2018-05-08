import numpy as np
import math
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import argparse
import collections
from collections import defaultdict
from io import StringIO

from PIL import Image
from munkres import Munkres

import cv2
import time
import csv
 
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class DetectorTracker:

    def __init__(self, tf_sess, detection_graph, testing_args):
        self.sess = tf_sess

        # Hungarian Algorithm object
        self.m = Munkres()
        self.detection_graph = detection_graph

        # setup output CSV format and file name globals
        self.output_format = ["Time", "Type", "Direction", "Total"]
        self.output_file = "tensor_output.csv"

        self.args = self.get_args(testing_args)

        # Choose Tracker Type - Currently using KCF
        tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
        self.tracker_type = tracker_types[2]

        self.untracked_cycles = {}
        #self.stationary = {}

        self.current_boxes = []
        self.counter = 0

        self.im_width = 640
        self.im_height = 480

    # setup commandline argument parsing for input video and parameters
    def get_args(self, testing_args):
        ap = argparse.ArgumentParser()

        ap.add_argument("--video", dest="in_file", type=str, help="filename of video to analyze")
        ap.add_argument("--detection-cycle", dest="DETECTION_CYCLE", default=20, type=int, help="how often to run the detection algorithm")
        ap.add_argument("--score-threshold", dest="MIN_SCORE_THRESH", default=20, type=float, help="minimum confidence needed to declare successful detection")
        ap.add_argument("--association-buffer", dest="UNTRACKED_THRESH", default=3, type=int, help="number of detections to check before destroying a tracker")
        ap.add_argument("--pixel-limit", dest="PIXEL_LIMIT", default=50, type=int, help="number of detections to check before destroying a tracker")
        ap.add_argument("--im-zoom", dest="IM_ZOOM", default=0.5, type=float, help="factor of image zoom")

        args = None
        if testing_args != None:
            in_file, DETECTION_CYCLE, MIN_SCORE_THRESH, UNTRACKED_THRESH, PIXEL_LIMIT, IM_ZOOM = testing_args
            #append path to file
            args = ap.parse_args(['--video', in_file, '--detection-cycle', str(DETECTION_CYCLE), '--score-threshold', str(MIN_SCORE_THRESH), '--association-buffer', str(UNTRACKED_THRESH), '--pixel-limit', str(PIXEL_LIMIT), '--im-zoom', str(IM_ZOOM)])
        else:
            args = ap.parse_args()

        if args.in_file == 'hopkins':
            args.in_file = constants.HOPKINS_STREAM
        elif args.in_file == 'druidhill':
            args.in_file = constants.DRUID_HILL_STREAM
        else:
            self.check_args(args)

        return args

    # error checking arguments
    def check_args(self, args):
        print("gets here")
        if not os.path.exists(args.in_file):
            raise Exception("data file specified by --video does not exist.")
        # TODO: should also check that all number arguments are positive numbers

    def get_count(self):
        return self.counter

    # detect object and return array of bbox tuples in format (x, y, w, h)
    def detect_and_get_bboxes(self, frame):

        # initialize for use outside loop
        bboxes = [] # array of bbox tuples in format (x, y, w, h)

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(frame, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents how level of confidence for each of the objects; Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        # Actual detection.
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        # if an object has been detected, then boxes will have a nonzero row value
        boxes_squeezed = boxes[0]
        if boxes_squeezed.shape[0] is not 0:
            scores_squeezed = scores[0]
            classes_squeezed = classes[0]
            eligible_exists = False

            # grabbing all valid detection above confidence of min_score_thresh bboxes
            for i in range(boxes_squeezed.shape[0]):
                if scores_squeezed[i] < self.args.MIN_SCORE_THRESH:
                    break
                if classes_squeezed[i] == 1:
                    ymin, xmin, ymax, xmax = tuple(boxes_squeezed[i].tolist()) # values are normalized from 0 to 1
                    bbox = (xmin*self.im_width, ymin*self.im_height, (xmax - xmin)*self.im_width, (ymax - ymin)*self.im_height) # bbox format: (x, y, w, h)

                    bboxes.append(bbox)
                    eligible_exists = True

            if not eligible_exists:
                #print("Failed to find accurate enough detections. Moving onto next frame.")
                return (False, bboxes)

            #print("Number of valid bboxes found:", len(bboxes))
            return (True, bboxes)
        else:
            #print("Failed to detect any objects. Moving onto next frame.")
            return (False, bboxes)

    # write data to CSV file
    def writeCSV(self, f, writer, prev_time, prev_count, total_count):
        current_time = time.time()
        direction = True # temporary

        # every 3ish seconds write to csv
        if current_time - prev_time > 3:
            # local time parsing
            local_time = time.localtime()
            time_string = time.strftime("%Y-%m-%d %H:%M:%S EST", local_time)
            delta_count = total_count - prev_count

            # write data to csv
            newrow = [time_string, "Pedestrian", direction, str(delta_count)]
            writer.writerow(newrow)
            f.flush()

            # update previous time
            prev_time = current_time
            prev_count = total_count

        return prev_time, prev_count


    # run hungarian algorithm
    def get_assignments(self, tracker_centers, detection_centers, association_limit):
        matrix = []
        limited_indexes = []
        
        # Generate distance matrix trackers=rows detections=column
        for tx,ty in tracker_centers:
            row = []
            for dx,dy in detection_centers:
                delta_x = dx - tx
                delta_y = dy - ty
                row.append(math.sqrt(delta_x*delta_x + delta_y*delta_y))
            matrix.append(row)

        indexes = self.m.compute(matrix)

        for x,y in indexes:
            if matrix[x][y] < association_limit:
                limited_indexes.append((x,y))

        return limited_indexes

    def run_detection(self, frame):
        # how many "new" people were found this time
        count = 0

        # get the valid detection's bounding boxes
        detection_found, bboxes = self.detect_and_get_bboxes(frame)

        detection_centers = []
        old_centers = []

        # get indexes of associated boxes
        indexes = []
        if (len(self.current_boxes) != 0 and len(bboxes) != 0):
            # generate centers
            for newbox in bboxes:
                xmin, ymin, width, height = newbox[:]
                center = (xmin + width/2, ymin + height/2)
                detection_centers.append(center)
            for box in self.current_boxes:
                xmin, ymin, width, height = box[:]
                center = (xmin + width/2, ymin + height/2)
                old_centers.append(center)
            indexes = self.get_assignments(old_centers, detection_centers, self.args.PIXEL_LIMIT)

        # the indexes of the matched pairs
        valid_trackers = []
        valid_detections = []
        data_remove = []

        for t,d in indexes:
            valid_trackers.append(t)
            valid_detections.append(d)

        # remove all unmatched trackers
        for i in range(0,len(self.current_boxes)):
            if i not in valid_trackers:
                self.untracked_cycles[self.current_boxes[i]] += 1
                if self.untracked_cycles[self.current_boxes[i]] >= self.args.UNTRACKED_THRESH:
                    data_remove.append(i)
            else:
                data_remove.append(i)

        for i in reversed(data_remove):
            self.current_boxes.remove(self.current_boxes[i])

        # add all unmatched detections
        for i in range(0,len(bboxes)):
            if i not in valid_detections:
                count += 1
            self.current_boxes.append(bboxes[i])
            self.untracked_cycles[bboxes[i]] = 0

        self.counter += count

    def get_video_stream(self, in_file):
        # Check if video is openable
        video = cv2.VideoCapture(in_file)
        if not video.isOpened():
            print("Could not open video")
            sys.exit()

        # get size of video to scale down size
        im_width = video.get(3)
        im_height = video.get(4)
        self.im_width = int(im_width * self.args.IM_ZOOM)
        self.im_height = int(im_height * self.args.IM_ZOOM)

        # Check if video is readable
        ok, frame = video.read()
        if not ok:
            print('Cannot read video file')
            sys.exit()

        return video

    # driving loop
    def analyze(self):

        # Initializations for CSV
        prev_time = time.time()
        direction = True

        # Initializations for detection
        prev_count = 0
        frame_num = 0
        name = "Druid Hill Project"

        video = self.get_video_stream(self.args.in_file)

        # begin detection + tracking cycle
        with open(self.output_file, "w") as f:
            writer = csv.writer(f, delimiter=',')

            while True:
                # print("========NEW FRAME===============")

                # Read a new frame
                ok, frame = video.read()
                if not ok:
                    break

                # count frame number
                frame_num += 1

                # resize frame to view better
                frame = cv2.resize(frame, (self.im_width, self.im_height), interpolation = cv2.INTER_LINEAR)

                # Re-do detection every n-th frame
                if frame_num % self.args.DETECTION_CYCLE == 0:

                    # print(">>> Re-doing detection...")

                    ok, frame = video.read()
                    if not ok:
                        print('Cannot read video file')
                        sys.exit()
                    frame = cv2.resize(frame, (self.im_width, self.im_height), interpolation = cv2.INTER_LINEAR)

                    self.run_detection(frame)

##                #Display tracker type, detection cycles, allowed unassociations, and total count of people on frame
##                cv2.putText(frame, self.tracker_type + " Tracker", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 170, 50), 2);
##                cv2.putText(frame, str(self.args.DETECTION_CYCLE) + " Cycles Per Detection", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (64,64,64), 2);
##                cv2.putText(frame, str(self.args.UNTRACKED_THRESH) + " Unassociations Allowed", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (64,64,64), 2);
##                cv2.putText(frame, "Total Pedestrians = " + str(self.counter), (50, self.im_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2);
##
##                #Display result and write to CSV
##                cv2.namedWindow(name, cv2.WINDOW_NORMAL);
##
##                #cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
##                
##                cv2.imshow(name, frame);
                prev_time, prev_count = self.writeCSV(f, writer, prev_time, prev_count, self.counter)

                # Exit conditions
                k = cv2.waitKey(1) & 0xff
                if k == 27 : #ESC
                    cv2.destroyAllWindows()
                    break
                # if cv2.getWindowProperty("Pedestrian Detection and Tracking", cv2.WND_PROP_VISIBLE) < 1: # X key on window
                #     break

        cv2.destroyAllWindows()
        print("complete: ", str(self.counter))

