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

import testing
 
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# # TENSORFLOW OBJECT DETECTION MODEL PREPARATION (ONLINE TRAINING) =======================================================

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/usr/local/lib/python3.6/site-packages/tensorflow/models/research/object_detection/data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

# ## Download Model
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())

# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

class DetectorTracker:

    def __init__(self, tf_sess, mode):
        self.sess = tf_sess

        # Hungarian Algorithm object
        self.m = Munkres()

        # setup output CSV format and file name globals
        self.output_format = ["Time", "Type", "Direction", "Total"]
        self.output_file = "tensor_output.csv"

        DEFAULT_STREAM = "https://stream-us1-alfa.dropcam.com:443/nexus_aac/7838408781384ee7bd8d1cc11695f731/chunklist_w1479032407.m3u8"

        self.args = None
        # get actual arguments
        if mode is not None:
            self.args.in_file, self.args.DETECTION_CYCLE, self.args.MIN_SCORE_THRESH, self.args.TRACKING_BUFFER, self.args.UNTRACKED_THRESH, self.args.PIXEL_LIMIT, self.args.IM_ZOOM = mode
        else:
            self.args = self.get_args(DEFAULT_STREAM)

        # Choose Tracker Type - Currently using KCF
        tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
        self.tracker_type = tracker_types[2]

        self.updates = {}
        self.untracked_cycles = {}
        #self.stationary = {}

        self.multitracker = []
        self.counter = 0

        self.im_width = 640
        self.im_height = 480

    # setup commandline argument parsing for input video and parameters
    def get_args(self, DEFAULT_STREAM):
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--video", dest="in_file", default=DEFAULT_STREAM, type=str, help="filename of video to analyze")
        ap.add_argument("-d", "--detection-cycle", dest="DETECTION_CYCLE", default=20, type=int, help="how often to run the detection algorithm")
        ap.add_argument("-s", "--score-threshold", dest="MIN_SCORE_THRESH", default=20, type=float, help="minimum confidence needed to declare successful detection")
        ap.add_argument("-t", "--tracking-buffer", dest="TRACKING_BUFFER", default=20, type=int, help="number of frames to check before calling a tracker successful (count increases)")
        ap.add_argument("-a", "--association-buffer", dest="UNTRACKED_THRESH", default=3, type=int, help="number of detections to check before destroying a tracker")
        ap.add_argument("-p", "--pixel-limit", dest="PIXEL_LIMIT", default=50, type=int, help="number of detections to check before destroying a tracker")
        ap.add_argument("-z", "--im-zoom", dest="IM_ZOOM", default=0.5, type=float, help="factor of image zoom")

        args = ap.parse_args()
        self.check_args(args, DEFAULT_STREAM)

        return args

    # error checking arguments
    def check_args(self, args, DEFAULT_STREAM):
        if args.in_file != DEFAULT_STREAM:
            if not os.path.exists(args.in_file):
                raise Exception("model file specified by --model-file does not exist.")
        # TODO: should also check that all number arguments are positive numbers

    def get_count(self):
        return self.counter

    # detect object and return array of bbox tuples in format (x, y, w, h)
    def detect_and_get_bboxes(self, frame):

        # initialize for use outside loop
        bboxes = [] # array of bbox tuples in format (x, y, w, h)

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(frame, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents how level of confidence for each of the objects; Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

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

    # create tracker of given type
    # TODO: be able to select different trackers through command line
    def create_tracker(self, frame, bbox):

        # TODO: write better code: tracker_type is currently global
        if self.tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if self.tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if self.tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if self.tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if self.tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if self.tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()

        ok = tracker.init(frame, bbox)
        self.updates[tracker] = 0
        self.untracked_cycles[tracker] = 0

        return tracker

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

    # re-run detection on video frame
    def run_detection(self, frame, current_boxes):
        # get the valid detection's bounding boxes
        detection_found, bboxes = self.detect_and_get_bboxes(frame)

        detection_centers = []
        tracker_centers = []

        # get indexes of associated boxes
        indexes = []
        if (len(current_boxes) != 0 and len(bboxes) != 0):
            # generate centers
            for newbox in bboxes:
                xmin, ymin, width, height = newbox[:]
                center = (xmin + width/2, ymin + height/2)
                detection_centers.append(center)
            for tracker,box in current_boxes:
                xmin, ymin, width, height = box[:]
                center = (xmin + width/2, ymin + height/2)
                tracker_centers.append(center)

            indexes = self.get_assignments(tracker_centers, detection_centers, self.args.PIXEL_LIMIT)

        # the indexes of the matched pairs
        valid_trackers = []
        valid_detections = []
        data_remove = []

        for t,d in indexes:
            valid_trackers.append(t)
            valid_detections.append(d)

        # remove all unmatched trackers
        for i in range(0,len(current_boxes)):
            # count many times the bbox fails to be associated
            if i not in valid_trackers:
                self.untracked_cycles[current_boxes[i][0]] += 1
                if self.untracked_cycles[current_boxes[i][0]] >= self.args.UNTRACKED_THRESH:
                    data_remove.append(i) # if it fails to be associated x times then stage for removal
            else:
                self.untracked_cycles[current_boxes[i][0]] = 0

        # actual removal of the tracker
        for i in reversed(data_remove):
            self.multitracker.remove(current_boxes[i][0])
            current_boxes.remove(current_boxes[i])

            #print("Tracker " + str(i) + " removed")

        # add all unmatched detections
        for i in range(0,len(bboxes)):
            if i not in valid_detections:
                self.multitracker.append(self.create_tracker(frame, bboxes[i]))

                #print("Box added: " + str(i))
        return current_boxes

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
    def begin_analysis(self):

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

                # Update tracker
                current_boxes = []
                failed_trackers = []
                for tracker in self.multitracker:
                    ok, updated_box = tracker.update(frame)

                    # Draw bounding box
                    if ok:
                        # Tracking success
                        p1 = (int(updated_box[0]), int(updated_box[1]))
                        p2 = (int(updated_box[0] + updated_box[2]), int(updated_box[1] + updated_box[3]))
                        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

                        current_boxes.append((tracker, updated_box))

                        self.updates[tracker] += 1
                        if self.updates[tracker] == self.args.TRACKING_BUFFER: # needs to be tracked throughout at least x frames in order to be counted as a person
                            self.counter += 1

                    else :
                        # Tracking failure
                        cv2.putText(frame, "Tracking Failure", (200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)

                        # if there is a tracking failure, try re-running the object detection algorithm and dont get rid of the tracker until the object detection
                        # officially says it's a bad tracker
                        failed_trackers.append(tracker)

                fail_count = len(failed_trackers)

                # REMOVING TRACKERS BASED UPON KCF REPORTED TRACKING FAILURES
                if not fail_count == 0:
                    for bad_tracker in failed_trackers:
                        self.multitracker.remove(bad_tracker)
                        self.updates.pop(bad_tracker)

                    # print(">>>", fail_count,"TRACKING FAILURES OCCURRED! Number of trackers after removal:", len(multitracker))

                #Display tracker type, detection cycles, allowed unassociations, and total count of people on frame
                cv2.putText(frame, self.tracker_type + " Tracker", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 170, 50), 2);
                cv2.putText(frame, str(self.args.DETECTION_CYCLE) + " Cycles Per Detection", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (64,64,64), 2);
                cv2.putText(frame, str(self.args.UNTRACKED_THRESH) + " Unassociations Allowed", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (64,64,64), 2);
                cv2.putText(frame, "Total Pedestrians = " + str(self.counter), (50, self.im_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2);

                #Display result and write to CSV
                cv2.namedWindow(name, cv2.WINDOW_NORMAL);

                #cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
                
                cv2.imshow(name, frame);
                prev_time, prev_count = self.writeCSV(f, writer, prev_time, prev_count, self.counter)

                # Re-do detection every n-th frame
                if frame_num % self.args.DETECTION_CYCLE == 0:

                    # print(">>> Re-doing detection...")

                    ok, frame = video.read()
                    if not ok:
                        print('Cannot read video file')
                        sys.exit()
                    frame = cv2.resize(frame, (self.im_width, self.im_height), interpolation = cv2.INTER_LINEAR)

                    current_boxes = self.run_detection(frame, current_boxes)

                # Exit conditions
                k = cv2.waitKey(1) & 0xff
                if k == 27 : #ESC
                    cv2.destroyAllWindows()
                    break
                # if cv2.getWindowProperty("Pedestrian Detection and Tracking", cv2.WND_PROP_VISIBLE) < 1: # X key on window
                #     break

        cv2.destroyAllWindows()
        print(str(self.counter))


from subprocess import Popen, PIPE

DATA_DIR = 'video_input'
VIDEOS = ['IMG_1295.MOV', 'IMG_1296.MOV', 'IMG_0900.MOV']

MIN_SCORE_THRESH = 0.6
IM_ZOOM = 0.5

if not os.path.exists(DATA_DIR):
    raise Exception('Data directory specified by DATA_DIR does not exist.')

data_format = ["DETECTION_CYCLE", "TRACKING_BUFFER", "UNTRACKED_THRESH", "PIXEL_LIMIT", "FILE", "COUNT"]
data_file = "data_output.csv"

def run_tests(sess):
    with open(data_file, "w") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow('MIN_SCORE_THRESH = 0.6, IM_ZOOM = 0.5')

        for DETECTION_CYCLE in ['20', '10', '5']:
            for UNTRACKED_THRESH in ['10', '5', '3', '0']:
                for PIXEL_LIMIT in ['150', '100', '80', '50']:
                    for TRACKING_BUFFER in ['100', '80', '50', '30', '10']:
                        for in_file in VIDEOS:
                            print('Running on %s for parameters --detection-cycle %s --score-threshold %s --tracking-buffer %s --association-buffer %s --pixel-limit %s --im-zoom %s' % (in_file, DETECTION_CYCLE, MIN_SCORE_THRESH, TRACKING_BUFFER, UNTRACKED_THRESH, PIXEL_LIMIT, IM_ZOOM))
                            file_path = os.path.join(DATA_DIR, in_file)

                            detector = DetectorTracker(sess, tuple([in_file, DETECTION_CYCLE, MIN_SCORE_THRESH, TRACKING_BUFFER, UNTRACKED_THRESH, PIXEL_LIMIT, IM_ZOOM]))
                            detector.begin_analysis()
                            count = detector.get_count()

                            p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE, stdin=PIPE)
                            output = p.stdout.readline().decode('ascii')
                            print(output)

                            newrow = [DETECTION_CYCLE, TRACKING_BUFFER, UNTRACKED_THRESH, PIXEL_LIMIT, in_file, output]
                            writer.writerow(newrow)
                            f.flush()

# main function
if __name__ == '__main__' :
    # setup detection session
    detection_graph.as_default()
    sess = tf.Session(graph=detection_graph)

    run_tests(sess)
    # end
