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

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# setup output CSV format and file name globals
output_format = ["Time", "Type", "Direction", "Total"]
output_file = "tensor_output.csv"

# Constant declarations
IM_WIDTH = 950
IM_HEIGHT = 600
DETECTION_CYCLE = 20 # how often to run the detection algo
MIN_SCORE_THRESH = 0.5  # initialize minimum confidence needed to call an object detection successful
TRACKING_BUFFER = 20
DEFAULT_STREAM = "https://stream-us1-alfa.dropcam.com:443/nexus_aac/7838408781384ee7bd8d1cc11695f731/chunklist_w1479032407.m3u8"

# # TENSORFLOW OBJECT DETECTION MODEL PREPARATION (ONLINE TRAINING) =======================================================

# ## Variables

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/usr/local/lib/python3.6/site-packages/tensorflow/models/research/object_detection/data', 'mscoco_label_map.pbtxt')
print(PATH_TO_LABELS)
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

# ## Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

# detect object and return array of bbox tuples in format (x, y, w, h)
def detect_and_get_bboxes(frame):

    # initialize for use outside loop
    bboxes = [] # array of bbox tuples in format (x, y, w, h)

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(frame, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
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
            if scores_squeezed[i] < MIN_SCORE_THRESH:
                break
            if classes_squeezed[i] == 1:
                ymin, xmin, ymax, xmax = tuple(boxes_squeezed[i].tolist()) # values are normalized from 0 to 1
                bbox = (xmin*IM_WIDTH, ymin*IM_HEIGHT, (xmax - xmin)*IM_WIDTH, (ymax - ymin)*IM_HEIGHT) # bbox format: (x, y, w, h)

                bboxes.append(bbox)
                eligible_exists = True

        if not eligible_exists:
            print("Failed to find accurate enough detections. Moving onto next frame.")
            return (False, bboxes)

        print("Number of valid bboxes found:", len(bboxes))
        return (True, bboxes)
    else:
        print("Failed to detect any objects. Moving onto next frame.")
        return (False, bboxes)

def writeCSV(prev_time):
    # WRITING TO CSV ================== MAKE SEPARATE FUNCTION
    current_time = time.time()
    # every 3ish seconds write to csv
    if current_time - prev_time > 3:
        # local time parsing
        local_time = time.localtime()
        time_string = time.strftime("%Y-%m-%d %H:%M:%S EST", local_time)

        # write data to csv
        newrow = [time_string, "Pedestrian", direction, str(counter)]
        writer.writerow(newrow)
        f.flush()

        # update previous time
        prev_time = current_time

    return prev_time

updates = {}
def create_tracker(frame, bbox):

    # bleh, write better code: tracker_type global

    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()

    ok = tracker.init(frame, bbox)
    updates[tracker] = 0

    return tracker

def get_assignments(tracker_centers, detection_centers, m):
    matrix = []
    
    # Generate distance matrix trackers=rows detections=column
    for tx,ty in tracker_centers:
        row = []
        for dx,dy in detection_centers:
            delta_x = dx - tx
            delta_y = dy - ty
            row.append(math.sqrt(delta_x*delta_x + delta_y*delta_y))
        matrix.append(row)

    indexes = m.compute(matrix)
    return indexes

if __name__ == '__main__' :

    # Hungarian Algorithm object
    m = Munkres()

    # # TRACKER SETUP=============================================================

    # Choose Tracker Type - Currently using KCF
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[2]

    # # DETECTION ================================================================

    detection_graph.as_default()
    sess = tf.Session(graph=detection_graph)

    # setup commandline argument parsing for input video
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", help="filename of video to analyze")
    args = vars(ap.parse_args())

    # if video is given, use as input, or if video isn't given, default to live stream
    in_file = DEFAULT_STREAM
    if args["input"]:
        in_file = args["input"]

    # Initializations for CSV
    prev_time = time.time()
    direction = True

    # Initializations for detection
    multitracker = []
    frame_num = 0
    counter = 0

    # Check if video is openable
    video = cv2.VideoCapture(in_file)
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Check if video is readable
    print("Starting Initial Detection...")
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # begin detection + tracking cycle
    with open(output_file, "w") as f:
        writer = csv.writer(f, delimiter=',')

        while True:
            print("========NEW FRAME===============")
            # Read a new frame
            ok, frame = video.read()
            if not ok:
                break

            # count frame number
            frame_num += 1

            # resize frame to view better
            frame = cv2.resize(frame, (IM_WIDTH, IM_HEIGHT))

            # TODO: every so often should run detection TOUGH PART. SIGH.
            if frame_num % DETECTION_CYCLE == 0:

                print(">>> Re-doing detection...")
                ok, frame = video.read()
                if not ok:
                    print('Cannot read video file')
                    sys.exit()
                frame = cv2.resize(frame, (IM_WIDTH, IM_HEIGHT))

                # get the valid detection's bounding boxes
                detection_found, bboxes = detect_and_get_bboxes(frame)

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
                    indexes = get_assignments(tracker_centers, detection_centers, m)

                # the indexes of the matched pairs
                valid_trackers = []
                valid_detections = []
                data_remove = []

                for t,d in indexes:
                    valid_trackers.append(t)
                    valid_detections.append(d)

                # remove all unmatched trackers
                for i in range(0,len(current_boxes)):
                    if i not in valid_trackers:
                        data_remove.append(i)

                for i in reversed(data_remove):
                    multitracker.remove(current_boxes[i][0])
                    current_boxes.remove(current_boxes[i])
                    print("Tracker " + str(i) + " removed")

                # add all unmatched detections
                for i in range(0,len(bboxes)):
                    if i not in valid_detections:
                        multitracker.append(create_tracker(frame, bboxes[i]))
                        print("Box added: " + str(i))

            # Start timer
            timer = cv2.getTickCount()

            # Update tracker
            current_boxes = []
            failed_trackers = []
            for tracker in multitracker:
                ok, updated_box = tracker.update(frame)

                # Draw bounding box
                if ok:
                    # Tracking success
                    p1 = (int(updated_box[0]), int(updated_box[1]))
                    p2 = (int(updated_box[0] + updated_box[2]), int(updated_box[1] + updated_box[3]))
                    cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

                    current_boxes.append((tracker, updated_box))

                    updates[tracker] += 1
                    if updates[tracker] == TRACKING_BUFFER:
                        counter += 1

                else :
                    # Tracking failure
                    cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
                    failed_trackers.append(tracker)

            print("Original Number of Trackers:", len(multitracker))
            fail_count = len(failed_trackers)

            # REMOVING TRACKERS BASED UPON KCF REPORTED TRACKING FAILURES
            if not fail_count == 0:
                for bad_tracker in failed_trackers:
                    multitracker.remove(bad_tracker)
                    updates.pop(bad_tracker)
                print(">>>", fail_count,"TRACKING FAILURES OCCURRED! Number of trackers after removal:", len(multitracker))

            # Display tracker type and FPS on frame
            cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 170,50), 2);
            cv2.putText(frame, "Total: " + str(counter), (50, IM_HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.50, (0,0,0), 2);

            # Display result and write to CSV
            cv2.imshow("Multi Object Tracking", frame)
            prev_time = writeCSV(prev_time)

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27 : break


print("RESULTS: " + str(counter) + " people were detected walking through the frame.")
