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

# Constant defaults
IM_ZOOM = 0.5
DETECTION_CYCLE = 10 # how often to run the detection algo
MIN_SCORE_THRESH = 0.5  # initialize minimum confidence needed to call an object detection successful
UNTRACKED_THRESH = 3 # how many detection cycles to permit unassociated trackers
PIXEL_LIMIT = 150     # allowed distance between associated trackers and detections
DEFAULT_STREAM = "https://stream-us1-charlie.dropcam.com:443/nexus_aac/44919a623bdd4086901ce942a60dbd27/playlist.m3u8"
#DEFAULT_STREAM = "https://stream-us1-alfa.dropcam.com:443/nexus_aac/7838408781384ee7bd8d1cc11695f731/chunklist_w1479032407.m3u8"

# # TENSORFLOW OBJECT DETECTION MODEL PREPARATION (ONLINE TRAINING) =======================================================

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/usr/local/lib/python3.6/site-packages/tensorflow/models/research/object_detection/data', 'mscoco_label_map.pbtxt')
#PATH_TO_LABELS = os.path.join('/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/tensorflow/research/object_detection/data', 'mscoco_label_map.pbtxt')
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

# setup commandline argument parsing for input video and parameters
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--video", dest="in_file", default=DEFAULT_STREAM, type=str, help="filename of video to analyze")
    ap.add_argument("-d", "--detection-cycle", dest="DETECTION_CYCLE", default=DETECTION_CYCLE, type=int, help="how often to run the detection algorithm")
    ap.add_argument("-s", "--score-threshold", dest="MIN_SCORE_THRESH", default=MIN_SCORE_THRESH, type=int, help="minimum confidence needed to declare successful detection")
    ap.add_argument("-a", "--association-buff", dest="UNTRACKED_THRESH", default=UNTRACKED_THRESH, type=int, help="number of detections to check before destroying a tracker")
    ap.add_argument("-p", "--pixel-limit", dest="PIXEL_LIMIT", default=PIXEL_LIMIT, type=int, help="number of detections to check before destroying a tracker")
    ap.add_argument("-z", "--im-zoom", dest="IM_ZOOM", default=IM_ZOOM, type=float, help="factor of image zoom")

    args = ap.parse_args()
    check_args(args)

    return args

# error checking arguments
def check_args(args):
    if args.in_file != DEFAULT_STREAM:
        if not os.path.exists(args.in_file):
            raise Exception("model file specified by --model-file does not exist.")
    # TODO: should also check that all number arguments are positive numbers

# detect object and return array of bbox tuples in format (x, y, w, h)
def detect_and_get_bboxes(frame, min_score_thresh):

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
            if scores_squeezed[i] < min_score_thresh:
                break
            if classes_squeezed[i] == 1:
                ymin, xmin, ymax, xmax = tuple(boxes_squeezed[i].tolist()) # values are normalized from 0 to 1
                bbox = (xmin*im_width, ymin*im_height, (xmax - xmin)*im_width, (ymax - ymin)*im_height) # bbox format: (x, y, w, h)

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

# write data to CSV file
def writeCSV(prev_time, prev_count, total_count):
    # WRITING TO CSV ================== MAKE SEPARATE FUNCTION
    current_time = time.time()
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
def get_assignments(tracker_centers, detection_centers, m, association_limit):
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

    indexes = m.compute(matrix)

    for x,y in indexes:
        if matrix[x][y] < association_limit:
            limited_indexes.append((x,y))

    return limited_indexes

# re-run detection on video frame
def run_detection(frame, pixel_limit, untracked_thresh, min_score_thresh):
    # how many "new" people were found this time
    count = 0

    # get the valid detection's bounding boxes
    detection_found, bboxes = detect_and_get_bboxes(frame, min_score_thresh)

    detection_centers = []
    old_centers = []

    # get indexes of associated boxes
    indexes = []
    if (len(current_boxes) != 0 and len(bboxes) != 0):
        # generate centers
        for newbox in bboxes:
            xmin, ymin, width, height = newbox[:]
            center = (xmin + width/2, ymin + height/2)
            detection_centers.append(center)
        for box in current_boxes:
            xmin, ymin, width, height = box[:]
            center = (xmin + width/2, ymin + height/2)
            old_centers.append(center)
        indexes = get_assignments(old_centers, detection_centers, m, pixel_limit)

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
            untracked_cycles[current_boxes[i]] += 1
            if untracked_cycles[current_boxes[i]] >= untracked_thresh:
                data_remove.append(i)
        else:
            data_remove.append(i)
            print("Found a match for " + str(i))

    for i in reversed(data_remove):
        current_boxes.remove(current_boxes[i])
        print("Detection " + str(i) + " removed")

    # add all unmatched detections
    for i in range(0,len(bboxes)):
        if i not in valid_detections:
            count += 1
        current_boxes.append(bboxes[i])
        untracked_cycles[bboxes[i]] = 0
        print("Box added: " + str(i))

    return count

def get_video_stream(in_file):
    # Check if video is openable
    video = cv2.VideoCapture(in_file)
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # get size of video to scale down size
    im_width = video.get(3)
    im_height = video.get(4)
    im_width = int(im_width * args.IM_ZOOM)
    im_height = int(im_height * args.IM_ZOOM)

    # Check if video is readable
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    return video, im_width, im_height

# main function
if __name__ == '__main__' :

    # Hungarian Algorithm object
    m = Munkres()

    # setup detection session
    detection_graph.as_default()
    sess = tf.Session(graph=detection_graph)

    # get command line arguments
    args = get_args()

    # Initializations for CSV
    prev_time = time.time()
    direction = True

    # Initializations for detection
    current_boxes = []
    untracked_cycles = {}
    frame_num = 0
    counter = 0
    prev_count = 0

    video, im_width, im_height = get_video_stream(args.in_file)

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
            frame = cv2.resize(frame, (im_width, im_height), interpolation = cv2.INTER_LINEAR)

            # Re-do detection every n-th frame
            if frame_num % args.DETECTION_CYCLE == 0:
                print(">>> Re-doing detection...")

                ok, frame = video.read()
                if not ok:
                    print('Cannot read video file')
                    sys.exit()
                frame = cv2.resize(frame, (im_width, im_height), interpolation = cv2.INTER_LINEAR)

                counter += run_detection(frame, args.PIXEL_LIMIT, args.UNTRACKED_THRESH, args.MIN_SCORE_THRESH)

            # Draw the boxes
            for box in current_boxes:
                p1 = (int(box[0]), int(box[1]))
                p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

            # Display detection cycles, allowed unassociations, and total count of people on frame
            cv2.putText(frame, str(args.DETECTION_CYCLE) + " Cycles Per Detection", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (64,64,64), 2);
            cv2.putText(frame, str(args.UNTRACKED_THRESH) + " Unassociations Allowed", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (64,64,64), 2);
            cv2.putText(frame, "Total Pedestrians = " + str(counter), (50, im_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2);

            # Display result and write to CSV
            cv2.imshow("Pedestrian Detection", frame)
            prev_time, prev_count = writeCSV(prev_time, prev_count, counter)

            # Exit conditions
            k = cv2.waitKey(1) & 0xff
            if k == 27 : #ESC
                cv2.destroyAllWindows()
                break
            if cv2.getWindowProperty("Pedestrian Detection", cv2.WND_PROP_VISIBLE) < 1: # X key on window
                break

    cv2.destroyAllWindows()


print("RESULTS: " + str(counter) + " people were detected walking through the frame.")
