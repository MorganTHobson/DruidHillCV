import numpy as np
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
#from matplotlib import pyplot as plt
from PIL import Image

import cv2
import time
import csv
 
# OBJECT DETECTION IMPORTS
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# # TENSORFLOW OBJECT DETECTION MODEL PREPARATION (ONLINE TRAINING) =======================================================

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/tensorflow/research/object_detection/data', 'mscoco_label_map.pbtxt')
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

    # Define an initial bounding box through detection
    # Using TensorFlow Object Detection Model
    with open(output_file, "w") as f:
        writer = csv.writer(f, delimiter=',')

        # initial detection
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:

                while True:
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
                    boxes_squeezed = np.squeeze(boxes)
                    if boxes_squeezed.shape[0] is not 0:
                        scores = np.squeeze(scores)
                        eligible_exists = False

                        # grabbing all valid detection above confidence of min_score_thresh bboxes
                        for i in range(boxes_squeezed.shape[0]):
                            if scores is None or scores[i] > min_score_thresh:
                                ymin, xmin, ymax, xmax = tuple(boxes_squeezed[i].tolist()) # values are normalized from 0 to 1
                                #print("Detected Box Normalized Coordinates:", ymin, xmin, ymax, xmax)
                                bbox = (xmin*IM_WIDTH, ymin*IM_HEIGHT, (xmax - xmin)*IM_WIDTH, (ymax - ymin)*IM_HEIGHT) # bbox format: (x, y, w, h)
                                #print("Final Bounding Box (x,y,w,h):", bbox)

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

# TODO: you could give these IDs using a dict.
updates = {}
def create_tracker(frame, bbox):
    # require use of OpenCV 3.3 or above
    #if int(minor_ver) < 3:
    #    tracker = cv2.Tracker_create(tracker_type)
    #else:

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

def is_match(newbox_size, currbox_size, newbox_center, currbox_center):
    SIZE_THRESHOLD = 0.05 # at least within 5% of its size
    CENTER_THRESHOLD = 0.15 # currbox_center[0]*0.05 + currbox_center[1]*0.05

    sim_size = False
    sim_center = False

    size_diff = abs(newbox_size - currbox_size)/currbox_size
    print("SIZE_RATIO:", size_diff)

    if size_diff < SIZE_THRESHOLD:
        sim_size = True

    newbox_x, newbox_y = newbox_center
    currbox_x, currbox_y = currbox_center
    center_diff = abs(newbox_x - currbox_x)/currbox_x + abs(newbox_y - currbox_y)/currbox_y

    print("CENTER_DIFF:", center_diff)

    if center_diff < CENTER_THRESHOLD:
        sim_center = True

    if sim_size and sim_center:
        return True
    else:
        return False

def subtractBackground(frame):
    fgmask = cv2.cvtColor(fgbg.apply(frame),cv2.COLOR_GRAY2RGB)
    return (255 - frame) * fgmask

if __name__ == '__main__' :

    # # TRACKER SETUP=============================================================

    # Choose Tracker Type - Currently using KCF
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[2]

    # # DETECTION ================================================================

    # setup commandline argument parsing for input video
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--videos", required=True, help="filename of video to analyze")
    args = vars(ap.parse_args())

    # initialize minimum confidence needed to call an object detection successful
    min_score_thresh = 0.5 #TODO: init better, maybe through default in function call?

    IM_WIDTH = 950
    IM_HEIGHT = 600
    DETECTION_CYCLE = 50 # how often to run the detection algo

    # setup output CSV format and file name globals
    output_format = ["Time", "Type", "Direction", "Total"]
    output_file = "tensor_output.csv"

    # temporary for CSV output
    direction = True

    # Read video
    video = cv2.VideoCapture(args["videos"])
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Initializations to detect first bounding box.
    detection_found = False
    ok = True
    multitracker = []
    frame = []
    bboxes = []

    # Initialize background subtraction
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    # detect first bbox
    print("Starting Initial Detection...")
    while not detection_found:
        # Read frame.
        ok, frame = video.read()
        if not ok:
            print('Cannot read video file')
            sys.exit()

        # get the first valid detection's bounding boxes
        detection_found, bboxes = detect_and_get_bboxes(frame)

    # resize for display 
    frame = cv2.resize(frame, (IM_WIDTH, IM_HEIGHT))
 
    # add trackers to multitracker with first valid frame and each bounding box to track
    for bbox in bboxes:
        ok = multitracker.append(create_tracker(frame, bbox))

    # Track Objects
    frame_num = 0
    counter = 0
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
        frame = subtractBackground(frame)

        # TODO: every so often should run detection TOUGH PART. SIGH.
        if frame_num % DETECTION_CYCLE == 0:

            print(">>> Re-doing detection...")
            ok, frame = video.read()
            if not ok:
                print('Cannot read video file')
                sys.exit()
            frame = cv2.resize(frame, (IM_WIDTH, IM_HEIGHT))
            frame = subtractBackground(frame)

            # get the valid detection's bounding boxes
            detection_found, bboxes = detect_and_get_bboxes(frame)

            if detection_found:
                for newbox in bboxes:
                    # see if the bbox is already being tracked
                    # IDEA: check overlap ratio (harder)

                    # IDEA: check center and check size. if both are similar enough, then call it the same and dont create a new tracker for it
                    xmin, ymin, width, height = newbox[:]
                    newbox_size = width * height
                    newbox_center = (xmin + width/2, ymin + height/2)

                    # check against each existing box to see if same object
                    match = False
                    for tracker, box in current_boxes:
                        xmin_curr, ymin_curr, w_curr, h_curr = box[:]
                        currbox_size = w_curr * h_curr
                        currbox_center = (xmin_curr + w_curr/2, ymin_curr + h_curr/2)

                        match = is_match(newbox_size, currbox_size, newbox_center, currbox_center)
                        if match:
                            print("MATCH - Object already being tracked. Moving onto next newly detected box.")
                            break

                    if not match:
                        print("Object newly detected. Adding to multitracker!")
                        ok = multitracker.append(create_tracker(frame, newbox)) # how to append only new bounding boxes? IDs? currently just removing and appending all

                    print("Number of trackers after re-detection:", len(multitracker))

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
                if updates[tracker] == 20: # arbitrary threshold
                    counter += 1
            else :
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
                failed_trackers.append(tracker)

                # remove tracker that used to be tracking that bounding box
                #multitracker.remove(tracker)
                #print(">>> TRACKING FAILURE OCCURRED! Number of trackers after removal:", len(multitracker))

        print("Original Number of Trackers:", len(multitracker))
        fail_count = len(failed_trackers)

        # if failed at least once, remove those trackers
        if not fail_count == 0:
            for bad_tracker in failed_trackers:
                multitracker.remove(bad_tracker)
                updates.pop(bad_tracker)
            print(">>>", fail_count,"TRACKING FAILURES OCCURRED! Number of trackers after removal:", len(multitracker))

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Display tracker type and FPS on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
        cv2.putText(frame, "Total: " + str(counter), (50, IM_HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.50, (0,0,0), 2);

        # Display result
        cv2.imshow("Multi Object Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
print "RESULTS: " + str(counter) + " people were detected walking through the frame."
