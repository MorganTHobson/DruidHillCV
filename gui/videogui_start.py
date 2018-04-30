import tkinter as tk
import time
from PIL import Image, ImageTk
import threading
from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
import sys
import re
from time import sleep
import os

import numpy as np
import math
import six.moves.urllib as urllib
import tensorflow as tf
import csv
import collections
from io import StringIO
import matplotlib; matplotlib.use('Agg')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class DruidGUI(object):

    def __init__(self, title, path):
        self.master = tk.Tk()
        self.path = path
        self.master.title(title)
        self.master.resizable(width=tk.FALSE, height=tk.FALSE)
        self.master.geometry('1024x768')
        self.master.config(background="black")

        self.DRUID_LOCKED = True
        self.DRUID_ANALYTICS_LOCKED = True

        self.DRUID_START_THREAD = False
        self.DRUID_ANALYTICS_START_THREAD = False

        # initialize minimum confidence needed to call an object detection successful
        self.min_score_thresh = 0.5 #

        self.IM_WIDTH = 950
        self.IM_HEIGHT = 600
        self.DETECTION_CYCLE = 10 # how often to run the detection algo
        self.UNTRACKED_THRESH = 3 # how many detection cycles to permit unassociated trackers
        self.PIXEL_LIMIT = 50     # allowed distance between associated trackers and detections

        # check for url string entered in the GUI entry box and validate 
        self.compiled_pattern = re.compile(
            r'^(?:http|ftp)s?://' # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain
            r'localhost|' #localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # or ip
            r'(?::\d+)?' # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

        self.MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
        self.PATH_TO_CKPT = self.MODEL_NAME + '/frozen_inference_graph.pb'
        self.PATH_TO_LABELS = os.path.join('/Users/kavya/miniconda3/lib/python3.6/site-packages/tensorflow/models/research/object_detection/data', 'mscoco_label_map.pbtxt')
        self.NUM_CLASSES = 90

        # top frame for labels and description
        self.top_frame = tk.Frame(self.master, width=1024, height=50, bg="brown")
        self.top_frame.pack()
        self.top_frame.pack_propagate(0)

        self.w = tk.Label(self.top_frame, text="Druid Hill Park - Object Detector", bg="brown", fg="white")
        self.w.pack()

        self.video_area = tk.Frame(self.master, width=1024, height=648)
        self.video_area.pack()
        self.video_area.pack_propagate(0)

        self.v_frame = tk.Label(self.video_area)
        self.v_frame.pack()
        self.v_frame.pack_propagate(0)

        # bottom frame for buttons
        self.bottom_frame = tk.Frame(self.master, width=1024, height=50, bg="brown")
        self.bottom_frame.pack()
        self.bottom_frame.pack_propagate(0)

        self.status_frame = tk.Frame(self.master, width=1024, height=20, bg="yellow")
        self.status_frame.pack()
        self.status_frame.pack_propagate(0)

        self.s = tk.Label(self.status_frame, text="DruidGUI is ready ...", bg="black", fg="white")
        self.s.pack()

        self.start_button = tk.Button(self.bottom_frame, text="start", command=self.start_thread)
        self.start_button.pack(fill=tk.X, padx=5, pady=5, side=tk.LEFT)

        self.stop_button = tk.Button(self.bottom_frame, text="stop", command=self.stop_thread)
        self.stop_button.pack(fill=tk.X, padx=5, pady=5, side=tk.LEFT)

        self.detector_button = tk.Button(self.bottom_frame, text="show analysis",
                                         command=self.start_thread_b)
        self.detector_button.pack(fill=tk.X, padx=5, pady=5, side=tk.LEFT)

        self.quit_button = tk.Button(self.bottom_frame, text="quit", command=self.close_all)
        self.quit_button.pack(fill=tk.X, padx=5, pady=5, side=tk.LEFT)

        self.submit_button = tk.Button(self.bottom_frame, text="load", command=self.submit_video)
        self.submit_button.pack(side=tk.RIGHT, padx=10)
        self.entry_field = tk.Entry(self.bottom_frame)
        self.entry_field.insert(0, 'pedestrian_orig.mp4')
        self.entry_field.focus_set()
        self.entry_field.pack(padx=2, pady=5, side=tk.RIGHT)
        self.entry_label = tk.Label(self.bottom_frame, text="Video URL or File:", fg="white", bg="brown")
        self.entry_label.pack(side=tk.RIGHT)

        self.video_name = self.entry_field.get()
        self.cap = cv2.VideoCapture(self.video_name)

        # initialize the HOG descriptor/person detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self.thread = threading.Thread(target=self.start_feed)
        self.thread_b = threading.Thread(target=self.start_feed_b)

        if self.path != '/dev/null':
            self.capture = cv2.VideoWriter_fourcc(*'mp4v')
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
            self.writer = cv2.VideoWriter(self.path, self.capture, 20.0, (self.width, self.height))
        #
        self.detection_graph = tf.Graph()
        #
        self.master.mainloop()

    def submit_video(self):
        trace_video = False
        self.video_name = self.entry_field.get()
        if self.compiled_pattern.match(self.video_name):
            print("submitted video URL : ", self.video_name)
            trace_video = True
        elif os.path.exists(self.video_name):
            print("submitted video : ", self.video_name)
            trace_video = True
        else:
            print("ERROR retrieving video from camera, looping ...")
            message = "FAILED video capture - error with path or file: " + self.video_name
            self.s.configure(text=message)
            trace_video = False
        if trace_video:
            self.cap = cv2.VideoCapture(self.video_name)
            if not self.cap.isOpened():
                message = "FAILED - error with video capture: " + self.video_name
                self.s.configure(text=message)
                return False
            else:
                message = "initiated video capture - proceed with start: " + self.video_name
                self.s.configure(text=message)
                return True
        else:
            return False

    def start_thread(self):
        print("thread A startup .....")
        self.DRUID_LOCKED = False
        self.DRUID_ANALYTICS_LOCKED = True
        if not self.DRUID_START_THREAD and self.cap.isOpened():
            print("thread startup .....")
            if self.path != '/dev/null':
                print("start thread: ", self.writer)

            self.thread.daemon = 1
            self.thread.start()
            self.DRUID_START_THREAD = True
        else:
            message = "FAILED video capture : " + self.video_name
            self.s.configure(text=message)

    def start_thread_b(self):
        print("thread B startup .....")
        self.DRUID_ANALYTICS_LOCKED = False
        self.DRUID_LOCKED = True
        if not self.DRUID_ANALYTICS_START_THREAD:
            print("thread b startup .....")
            if self.path != '/dev/null':
                print("start thread b: ", self.writer)

            self.thread_b.daemon = 1
            self.thread_b.start()
            self.DRUID_ANALYTICS_START_THREAD = True

    def stop_thread(self):
        print(type(self.s))
        message = 'pausing video stream...'
        self.s.configure(text=message)
        self.DRUID_LOCKED = True
        self.DRUID_ANALYTICS_LOCKED = True
        print(self.thread)

    def close_all(self):
        print(type(self.s))
        #self.cap.release()
        if self.path != '/dev/null':
            self.writer.release()

        cv2.destroyAllWindows()
        sys.exit(1)

    def start_feed(self):
        if not self.DRUID_LOCKED:
            _, my_frame = self.cap.read()
            cv2image = cv2.cvtColor(my_frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            img_tk = ImageTk.PhotoImage(image=img)
            self.v_frame.img_tk = img_tk
            self.v_frame.configure(image=img_tk)
            if my_frame is not None:
                if self.path != '/dev/null':
                    self.writer.write(my_frame)

                message = 'capturing video stream...'
                self.s.configure(text=message)

        self.v_frame.after(5, self.start_feed)

    def start_feed_b(self):
        if not self.DRUID_ANALYTICS_LOCKED and self.submit_video():
            message = 'initiating object analytics...'
            self.s.configure(text=message)
            sleep(2)
             
            message = 'loading tensor model to memory, please wait...'
            self.s.configure(text=message)
            self.detection_graph = tf.Graph()

            with self.detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')

            message = 'loading object category maps, please wait...'
            self.s.configure(text=message)
            label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
            categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
            category_index = label_map_util.create_category_index(categories)

            message = 'tensorflow session started ...'
            self.s.configure(text=message)

            self.detection_graph.as_default()
            sess = tf.Session(graph=self.detection_graph)

            prev_time = time.time()

            # Initializations to detect first bounding box.
            ok = True
            frame = []
            bboxes = []

            # detect first bbox
            while True:
                # Read frame.
                ok, frame = self.cap.read()
                if not ok:
                    message = 'ERROR - reading the capture frames...'
                    self.s.configure(text=message)
                    self.close_all()
                #
                # initialize for use outside loop
                bboxes = [] # array of bbox tuples in format (x, y, w, h)
                #bbox = (287, 23, 86, 320)

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(frame, axis=0)
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

                # Each box represents a part of the image where a particular object was detected.
                boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

                # Each score represents how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=2)

                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)
                img_tk = ImageTk.PhotoImage(image=img)
                self.v_frame.img_tk = img_tk
                self.v_frame.configure(image=img_tk)

                # provide the boxes detected count to the status window
                boxes_squeezed = boxes[0]
                if boxes_squeezed.shape[0] is not 0:
                    scores_squeezed = scores[0]
                    classes_squeezed = classes[0]
                    eligible_exists = False

                    # grabbing all valid detection above confidence of min_score_thresh bboxes
                    for i in range(boxes_squeezed.shape[0]):
                        if scores_squeezed[i] < self.min_score_thresh:
                            break
                        if classes_squeezed[i] == 1:
                            ymin, xmin, ymax, xmax = tuple(boxes_squeezed[i].tolist()) # values are normalized from 0 to 1
                            bbox = (xmin*self.IM_WIDTH, ymin*self.IM_HEIGHT, (xmax - xmin)*self.IM_WIDTH, (ymax - ymin)*self.IM_HEIGHT) # bbox format: (x, y, w, h)

                            bboxes.append(bbox)
                            eligible_exists = True

                    if not eligible_exists:
                        print("Failed to find accurate enough detections. Moving onto next frame.")
                        detection_found = False

                    print("Number of valid bboxes found:", len(bboxes))
                    message = 'Number of valid detections....' + str(len(bboxes))
                    self.s.configure(text=message)
