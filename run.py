from subprocess import Popen, PIPE
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

from detector_tracker import DetectorTracker
import constants

def setup():
    # (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    sys.path.append("..")

    # ## Download Model
    opener = urllib.request.URLopener()
    opener.retrieve(constants.DOWNLOAD_BASE + constants.MODEL_FILE, constants.MODEL_FILE)
    tar_file = tarfile.open(constants.MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())

    # ## Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(constants.PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # ## Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(constants.PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=constants.NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories) 

    return detection_graph

# main function
if __name__ == '__main__' :
    # setup detection session
    detection_graph = setup()

    detection_graph.as_default()
    sess = tf.Session(graph=detection_graph)

    detector = DetectorTracker(sess, detection_graph, None) # pass in none to indicate we are not running on testing mode
    detector.analyze() #automatically generates output file

    # end