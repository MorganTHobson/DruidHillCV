import os

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

DATA_DIR = 'video_input'
VIDEOS = ['IMG_1295.MOV', 'IMG_1296.MOV', 'IMG_0900.MOV']

# if not os.path.exists(DATA_DIR):
#     raise Exception('Data directory specified by DATA_DIR does not exist.')

MIN_SCORE_THRESH = 0.6
IM_ZOOM = 0.5
HOPKINS_STREAM = "https://stream-us1-charlie.dropcam.com/nexus_aac/44919a623bdd4086901ce942a60dbd27/chunklist_w948281787.m3u8"
DRUID_HILL_STREAM = "https://stream-us1-alfa.dropcam.com:443/nexus_aac/7838408781384ee7bd8d1cc11695f731/chunklist_w1479032407.m3u8"

#these should be in the testing file
DATA_FORMAT = ["DETECTION_CYCLE", "TRACKING_BUFFER", "UNTRACKED_THRESH", "PIXEL_LIMIT", "FILE", "COUNT"]
DATA_FILE = "data_output.csv"