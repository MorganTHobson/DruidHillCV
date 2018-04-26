import os
import csv
from subprocess import Popen, PIPE

DATA_DIR = 'video_input'
OUTPUT_DIR = 'output'
VIDEOS = ['IMG_1295.MOV', 'IMG_1296.MOV', 'IMG_0900.MOV']

MIN_SCORE_THRESH = 0.6
IM_ZOOM = 0.5

if not os.path.exists(DATA_DIR):
    raise Exception('Data directory specified by DATA_DIR does not exist.')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

output_format = ["DETECTION_CYCLE", "TRACKING_BUFFER", "UNTRACKED_THRESH", "PIXEL_LIMIT", "FILE", "COUNT"]
output_file = "data_output.csv"

with open(output_file, "w") as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['MIN_SCORE_THRESH = 0.6', 'IM_ZOOM = 0.5'])

    for DETECTION_CYCLE in ['30']:
        for UNTRACKED_THRESH in ['3']:
            for PIXEL_LIMIT in ['50']:
                for TRACKING_BUFFER in ['10']:
                    for in_file in VIDEOS:
                        print('Running on %s for parameters --detection-cycle %s --score-threshold %s --tracking-buffer %s --association-buff %s --pixel-limit %s --im-zoom %s' % (in_file, DETECTION_CYCLE, MIN_SCORE_THRESH, TRACKING_BUFFER, UNTRACKED_THRESH, PIXEL_LIMIT, IM_ZOOM))
                        file_path = os.path.join(DATA_DIR, in_file)

                        unformatted_cmd = 'python3 project.py --video %s --detection-cycle %s --score-threshold %s --tracking-buffer %s --association-buff %s --pixel-limit %s --im-zoom %s'
                        cmd = unformatted_cmd % (file_path, DETECTION_CYCLE, MIN_SCORE_THRESH, TRACKING_BUFFER, UNTRACKED_THRESH, PIXEL_LIMIT, IM_ZOOM)
                        # os.system(cmd)

                        p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE, stdin=PIPE)
                        output = p.stdout.readline()
                        print(output)

                        newrow = [DETECTION_CYCLE, TRACKING_BUFFER, UNTRACKED_THRESH, PIXEL_LIMIT, in_file, output]
                        writer.writerow(newrow)
                        f.flush()