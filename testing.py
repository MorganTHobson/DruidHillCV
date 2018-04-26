import os
import csv
from subprocess import Popen, PIPE

DATA_DIR = 'video_input'
VIDEOS = ['IMG_1295.MOV', 'IMG_1296.MOV', 'IMG_0900.MOV']

MIN_SCORE_THRESH = 0.6
IM_ZOOM = 0.5

if not os.path.exists(DATA_DIR):
    raise Exception('Data directory specified by DATA_DIR does not exist.')

data_format = ["DETECTION_CYCLE", "TRACKING_BUFFER", "UNTRACKED_THRESH", "PIXEL_LIMIT", "FILE", "COUNT"]
data_file = "data_output.csv"

def run_tests():
    # setup detection session
    detection_graph.as_default()
    sess = tf.Session(graph=detection_graph)

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

                            detector = DetectorTracker(sess, tuple(in_file, DETECTION_CYCLE, MIN_SCORE_THRESH, TRACKING_BUFFER, UNTRACKED_THRESH, PIXEL_LIMIT, IM_ZOOM))
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
    run_tests()
    # end