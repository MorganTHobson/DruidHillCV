""" trainer and detector utility for supporting cv2-based activities
    around identifying and tracking cars from real-time camera feed
"""
import cv2
import sys
from time import sleep

class Detector(object):
    """ Detector class with methods supporting capture
        and processing of live video feeds.
    """

    def __init__(self, hlsVideo, classifierFile):
        self.car_cascade = cv2.CascadeClassifier(classifierFile)
        self.capnode = cv2.VideoCapture(hlsVideo)
        self.flag = False

    def test_capture(self):
        """ ascertain that VideoCapture class can be used to read frames """
        if self.capnode.isOpened():
            self.flag, video_frame = self.capnode.read()
        else:
            self.flag = False

        if self.flag:
            return video_frame
        else:
            return self.flag

    def capture(self):
        """ capture video_frame by video_frame """
        self.flag, video_frame = self.capnode.read()

        if self.flag:
            # color converter convert frame to gray
            gray_video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)

            # gray_video_frame = video_frame
            # detect cars of different sizes
            cars = self.car_cascade.detectMultiScale(gray_video_frame, 1.1, 1)
            # To draw a rectangle in each cars
            for (x,y,w,h) in cars:
                cv2.rectangle(gray_video_frame,(x,y),(x+w,y+h),(0,0,255),2)
                return gray_video_frame
        else:
            return self.flag

    def handlers(self):
        """ signal handlers """
        signal.signal(signal.SIGINT, self.terminate)
        signal.signal(signal.SIGTERM, self.terminate)
        return

    def terminate(self, signal=None, frame=None):
        """ gracefully terminate """
        print("\nQuitting....")
        self.capnode.release()
        cv2.destroyAllWindows()
        sys.exit(0)
