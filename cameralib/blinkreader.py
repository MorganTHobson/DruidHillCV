import sys
from time import sleep
from blinkpy import blinkpy

__author__ = "Kavya Tumkur <ktumkur1@jhu.edu>"
__version__ = "1.0"

class BlinkReader(object):
    """ BlinkReader class with methods supporting capture and processing
        of motion-detected blink events from cloud
    """

    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.flag = False
        self.blink = False

    def reader(self):
        """ ascertain credentials and read frames """
        try:
            self.blink = blinkpy.Blink(username = self.username, password = self.password)
            self.blink.start()
        except:
            self.flag = Flase
            self.terminate()

    def processor(self):
        """ process frames """
        bcam = self.blink.cameras
        return bcam

    def handlers(self):
        """ signal handlers """
        signal.signal(signal.SIGINT, self.terminate)
        signal.signal(signal.SIGTERM, self.terminate)
        return

    def terminate(self, signal=None, frame=None) :
        """ gracefully terminate """
        print("\n Quitting...")
        sys.exit(0)
