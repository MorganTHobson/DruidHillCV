import sys
from cameralib import blinkreader
from time import sleep

__author__ = "Kavya Tumkur <ktumkur1@jhu.edu>"
__version__ = "1.0"

username = "##########"
password = "##########"

def main():
    """ instantiate blinkreader object and start processing frames """
    card = blinkreader.BlinkReader(username, password)

    try:
        card.reader()
        camera_hash = card.processor()
        camera_hash['outdoor'].video_to_file('/tmp/capture_blink.avi')
    except:
        card.terminate()

#start execution
if __name__ == '__main__':
    main()
