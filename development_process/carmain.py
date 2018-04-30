import cv2
import sys
from detectorlib import cardetector
from time import sleep

# path to live nest video camera
hlsVideo = "https://stream-us1-alfa.dropcam.com/nexus_aac/591b131879304cedbb64634cc754c7a2/chunklist_w861715763.m3u8"

# classifier file for car
#classifierFile = 'https://github.com/andrewssobral/vehicle_detection_haarcascades/blob/master/cars.xml'
classifierFile = 'cars.xml'
target_window = 'jhu-people-car-project'

def displayframe(frame):
    """ resize the frame by 3 folds and display the processed frame """
    height, width = frame.shape
    nh = height
    nw = width
    #nh = height * 3
    #nw = width * 3
    frame = cv2.resize(frame, (nw, nh))
    # display the processed and resized frame
    cv2.imshow(target_window, frame)

def main():
    """ instantiate cardetector object and start processing frames """
    card = cardetector.Detector(hlsVideo, classifierFile)
    frame = False
    try:
        frame = card.test_capture()
        if type(frame) != bool:
            while True:
                #
                frame = card.capture()
                if type(frame) != bool:
                    displayframe(frame)
                    # Wait a milisec and loop again
                    cv2.waitKey(1)
                #
                else:
                    print("no more frames to process...")
                    break
            #
            print("all frames processed...")
            card.terminate()
        else:
            print("failed capturing frame..")
    except:
        card.terminate()

# start execution
if __name__ == '__main__':
    main()
