# Pedestrian Detection Project for Druid Hill Park
Developed By: Jeesoo Kim, Morgan Hobson, Kavya Tumkur, Aidan Smith<br />

## Objective
Druid Hill Park in Baltimore is a historical park that has been maintained for decades. In order to continue its maintenance, we seek to aid the funding application process through park usage statistics provided by pedestrian detection and analysis. With the data that is gathered, we hope to prove the active use of Druid Hill Park, which will ultimately allow for the Parks and People Foundation of Baltimore to utilize this beautiful community space to its greatest potential. 

## Implementation
Python3 is the main language used. Although we had the option to use C++, we elected to use Python for quicker development. Implementation in C++ will likely show faster speeds, but for proof of concept, Python performs sufficiently well.

We use a variety of libraries in this software. The object detection and tracking library that is used is in TensorFlow's Object Detection API: https://github.com/tensorflow/models/tree/master/research/object_detection. We use SSD (single shot detection and the KCF (kernelized correlation filter) tracker in our implementation.

In addition, the main driver API utilized was OpenCV 3.4.0. This provided our video reading functions and created the framework in which we could incorporate object detection and tracking.

The GUI was created using TKinter for easier use of the program by non-programmers who simply seek to load a video and run analysis for data collection.

The live video stream that was used for testing is an HLS feed taken from a live Google Nest Cam Outdoor, placed near Druid Hill Park. Testing was also done on recorded video from smartphone cameras.

## Features and Performance
Coming soon.

## Installation
Detailed instructions coming soon.

## Credits
This project was developed for the project course 601.310 Software for Resilient Communities, led by Professor Yair Amir and (soon to be Dr.) Amy Babay at Johns Hopkins University. Much thanks to both for the guidance and advice towards creating a better project and inspiring us to constantly strive for nothing less than our highest potential. <br />

Partners for the project also included Jacob Green of Spread Concepts LLC, and the nonprofit Parks and People Foundation of Baltimore, particularly Valerie Rupp. Special thanks to Austin Reiter for helping with field advice and general guidance.

## Additional Information
Visit our project website here: http://www.dsn.jhu.edu/courses/cs310/baltimore/ <br />
Check out the Parks and People Foundation here: http://parksandpeople.org/
