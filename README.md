# DruidHillCV

Pedestrian Detection for Druid Hill Park<br />
Jeesoo Kim, Morgan Hobson, Kavya Tumkur, Aidan Smith<br />

## Objective
Druid Hill Park in Baltimore is a historical park that has been maintained for decades. In order to continue its maintenance, we seek to aid the funding application process through park usage statistics provided by pedestrian detection and analysis. With the data that is gathered, we hope to prove the active use of Druid Hill Park, which will ultimately allow for the Parks and People Foundation of Baltimore to utilize this beautiful community space to its greatest potential. 

## Implementation
Python3 is the main language used. Although we had the option to use C++, we elected to use Python for quicker development. Implementation in C++ will likely show faster speeds, but for proof of concept, Python performs sufficiently well.

We use a variety of libraries in this software. The main object detection and tracking library that is used is in TensorFlow's Object Detection API: https://github.com/tensorflow/models/tree/master/research/object_detection

In addition, the main driver API utilized was OpenCV 3.4.0. This provided our video reading functions and created the framework in which we could incorporate object detection and tracking.

## Features


## Installation


## Credits
This project was developed for the project course 601.310 Software for Resilient Communities, led by Professor Yair Amir and (soon to be Dr.) Amy Babay at Johns Hopkins University. Much thanks to both for the guidance and advice towards creating a better project and inspiring us to constantly strive for nothing less than our highest potential. <br />

Partners for the project also included Jacob Green of Spread Concepts LLC, and the nonprofit Parks and People Foundation of Baltimore, particularly Valerie Rupp. Special thanks to Austin Reiter for helping with field advice and general guidance.