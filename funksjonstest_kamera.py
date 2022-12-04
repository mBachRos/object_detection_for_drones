import sys
import time

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

#Nye linjer
#import os.path
from utils import midl
from utils import prob
from picamera import PiCamera

def capture_object():
    camera = PiCamera()
    camera.capture("/home/pi/examples/lite/examples/object_detection/raspberry_pi/img.jpg")