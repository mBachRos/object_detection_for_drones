# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the object detection routine."""
import argparse
import sys
import time

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

#Importing needed python packages
import os
import socket

#Importing global variables and functions
from utils import cat_name
from utils import prob
import get_relative_bearing
from get_relative_bearing import rel_bear
import data_storage
import sender


"""Defining detection-counter.
This variable keeps track of the number of
Detections for each run"""
n=0
"""Defining number of runs the program has gone
through. This is defined as a global variable later on"""
run_num=""

#et_svet = False


def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
  """Continuously run inference on images acquired from the camera.
    
  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
  """
  
  #Defininf number of detections as a global variable
  global n
  
  
  
  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Initialize the object detection model
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  detection_options = processor.DetectionOptions(
      max_results=1, #Normal is 3 detections
      score_threshold=0.4) #Normal is 0.3
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)

  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      ) 

    counter += 1
    

    image = cv2.flip(image, 1)
    
    

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    """This image collects rgb_image and converts it to s image
    that the model can train opon"""
    train_img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    # Create a TensorImage object from the RGB image.
    input_tensor = vision.TensorImage.create_from_array(rgb_image)

    # Run object detection estimation using the model.
    detection_result = detector.detect(input_tensor)

    # Draw keypoints and edges on input image
    image = utils.visualize(image, detection_result)
    
    # Writes the picture, category and the probability of it being the category to directory for the run
    if (utils.prob >= 0.4):
        #Sending detectioninformation to right directory
        data_storage.write_log_to_directory(utils.cat_name, utils.prob, n, run_num)
        #Sending images of detections to right directory
        data_storage.write_detection_to_directory(n, train_img, run_num)
        
        """Writes last detection to local webserver.
        This is written to a php program to keep track of latest detection"""
        cv2.imwrite("/var/www/html/Detection_img.png", image)
        #Number of detections increases by one every time
        n += 1

    #Function to communicate detection of interest
    if ((utils.prob >= 0.8) & (utils.cat_name == "person")):
        det_svet = True
        #Calculate the relative bearing
        get_relative_bearing.rel_bear_calculator()
        rel_bear = get_relative_bearing.rel_bear

        """Sending the data to RPi2.
        This is the RPi running ROS and is a part of
        the drone-swarm"""
        sender.background_controller(rel_bear, det_svet)
    else:
        """If the detection is not sufficient for
        sending it will keep the values as false and 0.0"""
        sender.background_controller(0.0, False)
        
        

    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow('object_detector', image)

  cap.release()
  cv2.destroyAllWindows()


def main():
  
  #Global value for number of runs
  global run_num
  #Logging this value in a txt file
  f = open("run_file.txt", "r+")
  #Reading the variable from the file
  variable_f = f.read()
  #Printing value to screen to keep track
  print (variable_f)
  #Defining the variable as the one logged in the file
  run_num = str(variable_f)
  f.seek(0)
  """Write a new number for next run.
  In this way we prepare the system for the next run."""
  f.write(str(int(variable_f)+1))
  #Closing the file
  f.close()
  #Creating the directory for this file at startup of the program
  data_storage.make_directory_for_run(run_num)

  
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU))
  


if __name__ == '__main__':
  main()
