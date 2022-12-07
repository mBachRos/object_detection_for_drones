'''
This module contains OS functions to
properly save needed data for further
training of the object detection model.

Questions: magnus.rosand@gmail.com

'''

import sys
import time
import os
import cv2


#Function to create new directory for each run
def make_directory_for_run(num_of_run):
        directory = "Directory_for_detectionrun_" + str(num_of_run)
        parent_dir = "/home/pi/examples/lite/examples/object_detection/raspberry_pi"
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)

#Function to create a log for each detection
#Each line in the txt file correlates to one detection
def write_log_to_directory(cat, cat_prob, num_det, num_of_run):
        #Find directory for this run
        directory_run = "Directory_for_detectionrun_" + str(num_of_run)
        #Defining path to this directory
        parent_dir_run = "/home/pi/examples/lite/examples/object_detection/raspberry_pi"
        #Coupling the path and the directory together
        path_dir_run = os.path.join(parent_dir_run, directory_run)
        #Define the name of the new file
        name_of_file = "Documented_detections.txt"
        #Coupling the name of the document and the path to parent directory
        complete_name = os.path.join(path_dir_run, name_of_file)
        #Append to the file
        f = open(complete_name, "a")
        #Write in the data we want to log
        f.write(cat + "\t" + str(cat_prob)+ "\t" + "test_img_num_" + str(num_det)+ "\n")
        f.close()
        
def write_detection_to_directory(num_det, img, num_of_run):
        #Finding the way to the directory for this run
        directory_run1 = "Directory_for_detectionrun_" + str(num_of_run) + "/"
        parent_dir_run1 = "/home/pi/examples/lite/examples/object_detection/raspberry_pi"
        path_dir_run1 = os.path.join(parent_dir_run1, directory_run1)
        
        #Sending pictures of detections to the directory
        cv2.imwrite(path_dir_run1 + "test_img_num_" + str(num_det) + ".png", img)