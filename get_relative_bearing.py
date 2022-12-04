'''
This module contains functions and variables
for calculation the relative position to detection
seen from the camera.

Questions: magnus.rosand@gmail.com

'''


import cv2
import numpy as np

import utils
#Implementing neccesary global variables from utils
from utils import width_of_box
from utils import start_of_box

#Number of pixels horisontally
pixels_x_axis = 640

#This is the angle the camera is covering horisontally
angle_range_camera = 62.2

#Defining the the number of pixels to the middle of the screen
#Later on we use this variable to define "origo" in the camera
#This is to make the bearing relative to the drones bow
mid_camera = 320

#Defining relative bearing before it is declared as a global
rel_bear = 0.0

#Finding number of angles per pixel
angles_pr_pixel = (angle_range_camera)/(pixels_x_axis)

def rel_bear_calculator ():
    
    #Defining as global variable for communication to the other RPi
    global rel_bear
    
    #Calculating the centre of the box in the x-axis in comparison to the middle of the camera
    mid_ship = -(((utils.width_of_box)/2) + (utils.start_of_box)) + mid_camera
    
    #Rel_bear is calculated by multiplying the centre of the detection by angles_pr_pixel to find it in degrees 
    rel_bear = angles_pr_pixel * float(mid_ship)
    
    return rel_bear
    