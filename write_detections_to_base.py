#import detect
#from detect import image

from threading import Timer
import sys
import time
import os
import cv2

def write_to_base():
    #img_path = "Detection_img.png"
    img_path = "/home/pi/examples/lite/examples/object_detection/raspberry_pi/Detection_img.png"
    directory_detection = "/var/www/html/Detection_img.png"
    os.rename(img_path, directory_detection)
    
    Timer(2, write_to_base).start()
    
    #Not choosing this method to keep the program alright
    #img = cv2.imread(img_path)
    #os.chdir(directory_images)
    #cv2.imwrite("Detection_img.png", img)
    
        
        