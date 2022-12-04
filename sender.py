'''
This module contains functions for opening
and maintaining a TCP connection to RPi2

Questions: magnus.rosand@gmail.com

'''

import socket
import time
import json

def background_controller(data, detection_svetlana):
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        
        
        #Connect to the server
        try:
            #Connecting to the server-RPi via IP adress and port 9091
            s.connect(("192.168.136.63", 9091));
        except:
            pass
        
        try:
            #Round up value to make it easier to work with
            data = round(data,2)
            #Categorizing the data to make it simpler to work with
            data = { "bearing" : data, "Detection" : detection_svetlana}
            
            #sende min_ID og relativ bearing
            
            #Make it into json
            data = json.dumps(data)
            #Encode and send the data in json form
            s.send(data.encode());
            #Receiving message if the message was received
            dataFromServer = s.recv(1024)
        
            #Print to console
            print(dataFromServer.decode())
        except:
            pass
    