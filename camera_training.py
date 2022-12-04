import cv2
from time import sleep

cam = cv2.VideoCapture(0)


cv2.namedWindow("Bilde_taking")
img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("Bilde_taking", frame)
    
    k = cv2.waitKey(1)
    if k%256 == 27:
        #Pressed ESC
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        #SPACE pressed
        img_name = "Testpicture_nr_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        
cam.release()
cv2.destroyAllWindows()