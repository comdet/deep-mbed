import cv2
import time
import picamera
import picamera.array
import test as m

# Create camera interface
camera = picamera.PiCamera()
rawCapture = picamera.array.PiRGBArray(camera)

while True:
    # Take the jpg image from camera
    #print "Capturing"
    #filename = '/home/pi/cap.jpg'
    # Show quick preview of what's being captured
    #camera.start_preview()
    #camera.capture(filename)
    #camera.stop_preview()
    
    # Run inception prediction on image
    #print "Predicting"
    #rawCapture=picamera.array.PiRGBArray(camera)
    camera.capture(rawCapture,format="rgb")
    img = rawCapture.array
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224,))
    cv2.imshow("Image",img)
    topn = m.predict_local(img)
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    # rint the top N most likely objects in image (default set to 5, change this in the function call above)
    #print topn
