import cv2
import time
import picamera
import picamera.array
import test as m

from picamera.array import PiRGBArray
from picamera import PiCamera
from threading import Thread
import cv2
 
class PiVideoStream:
    def __init__(self, resolution=(320, 240), framerate=32):
        # initialize the camera and stream
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture,
            format="bgr", use_video_port=True)
 
        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.stopped = False
    
    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self
 
    def update(self):
        # keep looping infinitely until the thread is stopped
        for f in self.stream:
            # grab the frame from the stream and clear the stream in
            # preparation for the next frame
            self.frame = f.array
            self.rawCapture.truncate(0)
 
            # if the thread indicator variable is set, stop the thread
            # and resource camera resources
            if self.stopped:
                self.stream.close()
                self.rawCapture.close()
                self.camera.close()
                return

    def read(self):
        # return the frame most recently read
        return self.frame
 
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
'''
# Create camera interface
camera = picamera.PiCamera()
camera.resolution = (320, 240)
camera.framerate = 32
rawCapture = picamera.array.PiRGBArray(camera, size=(320, 240))
stream = camera.capture_continuous(rawCapture, format="bgr",use_video_port=True)
'''
vs = PiVideoStream().start()
time.sleep(2.0)

while True:
    '''
    for (i, f) in enumerate(stream):
        img = f.array
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_show = img.copy()
        cv2.rectangle(img_show,(110,70),(210,170),(0,255,0),3)
        cv2.imshow("Image",img_show)

        img = img[70:170,110:210]    
        img = cv2.resize(img, (28, 28))
        cv2.imshow("Crop",img)
        #print("process and show time : %f" % m.tstop())
        m.predict_local(img)
        key = cv2.waitKey(1) & 0xFF
        #rawCapture.truncate(0)
        # rint the top N most likely objects in image (default set to 5, change this in the function call above)
        #print topn
        time.sleep(0.01)
        rawCapture.truncate(0)
    '''
       
        
    img = vs.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_show = img.copy()
    cv2.rectangle(img_show,(110,70),(210,170),(0,255,0),3)
    
    img = img[70:170,110:210]    
    img = cv2.resize(img, (28, 28))    
    #print("process and show time : %f" % m.tstop())
    #img = img + 50
    img = cv2.bitwise_not(img)
    img[img > 160] = 230
    img[img < 150] = 0
    #print(img[2,2])
    cv2.imshow("Crop",img)
    res = m.predict_local(img)
    
    cv2.putText(img_show,"res : %d c: %.2f" % ( int(res[0]),res[1] ), (5,20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
    cv2.imshow("Image",img_show)

    key = cv2.waitKey(1) & 0xFF
    #rawCapture.truncate(0)
    # rint the top N most likely objects in image (default set to 5, change this in the function call above)
    #print topn
    
    time.sleep(0.01)
    
# do a bit of cleanup
cv2.destroyAllWindows()
stream.close()
rawCapture.close()
camera.close()
#vs.stop()