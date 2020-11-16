#basic detection code using only jetson inference. Not as good as #using with OpenCV

import jetson_utils_python
import jetson_inference_python
import cv2
import time

timeStamp= time.time()
fpsFilter= 0

net= jetson_inference_python.detectNet('ssd-mobilenet-v2', threshold = 0.5)
dispW= 1280
dispH= 720
#cam = jetson_utils_python.gstCamera('/home/nvidia/Desktop/Project Files/CarsDrivingUnderBridge.mp4')
cam= jetson_utils_python.videoSource("/home/nvidia/Desktop/Project_Files/CarsDrivingUnderBridge.mp4")
display= jetson_utils_python.glDisplay()

while display.IsOpen():
    img, width, height= cam.Capture()
    detections= net.Detect(img, width, height)
    display.RenderOnce(img, width, height)
    dt= time.time()- timeStamp
    fps= 1/dt
    fpsFilter = .9*fpsFilter + .1*fps
    timeStamp= time.time()
    print(str(round(fps,1)+ ' fps'))