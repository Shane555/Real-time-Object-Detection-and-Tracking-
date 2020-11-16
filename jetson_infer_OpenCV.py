#Program outputs bounded box coordinates and class ID based on #the pre-trained ssd-mobilenet-v2 from jetson inference
#Combines detection from Jetson Inference with customised #computer vision modules from OpenCV

import jetson_utils_python
import jetson_inference_python
import cv2
import time
import numpy as np

#take time stamp
timeStamp= time.time()
fpsFilter= 0

#setup inference
net= jetson_inference_python.detectNet('ssd-mobilenet-v2', threshold = 0.5)
dispW= 1280
dispH= 720
font= cv2.FONT_HERSHEY_SIMPLEX

#uses video stream input instead of webcam but can easily #replace with webcam cv2.VideoCapture("/home/nvidia/Desktop/Project_Files/CarsDrivingUnderBridge.mp4")
cam.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)

while True:
#get video frame by frame 
    _, img= cam.read()
    height= img.shape[0]
    width= img.shape[1]
#data conversion as jetson inference uses RGBA float 32 whereas #OpenCV uses BGR int
    frame= cv2.cvtColor(img, cv2.COLOR_BGR2RGBA).astype(np.float32)
#converts frame from numpy to cuda for jetson inference
    frame= jetson_utils_python.cudaFromNumpy(frame)

#detect objects in frame based on model
    detections= net.Detect(frame, width, height)
    for detect in detections:
        ID= detect.ClassID
        item= net.GetClassDesc(ID)
        top= int(detect.Top)
        left= int(detect.Left)
        bottom= int(detect.Bottom)
        right= int(detect.Right)
        cv2.rectangle(img, (left, top), (right, bottom), (0,0,255), 2)

#get time stamp to calculate fps
    dt= time.time()- timeStamp
    fps= 1/dt
#stabilise fps using 'low pass filter'
    fpsFilter = .9*fpsFilter + .1*fps
    timeStamp= time.time()
    cv2.putText(img, str(round(fpsFilter, 1))+ 'fps', (0,30), font, 1, (0,0,255), 2)
#show frame with bounded boxes
    cv2.imshow('detCam', img)
    cv2.moveWindow('detCam', 0, 0)

#typical OpenCV end program code
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()