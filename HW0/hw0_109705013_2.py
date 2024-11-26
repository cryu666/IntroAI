#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np

# open video
stream = cv2.VideoCapture("./data/video.mp4")

if not stream.isOpened():
    print("No stream: ")
    exit()

# randomly selected frames in an array
frame_ids = stream.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=5)

# store selected frames in an array
frames = []
for fid in frame_ids:
    stream.set(cv2.CAP_PROP_POS_FRAMES, fid)
    
    # read the video frame
    success, frame = stream.read()
    
    # if no frames are returned, break the loop
    if not success:
        print(":( ")
        exit()
    frame = cv2.GaussianBlur(frame, (23, 23), 0)
    frames.append(frame)
    
# calculate the median along the time axis
median = np.median(frames, axis=0).astype(np.uint8)

# convert median frame to grayscale
# median = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
median[:, :, 0] = 0
median[:, :, 2] = 0


fps = stream.get(cv2.CAP_PROP_FPS)
width = int(int(stream.get(3))/4)
height = int(int(stream.get(4))/5)


# In[2]:


# reset frame number to 0
stream.set(cv2.CAP_PROP_POS_FRAMES, 0)

# loop  through the video frame
while True:
    # read the video frame
    success, frame = stream.read()
    
    # if no frames are returned, break the loop
    if not success:
        print("No more stream :( ")
        break
       
    frame_origin = frame.copy()
    # convert current frame to grayscale
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame[:, :, 0] = 0
    frame[:, :, 2] = 0    
        
    # calculate absolute difference of current frame and the median frame
    dif_frame = cv2.absdiff(frame, median)
    
    # treshold to binarize
    threshold, diff = cv2.threshold(dif_frame, 120, 255, cv2.THRESH_BINARY)
    
    # stack the frame horizontally
    hstacked_frames = np.hstack((frame_origin, diff))
    
    cv2.namedWindow("Remove the background", 0)
    cv2.resizeWindow("Remove the background", width, height)
    cv2.imshow("Remove the background", hstacked_frames)
    cv2.waitKey(10)
    
    # if the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) == ord("q"):
        break
    
# release video object
stream.release()

# destroy all windows
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




