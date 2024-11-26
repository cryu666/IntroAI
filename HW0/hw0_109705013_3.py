#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt

img = cv2.imread("./data./image.png")

img_cw_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

cv2.imshow("Rotation", img_cw_90)
cv2.imwrite("img_roration.jpg", img_cw_90)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[2]:


img_scaled = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation = cv2.INTER_LINEAR)

cv2.imshow("Scaling", img_scaled)
cv2.imwrite("img_scaling.jpg", img_scaled)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[3]:


import numpy as np

height, width = img.shape[:2]
quarter_height, quarter_width = height / 4, width / 4

T = np.float32([[1, 0, quarter_width], [0, 1, quarter_height]])
img_translation = cv2.warpAffine(img, T, (width, height))
  
cv2.imshow("Translation", img_translation)
cv2.imwrite("img_translation.jpg", img_translation)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[4]:


img_flip_lr = cv2.flip(img, 1)

cv2.imshow("Flipping", img_flip_lr)
cv2.imwrite("img_flipping.jpg", img_flip_lr)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[5]:


img_crop = img[608:721, 505:721]

cv2.imshow("Cropping", img_crop)
cv2.imwrite("img_cropping.jpg", img_crop)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




