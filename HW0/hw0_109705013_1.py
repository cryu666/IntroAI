#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image, ImageDraw

img = Image.open("./data/image.png")
img.show()


# In[2]:


shape1 = [608, 616, 721, 505]
shape2 = [836, 557, 916, 477]
shape3 = [1073, 726, 1180, 600]
shape4 = [985, 468, 1042, 417]
shape5 = [994, 383, 1042, 346]
shape6 = [1042, 314, 1088, 282]

img1 = ImageDraw.Draw(img)  
img1.rectangle(shape1, outline="red")
img1.rectangle(shape2, outline="red")
img1.rectangle(shape3, outline="red")
img1.rectangle(shape4, outline="red")
img1.rectangle(shape5, outline="red")
img1.rectangle(shape6, outline="red")
img.show()
img.save("hw0_109705013_1.png")


# In[ ]:




