#!/usr/bin/env python
# coding: utf-8

# In[2]:



import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tarfile
import cv2
import pickle
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


path = '/home/ohernand/Downloads/256_ObjectCategories/'
destdir ='/home/ohernand/Master/DL/preprocessed_images/'
picklepath = 'G:/Caltech256/'


# In[4]:


os.chdir(path)


# In[5]:


folders = os.listdir()


# In[6]:


folder_paths = []
all_images = []
all_classes = []


# In[7]:


img_size = 128


# In[8]:


from PIL import Image

def make_square(image, min_size=img_size, fill_color=(0, 0, 0, 0)):
    size = (min_size, min_size)
    image.thumbnail(size, Image.ANTIALIAS)
    background = Image.new('RGB', size, (255, 255, 255, 0))
    background.paste(
        image, (int((size[0] - image.size[0]) / 2), int((size[1] - image.size[1]) / 2))
    )

    new_img = np.array(background)
    new_img.flatten()
    return new_img


# In[9]:


for folder in range(len(folders)):
    folder_paths = path+str(folders[folder])+str('/')
    
    os.chdir(folder_paths)
    image_in_folder = os.listdir()

    for image in range(len(image_in_folder)):
        img = Image.open(image_in_folder[image])
        img = make_square(img)
        
        all_images.append(img.flatten()/255)
        all_classes.append(folders[folder])


# In[11]:


all_images_df = np.asarray(all_images)


# In[13]:


sys.getsizeof(all_images_df)
all_images_df1 = all_images_df[:10000,:]
all_images_df2 = all_images_df[10000:20000,:]
all_images_df3 = all_images_df[20000:,:]
print('all_images_df1:'+str(sys.getsizeof(all_images_df1)))
print('all_images_df2:'+str(sys.getsizeof(all_images_df2)))
print('all_images_df3:'+str(sys.getsizeof(all_images_df3)))


os.chdir(picklepath)

pickle_out = open("pickle_all_images_df1.pickle","wb")
pickle.dump(all_images_df1, pickle_out)
pickle_out.close()

pickle_out = open("pickle_all_images_df2.pickle","wb")
pickle.dump(all_images_df2, pickle_out)
pickle_out.close()

pickle_out = open("pickle_all_images_df3.pickle","wb")
pickle.dump(all_images_df3, pickle_out)
pickle_out.close()

pickle_out = open("pickle_all_classes.pickle","wb")
pickle.dump(all_classes, pickle_out)
pickle_out.close()

