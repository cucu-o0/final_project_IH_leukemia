# Libraries
import os
from tqdm import tqdm

import cv2 
import PIL 
from PIL import Image 

import numpy as np
import random
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns



#IMAGE CHECKER
def image_viz(path_1, path_2, title):

  '''
  Function that prints two images with their features
  '''
  
  # Var
  paths = [path_1, path_2]

  # Choose a random image from the top 1000
  num=random.choice(range(0,999))

  fig, ax = plt.subplots(ncols=len(paths),figsize=(10,10))

  for p,t in zip(paths,title):

    ind = paths.index(p)
    # Name of the first image
    image_0 = os.listdir(p)


    # Path of the location of the cell's image
    path_image = str(p)+'/'+str(image_0[num])

    # Image
    im = Image.open(path_image)
    img = cv2.imread(path_image)
    image = mpimg.imread(path_image)

    # Features
    detail= '''\
            {title}

            Name: {name}
            
            Format: {form} 
            Dimensions: {w} x {h} 
            Color composition: {col} 
                '''.format(title = t,
                          name=image_0[num], 
                          form=im.format, w=im.size[0], 
                          h=im.size[1], 
                          col=im.mode)
                
    # Viz
    ax[ind].set_title(detail, size=12, loc ='left')
    # ax[ind].set_xlabel(t, size=12)
    ax[ind].axis('off')
    ax[ind].yaxis.set_label_position('left')
    ax[ind].imshow(image)

  plt.show()

  
  
# LABELING-Training
def data_labels(folder,label,n_images):  

  '''
  Function that return a numpy arrays of images and labels
  '''

  Image = []
  Label = []

  for i in tqdm(range(n_images)):
    Image.append(cv2.imread(folder[i]))
    Label.append(label)
    
  return np.array(Image), np.array(Label) 



# SHUFFLE
def shuffle_selector(img,label,n):

  ''' Function that returns n shuffle images selected from a set'''

  shuffle_imgs=[]
  shuffle_lbls=[]

  for i in range(0, n):
    rand = np.random.randint(len(img))
    shuffle_imgs.append(img[rand])
    shuffle_lbls.append(label[rand])

  shuffle_imgs=np.array(shuffle_imgs)
  shuffle_lbls=np.array(shuffle_lbls)

  return shuffle_imgs, shuffle_lbls


  
# CELL COMPARATOR  
def shuffle_resized_images(Image,Label,w,h,n,title):

  '''
  Function that return a viz of n shuffle images with different resolutions
  '''
  
  fig, ax = plt.subplots(nrows = 1, ncols = n, figsize = (20,20))

  for i in range(0, len(Image)):

    img = cv2.resize(Image[i], (w,h))
    ax[i].imshow(img)
    ax[i].axis('off')
    
    a = Label[i]
    if a == 1:
        ax[i].set_title(title[1])
    else:
        ax[i].set_title(title[0])
        
  plt.show()

