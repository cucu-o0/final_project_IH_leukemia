# Libraries

import cv2 
import PIL 
from PIL import Image 

import numpy as np
import random
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns


# SHUFFLE TEST
def test_shuffle_selector(img,label,n,w):

  ''' 
  Function that returns n shuffle images selected from the test set
  '''

  shuffle_imgs=[]
  shuffle_imgs_res=[]
  shuffle_lbls=[]

  for i in range(0, n):
    rand = np.random.randint(len(img))
    img_res = cv2.resize(img[rand], (w,w))
    shuffle_imgs.append(img[rand])
    shuffle_imgs_res.append(img_res)
    shuffle_lbls.append(label[rand])

  return np.array(shuffle_imgs), np.array(shuffle_imgs_res), np.array(shuffle_lbls)


# PREDICTION
def prediction(n,img,lbl,pred):
  
  ''' 
  Function that compares the label of the real image with the prediction of the moel
  '''
  
  fig, ax = plt.subplots(nrows = 1, ncols = n, figsize = (15,15))

  for i in range(len(img)):
    ax[i].imshow(img[i])
    ax[i].axis('off')

    # Labels
    a = lbl[i]
    # Prediction
    b = pred[i]

    # Titles
    op1 = '''\
              Real Image: {re}
              Model Prediction: {pr}
                  '''.format(pr='LEUKEMIA', re='LEUKEMIA')
    op2 = '''\
              Real Image: {re}
              Model Prediction: {pr}
                '''.format(pr='HEALTHY', re='HEALTHY')

    op3 = '''\
              Real Image: {re}
              Model Prediction: {pr}
                '''.format(pr='LEUKEMIA', re='HEALTHY')
    op4 = '''\
              Real Image: {re}
              Model Prediction: {pr}
                '''.format(pr='HEALTHY', re='LEUKEMIA')
                  
    if a == b:
      if a == 1:
        ax[i].set_title(op1, size=12, loc ='left')
      else:
        ax[i].set_title(op2, size=12, loc ='left')

    else:
      if a == 1:
        ax[i].set_title(op3, size=12, loc ='left')
      else:
        ax[i].set_title(op4, size=12, loc ='left')

  plt.show()
