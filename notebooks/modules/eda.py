# Libraries
# import os

# import cv2 
# import PIL # python imaging library
# from PIL import Image 

# import random
# from sklearn.utils import shuffle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns



# CLASS DISTRIBUTION-GRAPH
def countplot_viz(df,target,label,palette,title,data):

  '''
  Barplot of the distribution of the classes
  '''

  fig = plt.figure(figsize=(6,4))

  # Plot
  viz=sns.countplot(x=target, data=df, palette=palette)

  # Show % values of each bar
  for a in viz.patches:
    percentage = '{:.1f}%'.format(100 * a.get_height()/len(df))
    viz.annotate(percentage, 
                (a.get_x() + a.get_width()/2, a.get_height()-10), 
                ha = 'center', va = 'center', 
                xytext = (0, -10), 
                textcoords = 'offset points', 
                fontsize=13,
                fontweight=500,
                color='white')

  # Add labels and titles:
  viz.set_xlabel('Target')
  viz.set(ylabel=None) 
  viz.tick_params(labelleft=False, left=False) 
  viz.set_title(title+' - '+data)
  
  plt.show();
  
  
  
# DATAFRAME
def dataframe (folder1,folder2,col,dim):

  '''Function that returns a DataFrame'''

  # Crate lists from numpy arrays
  folder1=list(folder1)
  folder2=list(folder2)

  # Create DF
  df=pd.Series(folder1+folder2).str.extract('([\w\-\.]+)$') 
  df['labels']  = df[0].str.contains('all').astype(int)
  df.rename(columns = {0:col}, inplace = True) 

  return df



# SUMMARY
def summary(df):

  '''
  Function that return the distribution of classes
  '''

  # 0: Healty cells
  # 1: Leukemia cells

  numeric=df.labels.value_counts()
  perc = round(numeric*100/len(df),1)

  print(f'LEUKEMIA vs HEALTHY CELLS')
  print(25*'-')
  print(f'Numeric')
  print(f'ALL (1): {numeric[1]}')
  print(f'HEM (0): {numeric[0]}')
  print(25*'-')
  print(f'Percentage')
  print(f'ALL%(1): {perc[1]}%')
  print(f'HEM%(0): {perc[0]}%')