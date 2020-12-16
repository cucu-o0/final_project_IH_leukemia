# Libraries
# Pandas and Numpy
import pandas as pd
import numpy as np

# Visualization tools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

# Sci-kit Learn

# Metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score
from sklearn.model_selection import validation_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# TensorFlow
import tensorflow as tf

# Keras
from keras import backend as K




# CONFUSION MATRIX
def confusion_mtx(y_test, y_pred, pal, xticks, yticks):

  '''
    Function that returns:
    - Viz of the confusion matrix
    - Results of the confusion matrix
  '''

  print(f'Confusion Matrix')
  cm = confusion_matrix(y_test, y_pred);
  # print(cm)
  
  print(45*'-')

  # Viz
  plt.figure(figsize=(5,4))
  res = sns.heatmap(cm, annot=True, vmin=0.0, vmax=1000, fmt='.1f', cmap=pal)
  
  plt.xticks([0.5,1.5], xticks, va='center')
  plt.yticks([0.5,1.5], yticks, va='center')
  plt.xlabel('PREDICT VALUES', fontsize=12)
  plt.ylabel('TEST VALUES', fontsize=12);

  
  # Results
  print(f'TN: {cm[0][0]} ({round(cm[0][0]*100/sum(cm[0]),2)}%)')  
  print(f'FN: {cm[1][0]} ({round(cm[1][0]*100/sum(cm[0]),2)}%)')
  
  print(f'TP: {cm[1][1]} ({round(cm[1][1]*100/sum(cm[1]),2)}%)')
  print(f'FP: {cm[0][1]} ({round(cm[0][1]*100/sum(cm[1]),2)}%)')
  
  
  # Sensitivity
  sens= cm[1][1]/(cm[1][1]+cm[1][0])
  # Specificity
  spec= cm[0][0]/(cm[0][0]+cm[0][1]) 
  return sens, spec




# TRAINING OVER EPOCHS
def plot_metric(model,epochs,epochs_es,col1,col2,w,h,step):
    
  """
  Function that return the visualization of:
  - Accuracy Curve (Training/Validation Set)
  - Loss Curve (Training/Validation Set)
  """

  plt.rcParams['figure.figsize'] = (w,h) 
  
  plt.subplot(1,2,1)  
  plt.plot(model.history['acc'],col1)  
  plt.plot(model.history['val_acc'],col2)  
  plt.xticks(np.arange(0, epochs_es+1,step))   
  plt.xlabel('Epochs')  
  plt.ylabel('Accuracy')  
  plt.title('Training Accuracy vs Validation Accuracy')  
  plt.legend(['train','validation'], loc='best')

  plt.subplot(1,2,2)  
  plt.plot(model.history['loss'],col1)  
  plt.plot(model.history['val_loss'],col2)  
  plt.xticks(np.arange(0, epochs_es+1, step))   
  plt.xlabel('Epochs')  
  plt.ylabel('Loss')  
  plt.title('Training Loss vs Validation Loss')  
  plt.legend(['train','validation'], loc='best')
  
 
  plt.show()
  

  
  
# ROC CURVE
def show_roc_curve(y_true, y_pred):
  
  '''
  Function that returns the ROC curve
  '''

  fpr, tpr, thresholds = roc_curve(y_true, y_pred)
  auc_val = auc(fpr, tpr)

  # plt.style.use(col)
  plt.figure(figsize=(9,5))
  plt.plot([0, 1], [0, 1], 'k--')
  plt.plot(fpr, tpr, label='AUC = {:.3f}'.format(auc_val), color='purple')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC curve')
  plt.legend(loc='best')
  plt.show()
  
  
  
  
# RESULTS
def recap(acc,er,sens,fpr,lim_acc, lim_tpr, lim_fpr):

  '''
  Function that resumes the results of the model:
  - Accuracy
  - Sensitivity (TPR)
  - Fall Out (FPR)
  '''  
  # Accuracy
  if acc > lim_acc:
    cap_acc= 'can correctly label'
  else:
    cap_acc= 'can NOT correctly label'

  # Sensitiviy
  if sens > lim_tpr:
    cap_sens= 'good'
  else:
    cap_sens= 'quite bad'

  # Specificity
  if fpr > lim_fpr:
    cap_spec= 'NOT correctly labels'
  else:
    cap_spec= 'correctly labels'

  # Accuracy
  print('RECAP')
  print(150*'-')
  print('ACCURACY')
  print(f'In our model, the Accuracy is {acc}% and the Error Rate is {er}%, so the model {cap_acc} a high percentage of the instances.')
  print(' ')

  # Sensitivity
  print('TPR_SENSITIVITY')
  print(f'The Sensitivity-TPR_True Positive Rate is {sens}%, which means that the model has a {cap_sens} capacity to detect the POSITIVE instances.')
  print(' ')
  print('FPR_FALL OUT')
  # Specificity
  print(f'Finally, the FPR_False Positive Rate is {fpr}%, which shows that the model {cap_spec} most of the NEGATIVE instances.')
  
  

  
  