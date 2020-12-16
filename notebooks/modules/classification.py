# Libraries

# TensorFlow
import tensorflow as tf

# Keras
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, Flatten
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam




# Classification_v2
def classification_model_2(input_layer):
  
  '''
  Implementation of a Classification Model 
  
  Variables:
  - input = Output U-Net
  - output = Binary prediction  
  '''
  
  # Adding fully connected layers --> Classification
  x = MaxPooling2D(4)(input_layer)
  x = Flatten()(x)

  x = Dense(512, activation='relu')(x)
  x = Dropout(0.5)(x) # Fully connected layer is followed by a dropout to try to avoid overfitting.

  x = Dense(256, activation = 'relu')(x)
  x = Dropout(0.5)(x) 

  x = Dense(128, activation = 'relu')(x)
  x = Dropout(0.2)(x) 
  
  # Output Classification
  output = Dense(1, activation='sigmoid')(x)

  return output

