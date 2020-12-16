# Libraries

# TensorFlow
import tensorflow as tf

# Keras
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.optimizers import Adam
from keras import backend as K


# U-Net
def model(input_layer, n_filters):
  
  '''
  Implementation of U-Net 
  
  Variables:
  - input = Original images 
  - output = Binary prediction
  '''
  
  # Contract Path
  conv1 = Conv2D(n_filters * 1, (3, 3), activation="relu", padding="same")(input_layer)
  conv1 = Conv2D(n_filters * 1, (3, 3), activation="relu", padding="same")(conv1)
  pool1 = MaxPooling2D((2, 2))(conv1)
  pool1 = Dropout(0.25)(pool1)

  conv2 = Conv2D(n_filters * 2, (3, 3), activation="relu", padding="same")(pool1)
  conv2 = Conv2D(n_filters * 2, (3, 3), activation="relu", padding="same")(conv2)
  pool2 = MaxPooling2D((2, 2))(conv2)
  pool2 = Dropout(0.5)(pool2)

  conv3 = Conv2D(n_filters * 4, (3, 3), activation="relu", padding="same")(pool2)
  conv3 = Conv2D(n_filters * 4, (3, 3), activation="relu", padding="same")(conv3)
  pool3 = MaxPooling2D((2, 2))(conv3)
  pool3 = Dropout(0.5)(pool3)

  conv4 = Conv2D(n_filters * 8, (3, 3), activation="relu", padding="same")(pool3)
  conv4 = Conv2D(n_filters * 8, (3, 3), activation="relu", padding="same")(conv4)
  pool4 = MaxPooling2D((2, 2))(conv4)
  pool4 = Dropout(0.5)(pool4)

  # Middle
  convm = Conv2D(n_filters * 16, (3, 3), activation="relu", padding="same")(pool4)
  convm = Conv2D(n_filters * 16, (3, 3), activation="relu", padding="same")(convm)
  
  # Expansive Path
  deconv4 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding="same")(convm)
  uconv4 = concatenate([deconv4, conv4])
  uconv4 = Dropout(0.5)(uconv4)
  uconv4 = Conv2D(n_filters * 8, (3, 3), activation="relu", padding="same")(uconv4)
  uconv4 = Conv2D(n_filters * 8, (3, 3), activation="relu", padding="same")(uconv4)

  deconv3 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
  uconv3 = concatenate([deconv3, conv3])
  uconv3 = Dropout(0.5)(uconv3)
  uconv3 = Conv2D(n_filters * 4, (3, 3), activation="relu", padding="same")(uconv3)
  uconv3 = Conv2D(n_filters * 4, (3, 3), activation="relu", padding="same")(uconv3)

  deconv2 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
  uconv2 = concatenate([deconv2, conv2])
  uconv2 = Dropout(0.5)(uconv2)
  uconv2 = Conv2D(n_filters * 2, (3, 3), activation="relu", padding="same")(uconv2)
  uconv2 = Conv2D(n_filters * 2, (3, 3), activation="relu", padding="same")(uconv2)

  deconv1 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
  uconv1 = concatenate([deconv1, conv1])
  uconv1 = Dropout(0.5)(uconv1)
  uconv1 = Conv2D(n_filters * 1, (3, 3), activation="relu", padding="same")(uconv1)
  uconv1 = Conv2D(n_filters * 1, (3, 3), activation="relu", padding="same")(uconv1)

  # Output U-Net
  output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
 
  return output_layer



# METRIC F1-SCORE
def get_f1(y_true, y_pred):
  
  '''
  Function that returns f1-score to evaluate keras model
  '''

  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  # Precision
  precision = true_positives / (predicted_positives + K.epsilon())
  # Recall
  recall = true_positives / (possible_positives + K.epsilon())
  # f1_score
  f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
  return f1_val







