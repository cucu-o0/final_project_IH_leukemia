# Libraries
import numpy as np
from tqdm import tqdm
import glob
import cv2


# JOIN TRAINING FOLDERS
def join_folders(folder1, folder2, path):

  '''
  Function that returns the training array
  '''

  im_paths=[]

  # Diseased cells
  type1=[]
  # NO diseased cells
  type2=[]

  for t in folder1:
    for c in folder2:

      # List of paths with training images
      im_paths.append(path+t+'/'+c+'/*.bmp')

  for i in range(len(im_paths)):
    # Creation of a two new list of training_data
    if i%2 ==0:
      type1.extend(glob.glob(im_paths[i]))

    else:
      type2.extend(glob.glob(im_paths[i]))

  arr1 = np.array(type1)
  arr2 = np.array(type2)

  return arr1, arr2




# LOAD FILES
def load_data(path,img,lbl,extension):

  '''
  Function that loads arrays saved in a specific folder
  '''
  imgs = np.load(path+img+extension, allow_pickle=True, fix_imports=True)

  if lbl == 0:
    return imgs

  else:
    lbls = np.load(path+lbl+extension, allow_pickle=True, fix_imports=True)
    return imgs, lbls





# NUMPY-ARRAYS
def data_manipulation(f, c, in_path, lbl, w,h):

  '''
  Function that return numpy arrays of:
  - Images
  - Labels
  '''

  if c!=0:

    # TRAINING SET
    # Create path list and folder with all the images
    im_paths=[]
    folder=[]

    im_paths.append(in_path+f+'/'+c+'/*.bmp')

    for i in range(len(im_paths)):
      folder.extend(glob.glob(im_paths[i]))

    # Load and label train images
    Image = []
    Label = []

    for i in tqdm(range(0,len(folder))):
      img = cv2.imread(folder[i])
      Image.append(cv2.resize(img, (w,h)))
      Label.append(lbl)

    fin_array_image = np.array(Image)
    fin_array_label = np.array(Label)

    x= f+'_'+c+'_'+'images.npy'
    y= f+'_'+c+'_'+'labels.npy'

    # Save Numpy Arrays
    xx = np.save(x, fin_array_image, allow_pickle=True, fix_imports=True)
    yx = np.save(y, fin_array_label, allow_pickle=True, fix_imports=True)

    return x,y

  else:

    # VALIDATION SET
    # Create path list and folder with all the images
    im_paths=[]
    folder=[]

    # Paths with validation images
    im_paths.append(in_path+'/*.bmp')
    #print(im_paths)

    for i in range(len(im_paths)):
      folder.extend(glob.glob(im_paths[i]))

    # Load validation images
    Image = []

    for i in tqdm(range(0,len(folder))):
      Image.append(cv2.resize(cv2.imread(folder[i]), (w,h)))

    val_array_image = np.array(Image)

    # Save Numpy Arrays
    xx = np.save('val_images.npy', val_array_image, allow_pickle=True, fix_imports=True)

    return 'val_images.npy'
