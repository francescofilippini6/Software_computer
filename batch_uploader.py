import numpy as np
import h5py
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

class DataGenerator(keras.utils.Sequence) :
  
  def __init__(self, X_image, labels, batch_size) :
    self.image = X_image
    self.labels = labels
    self.batch_size = batch_size
    
    
  def __len__(self) :
    return (np.ceil(len(self.image) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx) :
    """ Selecting the elements in image and label corresponding to a batch size
    """
    batch_x = self.image[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    
    return np.array(batch_x), np.array(batch_y)


    #return np.array([
    #        resize(imread('/content/all_images/' + str(file_name)), (80, 80, 3))
    #            for file_name in batch_x])/255.0, np.array(batch_y)
