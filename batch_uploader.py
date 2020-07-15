import numpy as np
import h5py
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import keras
class DataGenerator(keras.utils.Sequence) :
  def __init__(self, list_IDs, labels, batch_size=32, dim=(18,280,31), n_channels=1,n_classes=2, shuffle=True):
    self.dim = dim
    self.batch_size = batch_size
    self.labels = labels
    self.list_IDs = list_IDs
    self.n_channels = n_channels
    self.n_classes = n_classes
    self.shuffle = shuffle
    self.on_epoch_end()  
    #def __init__(self, X_image, labels, batch_size) :
    #self.image = X_image
    #self.labels = labels
    #self.batch_size = batch_size
    
    
  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    
  def __getitem__(self, index):
    """ Selecting the elements in image and label corresponding to a batch size
    """
    # Generate indexes of the batch
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    
    # Find list of IDs
    list_IDs_temp = [self.list_IDs[k] for k in indexes]
    
    # Generate data
    X, y = self.__data_generation(list_IDs_temp)
    
    return X, y 
      
    
  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.list_IDs))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)

  def __data_generation(self, list_IDs_temp):
    'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    # Initialization
    X = np.empty((self.batch_size, *self.dim, self.n_channels))
    y = np.empty((self.batch_size), dtype=int)
    
    # Generate data
    for i, ID in enumerate(list_IDs_temp):
      # Store sample
      #X[i,] = np.load('outputfolder_neutrino/' + str(ID) + '.npy')
      #X[i,] = np.array(
      df = pd.read_hdf('outputfolder_neutrino/concatenated_2.h5', 'cleanuserbase', start=0, stop=10)
      # Store class
      y[i] = self.labels[ID]
      
      return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
