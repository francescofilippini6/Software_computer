import numpy as np
import h5py
import keras


class DataGenerator(keras.utils.Sequence):

  'Generates batch of data to be fed in the CNN'
  def __init__(self, list_IDs, labels, batch_size=32, dim=(18,280,31), n_channels=1,n_classes=2, shuffle=True):
    """Class constructor"""
    self.dim = dim
    self.batch_size = batch_size
    self.labels = labels
    self.list_IDs = list_IDs
    self.n_channels = n_channels
    self.n_classes = n_classes
    self.shuffle = shuffle
    self.opens()
    self.on_epoch_end()

  
  def opens(self):
    
    """ Open all files once and prepare for read out. """

    filepath='/sps/km3net/users/ffilippi/ML/outputfolder_mupage/concatenated.h5'
    fil = h5py.File(filepath, "r")
    filepath1='/sps/km3net/users/ffilippi/ML/outputfolder_neutrino/concatenated.h5'
    fil2 = h5py.File(filepath1, "r")
    filelist=[fil,fil2]
    self.filelist=filelist
      
  def __len__(self):
    
    """Denotes the number of batches per epoch"""

    return int(np.floor(len(self.list_IDs) / self.batch_size))
  
  def __getitem__(self, index):
    
    """Generate one batch of data, thanks to a list of indexes"""
    # Generate indexes of the batch
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    
    # Find list of IDs
    list_IDs_temp = [self.list_IDs[k] for k in indexes]
    
    # Generate data
    X, y = self.__data_generation(list_IDs_temp)
    
    return X, y

  def on_epoch_end(self):
    
    """Updates indexes after each epoch"""
    self.indexes = np.arange(len(self.list_IDs))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)
  
  def __data_generation(self, list_IDs_temp):
    
    """Generates data containing batch_size samples"""
    
    # Initialization
    X = np.empty((self.batch_size, *self.dim, self.n_channels))
    y = np.empty((self.batch_size), dtype=int)
    
    # Generate data
    for i, ID in enumerate(list_IDs_temp):
      # Store sample
      pos = int("".join(filter(str.isdigit, ID)))
      
      if pos < len(self.filelist[0]):
        X[i,]=(np.array(self.filelist[0]['x'][pos])).reshape(18,280,31,1)
      else:
        X[i,]=(np.array(self.filelist[1]['x'][pos-(len(self.filelist[0]))])).reshape(18,280,31,1)
        
      # Store class
      y[i] = self.labels[ID]
    #return X, y
    return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
#if __name__=="__main__":
#  ci=DataGenerator()
