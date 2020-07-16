import numpy as np
import h5py
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import keras
from memory_profiler import profile



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
  
  #@profile
  def __data_generation(self, list_IDs_temp):
    'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    # Initialization
    X = np.empty((self.batch_size, *self.dim, self.n_channels))
    y = np.empty((self.batch_size), dtype=int)
    filepath='/sps/km3net/users/ffilippi/ML/outputfolder_mupage/concatenated_2.h5'
    fillo = h5py.File(filepath, "r")
    info1 = fillo['x']
    info2 = fillo['y']
    counter = 0
    #partial=[]
    #print(len(list_IDs_temp))
    for a in list_IDs_temp:
      #print(counter)
      X[counter,]=(np.array(info1['group_id'==a])).reshape(18,280,31,1)
      #print('partial',partial)
      #X[counter,]=X[counter,].reshape(18,280,31,1)
      #X[counter,]=partial
      #print (len(X[0][0][0][0]))
      y[counter]= (np.array(info2['group_id'==a])).reshape(1)
      #y[counter]= y[counter].reshape(1)
      #y[counter] = partial2
      #print(y[counter])
      counter+=1
    fillo.close()
    return X, y

  def release_list(self,a):
    del a[:]
    del a

if __name__=="__main__":
  ci=DataGenerator()
