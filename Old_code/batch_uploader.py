import numpy as np
import h5py
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import keras
from os import getcwd
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
    self.opens()
    self.counter=0
  def opens(self):
      """ Open all files and prepare for read out. """
      filepath='/sps/km3net/users/ffilippi/ML/outputfolder_mupage/concatenated.h5'
      fil = h5py.File(filepath, "r")
      filepath1='/sps/km3net/users/ffilippi/ML/outputfolder_neutrino/concatenated.h5'
      fil2 = h5py.File(filepath1, "r")
      print("dentro open")
      filelist=[fil,fil2]
      self.filelist=filelist
      

  def close(self):
    """ Close all files again. """
    for f in list(self.fileslist):
      f.close()
 
  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    
  def __getitem__(self, index):
    """ 
    Selecting the elements in image and label corresponding to a batch size
    """
   
    self.counter+=1
    print("counter",self.counter)
    # Generate indexes of the batch
    #indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    #indexes = self.indexes[counter*self.batch_size:(counter+1)*self.batch_size]
    start_batch_index=self.counter*self.batch_size
    print('dentro get')
    #start_batch_index.append(index*self.batch_size)
    # Find list of IDs
    list_IDs_temp = [self.list_IDs[k] for k in range(self.counter,(self.counter+1)*self.batch_size)]
    print(list_IDs_temp)
    #labels = np.array([self.labels[k] for k in indexes])
    # Generate data
    X, y = self.__data_generation(start_batch_index)
    
    return X, y 
      
    
  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.list_IDs))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)
    
  #@profile
  def __data_generation(self,list_IDs_temp):
    'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    # Initialization
    X = np.empty((self.batch_size, *self.dim, self.n_channels))
    y = np.empty((self.batch_size), dtype=int)
    print('dentro data, prima di allocare')
    cc=0
    for a in list_IDs_temp:
      X[cc,]=(np.array(self.filelist[0]['x'][a])).reshape(18,280,31,1)
      if a > len(self.filelist[0]):
        X[cc,]=(np.array(self.filelist[1]['x'][a-(len(self.filelist[0]))])).reshape(18,280,31,1)
      
      y[cc]= (np.array(self.labels['id_'+str(a)])).reshape(1)
      cc+=1
    print('dentro data, dopo di allocare')
    print(X.shape)
    return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
  

if __name__=="__main__":
  ci=DataGenerator()
