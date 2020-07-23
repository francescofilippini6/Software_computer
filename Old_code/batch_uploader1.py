import numpy as np
import h5py
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import keras
from os import getcwd
from memory_profiler import profile



class DataGenerator(keras.utils.Sequence) :
  def __init__(self, list_IDs, labels, batch_size=64, dim=(18,280,31), n_channels=1,n_classes=2, shuffle=True):
    self.dim = dim
    self.batch_size = batch_size
    self.labels = labels
    self.list_IDs = list_IDs
    self.n_channels = n_channels
    self.n_classes = n_classes
    self.shuffle = shuffle
    self.on_epoch_end()  
    self.file_mupage=self.reader()[0]
    self.file_neutrino=self.reader()[1]
    
  def open(self):
    """ Open all files and prepare for read out. """
    for input_key, file in self.files_dict.items():
      self._files[input_key] = h5py.File(file, 'r')
      self._store_file_length()
      self._store_batch_indices()
      
  def close(self):
    """ Close all files again. """
    for f in list(self._files.values()):
      f.close()
      
  def get_x_values(self, start_index):
    """
    Read one batch of samples from the files and zero center.
    Parameters
    ----------
    start_index : int
    The start index in the h5 files at which the batch will be read.
    The end index will be the start index + the batch size.
    Returns
    -------
    x_values : dict
    One batch of data for each input file.
    """
    x_values = {}
    for input_key, file in self._files.items():
      x_values[input_key] = file[self.key_x_values][start_index: start_index + self._batchsize]
      if self.xs_mean is not None:
        x_values[input_key] = np.subtract(x_values[input_key],self.xs_mean[input_key])
    return x_values
  def get_y_values(self, start_index):
    """
    Get y_values for the nn. Since the y_values are hopefully the same
    for all the files, use the ones from the first. TODO add check
    Parameters
    ----------
    start_index : int
    The start index in the h5 files at which the batch will be read.
    The end index will be the start index + the batch size.
    Returns
    -------
    y_values : ndarray
    The y_values, right from the files.
    """
    first_file = list(self._files.values())[0]
    try:
      y_values = first_file[self.key_y_values][
        start_index:start_index + self._batchsize]
    except KeyError:
      # can not look up y_values, lets hope we dont need them
      y_values = None
    return y_values

  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    
  def __getitem__(self, index):
    """ 
    Selecting the elements in image and label corresponding to a batch size
    """
    print(index)
    # Generate indexes of the batch
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    
    # Find list of IDs
    list_IDs_temp = [self.list_IDs[k] for k in indexes]
    #labels = np.array([self.labels[k] for k in indexes])
    # Generate data
    X, y = self.__data_generation(list_IDs_temp)
    
    return X, y 
      
  def reader(self):
    filepath=getcwd()+'/outputfolder_mupage/concatenated.h5'
    fil = h5py.File(filepath, "r")
    #image1 = fil['x'][position:position+batch_size]
    #image2 = fil['y'][position:position+batch_size]
    # fil.close()
    filepath1=getcwd()+'/outputfolder_neutrino/concatenated.h5'
    fil2 = h5py.File(filepath, "r")
    #image3 = fil2['x'][position:position+batch_size]
    #image4 = fil2['y'][position:position+batch_size]
    #file2.close()
    return fil, fil2
  
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
    counter = 0
    for a in list_IDs_temp:
      X[counter,]=(np.array(self.file_mupage['x']['group_id'==a])).reshape(18,280,31,1)
      if a not in self.file_mupage['event_info']['group_id']:
        X[counter,]=(np.array(self.file_neutrino['x']['group_id'==a])).reshape(18,280,31,1)
      
      y[counter]= (np.array(self.file_mupage['y']['group_id'==a])).reshape(1)
      if a not in self.file_mupage['group_id']:
        y[counter]=(np.array(self.file_neutrino['y']['group_id'==a])).reshape(1)
         
      #y[counter]= y[counter].reshape(1)
      #y[counter] = partial2
      #print(y[counter])
      counter+=1
    return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


if __name__=="__main__":
  ci=DataGenerator()
