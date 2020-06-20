import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt    # our plotting module
import pandas as pd    # the main HDF5 reader
import numpy as np    # must have
import km3pipe as kp    # some KM3NeT related helper functions
import seaborn as sns    
import km3pipe.style
import scipy.misc
from PIL import Image
km3pipe.style.use("km3pipe")

#to read the x group in the h5 file
def printfile():
    with h5py.File('/sps/km3net/users/ffilippi/ML/outputfolder/numuCC_hist.h5','r') as hdf:
        base_items = list(hdf.items())
        print('Items in the base directory', base_items)
        g2 = np.array(hdf.get('x'))
        #print(g2)
        print('shape',g2.shape)
        nevt=g2.shape[0]
        print(nevt)
        ybins=g2.shape[1]
        xbins=g2.shape[2]
        #taking the first image in g2 array and looking at its rows
        for u in g2[0]:
            # print(u)
            print (max(u))
    return g2
    
def appendLabelDataset(filename):
    with h5py.File('/sps/km3net/users/ffilippi/ML/outputfolder/'+filename,'a') as hdf:
        g2 = np.array(hdf.get('x'))
        nevt=g2.shape[0]
        yl=np.ones(nevt)
        hdf.create_dataset('y', data=yl, dtype=yl.dtype)

def newfile(g2):
    """
    Normalization of the g2 ndarray.
    Can be applied to the original file, appending the normalized one, or can bw called on the newfie function
    appending howevert the normalized one
    """ 
    nevt=g2.shape[0]
    yl=np.ones(nevt)
    xl=np.divide(g2, 255)
    print(xl)
    print (len(yl))
    path='/sps/km3net/users/ffilippi/ML/outputfolder/numuCC_x_y.h5'
    with h5py.File(path, 'w') as hdf:
        hdf.create_dataset('x', data=xl, dtype=xl.dtype)
        hdf.create_dataset('y', data=yl, dtype=yl.dtype)


ci=printfile()
newfile(ci)
#appendLabelDataset()
