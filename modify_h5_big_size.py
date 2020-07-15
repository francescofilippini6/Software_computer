import numpy as np
import h5py
import sys
import pandas as pd
import matplotlib.pyplot as plt    # our plotting module
import pandas as pd    # the main HDF5 reader
import numpy as np    # must have
import km3pipe as kp    # some KM3NeT related helper functions
import seaborn as sns    
import km3pipe.style
import scipy.misc
from PIL import Image
import glob


km3pipe.style.use("km3pipe")


typeofparticle = int(sys.argv[1])
#to read the x group in the h5 file
print(typeofparticle)
directory='o'
if typeofparticle==0:
    directory="outputfolder_mupage"
elif typeofparticle==1:
    directory="outputfolder_neutrino"
else:
    raise NameError('NO other particles taken into account!!')

general_path='/sps/km3net/users/ffilippi/ML/'
splitted_files=glob.glob(general_path+directory+'/'+"concatenated_*")
print(splitted_files)
filename=[]
outfile=[]
for i in splitted_files:
    filename.append(i)
    outfile.append(i+'aa')
print(filename)

def printfile():
    with h5py.File(filename,'r') as hdf:
        base_items = list(hdf.items())
        print('Items in the base directory', base_items)
        g2 = np.array(hdf.get('x'))
        #print(g2)
        print('shape:',g2.shape)
        nevt=g2.shape[0]
        print(nevt)
        ybins=g2.shape[1]
        xbins=g2.shape[2]
        #taking the first image in g2 array and looking at its rows
        for u in g2[0]:
            # print(u)
            print (max(u))
    return g2
    
def appendLabelDataset():
    for j in range(0,len(filename)):
        with h5py.File(filename[j],'a') as hdf:
            base_items = list(hdf.items())
            print('Items in the base directory', base_items)
            g2 = np.array(hdf.get('x'))

            nevt=g2.shape[0]
            if typeofparticle==0:
                yl=np.zeros(nevt)
            else:
                yl=np.ones(nevt)
            hdf.create_dataset('y', data=yl, dtype=yl.dtype)

def newfile():
    """
    Can be applied to the original file, appending the normalized one, or can be called on the newfie function
    appending however the normalized one
    """ 
    for j in range(0,len(filename)):
        with h5py.File(filename[j],'r') as hdf:
            base_items = list(hdf.items())
            print('Items in the base directory', base_items)
            g2 = np.array(hdf.get('x'))
    
            nevt=g2.shape[0]
            if typeofparticle==0:
                yl=np.zeros(nevt)
            else:
                yl=np.ones(nevt)
        
        with h5py.File(outfile[j], 'w') as hdf:
            hdf.create_dataset('x', data=g2, dtype=g2.dtype)
            hdf.create_dataset('y', data=yl, dtype=yl.dtype)
#from sklearn.model_selection import train_test_split
#def split_train_test():
    
#ci=printfile()
#newfile()
appendLabelDataset()
#appendLabelDataset()
