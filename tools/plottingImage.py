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

with h5py.File('/sps/km3net/users/ffilippi/ML/outputfolder_mupage/concatenated.h5','r') as hdf:
    base_items = list(hdf.items())
    print('Items in the base directory', base_items)
    g2 = np.array(hdf.get('x'))
    print(g2)
    print('shape',g2.shape)
    nevt=g2.shape[0]
    print(nevt)
    ybins=g2.shape[1]
    xbins=g2.shape[2]
    for i in range(0,nevt):
        A=np.sum(g2[i],axis = 2)
        print (A)
        imgplot = plt.imshow(A,cmap="viridis",aspect='auto',origin='lower')
        name='image'+str(i)+'done.jpg'
        plt.ylabel('z (m)')
        plt.xlabel('time (ns)')
        plt.title('z-t plane image')
        plt.savefig('/sps/km3net/users/ffilippi/ML/outputfolder_mupage/'+name)
        print(name)
