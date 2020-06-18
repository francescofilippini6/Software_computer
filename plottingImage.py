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

with h5py.File('numuCC_hist.h5','r') as hdf:
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
        imgplot = plt.imshow(g2[i],cmap="viridis")
        plt.colorbar()
        name='image'+str(i)+'done.jpg'
        plt.savefig(name)
        print(name)