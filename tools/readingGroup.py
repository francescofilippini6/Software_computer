import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt    # our plotting module
import pandas as pd    # the main HDF5 reader
import numpy as np    # must have
import km3pipe as kp    # some KM3NeT related helper functions
import seaborn as sns    
import km3pipe.style
km3pipe.style.use("km3pipe")

with h5py.File('outfile.h5','r') as hdf:
    base_items = list(hdf.items())
    print('Items in the base directory', base_items)
    g2 = np.array(hdf.get('x'))
    print(g2)
    print('shape',g2.shape)
    print(g2[0])
    #plt.hist2d(primaries.pos_x, primaries.pos_y, bins=100, cmap='viridis')
    #plt.xlabel('x [m]')
    #plt.ylabel('y [m]')
    #plt.title('2D Plane')
    #plt.colorbar()    
