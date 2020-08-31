from collections import defaultdict, Counter
from os import listdir, makedirs, getcwd
from os.path import  join, exists, basename
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
import km3pipe as kp
from km3pipe.dataclasses import Table
from km3pipe.math import pld3
from km3modules.common import StatusBar
import km3pipe.style
from  tables import nodes
km3pipe.style.use("km3pipe")

#path='/sps/km3net/users/ffilippi/ML/nu_gehen/'
directory='/sps/km3net/users/ffilippi/ML/regression/neutrino_binned/'

def listing(mypath):
    """
    function generating the listing of the *.h5 files in the selected directory, returning the abs path
    """
    files = [join(mypath, f) for f in listdir(mypath) if f.endswith(".h5")]
    return(files)



class Neutrino_zenith(kp.Module):
    """class for extracting the cos(zenith) angle from each neutrino event in MC file and append 
       to the binned files. Usual implementation of workflow implemented in km3pipe:
       configure
       process
       finish
    """
    def configure(self):
        self.zenith_list = []
        self.filename=self.get("filee")

    def process(self, blob):
        tracks = blob['McTracks']
        dir_x = tracks['dir_x'][0]
        dir_y = tracks['dir_y'][0]
        dir_z = tracks['dir_z'][0]
        dir_vector = (dir_x, dir_y,dir_z)
        zenith=np.cos(kp.math.zenith(dir_vector))
        self.zenith_list.append(zenith)
        return blob

    def finish(self):
        print(self.filename)
        with h5py.File(self.filename,'a') as hdf:
            lenofg2 = len(hdf['x'])
            yl=self.zenith_list
            hdf.create_dataset('y', data=yl)
            hdf.close()        




for i in listing(directory):
    a = basename(i)
    # print(a)
    originalname=a.split("_")
    e='/sps/km3net/users/ffilippi/ML/nu_gehen/'+originalname[0]+"_"+originalname[1]+"_"+originalname[2]+'.h5'
    #print(e)
    with h5py.File(i,'r') as check:
        keys=list(check.keys())
        if 'y' in keys:
            print(i,"labels already exists")
            check.close()
            continue
    pipe = kp.Pipeline()
    pipe.attach(kp.io.HDF5Pump, filename=e)
    pipe.attach(StatusBar, every=100)
    pipe.attach(Neutrino_zenith, filee=i)
    pipe.drain()

