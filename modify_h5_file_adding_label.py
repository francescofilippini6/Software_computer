import numpy as np
import h5py
import sys
from binner import path_generator
def filename_gen(particle):
    """generation of the abs path of the existing file: concatenated and of the possible new one: concatenated_x_y.h5."""

    filename=path_generator(particle)[1]+'concatenated.h5'
    newfile=path_generator(particle)[1]+'concatenated_x_y.h5'
    
    return filename, newfile

def printfile(particle):
    """printing some statistics of the datasets in the file."""

    with h5py.File(filename_gen(particle)[0],'r') as hdf:
        base_items = list(hdf.items())
        print('Items in the base directory', base_items)
        g2=hdf['x']
        nevt=len(g2)
        zbins=len(g2[0])
        tbins=len(g2[0][0])
        pmtbins=len(g2[0][0][0])
        return nevt, zbins, tbins, pmtbins
    
def appendLabelDataset(particle):
    """appending to concatenated.h5 file a coloumn of labels."""

    with h5py.File(filename_gen(particle)[0],'a') as hdf:
        keys=list(hdf.keys())
        if 'y' in keys:
            print("labels already exists")
            hdf.close()
            return len(hdf['y'])
        else:
            lenofg2 = len(hdf['x'])
            print (lenofg2)
            if particle==0:
                yl=np.zeros(lenofg2)
            else:
                yl=np.ones(lenofg2)
            hdf.create_dataset('y', data=yl, dtype=yl.dtype)
            hdf.close()
            return len(yl) 

def newfile(particle):
    """Can be applied to the original file, or can be called on the newfie function appending however the normalized one."""
 
    with h5py.File(filename_gen(particle)[0],'r') as hdf:
        g2 = len(hdf['x'])
        if particle==0:
            yl=np.zeros(g2)
        else:
            yl=np.ones(g2)
    
        path=filename_gen(particle)[0]
        with h5py.File(path, 'w') as hdff:
            hdff.create_dataset('x', data=hdf['x'])
            hdff.create_dataset('y', data=yl)
        

if __name__=="__main__":
    typeofparticle = int(sys.argv[1])
    #newfile()
    printfile(typeofparticle)
    appendLabelDataset(typeofparticle)
