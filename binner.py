import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, exists
from orcasong.core import FileBinner
from orcasong.tools import FileConcatenator

general_path='/sps/km3net/users/ffilippi/ML/'

inpath=general_path +'mupage_root_files_from_irods/'
directory='outputfolder_mupage'

if not exists(directory):
    makedirs(directory)

outpath=general_path+directory+'/'


def listing(mypath):
    files = [join(mypath, f) for f in listdir(mypath) if f.endswith(".h5")]
    return(files)

def binning():
    bin_edges_list = [
        ["pos_z", np.linspace(0, 700, 19)],
        ["time", np.linspace(-100, 1000, 111)],
    ]
    fb = FileBinner(bin_edges_list) 
    fb.run_multi(listing(inpath),outpath,True)
    return(0)

def concatenate(path):
    infiles=listing(path)
    ci=FileConcatenator(infiles)
    ci.concatenate(path+'concatenated.h5')
    return(1)

binning()
concatenate(outpath)
