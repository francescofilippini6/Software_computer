import numpy as np
import sys
from os import listdir, makedirs, getcwd
from os.path import isfile, join, exists
from orcasong.core import FileBinner
from orcasong.tools import FileConcatenator



def path_generator():
    """
    generating the path of the folders from which taking data and where saving the binned .h5 files:
    0 -> MUONs
    1 -> NEUTRINOs
    """
    general_path='/sps/km3net/users/ffilippi/ML/'
    #general_path=getcwd()+'/'
    directory='o'
    particle_selector=int(sys.argv[1])
    if particle_selector==0:   #taking the external argument  0 = muon
        inpath=general_path +'mupage_root_files_from_irods/'
        directory='outputfolder_mupage'
    elif particle_selector==1:  #taking external argument    1=neutrino
        inpath=general_path +'nu_gehen/'
        directory='outputfolder_neutrino'
    else:
        raise NameError('NO other particles taken into account!!')  #if inserted another numerical value return an error (only two particles)
    
    if not exists(directory):   # creating the output directory if not already exist
        makedirs(directory)
    
    outpath=general_path+directory+'/'
    return (inpath, outpath)  # return a tuple with (inpath and outpath)
    
    

def listing(mypath):
    """
    function generating the listing of the *.h5 files in the selected directory
    """
    files = [join(mypath, f) for f in listdir(mypath) if f.endswith(".h5")]
    #new_start=files.index('/sps/km3net/users/ffilippi/ML/mupage_root_files_from_irods/mupage_348.h5')
    #files2=files[new_start:]
    return(files)

def binning():
    """
    function that takes the events of the .h5 file and create the histograms = "images", saved in the outpath
    """
    bin_edges_list = [
        ["pos_z", np.linspace(47, 695, 19)],
        ["time", np.linspace(-200, 1200, 281)],
        ["channel_id",np.linspace(-0.5,31.5,32)],
    ]
    
    fb = FileBinner(bin_edges_list) 
    fb.run_multi(listing(path_generator()[0]),path_generator()[1],True)
    return()

def concatenate():
    """
    function that concatenate all the binned files in a single one
    """
    outpath=path_generator()[1]
    infiles=listing(outpath)
    print(infiles[0])
    ci=FileConcatenator(infiles)
    ci.concatenate(outpath+'concatenated.h5')
    return()


if __name__ == "__main__":
    #listing()
    #binning()
    concatenate()
