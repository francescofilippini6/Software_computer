import numpy as np
import sys
import pandas as pd
import km3pipe as kp
from os import listdir, makedirs, getcwd
from os.path import isfile, join, exists
from orcasong.core import FileBinner
from orcasong.tools import FileConcatenator




def path_generator(particle):
    """
    generating the path of the folders from which taking data and where saving the binned .h5 files:
    0 -> MUONs
    1 -> NEUTRINOs
    """
    #general_path='/sps/km3net/users/ffilippi/ML/'
    general_path=getcwd()+'/'
    #print(general_path)
    directory='o'
    if particle==0:   #taking the external argument  0 = muon
        inpath=general_path +'mupage_root_files_from_irods/'
        directory='outputfolder_mupage'
    elif particle==1:  #taking external argument    1=neutrino
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
    function generating the listing of the *.h5 files in the selected directory, returning the abs path
    """
    files = [join(mypath, f) for f in listdir(mypath) if f.endswith(".h5")]
    return(files)

def cut_zenith(particle):
    if particle == 0:
        print("Muons come only from -pi/2 to pi/2")
    if particle == 1:
        directions=[]
        tracks=0
        for filepath in listing(path_generator(particle)[0]):
            tracks = pd.read_hdf(filepath, '/mc_tracks')
            primaries = tracks.groupby('group_id').first()
            zeniths = kp.math.zenith(primaries.filter(regex='^dir_.?$'))
            primaries['zenith'] = np.cos(zeniths)
            primaries=primaries[primaries['zenith'] < 0]
     #       primaries.to_hdf('data.h5', key='df', mode='w')
            
        

def binning(particle):
    """
    function that takes the events of the .h5 file and create the histograms = "images", saved in the outpath
    """
    bin_edges_list = [
        ["pos_z", np.linspace(47, 695, 19)],
        ["time", np.linspace(-200, 1200, 281)],
        ["channel_id",np.linspace(-0.5,31.5,32)],
    ]
    
    fb = FileBinner(bin_edges_list) 
    fb.run_multi(listing(path_generator(particle)[0]),path_generator(particle)[1],True)
    

def concatenate(particle):
    """
    concatenate .h5 files created from binner in a single file: 1 for muon and 1 for neutrino  
    """
    outpath=path_generator(particle)[1]
    infiles=listing(outpath)
    ci=FileConcatenator(infiles)
    ci.concatenate(outpath+'concatenated.h5')
    

def concatenate_batch(particle,num_files):
    """
    concatenate .h5 files created from binner in different and smaller files
    """
    outpath=path_generator(particle)[1]
    infiles=listing(outpath)
    lists = np.array_split(np.array(infiles),num_files)
    counter=1
    for infile in lists:
        print(infile)
        counter+=1
        ci=FileConcatenator(infile)
        name='concatenated_'+str(counter)+'.h5'
        ci.concatenate(outpath+name)
    

if __name__ == "__main__":
    particle_selector=int(sys.argv[1])
    #path_generator(particle_selector)
    #cut_zenith(particle_selector)
    #listing()
    #binning()
    #concatenate(particle_selector)
    #concatenate_batch()
