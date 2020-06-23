from orcasong.core import FileBinner
import numpy as np

bin_edges_list = [
    ["pos_z", np.linspace(0, 700, 19)],
    ["time", np.linspace(-100, 1000, 111)],
]

fb = FileBinner(bin_edges_list) #,det_file="KM3NeT_00000042_00008494.detx")
infiles=['/sps/km3net/users/ffilippi/ML/nu_gehen/nu_mu8444.h5','/sps/km3net/users/ffilippi/ML/nu_gehen/nu_mu8445.h5','/sps/km3net/users/ffilippi/ML/nu_gehen/nu_mu8446.h5','/sps/km3net/users/ffilippi/ML/nu_gehen/nu_mu8447.h5','/sps/km3net/users/ffilippi/ML/nu_gehen/nu_mu8448.h5','/sps/km3net/users/ffilippi/ML/nu_gehen/nu_mu8449.h5','/sps/km3net/users/ffilippi/ML/nu_gehen/nu_mu8450.h5','/sps/km3net/users/ffilippi/ML/nu_gehen/nu_mu8451.h5','/sps/km3net/users/ffilippi/ML/nu_gehen/nu_mu8452.h5','/sps/km3net/users/ffilippi/ML/nu_gehen/nu_mu8453.h5','/sps/km3net/users/ffilippi/ML/nu_gehen/nu_mu8454.h5']
fb.run_multi(infiles,'/sps/km3net/users/ffilippi/ML/outputfolder1',True)

