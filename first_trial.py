from orcasong.core import FileBinner
import numpy as np
bin_edges_list = [
    ["pos_z", np.linspace(0, 700, 19)],
    ["time", np.linspace(-100, 2500, 1200)],
]

fb = FileBinner(bin_edges_list, det_file="KM3NeT_00000042_00008494.detx")
infiles=['/sps/km3net/users/ffilippi/ML/test.h5','/sps/km3net/users/ffilippi/ML/test2.h5','/sps/km3net/users/ffilippi/ML/numuCC.h5']
#fb.run('test.h5','outfile3333.h5')
fb.run_multi(infiles,'/sps/km3net/users/ffilippi/ML/outputfolder3',True)

