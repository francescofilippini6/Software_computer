from orcasong.tools import FileConcatenator
from os import listdir
from os.path import isfile, join
import glob
mypath='/sps/km3net/users/ffilippi/ML/'
infiles=('/sps/km3net/users/ffilippi/ML/outputfolder_neutrino/concatenated_x_y.h5','/sps/km3net/users/ffilippi/ML/outputfolder_mupage/concatenated_x_y.h5')

ci=FileConcatenator(infiles)
ci.concatenate(mypath+'finaldataset.h5')
