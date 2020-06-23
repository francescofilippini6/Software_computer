from orcasong.tools import FileConcatenator
from os import listdir
from os.path import isfile, join
import glob
mypath='/sps/km3net/users/ffilippi/ML/outputfolder1/'
infiles=glob.glob(mypath+'*.h5')

#infiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#print(infiles)
ci=FileConcatenator(infiles)
ci.concatenate(mypath+'concatenated.h5')
