import sys
from os import getcwd
path=getcwd()
sys.path.insert(0, path)
from modify_h5_file_adding_label import filename_gen, printfile

class Test_filename_gen:
    def test_filename1(self):
       assert filename_gen(1) == (getcwd()+'/outputfolder_neutrino/concatenated.h5',getcwd()+'/outputfolder_neutrino/concatenated_x_y.h5')

    def test_filename0(self):
        assert filename_gen(0) == (getcwd()+'/outputfolder_mupage/concatenated.h5',getcwd()+'/outputfolder_mupage/concatenated_x_y.h5')
       
def test_printfile():
    image=(printfile(0)[1],printfile(0)[2],printfile(0)[3])
    assert image == (18,280,31)
    
#def test_appendDataset():
#    assert appendLabelDataset(0) == printfile(0)[0]

    
