import sys
from os import getcwd
import os
import pytest
from hypothesis import given
import hypothesis.strategies as st
path=getcwd()
sys.path.insert(0, path)
from binner import listing, path_generator

class Test_listing:

    """class test for listing function. """
    def test_listing(self):
        """ Testing that if not empty, we have a list of strings."""
        #path=os.sep.join(os.getcwd().split(os.sep)[:-1])
        path=getcwd()
        path=path+'/outputfolder_mupage'
        files=listing(path)
        assert all(isinstance(x,str) for x in files)

    def test_2listing(self):
        """Testing that elements return the absolute path."""
        #path=os.sep.join(os.getcwd().split(os.sep)[:-1])
        path=getcwd()
        path=path+'/outputfolder_mupage'
        files=listing(path)
        assert os.path.isabs(files[0])

class Test_path_gen:
  
    """class test for path_generator function."""
    def test_path_generator1(self):
        """Testing neutrino path generation."""
        assert path_generator(1) == (getcwd()+'/nu_gehen/',getcwd()+'/outputfolder_neutrino/')
 
    def test_path_generator2(self):
        """Testing muon path generation."""
        assert path_generator(0) == (getcwd()+'/mupage_root_files_from_irods/',getcwd()+'/outputfolder_mupage/')
    
    @given(value=st.integers())
    def test_path_generator3(self,value):
        """Testing raising error if not 0 or 1 are inserted."""
        if value > 1:
            with pytest.raises(NameError):
                path_generator(value)

