import numpy as np
import sys
from os import getcwd
import os
import pytest
import unittest
from hypothesis import given
import hypothesis.strategies as st
path=getcwd()
sys.path.insert(0, path)
from binner import listing, path_generator, binning

class Test_listing:
    """
    class test for listing function
    """
    def test_listing(self):
        """
        testing that if not empty, we have a list of strings
        """
        #path=os.sep.join(os.getcwd().split(os.sep)[:-1])
        path=getcwd()
        path=path+'/outputfolder_mupage'
        
        files=listing(path)
        assert all(isinstance(x,str) for x in files)

    def test_2listing(self):
        """
        testing that elements return the absolute path 
        """
        
        #path=os.sep.join(os.getcwd().split(os.sep)[:-1])
        path=getcwd()
        path=path+'/outputfolder_mupage'
                
        files=listing(path)
        assert os.path.isabs(files[0])

class Test_path_gen:
    """
    class test for path_generator function
    """
   
    def test_path_generator1(self):
        """
        testing neutrino path generation
        """
        # assert path_generator(1) == ('/sps/km3net/users/ffilippi/ML/test/nu_gehen/','/sps/km3net/users/ffilippi/ML/test/outputfolder_neutrino/')
        assert path_generator(1) == ('/sps/km3net/users/ffilippi/ML/nu_gehen/','/sps/km3net/users/ffilippi/ML/outputfolder_neutrino/')
 
    def test_path_generator2(self):
        """
        testing muon path generation
        """
        # assert path_generator(0) == ('/sps/km3net/users/ffilippi/ML/test/mupage_root_files_from_irods/','/sps/km3net/users/ffilippi/ML/test/outputfolder_mupage/')
        assert path_generator(0) == ('/sps/km3net/users/ffilippi/ML/mupage_root_files_from_irods/','/sps/km3net/users/ffilippi/ML/outputfolder_mupage/')
    
    @given(value=st.integers())
    def test_path_generator2(self,value):
        """
        testing raising error if not 0 or 1 are inserted
        """
        if value > 1:
            with pytest.raises(NameError):
                path_generator(value)

#Classll Test_binning():
#    def test_binning(self):
#        assert binning(0)[0]==18



#class Test_concatenate:
#    def test_concatenate(self):
#        assert concatenate(0)

