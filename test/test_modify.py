import numpy as np
import sys
from binner import path_generator
from modify_h5_file_adding_label import filename_gen, printfile, appendLabelDataset, newfile
from os import getcwd
import os
import pytest
import unittest
from hypothesis import given
import hypothesis.strategies as st

class Test_filename_gen:
    def test_filename1(self):
       assert filename_gen(1) == (getcwd()+'/outputfolder_neutrino/concatenated.h5',getcwd()+'/outputfolder_neutrino/concatenated_x_y.h5')

    def test_filename0(self):
        assert filename_gen(0) == (getcwd()+'/outputfolder_mupage/concatenated.h5',getcwd()+'/outputfolder_mupage/concatenated_x_y.h5')
       
def test_printfile():
    image=(printfile(0)[1],printfile(0)[2],printfile(0)[3])
    assert image == (18,280,31)
    
def test_appendDataset():
     assert appendLabelDataset(0) == printfile(0)[0]


    
