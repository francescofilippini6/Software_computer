import numpy as np
import sys
from binner import listing
from os import getcwd
import os

def test_listing():
    """
    testing that if not empty, we have a list of strings
    """
    path=getcwd()
    files=listing(path)
    assert all(isinstance(x,str) for x in files)

def test_2listing():
    """
    testing that elements return the absolute path 
    """
    path=getcwd()
    files=listing(path)
    os.path.abspath()==
#def test_generator():



#def test_binning():


#def test_concatenate():



