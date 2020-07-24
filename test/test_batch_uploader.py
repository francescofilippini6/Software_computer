import numpy as np
import sys
sys.path.insert(0, '/sps/km3net/users/ffilippi/ML/')
from batch_uploader_keras import DataGenerator
from os import getcwd
import h5py
import os
import pytest
import unittest
from hypothesis import given
import hypothesis.strategies as st


def dummy_list():
    group_id=[]
    labels=[]
    with h5py.File('/sps/km3net/users/ffilippi/ML/test/outputfolder_mupage/concatenated.h5','r') as hdf:
        labels=np.array(hdf['y'])
    for x in range(0,(len(labels))):
        name='id_'+str(x)
        group_id.append(name)    
    dict_label ={group_id[i]:labels[i] for i in range(len(group_id))}
    return group_id, dict_label, len(labels)

a=DataGenerator(dummy_list()[0],dummy_list()[1])

def test_len():
    assert a.__len__() == int(dummy_list()[2]/a.batch_size)

@given(value=st.integers())
def test_data_values(value):
    """
    testing the binary behaviour of labels. On a small batch (e.g.=8) we can sample only 1 type of particle (only 0 or 1)
    """
    if value < a.__len__() and value > 0:
        assert a.__getitem__(value)[1].shape==(a.batch_size,2)


@given(value=st.integers())
def test_getitem_dimensions(value):
    """
    check of dimensions of the resulting batches of data-> x=images and labels
    """
    if value < a.__len__() and value > 0:
        assert a.__getitem__(value)[0].shape==(a.batch_size,18,280,31,1)
    
                
