import numpy as np
import h5py
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import os

from orcanet.core import Organizer



def make_dummy_model():
    """
    Build and compile a small dummy model.
    """
    #batch size=number of samples in the network, 
    #img dimensions, last parameter (RGB)
    input_shape = (2243,18,299,3)

    inp = Input(input_shape, name="random_numbers")
    

    x = Dense(10)(inp)
    
    outp = Dense(2, name="sum")(x)

    model = Model(inp, outp)
    model.compile("sgd", loss="mae")

    return model


def use_orcanet():
    temp_folder = "output/"
    os.mkdir(temp_folder)
    
    list_file = "example_list.toml"

    organizer = Organizer(temp_folder + "sum_model", list_file)
    organizer.cfg.train_logger_display = 10

    model = make_dummy_model()
    organizer.train_and_validate(model, epochs=3)

    organizer.predict()


use_orcanet()
