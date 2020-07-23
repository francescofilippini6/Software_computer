import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import h5py
import os
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.applications import VGG16
from keras.optimizers import Adam
from keras.utils import plot_model


model = Sequential()
"""Parameters of Conv2D -> (5,5)kernel size
                 32 -> output space
                 strides -> step of the cnn
                 input shape -> (batch size, channels, rows,cols)
""" 
model.add(Conv2D(32, (3,3), strides = (1,1), name = 'conv0_0', input_shape = (18,110,1)))
model.add(Conv2D(32,(3,3), strides=1, name='conv0_1'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2), strides=2, name='max_pool'))


model.add(Conv2D(64,(3,3), strides = 1, name="conv1_0"))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2),strides=2, name='max_pool_2'))




#model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(4096,activation="relu"))
model.add(Dense(1,activation='sigmoid', name='sm'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

 

X_tr=[]
Y_tr=[]

with h5py.File('/sps/km3net/users/ffilippi/ML/outputfolder_mupage/concatenated_2_x_y.h5','r') as hdf:
    X_t = np.array(hdf.get('x'))
    Y_t = np.array(hdf.get('y'))

X_tr_nu = X_t.reshape(161734,18,110,1)
Y_tr_nu = Y_t.reshape(161734,1)

with h5py.File('/sps/km3net/users/ffilippi/ML/outputfolder_neutrino/concatenated_2_x_y.h5','r') as hdf:
    X_t = np.array(hdf.get('x'))
    Y_t = np.array(hdf.get('y'))

X_t = X_t.reshape(124098,18,110,1)
Y_t = Y_t.reshape(124098,1)
X_tr=np.concatenate((X_tr_nu,X_t))
Y_tr=np.concatenate((Y_tr_nu,Y_t))

print('shape of the final dataset',X_tr.shape)


batch_size = 32
nb_epoch = 10




history = model.fit(X_tr, Y_tr,
              batch_size=batch_size,
              epochs=nb_epoch,
              validation_split=0.2,
              shuffle=True,
              verbose=2)
#Print a table with the input and output
print(model.summary())
#saving the metrics in a .json file
hist_df = pd.DataFrame(history.history)
hist_json_file = '/sps/km3net/users/ffilippi/ML/history_1.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)
