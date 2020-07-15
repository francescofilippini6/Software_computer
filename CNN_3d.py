import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import h5py
import os
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv3D
from keras.layers import AveragePooling3D, MaxPooling3D, Dropout, GlobalAveragePooling3D
from keras.models import Model
from keras.applications import VGG16
from keras.optimizers import Adam
from keras.utils import plot_model, to_categorical


model = Sequential()
"""Parameters of Conv2D -> (5,5)kernel size
                 32 -> output space
                 strides -> step of the cnn
                 input shape -> (batch size, channels, rows,cols)
""" 
model.add(Conv3D(16, 3, strides = 1, padding='same', name = 'conv0_0', input_shape = (18,280,31,1)))
model.add(Conv3D(16, 3, strides = 1, padding='same', name='conv0_1'))
model.add(Activation('relu'))
model.add(MaxPooling3D(2, strides=1, name='max_pool'))


model.add(Conv3D(32, 3, strides = 1, padding='same', name = 'conv1_0'))
model.add(Conv3D(32,3, strides=1, name='conv1_1'))
model.add(Activation('relu'))
model.add(MaxPooling3D(2, strides=2, name='max_pool_1'))
#model.add(Dropout(0.25))


model.add(Conv3D(64,3, strides = 1, name="conv2_0"))
model.add(Activation('relu'))
model.add(MaxPooling3D(2,strides=2, name='max_pool_2'))




model.add(Flatten())
#model.add(GlobalAveragePooling2D())
model.add(Dense(4096,activation="relu"))
model.add(Dense(2,activation='softmax', name='sm'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

 

X_tr=[]
Y_tr=[]

with h5py.File('/sps/km3net/users/ffilippi/ML/outputfolder_mupage/concatenated_2.h5','r') as hdf:
    X_t = np.array(hdf.get('x'))
    Y_t = np.array(hdf.get('y')) 
X_tr_nu = X_t.reshape(len(X_t),18,280,31,1)
Y_tr_nu = Y_t.reshape(len(X_t),1)

with h5py.File('/sps/km3net/users/ffilippi/ML/outputfolder_neutrino/concatenated_2.h5','r') as hdf:
    X_t = np.array(hdf.get('x'))
    Y_t = np.array(hdf.get('y'))

X_t = X_t.reshape(len(X_t),18,280,31,1)
Y_t = Y_t.reshape(len(X_t),1)
X_tr=np.concatenate((X_tr_nu,X_t))
Y_tr=np.concatenate((Y_tr_nu,Y_t))
#X_tr=np.divide(X_tr,255)
Y_tr=to_categorical(Y_tr,num_classes=2)
print('shape of the final dataset',X_tr.shape)



batch_size = 32
nb_epoch = 16


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

#from keras.callbacks import Callback
#out_batch = NBatchLogger()
#class NBatchLogger(Callback):

 #   def on_train_begin(self,logs={}):
 #       self.loss_train = []
 #       self.accuracy_train = []
 #       self.loss_val = []
 #       self.accuracy_val = []

 #   def on_batch_end(self, batch, logs={}):
 #       self.loss_train.append(logs.get('loss'))
 #       self.accuracy_train.append(logs.get('accuracy'))
 #       self.loss_val.append(logs.get('val_loss'))
 #       self.accuracy_val.append(logs.get('val_accuracy'))
