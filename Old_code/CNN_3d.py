import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import h5py
import os
import random
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.python.client import device_lib
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv3D
from keras.layers import AveragePooling3D, MaxPooling3D, Dropout, GlobalAveragePooling3D
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard, LambdaCallback
from keras.applications import VGG16
from keras.optimizers import Adam
from keras.utils import plot_model, to_categorical
#from batch_uploader import DataGenerator
from batch_uploader_keras import DataGenerator

#@profile
def get_available_devices():
   local_device_protos = device_lib.list_local_devices()
   print( [x.name for x in local_device_protos])


get_available_devices()




model = Sequential()
"""Parameters of Conv2D -> (5,5)kernel size
                 32 -> output space
                 strides -> step of the cnn
                 input shape -> (batch size, channels, rows,cols)
""" 
model.add(Conv3D(16, 3, strides = 1, name = 'conv0_0', input_shape = (18,280,31,1)))
model.add(Conv3D(16, 3, strides = 1, padding='same', name='conv0_1'))
model.add(Activation('relu'))
model.add(MaxPooling3D(2, strides=1, name='max_pool'))


model.add(Conv3D(32, 3, strides = 1, padding='same', name = 'conv1_0'))
model.add(Conv3D(32,3, strides=1, name='conv1_1'))
model.add(Activation('relu'))
model.add(MaxPooling3D(2, strides=2, name='max_pool_1'))



model.add(Conv3D(64,3, strides = 1, name="conv2_0"))
model.add(Activation('relu'))
model.add(MaxPooling3D(2,strides=2, name='max_pool_2'))



#model.add(BatchNormalization())
#model.add(Flatten())
model.add(GlobalAveragePooling3D())
model.add(Dense(128,activation="relu"))
model.add(Dense(2,activation='softmax', name='sm'))


#callbacks = [LearningRateScheduler(exp_decay)]

model.compile(optimizer=Adam(lr=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#@profile
def batch_uplo(): 
   hdf1=h5py.File('/sps/km3net/users/ffilippi/ML/outputfolder_mupage/concatenated.h5','r')
   #with h5py.File('/sps/km3net/users/ffilippi/ML/outputfolder_mupage/concatenated.h5','r') as hdf:
   #event_info_mu=np.array(hdf['event_info'])
   #group_id_m = np.array(event_info_mu['group_id'])
   #group_id_mu = group_id_m[:1000]
   #print(group_id_mu)
   labels_m=np.array(hdf1['y'])    
   labels_mu=labels_m[:1000]
   hdf1.close()
   hdf2=h5py.File('/sps/km3net/users/ffilippi/ML/outputfolder_neutrino/concatenated.h5','r')
   #with h5py.File('/sps/km3net/users/ffilippi/ML/outputfolder_neutrino/concatenated.h5','r') as hdf:
   #event_info_nu=np.array(hdf['event_info'])
   #group_id_n = np.array(event_info_nu['group_id'])+len(group_id_mu)
   #group_id_nu=group_id_n[:1000]
   #print(group_id_nu)
   labels_n=np.array(hdf2['y'])    
   labels_nu=labels_n[:1000]
   hdf2.close()
   #group_id = np.concatenate([group_id_mu, group_id_nu])
   labels = np.concatenate([labels_mu, labels_nu])
   
   #print ("labels",labels)
   group_id=[]
   for x in range(0,(len(labels_nu)+len(labels_mu))):
      name='id_'+str(x)
      group_id.append(name)
      
   #print ("grap_id",group_id)
   print(group_id)
   #operate on the id of the event the split train and test?
   labelsa ={group_id[i]:labels[i] for i in range(len(group_id))} 
   #group_id=random.shuffle(group_id)
   #label_s = [labelsa[x] for x in group_id]

   #label_s = [x for x in labelsa[ if x==group_id]
   #print("CC")
   print(labelsa)
   X_train, X_test, y_train, y_test = train_test_split(group_id,labels, test_size=0.2)
   dx = {'train' : X_train, 'validation': X_test}


   training_generator = DataGenerator(dx['train'], labelsa)
   print('Primo') 
   validation_generator = DataGenerator(dx['validation'], labelsa)
   print('secondo')
   return training_generator , validation_generator

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                            patience=2,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.0000001)


early_stop = EarlyStopping(monitor="val_loss",
                           mode="min",
                           patience=5,
                           restore_best_weights=True)

checkpoint = ModelCheckpoint('./model_weights.hdf5',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min',
                             save_weights_only=True)
#batch_size = 16
nb_epoch = 2

train,val=batch_uplo()
#val=batch_uplo()[1]
model.fit_generator(generator=train,
                    validation_data=val,
                    epochs=nb_epoch,verbose=1
                    )#callbacks =[learning_rate_reduction, early_stop, checkpoint]) #,use_multiprocessing=True,
                    #workers=4)


#Print a table with the input and output
print(model.summary())
#saving the metrics in a .json file
hist_df = pd.DataFrame(history.history)
hist_json_file = '/sps/km3net/users/ffilippi/ML/history_1.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)



#def all_file():
#   X_tr=[]
#   Y_tr=[]

#   with h5py.File('/sps/km3net/users/ffilippi/ML/outputfolder_mupage/concatenated.h5','r') as hdf:
#      X_t = np.array(hdf['x'])
#      Y_t = np.array(hdf['y']) 
#      X_tr_nu = X_t.reshape(len(X_t),18,280,31,1)
#      Y_tr_nu = Y_t.reshape(len(X_t),1)

#   with h5py.File('/sps/km3net/users/ffilippi/ML/outputfolder_neutrino/concatenated.h5','r') as hdf:
#      X_t = np.array(hdf['x'])
#      Y_t = np.array(hdf['y'])
#      X_t = X_t.reshape(len(X_t),18,280,31,1)
#      Y_t = Y_t.reshape(len(X_t),1)
#   X_tr=np.concatenate((X_tr_nu,X_t))
#   Y_tr=np.concatenate((Y_tr_nu,Y_t))
#   Y_tr=to_categorical(Y_tr,num_classes=2)
#   print('shape of the final dataset',X_tr.shape)
#   return X_tr, Y_tr

#history = model.fit(all_file()[0], all_file()[1],
#              batch_size=batch_size,
#              epochs=nb_epoch,
#              validation_split=0.2,
#              shuffle=True,
#              verbose=2)

