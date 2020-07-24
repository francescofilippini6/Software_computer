import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
from os import getcwd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from tensorflow.python.client import device_lib
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Dense, Activation, BatchNormalization, Flatten, Conv3D
from keras.layers import AveragePooling3D, MaxPooling3D, Dropout, GlobalAveragePooling3D
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, LambdaCallback
from keras.optimizers import Adam
from keras.utils import to_categorical
#from batch_uploader import DataGenerator
from batch_uploader_keras import DataGenerator

#@profile
def get_available_devices():
   local_device_protos = device_lib.list_local_devices()
   print( [x.name for x in local_device_protos])


#get_available_devices()

model = Sequential()
"""Parameters of Conv3D  
                 16 -> filter number
                  3 -> kernel size
                 strides -> step of the cnn
                 input shape -> (iumage size, channels)
""" 

model.add(Conv3D(16, 3, strides = 1, name = 'conv0_0', input_shape = (18,280,31,1)))
model.add(Conv3D(16, 3, strides = 1, name='conv0_1'))
model.add(Activation('relu'))
model.add(MaxPooling3D(2, strides=1, name='max_pool'))


model.add(Conv3D(32, 3, strides = 1, name = 'conv1_0'))
model.add(Conv3D(32, 3, strides=1, name='conv1_1'))
model.add(Activation('relu'))
model.add(MaxPooling3D(2, strides=2, name='max_pool_1'))



#model.add(Conv3D(64,3, strides = 1, name="conv2_0"))
#model.add(Activation('relu'))
#model.add(MaxPooling3D(2,strides=2, name='max_pool_2'))



#model.add(BatchNormalization())
model.add(Flatten())
#model.add(GlobalAveragePooling3D())
model.add(Dense(128,activation="relu"))
model.add(Dense(2,activation='softmax', name='sm'))


#callbacks = [LearningRateScheduler(exp_decay)]

model.compile(optimizer=Adam(learning_rate=0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#@profile
def batch_uploader_caller():
 
   """
   creating the dictionary and position array to be passed to batch_uploader.py -> DataGenerator()
   reading muon events
   reading the neutrino events
   extracting labels and dimensions
   """
   hdf1=h5py.File('/sps/km3net/users/ffilippi/ML/outputfolder_mupage/concatenated.h5','r')
   labels_m=np.array(hdf1['y'])    
   labels_mu=labels_m[:10000]
   hdf1.close()
   hdf2=h5py.File('/sps/km3net/users/ffilippi/ML/outputfolder_neutrino/concatenated.h5','r')
   labels_n=np.array(hdf2['y'])    
   labels_nu=labels_n[:10000]
   hdf2.close()
   labels = np.concatenate([labels_mu, labels_nu])
   
   group_id=[]
   for x in range(0,(len(labels_nu)+len(labels_mu))):
      name='id_'+str(x)
      group_id.append(name)
      
  
   labelsa ={group_id[i]:labels[i] for i in range(len(group_id))} 
  
   X_train, X_test, y_train, y_test = train_test_split(group_id,labels, test_size=0.2)
   dx = {'train' : X_train, 'validation': X_test}


   training_generator = DataGenerator(dx['train'], labelsa)
   
   validation_generator = DataGenerator(dx['validation'], labelsa)
   
   return training_generator , validation_generator

#learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
#                                            patience=2,
#                                            verbose=1,
#                                            factor=0.5,
#                                            min_lr=0.0000001)


#early_stop = EarlyStopping(monitor="val_loss",
#                           mode="min",
#                           patience=5,
#                           restore_best_weights=True)

#checkpoint = ModelCheckpoint('./model_weights.hdf5',
#                             monitor='val_loss',
#                             verbose=1,
#                             save_best_only=True,
#                             mode='min',
#                             save_weights_only=True)


nb_epoch = 2

train,val=batch_uploader_caller()

history=model.fit_generator(generator=train,
                    validation_data=val,
                    epochs=nb_epoch,verbose=1
                    ,use_multiprocessing=True,
                    workers=4)



#Print a table with the input and output
print(model.summary())
#saving the metrics in a .json file
hist_df = pd.DataFrame(history.history)
hist_json_file = getcwd()+'/history_1.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)



