import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import h5py
import os
from tqdm import tqdm, tqdm_notebook
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
model.add(Conv2D(32, (3,3) ,strides = (1,1), name = 'conv0', input_shape = (18,110,1)))
model.add(BatchNormalization(axis = 3, name = 'bn0'))
model.add(Activation('relu'))

model.add(MaxPooling2D((2,2), name='max_pool'))
model.add(Conv2D(64,(4,4), strides = 1, name="conv1"))
model.add(Activation('relu'))
model.add(AveragePooling2D((3,3), name='avg_pool'))

model.add(GlobalAveragePooling2D())
model.add(Dense(300, activation="relu", name='rl'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid', name='sm'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

 
#X_tr = []
#Y_tr = []
#imges = train_df['id'].values
#for img_id in tqdm_notebook(imges):
#    temp = cv2.imread(train_dir + img_id)
#    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
#    temp = np.expand_dims(temp, axis=2)
#    X_tr.append(temp)    
#    Y_tr.append(train_df[train_df['id'] == img_id]['has_cactus'].values[0])  
#X_tr = np.asarray(X_tr)
#X_tr = X_tr.astype('float32')
#X_tr /= 255
#Y_tr = np.asarray(Y_tr)


X_tr=[]
Y_tr=[]

with h5py.File('/sps/km3net/users/ffilippi/ML/outputfolder/numuCC_x_y.h5','r') as hdf:
    X_t = np.array(hdf.get('x'))
    Y_t = np.array(hdf.get('y'))
X_tr_nu = X_t.reshape(2243,18,110,1)
Y_tr_nu = Y_t.reshape(2243,1)
with h5py.File('/sps/km3net/users/ffilippi/ML/outputfolder/mupage_x_y.h5','r') as hdf:
    X_t = np.array(hdf.get('x'))
    Y_t = np.array(hdf.get('y'))

X_t = X_t.reshape(2339,18,110,1)
Y_t = Y_t.reshape(2339,1)
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
#Plotting the model in a schematic way
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# summarize history for accuracy
#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#history_df = pd.DataFrame(history.history)
#history_df[['loss', 'val_loss']].plot()
#history_df[['acc', 'val_acc']].plot()
