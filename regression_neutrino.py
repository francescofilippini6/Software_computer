import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import h5py
import os
from keras.models import Sequential, model_from_json
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint,Callback,EarlyStopping
from keras.optimizers import Adam
from keras.utils import plot_model, to_categorical

model = Sequential()
"""Parameters of Conv2D -> (5,5)kernel size
                 32 -> output space
                 strides -> step of the cnn
                 input shape -> (batch size, channels, rows,cols)
"""
model.add(Conv2D(16, (3,3), strides = (1,1),padding='same', name = 'conv0_0', input_shape = (18,280,1)))
model.add(Conv2D(16,(3,3), strides=1,padding='same',name='conv0_1'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2), strides=1, name='max_pool'))
model.add(Dropout(0.25))
model.add(Conv2D(16, (5,5), strides = (1,1),padding='same', name = 'conv1_0'))
model.add(Conv2D(16,(5,5), strides=1,padding='same',name='conv1_1'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2), strides=1, name='max_pool2'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(32,activation="relu"))
model.add(Dense(64,activation="relu"))
model.add(Dense(1,activation='linear', name='sm'))
model.compile(optimizer='adam',
              loss='mean_squared_error')

class LossHistory(Callback):
  def on_train_begin(self, logs={}):
    self.losses = []
    self.val_losses = []

  def on_batch_end(self, batch, logs={}):
    self.losses.append(logs.get('loss'))
    self.val_losses.append(logs.get('val_loss'))

X_tr=[]
Y_tr=[]

with h5py.File('/content/drive/My Drive/z-t_colab_2.0/neutrino-concatenated.h5','r') as hdf:
    X_tr = np.array(hdf['x'][:900000]).reshape(900000,18,280,1)
    Y_tr = np.array(hdf['y'][:900000]).reshape(900000,1)
    hdf.close()
print('shape of the final dataset',X_tr.shape)

batch_size = 32
nb_epoch = 30
checkpoint = ModelCheckpoint("best_model.hdf5", monitor='val_loss', verbose=1,
    save_best_only=True, mode='auto', period=1)
historyy = LossHistory()
es = EarlyStopping(monitor='val_loss', verbose=1, patience=10)
cb_list = [historyy, es, checkpoint]
history = model.fit(X_tr, Y_tr,             
          batch_size=batch_size,
          epochs=nb_epoch,
          validation_split=0.25,
          shuffle=True,
          verbose=1,
          callbacks=cb_list)
#-------------------------------------------------
#plotting val_loss and loss value registered at each epoch end
#-------------------------------------------------
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
#-------------------------------------------------
#plotting loss and val_loss values registered at each batch end
#-------------------------------------------------

x=np.linspace(0,len(historyy.losses),len(historyy.losses))
plt.plot(historyy.losses,x,'r-', label='loss')
plt.plot(historyy.val_losses,x,'b-', label='val_loss')
plt.title('model loss')
plt.ylabel('loss value')
plt.xlabel('epochs')
plt.legend(loc='best')
#plt.savefig('loss_each_epoch_end.png')
#-------------------------------------------------
#predicting the labels of 100.000 events and plot predicted vs MonteCarlo(real) value
#-------------------------------------------------
with h5py.File('/content/drive/My Drive/z-t_colab_2.0/neutrino-concatenated.h5','r') as hdfa:
  X_predict = np.array(hdfa['x'][600001:700001]).reshape(100000,18,280,1)
  Y_predict=model.predict(X_tra)
  
  Y_predict=y.reshape(100000)
  Y_real = np.array(hdfa['y'][600001:700001])#.reshape(100000,1)
  
  print("predicted_y",Y_predicted)
  print("real_y",Y_real)
  #plt.hist2d(y, Y_tra, bins=(50, 50), cmap=plt.cm.jet)
  plt.scatter(Y_predicted[:500], Y_real[:500])
  plt.title('predicted_vs_real')
  plt.ylabel('real value')
  plt.xlabel('predicted value')
  plt.figure(figsize=(15,10))
  plt.tight_layout()
  #weights = np.ones_like(Y_predicted) / len(Y_predicted)
  #weights1 = np.ones_like(Y_real) / len(Y_real)

  a=np.histogram(y,bins=200)
  b=np.histogram(Y_tra,bins=200)
  #print(a[0])
  #print(b[0])
  import seaborn as seabornInstance 
  seabornInstance.distplot(Y_predicted,hist_kws={'range': (-2.0, 2.0)},label = 'predicted',kde=False,bins=200)
  seabornInstance.distplot(Y_real, hist_kws={'range': (-2.0, 2.0)},label='real',kde=False,bins=200)
  plt.title('cos distribution')
  plt.ylabel('entries')
  plt.xlabel('cos(zenith)')
  plt.legend()
  from scipy.stats import chisquare
  c=chisquare(a[0], f_exp=b[0])
  print(c[0])
#model.save_weights("model.h5")
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
#-------------------------------------------------
#loading the model and weights
#-------------------------------------------------
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
# load weights into new model
#loaded_model.load_weights("best_model.hdf5")
#loaded_model.compile(optimizer='adam',loss='mean_squared_error')
