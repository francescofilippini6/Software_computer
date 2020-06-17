import keras
#from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import h5py
batch_size = 128
num_classes = 10
epochs = 12


with h5py.File('numuCC_hist.h5','r') as hdf:
    base_items = list(hdf.items())
    print('Items in the base directory', base_items)
    g2 = np.array(hdf.get('x'))
    print(g2)
    print('shape',g2.shape)
    nevt=g2.shape[0]
    print(nevt)
    ybins=g2.shape[1]
    xbins=g2.shape[2]

# input image dimensions
img_rows, img_cols = ybins, xbins
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)
#test_dataset, training_dataset = train_test_split(g2, train_size=5, test_size=5)
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = g2 

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
 
