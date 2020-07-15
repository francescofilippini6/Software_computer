import numpy as np
import h5py
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def concatenate():
    X_tr=[]
    Y_tr=[]
    
    with h5py.File('/sps/km3net/users/ffilippi/ML/outputfolder_mupage/concatenated_x_y.h5','r') as hdf:
        X_t = np.array(hdf.get('x'))
        Y_t = np.array(hdf.get('y'))
        
    X_tr_nu = X_t.reshape(161734,18,110,1)
    Y_tr_nu = Y_t.reshape(161734,1)

    with h5py.File('/sps/km3net/users/ffilippi/ML/outputfolder_neutrino/concatenated_x_y.h5','r') as hdf:
        X_t = np.array(hdf.get('x'))
        Y_t = np.array(hdf.get('y'))

    X_t = X_t.reshape(124098,18,110,1)
    Y_t = Y_t.reshape(124098,1)
    X_tr=np.concatenate((X_tr_nu,X_t))
    Y_tr=np.concatenate((Y_tr_nu,Y_t))
    X_tr_shuffled, Y_tr_shuffled=shuffle(X_tr,Y_tr)
    print('shape of the final dataset',X_tr_shuffled.shape)
    #with h5py.File('final_dataset_shuffled.h5', 'w') as hdf:
    #    hdf.create_dataset('x', data=X_tr_shuffled)
    #    hdf.create_dataset('y', data=Y_tr_shuffled)
    
    X_train, X_val, Y_train, Y_val=train_test_split(X_tr_shuffled, Y_tr_shuffled, test_size=0.2, random_state=1)
    with h5py.File('final_training_dataset_shuffled.h5', 'w') as hdf:
        hdf.create_dataset('x', data=X_train)
        hdf.create_dataset('y', data=Y_train)
    with h5py.File('final_validation_dataset_shuffled.h5', 'w') as hdf:
        hdf.create_dataset('x', data=X_val)
        hdf.create_dataset('y', data=Y_val)
    
concatenate()
    
