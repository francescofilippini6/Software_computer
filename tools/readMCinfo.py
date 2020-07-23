import matplotlib.pyplot as plt  
import pandas as pd   
import numpy as np    
import km3pipe as kp    # some KM3NeT related helper functions
import seaborn as sns   
import km3pipe.style
import h5py
km3pipe.style.use("km3pipe")

filepath='/sps/km3net/users/ffilippi/ML/nu_gehen/anu_mu_222.h5'

#primaries = pd.read_hdf(filepath, '/mc_tracks')
info = pd.read_hdf(filepath, '/mc_tracks')
print(info)
primariess = info.groupby('group_id').first()
print(primariess.keys())
zeniths = kp.math.zenith(primariess.filter(regex='^dir_.?$'))
print (zeniths)
#a = h5py.File(filepath, "r")
#print(a.keys())
#xa=info['group_id']
#if len(xa) != len(set(xa)):
#    print("aiia")
#else:
print("perfect")
#print(info.head(10))
#how to group a muon bundle????

#print(len(primariess))
#info1 = a['y']

#print("ciccio",info1['group_id'==50000])

plt.figure()
plt.subplot(141)
plt.hist(primaries.energy, bins=100, log=True)
plt.xlabel('energy [GeV]')
plt.ylabel('number of events')
plt.title('Energy Distribution')
plt.subplot(142)
zeniths = kp.math.zenith(primaries.filter(regex='^dir_.?$'))
primaries['zenith'] = zeniths

plt.hist(np.cos(primaries.zenith), bins=21, histtype='step', linewidth=2)
plt.xlabel(r'cos($\theta$)')
plt.ylabel('number of events')
plt.title('Zenith Distribution')
plt.subplot(143)
plt.hist2d(primaries.pos_x, primaries.pos_z, bins=100, cmap='viridis')
plt.xlabel('Pos x [m]')
plt.ylabel('pos y [m]')
plt.title('2D Plane')
plt.colorbar()
plt.subplot(144)
plt.hist2d(primaries.dir_y, primaries.dir_x, bins=100, cmap='viridis')
plt.xlabel('Direction x [m]')
plt.ylabel('Direction y [m]')
plt.title('2D Plane')
plt.colorbar()

plt.show()
