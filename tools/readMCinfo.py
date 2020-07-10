import matplotlib.pyplot as plt    # our plotting module
import pandas as pd    # the main HDF5 reader
import numpy as np    # must have
import km3pipe as kp    # some KM3NeT related helper functions
import seaborn as sns    # beautiful statistical plots!


import km3pipe.style
km3pipe.style.use("km3pipe")
filepath='numuCC.h5'
#calibration step already performed
#cal = kp.calib.Calibration(filename="KM3NeT_00000042_00008494.detx")

primaries = pd.read_hdf(filepath, '/mc_tracks')
info = pd.read_hdf(filepath, '/event_info')
print(primaries.head(10))
#how to group a muon bundle????
primariess = primaries.groupby('group_id').first()
print(len(primariess))

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