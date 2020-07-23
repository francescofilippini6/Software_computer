import matplotlib.pyplot as plt
import pandas as pd


#with open('history.json', 'r') as myfile:
#    history=myfile.read()
history=pd.read_json('history_1.json')
kk=history.keys()
print(kk)
fig = plt.figure()
# summarize history for accuracy
ax1=fig.add_subplot(1,2,1)
ax1.plot(history['accuracy'])
ax1.plot(history['val_accuracy'])
ax1.set_title('model accuracy')
ax1.set_ylabel('accuracy')
ax1.set_xlabel('epoch')
#ax1.legend(['train', 'test'], loc='upper left')
# summarize history for loss
ax2=fig.add_subplot(1,2,2)
ax2.plot(history['loss'])
ax2.plot(history['val_loss'])
ax2.set_title('model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
#ax2.legend(['train', 'test'], loc='upper left')
#fig.savefig('/sps/km3net/users/ffilippi/ML/loss.png')
plt.show()
