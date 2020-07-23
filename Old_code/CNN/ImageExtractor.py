import numpy as np
import os
from PIL import Image
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

path='path to the directory of the images'
listing =os.listdir(path)
num_samples=size(listing)
print (num_samples)



img_rows, img_cols = 200,200
for file in listing:
    im=Image.open(path+'\\'+file)
    img=im.resize((img_rows, img_cols))
    gray= img.convert('L')
    gay.save(path2+'\\'+file,"JPEG")
imlist=os.listdir(path2)

im1=array(Image.open('input_data_resized'+'\\'+imlist[0]))
m,n =im1.shape[0:2]
imnbr=len(imlist)
#flattening the image matrix
"""from this 1 4 5
             3 5 0  to [1 4 5 3 5 0 1 1 6]
             1 1 6 
from 2

"""

immatrix=array([array(Image.open('input_data_resized'+'\\'+im2)).flatten() for im2 in imlist],'f')

label=np.ones((num_samples,)dtype=int)
#definisce le labels a mano
label[0:89]=0
label[89:187]=1
label[187:]=2
#need to shuffle the data in order to make the algorithm learns better. Random state to make sure that each time it is shuffled in the same manner
data,Label=shuffle(immatrix,label, random_state=2)
train_data=[data,label]

img=immatrix[167].reshape(img_rows, img_cols)
plt.imshow(img)
plt.imshow(img, cmap='gray')


batch_size = 32
nb_classes = 3  
nb_epochs = 20
