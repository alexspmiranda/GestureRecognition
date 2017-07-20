#%%

########################### LIBRARY ###########################################

#KERAS
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
#from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils

import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib
import os
import theano
from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn import cross_validation
#from sklearn.linear_model import LogisticRegression

#%%

# input image dimensions
rows, columns = 64, 64

# number of channels
img_channels = 1

#batch_size to train
batch_size = 64
# number of output classes
nb_classes = 5
# number of epochs to train
nb_epoch = 2


# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

#  data
path1 = 'D:\Estudos\IME\Disciplinas\IA\PythonScripts\input-1500'    #path of folder of images    
path2 = 'D:\Estudos\IME\Disciplinas\IA\PythonScripts\output'  #path of folder to save images    

listing = os.listdir(path1)
num_samples = size(listing)
print num_samples

for file in listing:
    im = Image.open(path1 + '\\' + file)  
    img = im.resize((rows,columns))
    gray = img.convert('L')
                #need to do some more processing here          
    gray.save(path2 +'\\' +  file, "JPEG")

imlist = os.listdir(path2)


im1 = array(Image.open(path2 + '\\'+ imlist[0])) # open one image to get size

m,n = im1.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images

# create matrix to store all flattened images
immatrix = array([array(Image.open(path2 + '\\' + im2)).flatten()
              for im2 in imlist],'f')
label=np.ones((num_samples,),dtype = int)

'''

label[0:4656]=0
label[4657:8769]=1
label[8780:13767]=2
label[13768:17333]=3
label[17334:]=4

'''

label[0:299]=0
label[300:599]=1
label[600:899]=2
label[900:1199]=3
label[1200:]=4

#%%

data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]

(X, y) = (train_data[0],train_data[1])


# STEP 1: split X and y into training and testing sets

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.999 , random_state=3)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,  test_size=0.999, random_state=5)

X_train = X_train.reshape(X_train.shape[0], 1, rows, columns)
X_test = X_test.reshape(X_test.shape[0], 1, rows, columns)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#%%

kfold = StratifiedKFold(y, n_folds=2, shuffle=True, random_state=7)
cvscores = []

for i in enumerate(kfold):
    # create model
    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='valid', input_shape=(1, rows, columns)))
    convout1 = Activation('relu')
    model.add(convout1)
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    convout2 = Activation('relu')
    model.add(convout2)
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.1))
    
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
	# Fit the model
    model.fit(X_test, Y_test, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))	# evaluate the model
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
 
print "%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores))

# Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict_classes(X_test)

target_names = ['class 0(C)', 'class 1(Closed)', 'class 2(Opened)', 'class 3(Pointer)', 'class 4(V)']
print(classification_report(np.argmax(Y_test,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(Y_test,axis=1), y_pred))