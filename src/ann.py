import os
####*IMPORANT*: Have to do this line *before* importing tensorflow
os.environ['PYTHONHASHSEED']=str(1)

# libraries
import numpy as np 
import tensorflow as tf 
import random
import keras
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout

def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(1)
   tf.random.set_seed(1)
   np.random.seed(1)
   random.seed(1)

#make some random data
reset_random_seeds()

# data parameters
num_classes = 10
input_shape = (784, 1)

# import data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# scale images
x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255

x_train = np.reshape(x_train, (60000, -1))
x_test = np.reshape(x_test, (10000, -1))

def ANN():
	inp = keras.Input(shape = x_train[0].shape)
	x = Dense(128, activation = "relu")(inp)
	x = Dropout(0.15)(x)
	out = Dense(10, activation = "softmax")(x)

	model = Model(inputs = inp, outputs = out)

	return model 

classifier = ANN()

batch_size = 32
epochs = 100

mcp_save = keras.callbacks.callbacks.ModelCheckpoint('../models/dl1.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks_list = [mcp_save, reduce_lr]

classifier.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

classifier.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (x_test, y_test), callbacks = callbacks_list)

'''Results:

DL Model 1 (dl1.h5): (Input, 128, Output): BS:32, Epochs: 100: loss: 3.1591e-05 - accuracy: 1.0000 - val_loss: 0.1099 - val_accuracy: 0.9827 (Overfitting a little bit)

'''