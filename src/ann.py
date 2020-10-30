import os
####*IMPORANT*: Have to do this line *before* importing tensorflow
os.environ['PYTHONHASHSEED']=str(1)

# libraries
import numpy as np 
import matplotlib.pyplot as plt
import pickle
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

# for i in range(len(x_train)):
# 	if x_train[i].any() < 0:
# 		print("YAYYY")
# scale images
x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255

x_train = np.reshape(x_train, (60000, -1))
x_test = np.reshape(x_test, (10000, -1))

def ANN1():
	inp = keras.Input(shape = x_train[0].shape)
	x = Dense(10, activation = "tanh")(inp)
	x = Dense(10, activation = "tanh")(x)
	x = Dense(10, activation = "tanh")(x)
	x = Dense(10, activation = "tanh")(x)
	x = Dense(10, activation = "tanh")(x)
	x = Dense(10, activation = "tanh")(x)
	x = Dense(10, activation = "tanh")(x)
	x = Dense(10, activation = "tanh")(x)
	x = Dense(10, activation = "tanh")(x)
	x = Dropout(0.15)(x)
	out = Dense(1, activation = "linear")(x)

	model = Model(inputs = inp, outputs = out)

	return model 

def ANN2():
	inp = keras.Input(shape = x_train[0].shape)
	x = Dense(50, activation = "tanh")(inp)
	x = Dense(50, activation = "tanh")(x)
	x = Dense(50, activation = "tanh")(x)
	x = Dense(50, activation = "tanh")(x)
	x = Dense(50, activation = "tanh")(x)
	x = Dropout(0.15)(x)
	out = Dense(1, activation = "linear")(x)

	model = Model(inputs = inp, outputs = out)

	return model 

def ANN3():
	inp = keras.Input(shape = x_train[0].shape)
	x = Dense(25, activation = "tanh")(inp)
	x = Dense(25, activation = "tanh")(x)
	x = Dense(25, activation = "tanh")(x)
	x = Dense(25, activation = "tanh")(x)
	x = Dense(25, activation = "tanh")(x)
	x = Dense(25, activation = "tanh")(x)
	x = Dense(25, activation = "tanh")(x)
	x = Dropout(0.15)(x)
	out = Dense(1, activation = "linear")(x)

	model = Model(inputs = inp, outputs = out)

	return model 

def ANN4():
	inp = keras.Input(shape = x_train[0].shape)
	x = Dense(128, activation = "tanh")(inp)
	x = Dense(128, activation = "tanh")(x)
	x = Dropout(0.5)(x)
	x = Dense(128, activation = "tanh")(x)
	x = Dropout(0.7)(x)
	out = Dense(1, activation = "linear")(x)

	model = Model(inputs = inp, outputs = out)

	return model 

classifier = ANN4()

batch_size = 512
epochs = 100

mcp_save = keras.callbacks.callbacks.ModelCheckpoint('../models/dl48.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks_list = [mcp_save, reduce_lr]

classifier.compile(loss = "mse", optimizer = "adam", metrics = ["accuracy"])

history = classifier.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (x_test, y_test), callbacks = callbacks_list)

# pickle history file
filename = '../model_history/dl48.pickle'
outfile = open(filename, 'wb')
pickle.dump(history ,outfile)
outfile.close()

# Plot History
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

'''Results:

ANN1: BS:256, Epochs: 100: (ReLU)  loss: 0.3238 - accuracy: 0.9104 - val_loss: 0.3802 - val_accuracy: 0.9178
ANN1: BS:256, Epochs: 100: (softplus)  loss: 0.2434 - accuracy: 0.9301 - val_loss: 0.3313 - val_accuracy: 0.9288
ANN1: BS:256, Epochs: 100: (tanh) loss: 0.1837 - accuracy: 0.9549 - val_loss: 0.2359 - val_accuracy: 0.9445
'''