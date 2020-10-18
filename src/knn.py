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
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

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

kVals = range(1, 30, 2)
accuracies = []
 
# loop over various values of `k` for the k-Nearest Neighbor classifier
for k in range(1, 30, 2):
	# train the k-Nearest Neighbor classifier with the current value of `k`
	model = KNeighborsClassifier(n_neighbors=k)
	model.fit(x_train, y_train)
 
	# evaluate the model and update the accuracies list
	score = model.score(x_test, y_test)
	print("k=%d, accuracy=%.2f%%" % (k, score * 100))
	accuracies.append(score)
 
# find the value of k that has the largest accuracy
i = int(np.argmax(accuracies))
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
	accuracies[i] * 100))


'''
k=1, accuracy=96.91%
k=3, accuracy=97.05%
k=5, accuracy=96.88%
k=7, accuracy=96.94%
k=9, accuracy=96.59%
k=11, accuracy=96.68%
k=13, accuracy=96.53%
k=15, accuracy=96.33%
k=17, accuracy=96.30%
k=19, accuracy=96.32%
  k=21, accuracy=96.30%
k=23, accuracy=96.19%
k=25, accuracy=96.09%
k=27, accuracy=96.04%
k=29, accuracy=95.93%
k=3 achieved highest accuracy of 97.05% on validation data
'''