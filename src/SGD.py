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

# scale images
x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255

x_train = np.reshape(x_train, (60000, -1))
x_test = np.reshape(x_test, (10000, -1))

#Model with loss -> Hinge

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(loss='hinge', random_state=42)
sgd_clf.fit(x_train, y_train)
print("SGD hinge fit done")

from sklearn.metrics import accuracy_score

sgd_svm_pred = sgd_clf.predict(x_test)
print(sgd_svm_pred)

sgd_accuracy = accuracy_score(y_test, sgd_svm_pred)
print(sgd_accuracy)


#Model with loss -> Log
sgd_clf2 = SGDClassifier(loss='log', random_state=42)
sgd_clf2.fit(x_train, y_train)
print("SGD log fit done")

sgd_svm_pred2 = sgd_clf2.predict(x_test)
print(sgd_svm_pred2)

sgd_accuracy2 = accuracy_score(y_test, sgd_svm_pred2)
print(sgd_accuracy2)


'''
SGD with hinge loss -> 0.9174
SGD with log loss -> 0.9154
'''
