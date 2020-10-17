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

from sklearn import svm
from sklearn import metrics

# an initial SVM model with linear kernel   
svm_linear = svm.SVC(kernel='linear')

# fit
svm_linear.fit(x_train, y_train)

# predict
predictions1 = svm_linear.predict(x_test)
print(predictions1[:10])

print(metrics.accuracy_score(y_true=y_test, y_pred=predictions1))

class_wise = metrics.classification_report(y_true=y_test, y_pred=predictions1)
print(class_wise)

# rbf kernel with other hyperparameters kept to default 
svm_rbf = svm.SVC(kernel='rbf')
svm_rbf.fit(x_train, y_train)

# predict
predictions2 = svm_rbf.predict(x_test)

# accuracy 
print(metrics.accuracy_score(y_true=y_test, y_pred=predictions2))

class_wise2 = metrics.classification_report(y_true=y_test, y_pred=predictions2)
print(class_wise2)

'''
linear ->
0.9404
              precision    recall  f1-score   support

           0       0.95      0.98      0.96       980
           1       0.97      0.99      0.98      1135
           2       0.93      0.94      0.93      1032
           3       0.91      0.94      0.92      1010
           4       0.94      0.96      0.95       982
           5       0.91      0.90      0.91       892
           6       0.96      0.95      0.95       958
           7       0.95      0.93      0.94      1028
           8       0.94      0.90      0.92       974
           9       0.95      0.91      0.93      1009

    accuracy                           0.94     10000
   macro avg       0.94      0.94      0.94     10000
weighted avg       0.94      0.94      0.94     10000


rbf->
 0.9792
               precision    recall  f1-score   support

            0       0.98      0.99      0.99       980
            1       0.99      0.99      0.99      1135
            2       0.98      0.97      0.98      1032
            3       0.97      0.99      0.98      1010
            4       0.98      0.98      0.98       982
            5       0.99      0.98      0.98       892
            6       0.99      0.99      0.99       958
            7       0.98      0.97      0.97      1028
            8       0.97      0.98      0.97       974
            9       0.97      0.96      0.97      1009

     accuracy                           0.98     10000
    macro avg       0.98      0.98      0.98     10000
 weighted avg       0.98      0.98      0.98     10000
 
