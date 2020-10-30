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

from sklearn.linear_model import LogisticRegression
ridge_classifier = LogisticRegression(fit_intercept=True,
                        multi_class='auto',
                        penalty='l2', #ridge regression
                        solver='saga',
                        max_iter=10000,
                        C=50)
print(ridge_classifier)

ridge_classifier.fit(x_train, y_train)
print(ridge_classifier.classes_)

# predict
predictions1 = ridge_classifier.predict(x_test)
print(predictions1[:10])

print(metrics.accuracy_score(y_true=y_test, y_pred=predictions1))

class_wise1 = metrics.classification_report(y_true=y_test, y_pred=predictions1)
print(class_wise1)


lasso_classifier = LogisticRegression(fit_intercept=True,
                        multi_class='auto',
                        penalty='l1', #lasso regression
                        solver='saga',
                        max_iter=1000,
                        C=50,
                        verbose=2, # output progress
                        n_jobs=5, # parallelize over 5 processes
                        tol=0.01
                         )
lasso_classifier.fit(x_train, y_train)
print(lasso_classifier.classes_)

# predict
predictions2 = lasso_classifier.predict(x_test)
print(predictions2[:10])

print(metrics.accuracy_score(y_true=y_test, y_pred=predictions2))

class_wise2 = metrics.classification_report(y_true=y_test, y_pred=predictions2)
print(class_wise2)

'''

Ridge regression results

LogisticRegression(C=50, max_iter=10000, solver='saga')

0.9251--> accuracy
              precision    recall  f1-score   support

           0       0.95      0.97      0.96       980
           1       0.96      0.98      0.97      1135
           2       0.92      0.90      0.91      1032
           3       0.90      0.91      0.91      1010
           4       0.94      0.94      0.94       982
           5       0.90      0.87      0.88       892
           6       0.95      0.95      0.95       958
           7       0.93      0.92      0.93      1028
           8       0.88      0.88      0.88       974
           9       0.91      0.91      0.91      1009

    accuracy                           0.93     10000
   macro avg       0.92      0.92      0.92     10000
weighted avg       0.92      0.93      0.92     10000

Lasso regression results

0.9262--> accuracy
              precision    recall  f1-score   support

           0       0.95      0.98      0.97       980
           1       0.96      0.98      0.97      1135
           2       0.93      0.90      0.91      1032
           3       0.90      0.91      0.91      1010
           4       0.94      0.93      0.94       982
           5       0.89      0.87      0.88       892
           6       0.94      0.95      0.95       958
           7       0.94      0.92      0.93      1028
           8       0.89      0.88      0.88       974
           9       0.91      0.92      0.92      1009

    accuracy                           0.93     10000
   macro avg       0.93      0.93      0.93     10000
weighted avg       0.93      0.93      0.93     10000
'''

