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

from sklearn import tree

clf1 = tree.DecisionTreeClassifier(random_state=0, max_depth=5);
clf1= clf1.fit(x_train, y_train)
y_pred1 = clf1.predict(x_test)
print(y_pred1[0:20], ".....")
print(y_test[0:20], ".....")
print("Accuracy with max depth: 5 -->")
print(metrics.accuracy_score(y_test, y_pred1))


clf2 = tree.DecisionTreeClassifier(random_state=0, max_depth=10);
clf2= clf2.fit(x_train, y_train)
y_pred2 = clf2.predict(x_test)
print(y_pred2[0:20], ".....")
print(y_test[0:20], ".....")
print("Accuracy with max depth: 10 -->")
print(metrics.accuracy_score(y_test, y_pred2))

clf3 = tree.DecisionTreeClassifier(random_state=0, max_depth=15);
clf3= clf3.fit(x_train, y_train)
y_pred3 = clf3.predict(x_test)
print(y_pred3[0:20], ".....")
print(y_test[0:20], ".....")
print("Accuracy with max depth: 15 -->")
print(metrics.accuracy_score(y_test, y_pred3))

clf4 = tree.DecisionTreeClassifier(random_state=0, max_depth=20);
clf4= clf4.fit(x_train, y_train)
y_pred4 = clf4.predict(x_test)
print(y_pred4[0:20], ".....")
print(y_test[0:20], ".....")
print("Accuracy with max depth: 20 -->")
print(metrics.accuracy_score(y_test, y_pred4))

clf5 = tree.DecisionTreeClassifier(random_state=0, max_depth=25);
clf5= clf5.fit(x_train, y_train)
y_pred5 = clf5.predict(x_test)
print(y_pred5[0:20], ".....")
print(y_test[0:20], ".....")
print("Accuracy with max depth: 25 -->")
print(metrics.accuracy_score(y_test, y_pred5))

'''
[7 6 1 0 9 1 5 9 6 9 0 6 9 0 1 5 9 7 6 4] .....
[7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4] .....
Accuracy with max depth: 5 -->
0.6747
[7 2 1 0 4 1 4 7 6 9 0 6 9 0 1 5 9 7 3 4] .....
[7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4] .....
Accuracy with max depth: 10 -->
0.8658
[7 2 1 0 4 1 4 7 8 9 0 6 9 0 1 5 9 7 6 4] .....
[7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4] .....
Accuracy with max depth: 15 -->
0.882
[7 2 1 0 4 1 4 7 5 9 0 6 9 0 1 5 9 7 9 4] .....
[7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4] .....
Accuracy with max depth: 20 -->
0.8795
[7 2 1 0 4 1 4 7 5 9 0 6 9 0 1 5 9 7 6 4] .....
[7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4] .....
Accuracy with max depth: 25 -->
0.8801
'''
