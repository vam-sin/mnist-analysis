import numpy as np 
import matplotlib.pyplot as plt
import pickle

file = open("../model_history/dl3.pickle",'rb')
history1 = pickle.load(file)
file.close()

file = open("../model_history/dl14.pickle",'rb')
history2 = pickle.load(file)
file.close()

file = open("../model_history/dl32.pickle",'rb')
history3 = pickle.load(file)
file.close()

file = open("../model_history/dl38.pickle",'rb')
history4 = pickle.load(file)
file.close()

# plt.plot(history.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.plot(history3.history['val_accuracy'])
plt.plot(history4.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['ANN 1', 'ANN 2', 'ANN 3', 'ANN 4'], loc='bottom right')
plt.show()
# summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()