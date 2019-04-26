import numpy as np
import scipy
import matplotlib.pyplot as plt
from keras.datasets import mnist
from util import func_confusion_matrix



# load (downloaded if needed) the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# transform each image from 28 by28 to a 784 pixel vector
pixel_count = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], pixel_count).astype('float32')
x_test = x_test.reshape(x_test.shape[0], pixel_count).astype('float32')

# normalize inputs from gray scale of 0-255 to values between 0-1
x_train = x_train / 255
x_test = x_test / 255

# Please write your own codes in responses to the homework assignment 4
batch_size = 512
num_classes = 10
epochs = 20

import random
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical

indicies = random.sample(range(0, 60000), 10000)
xval = [row for (i,row) in enumerate(x_train) if i in indicies]
yval = [row for (i,row) in enumerate(y_train) if i in indicies]
xval = np.array(xval)
#yval = to_categorical(yval, num_classes=num_classes, dtype='int')


xtrain = [row for (i,row) in enumerate(x_train) if i not in indicies]
ytrain = [row for (i,row) in enumerate(y_train) if i not in indicies]
xtrain = np.array(xtrain)
ytrain = to_categorical(ytrain, num_classes=num_classes, dtype='int')






#Model 1
#1 hidden layer
#relu activation
#softmax output
model1 = Sequential()
model1.add(Dense(128, activation='relu', input_shape=(784,)))
model1.add(Dense(num_classes, activation='softmax'))

model1.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
print(model1.summary())

model1.fit(xtrain, ytrain,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)

ypred = model1.predict_classes(xval)

conf, acc, rec, prec = func_confusion_matrix(yval,ypred)
print(conf)
print(acc)
print(rec)
print(prec)


#model 2
#2 hidden layers
#sigmoid activation
#sigmoid output
model2 = Sequential()
model2.add(Dense(128, activation='sigmoid', input_shape=(784,)))
model2.add(Dropout(0.2))
model2.add(Dense(64, activation='sigmoid'))
model2.add(Dense(num_classes, activation='sigmoid'))

model2.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

model2.fit(xtrain, ytrain,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)

ypred = model2.predict_classes(xval)

conf, acc, rec, prec = func_confusion_matrix(yval,ypred)
print(conf)
print(acc)
print(rec)
print(prec)



#model 3
#3 hidden layers
#relu activation
#softmax output
model3 = Sequential()
model3.add(Dense(256, activation='relu', input_shape=(784,)))
model3.add(Dropout(0.2))
model3.add(Dense(128, activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(64, activation='relu'))
model3.add(Dense(num_classes, activation='softmax'))

model3.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

model3.fit(xtrain, ytrain,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)

ypred = model3.predict_classes(xval)

conf, acc, rec, prec = func_confusion_matrix(yval,ypred)
print(conf)
print(acc)
print(rec)
print(prec)


#use model 3 to predict the test set
ypred = model3.predict_classes(x_test)
conf, acc, rec, prec = func_confusion_matrix(y_test,ypred)
print(conf)
print(acc)
print(rec)
print(prec)

dif = []
for i in range(len(y_test)):
    if(y_test[i] != ypred[i]):
        dif.append(i)
        
image_index = 4444
for i in range(10):
    plt.imshow(x_test[dif[i]].reshape(28, 28),cmap='Greys')
    plt.show()
    print("predicted",ypred[dif[i]])
    print('actual',y_test[dif[i]])


plt.imshow(x_test[dif[1]].reshape(28, 28),cmap='Greys')

