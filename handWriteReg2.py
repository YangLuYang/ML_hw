import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPool2D, Flatten, Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, Adagrad
from keras.utils import np_utils
from keras.datasets import mnist

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data('mnist.npz')
    number = 10000
    x_train = x_train[0:number]
    y_train = y_train[0:number]
    x_train = x_train.reshape(number, 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    #归一化
    x_train = x_train / 255
    x_test = x_test / 255
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data()
model = Sequential()
model.add(Convolution2D(25, (5,5), input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))

model.add(Convolution2D(50, (5, 5)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))
model.add(Flatten())

model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=100, epochs=10)

train_res = model.evaluate(x_train, y_train)
print('Train Acc:%f'%(train_res[1]))
res = model.evaluate(x_test, y_test)
print('Test Acc:%f'%res[1])
