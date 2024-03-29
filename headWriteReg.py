import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.optimizers import SGD, Adam, Adagrad
from keras.utils import np_utils
from keras.datasets import mnist

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data('mnist.npz')
    number = 10000
    x_train = x_train[0:number]
    y_train = y_train[0:number]
    x_train = x_train.reshape(number, 28*28)
    x_test = x_test.reshape(x_test.shape[0], 28*28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train
    x_test = x_test
    #归一化
    x_train = x_train / 255
    x_test = x_test / 255
    #添加噪声
    x_test = np.random.normal(x_test)
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data()
model = Sequential()
model.add(Dense(input_dim=28*28, units=500, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(units=500, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(units=500, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adagrad(lr=0.01), metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=100, epochs=20)

train_res = model.evaluate(x_train, y_train)
print('Train Acc:%f'%(train_res[1]))
res = model.evaluate(x_test, y_test)
print('Test Acc:%f'%res[1])



