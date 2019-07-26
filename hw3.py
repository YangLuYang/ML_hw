import tensorflow as tf
import keras as ks
import numpy as np
import cv2
import pandas as pd

# def cleanData():
#     data = np.loadtxt('./data/train(3).csv', skiprows=1, delimiter=' ', dtype=str)
#     # data = np.loadtxt('./data/hw3.csv', skiprows=1, delimiter=' ', dtype=str)
#     x_data = np.array(data[:,1:])
#     y_data = []
#     for i in range(data.shape[0]):
#         y_data.append(data[i, 0].split(','))
#     y_data = np.array(y_data).reshape(data.shape[0], 2)
#     x_data = np.concatenate((y_data[:,1].reshape(data.shape[0],1), x_data), axis=1).astype('float64')
#     y_data = y_data[:,0]
#     y_data = ks.utils.to_categorical(y_data, 7)
#     print(x_data.shape)
#     print(y_data.shape)
#     saveAsImg(x_data)
#     np.savetxt('./data/hw3_x.csv', x_data)
#     np.savetxt('./data/hw3_y.csv', y_data)
# def saveAsImg(array):
#     path = './data/face'
#     for i in range(array.shape[0]):
#         face = array[i].reshape(48, 48)
#         cv2.imwrite(path+'/'+'{}.jpg'.format(i), face)
#
# def loadData():
#     x_data = np.loadtxt('./data/hw3_x.csv')
#     y_data = np.loadtxt('./data/hw3_y.csv')
#     return x_data[:20000,:],x_data[20000:,:], y_data[:20000,:],y_data[20000:,:]

def readData():
    data = pd.read_csv('./data/train(3).csv')
    # data = pd.read_csv('./data/hw3.csv')
    labels = np.array(data['label'])
    features = np.array(data['feature'])
    images = []
    for i in range(features.shape[0]):
        images.append(features[i].split(' '))
    images = np.array(images, dtype=float).reshape(features.shape[0], 48, 48, 1) / 255
    labels = ks.utils.to_categorical(labels, 7)
    return images, labels

x,y = readData()
x_train = x[:20000]
y_train = y[:20000]
x_test = x[20000:]
y_test = y[20000:]

model = ks.Sequential()
#卷积
# 100x46x46
model.add(ks.layers.Convolution2D(filters=100, kernel_size=(3, 3), input_shape=(48, 48, 1)))
# 100x23x23
model.add(ks.layers.MaxPooling2D(pool_size=(2, 2)))
# 200x20x20
model.add(ks.layers.Convolution2D(filters=200, kernel_size=(4, 4)))
# 200x10x10
model.add(ks.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(ks.layers.Flatten())

#全连接
model.add(ks.layers.Dense(units=200, activation='relu'))
model.add(ks.layers.Dense(units=7, activation='softmax'))

model.summary()

model.compile(optimizer=ks.optimizers.Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, 100, 10)

train_res = model.evaluate(x_train, y_train)
print('Train Acc:%f'%(train_res[1]))
res = model.evaluate(x_test, y_test)
print('Test Acc:%f'%res[1])

model.save('hw3.h5')