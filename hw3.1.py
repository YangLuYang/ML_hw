import os
from utility import clean_data
import keras as ks
import numpy as np
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def readData():
    data = pd.read_csv('./data/train(3).csv')
    # data = pd.read_csv('./data/hw3.csv')
    labels = np.array(data['label'])
    features = np.array(data['feature'])
    images = []
    for i in range(features.shape[0]):
        images.append(features[i].split(' '))
    images = np.array(images, dtype=float).reshape(features.shape[0], 48, 48, 1) / 255.0
    labels = ks.utils.to_categorical(labels, 7)

    return images, labels

# x,y = readData()
train_data = clean_data('./data/train(3).csv')
train = train_data.feature.reshape(-1, 48, 48, 1) / 255.0
TRAIN_SIZE = 24000
x_train = train[:TRAIN_SIZE]
y_train = train_data.onehot[:TRAIN_SIZE]
x_test = train[TRAIN_SIZE:]
y_test = train_data.onehot[TRAIN_SIZE:]

model = ks.Sequential()
model.add(ks.layers.Convolution2D(filters=64, kernel_size=(3, 3), input_shape=(48, 48, 1), padding='same'))
model.add(ks.layers.Activation('relu'))
model.add(ks.layers.BatchNormalization())
model.add(ks.layers.MaxPooling2D(pool_size=(2, 2)))


model.add(ks.layers.Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
model.add(ks.layers.Activation('relu'))
model.add(ks.layers.BatchNormalization())
model.add(ks.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(ks.layers.Convolution2D(filters=256, kernel_size=(3, 3), padding='same'))
model.add(ks.layers.Activation('relu'))
model.add(ks.layers.BatchNormalization())
model.add(ks.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(ks.layers.Flatten())

model.add(ks.layers.Dense(4096, activation='relu'))
model.add(ks.layers.BatchNormalization())
model.add(ks.layers.Dropout(0.5))
model.add(ks.layers.Dense(1024, activation='relu'))
model.add(ks.layers.BatchNormalization())
model.add(ks.layers.Dropout(0.2))
model.add(ks.layers.Dense(256, activation='relu'))
model.add(ks.layers.BatchNormalization())
model.add(ks.layers.Dense(7))
model.add(ks.layers.Activation('softmax'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, 64, 40)
model.save('hw3.h5')

train_res = model.evaluate(x_train, y_train)
print('Train Acc:%f'%(train_res[1]))
res = model.evaluate(x_test, y_test)
print('Test Acc:%f'%res[1])

