##
## Generative Model
##
import numpy as np
from numpy import linalg
def pre_process():
    train_data = np.loadtxt('./data/spam_train.csv', delimiter=',', dtype=float)
    train_data = train_data[:,1:]
    x = train_data[:, :-1]
    y = train_data[:, -1]
    return x,y

def train(x, y):
    x = np.transpose(x)
    #数据总数
    num = y.shape[0]

    n1 = 0 # $==1
    n2 = 0 # $==0
    u1 = np.zeros(57).reshape(57,1)
    u2 = np.zeros(57).reshape(57,1)
    m1 = np.zeros(57 * 57).reshape(57, 57)
    m2 = np.zeros(57 * 57).reshape(57, 57)

    #计算n1 和 n2
    for i in range(num):
        if y[i] == 1:
            n1 += 1
            u1 += x[:,i].reshape(57,1)
        else:
            n2 += 1
            u2 += x[:,i].reshape(57,1)
    u1 /= n1
    u2 /= n2
    #计算sigma
    for i in range(num):
        if y[i] == 1:
            #np.tranpose() == x.T 求转置
            m1 += np.dot((x[:,i].reshape(57,1) - u1), (x[:,i].reshape(57,1) - u1).T)
        else:
            m2 += np.dot((x[:,i].reshape(57,1) - u2), (x[:,i].reshape(57,1) - u2).T)
    m1 /= n1
    m2 /= n2

    m = (m1 * (float(n1)/(n1 + n2)) + m2 * (float(n1)/(n1 + n2)))
    m_inv = linalg.inv(m)
    # print('n1=%d, n2=%d'%(n1, n2))
    # np.dot 与 np.matmul相似 均为矩阵乘法
    w = np.dot((u1 - u2).T, linalg.inv(m))
    b = - 0.5 * np.dot(np.dot(u1.T, m_inv), u1) \
        + 0.5 * np.dot(np.dot(u2.T, m_inv), u2) + np.log(float(n1) / n2)

    return w,b

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def validate(w, b):
    acc = 0
    num = y_val.shape[0]
    result = np.zeros(num)
    z = np.dot(w, x_val.T) + b
    f = sigmoid(z)
    for i in range(num):
        result[i] = np.round(f[0,i])
        if result[i] == y_val[i]:
            acc += 1.0
    return acc / num

x, y = pre_process()
TRAIN_SIZE = 3600
x_train = x[:TRAIN_SIZE, :]
y_train = y[:TRAIN_SIZE]
x_val = x[TRAIN_SIZE:, :]
y_val = y[TRAIN_SIZE:]
w,b = train(x_train, y_train)
print(validate(w, b))