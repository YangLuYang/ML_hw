import numpy as np

def pre_process():
    train_data = np.loadtxt('./data/spam_train.csv', delimiter=',', dtype=float)
    train_data = train_data[:,1:]
    x = train_data[:, :-1]
    x[:, -1] /= np.mean(x[:, -1])
    x[:, -2] /= np.mean(x[:, -2])
    y = train_data[:, -1]
    return x,y

def train(x, y, epoch):
    num = y.shape[0]
    w = np.ones(57)
    b = 0
    rate = 0.001
    batch = 32
    for i in range(epoch):
        grad_b = 0
        for j in range(0, batch, num):
            x = x[j*batch: (j+1)*batch]
            y = y[j*batch: (j+1)*batch]

            f = 1 / (1.0 + np.exp(-(np.dot(x, w)) + b))
            grad = np.sum(-1 * x * (y - f).reshape((batch, 1)), axis=0)
            grad_b = -np.sum(y - f)
            w = w - rate * grad
            b = b - rate * grad_b
        if i%10 == 0:
            print(validate(w, b))
    return w, b

def validate(w, b):
    num = y_val.shape[0]
    cal = 0
    for i in range(num):
        res = 1 / (1 + np.exp(-(np.dot(w, x_val[i].T) + b)))
        if res.round() == y_val[i]:
            cal += 1.0
    return cal / num


x, y = pre_process()
TRAIN_SIZE = 3200
x_train = x[:TRAIN_SIZE]
y_train = y[:TRAIN_SIZE]
x_val = x[TRAIN_SIZE:]
y_val = y[TRAIN_SIZE:]
EPOCH = 10000
w, b = train(x_train, y_train, EPOCH)
print(validate(w, b))