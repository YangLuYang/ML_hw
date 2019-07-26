import numpy as np
def pre_process():
    data = np.loadtxt('./data/train(1).csv', skiprows=1, delimiter=',', dtype=str)
    data = np.delete(data, np.arange(3), axis=1)
    data[data == 'NR'] = '0.0'
    # print(data)
    #data[数据，小时，月份]
    data = data.astype(float).reshape((103680,)).reshape((4320,24))
    d = np.zeros(216 * 480).reshape(216, 480)
    x = np.zeros((101736, 9))
    y = []
    #20
    for j in range(240):
        d[(j//20)*18:(j//20)*18 + 18, (j%20)*24:(j%20)*24+24] = data[(j//20)*18:(j//20)*18 + 18, 0:]
    np.savetxt('data(1).txt', d)
    index = 0
    for i in range(12):
        for j in range(471):
            # print('i=%d, j=%d'%(i,j))
            x[index*18:index*18+18, :9] = d[i*18:i*18+18, j:j+9]
            y.append(d[i*18+9, j+9])
            index += 1
    return x,y

def train(x, y, epoch):
    b = 1
    #仅将前9个小时的pm2.5作为变量
    w = np.ones(9)
    learning_rate = 0.2
    reg_rate = 0.001
    #存放梯度和
    w_sum = np.ones(9)
    b_sum = 0
    for j in range(epoch):
        w_s = np.zeros(9)
        b_s = 0
        for i in range(9, 99000, 18):
            #全局计数
            count = (i - 9)//18
            #Func
            f = b + x[i, :].dot(w)
            #计算梯度
            for j in range(9):
                w_s[j] += np.sum((y[count * 9:count * 9 + 9] - f) * (-x[i, j]))
            b_s += -1 * np.sum(y[count * 9:count * 9 + 9] - f)

        # 计算平均值
        w_s /= 5500
        b_s /= 5500

        #L1正则
        for j in range(8):
            w_s[j] += reg_rate * np.sum(w)

        # AdaG
        w_sum += w_s ** 2
        b_sum += b_s ** 2

        w -= learning_rate / np.sqrt(w_sum) * w_s
        b -= learning_rate / np.sqrt(b_sum) * b_s
        print(validate(w, b))
    return w, b

def validate(w, b):
    loss = 0
    for i in range(152):
        loss += (y_val[i] - w.dot(x_val[9 + i*18, :]) - b) ** 2
    return loss/152
x, y = pre_process()
#划分训练集和验证集
x_train, y_train = x[:5500*18, :], np.array(y[:5500], dtype=float)
x_val, y_val = x[:5501*18, :], y[5500:]
#训练轮数
epoch = 100
#开始训练
w, b = train(x_train, y_train, epoch)
print('w ==> ',w)
print('b ==> %f'%b)
#计算loss
loss = validate(w, b)
print(loss)