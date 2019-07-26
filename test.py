import tensorflow as tf
if __name__ == "__main__":
    # hello = tf.constant('Hello, tensorflow!')
    # sess = tf.Session()
    # print(sess.run(hello))
    # 基本常量操作
    # T构造函数返回的值就是常量节点(Constant op)的输出.
    a = tf.constant(2)
    b = tf.constant(3)

    # 启动默认的计算图
    with tf.Session() as sess:
        print("a=2, b=3")
        print("常量节点相加: %i" % sess.run(a + b))
        print("常量节点相乘: %i" % sess.run(a * b))

    # 使用变量(variable)作为计算图的输入
    # 构造函数返回的值代表了Variable op的输出 (session运行的时候，为session提供输入)
    # tf Graph input
    a = tf.placeholder(tf.int16)
    b = tf.placeholder(tf.int16)

    # 定义一些操作
    add = tf.add(a, b)
    mul = tf.multiply(a, b)

    # 启动默认会话
    with tf.Session() as sess:
        # 把运行每一个操作，把变量输入进去
        print("变量相加: %i" % sess.run(add, feed_dict={a: 2, b: 3}))
        print("变量相乘: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))

    # 矩阵相乘(Matrix Multiplication)
    # 创建一个 Constant op ，产生 1x2 matrix.
    # 该op会作为一个节点被加入到默认的计算图
    # 构造器返回的值 代表了Constant op的输出
    matrix1 = tf.constant([[3., 3.]])
    # 创建另一个 Constant op 产生  2x1 矩阵.
    matrix2 = tf.constant([[2.], [2.]])
    # 创建一个 Matmul op 以 'matrix1' 和 'matrix2' 作为输入.
    # 返回的值, 'product', 表达了矩阵相乘的结果
    product = tf.matmul(matrix1, matrix2)
    # 为了运行 matmul op 我们调用 session 的 'run()' 方法, 传入 'product'
    # ‘product’表达了 matmul op的输出. 这表明我们想要取回(fetch back)matmul op的输出
    # op 需要的所有输入都会由session自动运行. 某些过程可以自动并行执行
    #
    # 调用 'run(product)' 就会引起计算图上三个节点的执行：2个 constants 和一个 matmul.
    # ‘product’op 的输出会返回到 'result'：一个 numpy `ndarray` 对象.
    with tf.Session() as sess:
        result = sess.run(product)
        print('矩阵相乘的结果：', result)
        # ==> [[ 12.]]