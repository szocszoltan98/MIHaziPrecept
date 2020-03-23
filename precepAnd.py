import numpy as np
import matplotlib.pyplot as plt
import time


def hardlim(val):
    if val < 0:
        return 0
    return 1

def init_figure(data):
    plt.ion()
    figure = plt.figure()
    figure.suptitle('And perceptron')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim((-0.5, 1.5))
    plt.ylim((-0.5, 1.5))
    plt.grid(True)
    plt.scatter(data[:, 1], data[:, 2])
    return figure


def perceptron_learning(data, expected_output):
    N, n = data.shape
    lr = .1
    w = np.random.randn(n, 1)
    E = 1
    figure = init_figure(data)
    x = np.linspace(-5, 5, 50)
    while E != 0:
        E = 0
        for i in range(N):
            yi = hardlim(np.dot(data[i], w))
            ei = expected_output[i] - yi
            w += lr * ei * data[i].reshape(n, 1)
            E += ei ** 2
        a = [0, -w[0] / w[2]]
        c = [-w[0] / w[1], 0]
        m = (a[1] - a[0]) / (c[1] - c[0])
        line, = plt.plot(x, x * m + a[1])
        line.set_ydata(x * m + a[1])
        figure.canvas.draw()
        time.sleep(0.5)
        line.remove()
        figure.canvas.flush_events()

def main ():
    data_array = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
    label_array = [0, 0, 0, 1]
    perceptron_learning(data_array, label_array)

main()
